"""
droplet_geometry -- training-style droplet geometry for the inference pipeline.

Inference builds droplets by thresholding the model's droplet head and taking
connected components. That is not how training defined a droplet, and on real
data it produced merged parents of ~13 million pixels. This module instead
reuses the training notebook's own geometry (NPC watershed inventory, radial
wall extraction, RANSAC circle fits, robust sphere profile, `p2_circle`
policy) on the raw image, without touching any nuclear mask.

Three steps, each resumable:

    compute_geometry      droplet geometry for every timepoint, cached per
                          timepoint with code / configuration / input provenance
    assign_components     reassign each saved nuclear component to a fitted
                          droplet support and re-measure enrichment / solidity
    (droplet_population)  apply the rules and build the population tables

Nothing here runs neural inference, retrains anything, or opens a mask, image
or baseline table for writing. Images and masks are memory-mapped read-only.

Two different Z scales appear and must not be merged:
    geometry_z_step_um   empirical axial scale of the droplet sphere fit (2.18)
    cfg.z_step_um        acquisition Z spacing (2.0); not used here at all
"""

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .droplet_population import (GEOM_AMBIGUOUS, GEOM_ASSIGNED, GEOM_MISSING,
                                 GEOM_NO_TIMEPOINT, GEOM_PARTIAL, OBSERVED, PREDICTED)

# The training functions that make up the geometry workflow, in the order the
# reviewed audit hashed them. Changing this tuple changes the code hash and so
# invalidates every geometry cache; do it deliberately.
GEOMETRY_FUNCTION_NAMES = (
    "extract_plane", "clip_histogram", "_circularity", "detect_droplets_npc_watershed",
    "_circle_from_3", "_fit_circle_kasa", "fit_circle_ransac", "smooth_npc_plane",
    "wall_points_on_smoothed", "fit_droplet_circle_on_smoothed", "_sphere_lsq",
    "fit_sphere_profile", "fit_sphere_ransac", "compute_droplet_geometry",
    "sphere_radius_px", "predicted_circle", "p2_circle")

GEOMETRY_MANIFEST = "geometry_signature.json"
COMPONENT_SCHEMA_VERSION = 1

CIRCLE_COLUMNS = ["t", "z", "parent", "cx", "cy", "radius_px", "support_pixels", "source"]
COMPONENT_COLUMNS = [
    "t", "z", "label", "nucleus_3d_id", "area_pixels", "parent_candidates",
    "candidate_parents", "parent", "parent_source", "parent_overlap_fraction",
    "geometry_status", "geometry_gate_pass", "contained_gate_pass", "enrichment",
    "solidity", "nucleus_mean", "cytoplasm_mean", "ring_pixels", "reference_ok",
    "gate_reason"]


# ── Enrichment / solidity gate (one canonical copy of the training formula) ──
@dataclass(frozen=True)
class ArtifactGateSettings:
    min_enrichment: float = 1.20
    min_solidity: float = 0.80
    ring_gap_um: float = 0.5
    ring_width_um: float = 2.0
    min_ring_pixels: int = 50


def measure_segment_gate(image, candidate, support, raw_foreground, pixel_size_um,
                         settings=ArtifactGateSettings(), solidity=None):
    """Raw-NLS enrichment and solidity of one candidate nucleus.

    Training formula, unchanged. All masks share one crop that is padded by at
    least the outer ring radius. The reference ring is the band between
    `ring_gap_um` and `ring_gap_um + ring_width_um` outside the candidate,
    restricted to the droplet support and with raw Otsu foreground removed.
    Returns (measurements, ring mask). `gate_pass` is False when the reference
    ring is too small to measure; that is "unassessable", not an artifact.
    """
    from skimage import measure, morphology

    gap = max(1, int(np.ceil(settings.ring_gap_um / pixel_size_um)))
    outer = gap + max(1, int(np.ceil(settings.ring_width_um / pixel_size_um)))
    ring = (morphology.binary_dilation(candidate, morphology.disk(outer)) &
            ~morphology.binary_dilation(candidate, morphology.disk(gap)) &
            support & ~raw_foreground)
    nr = int(ring.sum())
    inside = float(np.mean(image[candidate], dtype=np.float64)) if candidate.any() else np.nan
    outside = float(np.mean(image[ring], dtype=np.float64)) if nr else np.nan
    valid = nr >= settings.min_ring_pixels and np.isfinite(outside) and outside > 0 and np.isfinite(inside)
    enrichment = inside / outside if valid else np.nan
    if solidity is None:
        regions = measure.regionprops(candidate.astype(np.uint8))
        solidity = float(regions[0].solidity) if regions else np.nan
    reasons = []
    if not valid:
        reasons.append("unassessable_cytoplasm")
    elif enrichment < settings.min_enrichment:
        reasons.append("enrichment")
    if not np.isfinite(solidity) or solidity < settings.min_solidity:
        reasons.append("solidity")
    return dict(enrichment=enrichment, solidity=solidity, nucleus_mean=inside,
                cytoplasm_mean=outside, ring_pixels=nr, reference_ok=bool(valid),
                gate_pass=not reasons,
                gate_reason="keep" if not reasons else " + ".join(reasons)), ring


# ── Loading the training geometry ───────────────────────────────────────────
@dataclass
class TrainingGeometry:
    """The training notebook's geometry functions plus the settings they read."""
    namespace: dict
    config: SimpleNamespace
    settings: dict
    code_sha256: str
    notebook: Optional[Path] = None


def load_training_geometry(training_notebook, pixel_size_um: float,
                           npc_channel_index: int) -> TrainingGeometry:
    """Lift the named geometry functions and declared defaults out of the notebook.

    The notebook is parsed, never run: only the function definitions listed in
    GEOMETRY_FUNCTION_NAMES are compiled, and only literal defaults named in
    PipelineConfig._GEOM_FIELDS (plus erosion_px) are read. No training cell,
    setup cell or pipeline object is executed. Pixel size and NPC channel come
    from the inference configuration, not from the training notebook.
    """
    from scipy import ndimage
    from skimage import filters, measure, morphology, segmentation
    from skimage.morphology import h_maxima

    training_notebook = Path(training_notebook)
    source = training_notebook.read_text(encoding="utf8")
    functions, config = {}, None
    for cell in json.loads(source)["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = ast.get_source_segment(text, node)
            if isinstance(node, ast.ClassDef) and node.name == "PipelineConfig":
                config = node
    absent = [name for name in GEOMETRY_FUNCTION_NAMES if name not in functions]
    if absent:
        raise ValueError(f"{training_notebook.name} does not define: {absent}")
    if config is None:
        raise ValueError(f"{training_notebook.name} has no PipelineConfig class")

    fields = {n.target.id: n.value for n in config.body if isinstance(n, ast.AnnAssign)}
    geom_fields = next(
        (ast.literal_eval(n.value) for n in config.body
         if isinstance(n, ast.Assign)
         and any(isinstance(t, ast.Name) and t.id == "_GEOM_FIELDS" for t in n.targets)),
        None)
    if geom_fields is None:
        raise ValueError("Training PipelineConfig declares no _GEOM_FIELDS")
    settings = {k: ast.literal_eval(fields[k]) for k in tuple(geom_fields) + ("erosion_px",)}
    settings.update(pixel_size_um=float(pixel_size_um), npc_channel_idx=int(npc_channel_index))

    gc = SimpleNamespace(**settings)
    gc.min_droplet_area_px = lambda: gc.min_droplet_area_um2 / gc.pixel_size_um ** 2
    selected = "\n\n".join(functions[name] for name in GEOMETRY_FUNCTION_NAMES)
    namespace = dict(np=np, filters=filters, morphology=morphology, measure=measure,
                     segmentation=segmentation, ndimage=ndimage, h_maxima=h_maxima,
                     copy=copy, itertools=itertools, cfg=gc)
    exec(compile(selected, str(training_notebook) + ":geometry-only", "exec"), namespace)
    return TrainingGeometry(namespace=namespace, config=gc, settings=settings,
                            code_sha256=hashlib.sha256(selected.encode()).hexdigest(),
                            notebook=training_notebook)


# ── Provenance ──────────────────────────────────────────────────────────────
def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _file_identity(path) -> list:
    path = Path(path)
    st = path.stat()
    return [str(path.resolve()), st.st_size, st.st_mtime_ns]


def geometry_signature(training: TrainingGeometry, image_path) -> dict:
    """What a cached geometry depends on: settings, function source, input file.

    Same layout as the reviewed section-38g audit, so its caches stay usable.
    """
    return dict(settings=training.settings, code_sha256=training.code_sha256,
                image=_file_identity(image_path))


def signature_hash(signature: Mapping) -> str:
    return hashlib.sha256(_canonical(signature).encode()).hexdigest()[:12]


def _same_signature(a: Mapping, b: Mapping) -> bool:
    return json.loads(_canonical(a)) == json.loads(_canonical(b))


def _atomic_pickle(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    pd.to_pickle(obj, tmp)
    os.replace(tmp, path)


def _prepare_cache_dir(cache_dir: Path, signature: Mapping) -> None:
    """Create the cache, or verify that an existing one has matching provenance."""
    manifest = cache_dir / GEOMETRY_MANIFEST
    if manifest.exists():
        if not _same_signature(json.loads(manifest.read_text(encoding="utf8")), signature):
            raise ValueError(
                f"Geometry cache provenance mismatch in {cache_dir}. The training geometry "
                "code, its settings, or the input image changed. Use a fresh cache directory.")
        return
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise ValueError(f"{cache_dir} holds files but no {GEOMETRY_MANIFEST}; use a fresh directory")
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(signature, indent=2, default=str), encoding="utf8")


def find_geometry_caches(search_root, signature: Mapping) -> pd.DataFrame:
    """Earlier geometry caches under `search_root` and whether they match.

    Looks for geometry_signature.json up to two folders deep (this is where the
    section-38g audit left its `ransac_droplet_<timestamp>` folders). Read-only.
    """
    rows = []
    root = Path(search_root)
    if root.exists():
        found = sorted(set(root.glob(f"*/{GEOMETRY_MANIFEST}")) |
                       set(root.glob(f"*/*/{GEOMETRY_MANIFEST}")))
        for manifest in found:
            try:
                same = _same_signature(json.loads(manifest.read_text(encoding="utf8")), signature)
            except (OSError, ValueError):
                same = False
            times = sorted(int(p.stem.split("_t")[-1]) for p in manifest.parent.glob("geometry_t*.pkl"))
            rows.append(dict(cache_dir=manifest.parent, provenance_matches=same, timepoints=times))
    return pd.DataFrame(rows, columns=["cache_dir", "provenance_matches", "timepoints"])


# ── Step 1: geometry for every timepoint ────────────────────────────────────
def _geometry_for_timepoint(hs, t: int, training: TrainingGeometry, progress: bool):
    ns, gc = training.namespace, training.config
    plane = ns["extract_plane"](hs, t, gc.inventory_ref_z, gc.npc_channel_idx)
    inventory = ns["detect_droplets_npc_watershed"](plane, gc, compact=True)
    if not inventory:
        raise RuntimeError("T%d: no droplet inventory; refusing to fall back to model droplets" % t)
    geometry = ns["compute_droplet_geometry"](hs, t, inventory, gc, progress=progress)
    return geometry, len(inventory)


def _geometry_worker(job: dict) -> dict:
    """Compute and cache one timepoint. Module-level so a spawned process can run it."""
    import tifffile

    started = time.perf_counter()
    t = int(job["t"])
    try:
        training = load_training_geometry(job["training_notebook"], job["pixel_size_um"],
                                          job["npc_channel_index"])
        if training.code_sha256 != job["code_sha256"]:
            raise RuntimeError("Training geometry source changed while the stage was running")
        hs = tifffile.memmap(job["image_path"], mode="r")
        geometry, n_inventory = _geometry_for_timepoint(hs, t, training, progress=False)
        _atomic_pickle(dict(geometry=geometry, n_inventory=n_inventory), Path(job["cache_file"]))
        return dict(t=t, error=None, seconds=time.perf_counter() - started)
    except Exception as exc:  # reported to the parent, which decides what to do
        return dict(t=t, error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - started)


def compute_geometry(image_path, training_notebook=None, *, pixel_size_um: float,
                     npc_channel_index: int, cache_dir, timepoints: Optional[Iterable[int]] = None,
                     reuse_cache_dirs: Sequence = (), n_workers: int = 1,
                     on_error: str = "raise", progress: bool = True,
                     training: Optional[TrainingGeometry] = None) -> dict:
    """Training-style droplet geometry for every requested timepoint.

    timepoints=None means every timepoint in the image. Each finished timepoint
    is written to `cache_dir/geometry_tNNN.pkl`; an interrupted run resumes
    from the files already there. The cache carries a provenance signature
    (geometry settings, source of the geometry functions, image path/size/mtime)
    and is refused if it does not match.

    reuse_cache_dirs: older caches (for example the T0-T2 audit folders). A
    timepoint missing from `cache_dir` is copied from the first of these whose
    provenance matches; a mismatched one is ignored with a message.

    n_workers > 1 computes timepoints in separate spawned processes (never
    forked: the notebook kernel has TensorFlow loaded). Each worker holds one
    timepoint's smoothed NPC planes in memory. Per-plane progress is only
    printed when n_workers == 1.

    on_error="raise" stops at the first timepoint that fails, after the others
    in flight have been cached. on_error="record" finishes the rest and lists
    the failure in the coverage table; that timepoint's components then come
    out as unresolved (`no_timepoint_geometry`), never as artifacts.
    """
    import tifffile

    if on_error not in ("raise", "record"):
        raise ValueError("on_error must be 'raise' or 'record'")
    if training is None:
        if training_notebook is None:
            raise ValueError("Give training_notebook (or a loaded TrainingGeometry)")
        training = load_training_geometry(training_notebook, pixel_size_um, npc_channel_index)
    elif int(n_workers) > 1 and training.notebook is None:
        raise ValueError("n_workers > 1 needs a TrainingGeometry loaded from a notebook file")
    gc = training.config
    hs = tifffile.memmap(str(image_path), mode="r")
    if hs.ndim != 5:
        raise ValueError(f"Expected a TZCYX hyperstack, got shape {hs.shape}")
    if not 0 <= gc.inventory_ref_z < hs.shape[1]:
        raise ValueError("Training inventory_ref_z lies outside the stack")
    ts = sorted(set(range(hs.shape[0]) if timepoints is None else map(int, timepoints)))
    if not ts or ts[0] < 0 or ts[-1] >= hs.shape[0]:
        raise ValueError(f"Invalid timepoints {ts} for an image with {hs.shape[0]} timepoints")

    signature = geometry_signature(training, image_path)
    cache_dir = Path(cache_dir)
    _prepare_cache_dir(cache_dir, signature)
    if progress:
        print(f"Geometry cache (resumable): {cache_dir}", flush=True)

    donors = []
    for other in reuse_cache_dirs:
        other = Path(other)
        manifest = other / GEOMETRY_MANIFEST
        if other.resolve() == cache_dir.resolve():
            continue
        if manifest.exists() and _same_signature(json.loads(manifest.read_text(encoding="utf8")), signature):
            donors.append(other)
        elif progress:
            print(f"Not reusing {other}: provenance missing or different", flush=True)

    files = {t: cache_dir / ("geometry_t%03d.pkl" % t) for t in ts}
    origin, seconds, errors = {}, {}, {}
    todo = []
    for t in ts:
        if files[t].exists():
            origin[t] = "cache"
            continue
        donor = next((d / files[t].name for d in donors if (d / files[t].name).exists()), None)
        if donor is not None:
            _atomic_pickle(pd.read_pickle(donor), files[t])
            origin[t] = f"reused from {donor.parent.name}"
            continue
        todo.append(t)
    if progress:
        print(f"Timepoints: {len(ts)} requested, {len(ts) - len(todo)} already cached, "
              f"{len(todo)} to compute", flush=True)

    def finished(t, error, took):
        seconds[t] = took
        if error is None:
            origin[t] = "computed"
        else:
            origin[t], errors[t] = "failed", error
        if progress:
            done = sum(1 for x in todo if x in seconds)
            mean = float(np.mean([seconds[x] for x in todo if x in seconds]))
            left = (len(todo) - done) * mean / max(1, min(int(n_workers), len(todo)))
            state = "done" if error is None else f"FAILED ({error})"
            print(f"T{t} geometry {state} in {took / 60:.1f} min | {done}/{len(todo)} computed, "
                  f"about {left / 60:.0f} min left", flush=True)

    if todo and int(n_workers) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        jobs = [dict(t=t, training_notebook=str(training.notebook), pixel_size_um=float(pixel_size_um),
                     npc_channel_index=int(npc_channel_index), code_sha256=training.code_sha256,
                     image_path=str(image_path), cache_file=str(files[t])) for t in todo]
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(int(n_workers), len(todo)), mp_context=context) as pool:
            for future in as_completed([pool.submit(_geometry_worker, job) for job in jobs]):
                result = future.result()
                finished(result["t"], result["error"], result["seconds"])
    else:
        for t in todo:
            started = time.perf_counter()
            try:
                geometry, n_inventory = _geometry_for_timepoint(hs, t, training, progress)
                _atomic_pickle(dict(geometry=geometry, n_inventory=n_inventory), files[t])
                finished(t, None, time.perf_counter() - started)
            except Exception as exc:
                finished(t, f"{type(exc).__name__}: {exc}", time.perf_counter() - started)
                if on_error == "raise":
                    raise
    if errors and on_error == "raise":
        raise RuntimeError("Geometry failed for timepoint(s): " +
                           "; ".join(f"T{t}: {e}" for t, e in sorted(errors.items())))

    geometry, coverage = {}, []
    for t in ts:
        if t in errors:
            coverage.append(dict(t=t, status="failed", inventory=0, validated_geometry=0,
                                 origin=origin[t], minutes=seconds.get(t, np.nan) / 60,
                                 error=errors[t]))
            continue
        saved = pd.read_pickle(files[t])
        geometry[t] = saved["geometry"]
        coverage.append(dict(t=t, status="ok", inventory=int(saved["n_inventory"]),
                             validated_geometry=len(saved["geometry"]), origin=origin[t],
                             minutes=seconds.get(t, np.nan) / 60, error=""))
    return dict(geometry=geometry, coverage=pd.DataFrame(coverage), signature=signature,
                signature_hash=signature_hash(signature), cache_dir=cache_dir,
                settings=dict(training.settings), training=training,
                image_path=Path(image_path), timepoints=ts)


# ── Step 2: reassign components to fitted supports ──────────────────────────
def circle_crop(cx, cy, radius, shape):
    """Bounding box of a disc clipped to the image, and the disc mask inside it."""
    h, w = shape
    a, b = max(0, int(np.floor(cy - radius))), max(0, int(np.floor(cx - radius)))
    c, d = min(h, int(np.ceil(cy + radius)) + 1), min(w, int(np.ceil(cx + radius)) + 1)
    a, b, c, d = min(a, h), min(b, w), max(c, 0), max(d, 0)
    yy, xx = np.ogrid[a:c, b:d]
    return (a, b, c, d), (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def plane_circles(geometry_t: Mapping, z: int, shape, training: TrainingGeometry) -> list:
    """Droplet circles on one plane under the training `p2_circle` policy.

    A valid observed circle is used where the plane is a sphere inlier;
    otherwise the sphere prediction. `source` records which, so a predicted
    circle is never presented as an observed boundary. `r` is the un-eroded
    radius; `mask` is the support eroded by the training `erosion_px`.
    """
    gc = training.config
    circles = []
    for did, g in geometry_t.items():
        selected = training.namespace["p2_circle"](g, z, gc)
        if selected is None:
            continue
        kind, circ = selected
        if kind not in ("fitted", "predicted"):
            raise ValueError("Unsupported training p2_circle contract")
        if circ[2] <= gc.erosion_px:
            continue
        cx, cy, r = map(float, circ)
        box, mask = circle_crop(cx, cy, r - gc.erosion_px, shape)
        if mask.any():
            circles.append(dict(parent=int(did) + 1, cx=cx, cy=cy, r=r,
                                source=OBSERVED if kind == "fitted" else PREDICTED,
                                box=box, mask=mask))
    return circles


def included_planes(segmentation_index: pd.DataFrame, min_z: int) -> set:
    """(t, z) planes that segmentation committed and that are at or above min_z."""
    included = segmentation_index["included"].astype(str).str.lower().isin(["true", "1"])
    planes = segmentation_index.loc[included & (segmentation_index.z >= int(min_z)), ["t", "z"]]
    return set(map(tuple, planes.astype(int).to_numpy()))


def _assign_timepoint(hs, labels, group_t: pd.DataFrame, geometry_t: Optional[Mapping],
                      training: TrainingGeometry, nuclear_channel_index: int,
                      pixel_size_um: float, params: ArtifactGateSettings,
                      min_parent_overlap: float, progress: bool):
    from skimage import filters, measure

    gc = training.config
    t = int(group_t.t.iloc[0])
    rows, circle_rows = [], []
    gap = max(1, int(np.ceil(params.ring_gap_um / pixel_size_um)))
    pad = gap + max(1, int(np.ceil(params.ring_width_um / pixel_size_um)))
    for z, group in group_t.groupby("z", sort=True):
        z = int(z)
        lab = np.asarray(labels[t, z])
        regions = {r.label: r for r in measure.regionprops(lab)}
        circles, assignments, support = [], {}, {}
        if geometry_t is not None:
            raw = np.asarray(hs[t, z, nuclear_channel_index])
            circles = plane_circles(geometry_t, z, lab.shape, training)
            for circle in circles:
                did = circle["parent"]
                a, b, c, d = circle["box"]
                mask = circle["mask"]
                ids, counts = np.unique(lab[a:c, b:d][mask], return_counts=True)
                for label, count in zip(ids, counts):
                    if label:
                        assignments.setdefault(int(label), []).append((int(count), did))
                values = raw[a:c, b:d][mask]
                threshold = (float(filters.threshold_otsu(values))
                             if values.max() > values.min() else np.nan)
                support[did] = (circle, threshold)
                circle_rows.append(dict(t=t, z=z, parent=did, cx=circle["cx"], cy=circle["cy"],
                                        radius_px=circle["r"], support_pixels=int(mask.sum()),
                                        source=circle["source"]))
        for row in group.itertuples():
            label = int(row.label)
            region = regions.get(label)
            if region is None or int(region.area) != int(row.area_px):
                raise ValueError(
                    f"T{t} Z{z} label {label}: the object table disagrees with the saved nuclear "
                    "masks. They must come from the same segmentation run.")
            candidates = sorted(assignments.get(label, []), reverse=True)
            eligible = [(n, did) for n, did in candidates if n / region.area >= min_parent_overlap]
            rec = dict(
                t=t, z=z, label=label, nucleus_3d_id=int(row.nucleus_3d_id),
                area_pixels=int(region.area), parent_candidates=len(candidates),
                candidate_parents=";".join("%d:%.4f" % (did, n / region.area) for n, did in candidates),
                parent=0, parent_source="none", parent_overlap_fraction=0.0,
                geometry_status=GEOM_MISSING if geometry_t is not None else GEOM_NO_TIMEPOINT,
                geometry_gate_pass=None, contained_gate_pass=None, enrichment=np.nan,
                solidity=float(region.solidity), nucleus_mean=np.nan, cytoplasm_mean=np.nan,
                ring_pixels=0, reference_ok=False, gate_reason="unassessable geometry")
            # More than one circle holding the component is ambiguous, as is an
            # exact tie for the largest share. The component is then left
            # unassigned; it is never handed to one of the candidates.
            tied = not eligible and len(candidates) > 1 and candidates[0][0] == candidates[1][0]
            if candidates and len(eligible) <= 1 and not tied:
                n, did = eligible[0] if eligible else candidates[0]
                circle, threshold = support[did]
                overlap = n / region.area
                a, b, c, d = region.bbox
                a, b = max(0, a - pad), max(0, b - pad)
                c, d = min(lab.shape[0], c + pad), min(lab.shape[1], d + pad)
                yy, xx = np.ogrid[a:c, b:d]
                inside = ((xx - circle["cx"]) ** 2 + (yy - circle["cy"]) ** 2
                          <= (circle["r"] - gc.erosion_px) ** 2)
                crop = raw[a:c, b:d]
                candidate = lab[a:c, b:d] == label
                foreground = ((crop > threshold) & inside if np.isfinite(threshold)
                              else np.zeros_like(inside))
                metric, _ = measure_segment_gate(crop, candidate, inside, foreground,
                                                 pixel_size_um, params, float(region.solidity))
                gate_pass = bool(metric.pop("gate_pass"))
                rec.update(metric)
                rec.update(parent=did, parent_source=circle["source"],
                           parent_overlap_fraction=overlap,
                           geometry_status=GEOM_ASSIGNED if overlap >= min_parent_overlap else GEOM_PARTIAL,
                           geometry_gate_pass=gate_pass,
                           contained_gate_pass=bool(gate_pass and overlap >= min_parent_overlap))
            elif len(eligible) > 1 or tied:
                rec["geometry_status"] = GEOM_AMBIGUOUS
            rows.append(rec)
        if progress:
            print("  T%d Z%d: %d fitted/predicted supports, %d nuclear components"
                  % (t, z, len(circles), len(group)), flush=True)
    return (pd.DataFrame(rows, columns=COMPONENT_COLUMNS),
            pd.DataFrame(circle_rows, columns=CIRCLE_COLUMNS))


def _concat(frames, columns) -> pd.DataFrame:
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def assign_components(image_path, labels_path, components: pd.DataFrame, geometry: Mapping, *,
                      nuclear_channel_index: int, pixel_size_um: float,
                      gate_settings: ArtifactGateSettings = ArtifactGateSettings(),
                      min_parent_overlap: float = 0.85, valid_planes: Optional[set] = None,
                      cache_dir=None, progress: bool = True) -> dict:
    """Reassign saved nuclear components to fitted droplet supports and re-measure them.

    `components` is the pipeline's linked object table (grouped_z_df): one row
    per saved nuclear instance with t, z, label, area_px and nucleus_3d_id. It
    is read, never changed. `geometry` is the result of compute_geometry.

    For each component the fitted (eroded) circles covering it are counted:
        assigned          one circle holds >= min_parent_overlap of its pixels
        partial_parent    its best circle holds less than that
        ambiguous_parent  two or more circles hold >= min_parent_overlap, or
                          the best share is an exact tie
        missing_parent    no fitted circle touches it
    `min_parent_overlap` (0.85) is an exploratory figure: it decides whether a
    parent counts as resolved, it is not a validated biological cut-off.

    Circles are evaluated on the planes that carry at least one nuclear
    component, which is exactly the set the reviewed overlap diagnostic used.

    Per-timepoint results are cached under `cache_dir` and reused only when the
    geometry signature, mask file, gate settings and that timepoint's object
    rows are all unchanged.
    """
    import tifffile

    if not 0 < min_parent_overlap <= 1:
        raise ValueError("min_parent_overlap must be in (0, 1]")
    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive")
    need = {"t", "z", "label", "area_px", "nucleus_3d_id"}
    if need - set(components.columns):
        raise KeyError(f"components is missing columns: {sorted(need - set(components.columns))}")
    if components.duplicated(["t", "z", "label"]).any():
        raise ValueError("Duplicate component keys (t, z, label)")
    hs = tifffile.memmap(str(image_path), mode="r")
    labels = tifffile.memmap(str(labels_path), mode="r")
    if hs.shape[:2] + hs.shape[-2:] != labels.shape:
        raise ValueError(f"Raw image {hs.shape} and nuclear masks {labels.shape} do not match")
    training = geometry["training"]
    wanted = set(geometry["timepoints"])
    present = set(components.t.astype(int))
    if not present <= wanted:
        raise ValueError(f"No geometry was requested for timepoint(s) {sorted(present - wanted)}. "
                         "Run compute_geometry on them, or pass only the rows you want judged.")
    if valid_planes is not None:
        planes = set(map(tuple, components[["t", "z"]].drop_duplicates().astype(int).to_numpy()))
        bad = sorted(planes - set(valid_planes))
        if bad:
            raise ValueError(f"Components lie on planes segmentation did not include: {bad[:5]}")

    table = components[["t", "z", "label", "area_px", "nucleus_3d_id"]].sort_values(["t", "z", "label"]).reset_index(drop=True)
    shared = dict(schema=COMPONENT_SCHEMA_VERSION, geometry=geometry["signature_hash"],
                  labels=_file_identity(labels_path), gate=asdict(gate_settings),
                  min_parent_overlap=float(min_parent_overlap),
                  nuclear_channel_index=int(nuclear_channel_index),
                  pixel_size_um=float(pixel_size_um))
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        if progress:
            print(f"Component cache (resumable): {cache_dir}", flush=True)

    frames, circle_frames, hashes = [], [], {}
    for t, group_t in table.groupby("t", sort=True):
        t = int(t)
        rows_hash = hashlib.sha256(pd.util.hash_pandas_object(group_t, index=False).values.tobytes()).hexdigest()
        signature = dict(shared, t=t, rows=rows_hash,
                         has_geometry=t in geometry["geometry"])
        hashes[t] = signature_hash(signature)
        cache_file = cache_dir / ("components_t%03d.pkl" % t) if cache_dir is not None else None
        if cache_file is not None and cache_file.exists():
            saved = pd.read_pickle(cache_file)
            if _same_signature(saved.get("signature", {}), signature):
                frames.append(saved["components"])
                circle_frames.append(saved["circles"])
                if progress:
                    print(f"T{t}: loaded {len(saved['components'])} cached component rows", flush=True)
                continue
            if progress:
                print(f"T{t}: cached components have different provenance; recomputing", flush=True)
        started = time.perf_counter()
        comp, circ = _assign_timepoint(hs, labels, group_t, geometry["geometry"].get(t), training,
                                       int(nuclear_channel_index), float(pixel_size_um),
                                       gate_settings, float(min_parent_overlap), progress)
        if cache_file is not None:
            _atomic_pickle(dict(signature=signature, components=comp, circles=circ), cache_file)
        frames.append(comp)
        circle_frames.append(circ)
        if progress:
            print(f"T{t}: {len(comp)} components against {circ.parent.nunique() if len(circ) else 0} "
                  f"droplets in {(time.perf_counter() - started) / 60:.1f} min", flush=True)

    result = _concat(frames, COMPONENT_COLUMNS)
    circles = _concat(circle_frames, CIRCLE_COLUMNS)
    # Everything the population inherits from the measurement steps. It goes into
    # the population tag, so two populations measured differently never share a folder.
    upstream = dict(geometry=geometry["signature_hash"],
                    components=signature_hash(dict(shared, per_t=hashes)),
                    timepoints=sorted(present),
                    gate_measurement=asdict(gate_settings),
                    assignment_min_containment=float(min_parent_overlap),
                    geometry_settings=dict(geometry["settings"]),
                    image=str(Path(image_path)), nuclear_masks=str(Path(labels_path)))
    return dict(components=result, circles=circles, upstream=upstream, cache_dir=cache_dir)
