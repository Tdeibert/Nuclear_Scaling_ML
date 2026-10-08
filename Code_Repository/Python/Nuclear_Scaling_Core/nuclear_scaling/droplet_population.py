"""
droplet_population -- the selected analysis population for the Vulcan pipeline.

This module turns per-component measurements (one row per segmented nuclear
component on one Z plane) into three explicit, mutually exclusive tables:

    accepted     passes every enabled rule against a resolved parent droplet
    excluded     fails at least one definite rule; the reason is recorded
    unresolved   cannot be judged (no parent, ambiguous parent, ...);
                 NOT a confirmed artifact and NOT part of the accepted set

It never touches images, masks or the pipeline's baseline tables: everything
here is numpy/pandas on tables that `droplet_geometry` produced. Baseline
tables are filtered on the way out with `select_population`; they are never
rewritten.

What is approved, and what is not
---------------------------------
* APPROVED (user review, 2026-10): whole-droplet exclusion when two fitted
  droplet circles overlap by MORE THAN 5 % of the smaller full circle.
    - both circles must be observed sphere-inlier fits; a sphere-predicted
      circle can never trigger the rule
    - un-eroded circles, analytic intersection, no clipping to the image
    - strict comparison: overlap > tolerance
    - a hit on any evaluated Z plane flags BOTH droplet IDs on every Z plane
      of that timepoint; identity is never carried across timepoints
* EXISTING GATE (training-like thresholds, formulas preserved):
  raw-NLS enrichment >= 1.20 and solidity >= 0.80.
* EXPLORATORY (kept visible, never promoted silently):
    - the 85 % containment figure. It is used only to decide whether a
      component's parent droplet is resolved. A component below it is routed
      to `unresolved` (reason `partial_parent`), never called an artifact.
    - removal of exactly-one-pixel components (off by default).
  Probability and NPC-contrast cut-offs are deliberately absent: neither
  separated the reviewed artifacts from the reviewed real nuclei.

Known gaps this module does NOT close
-------------------------------------
* Z-linking (`group_nuclei_across_z`) is used as-is. A linked ID is not
  guaranteed to be one biological nucleus; branch flags are reported.
* Droplets holding more than one nucleus are not excluded.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

# ── Vocabulary ──────────────────────────────────────────────────────────────
OBSERVED = "observed_inlier"       # circle fitted on a sphere-inlier plane
PREDICTED = "sphere_predicted"     # circle implied by the sphere, not observed

STATUS_ACCEPTED = "accepted"
STATUS_EXCLUDED = "excluded"
STATUS_UNRESOLVED = "unresolved"
STATUSES = (STATUS_ACCEPTED, STATUS_EXCLUDED, STATUS_UNRESOLVED)

# Parent-assignment states written by droplet_geometry.assign_components.
GEOM_ASSIGNED = "assigned"
GEOM_PARTIAL = "partial_parent"
GEOM_MISSING = "missing_parent"
GEOM_AMBIGUOUS = "ambiguous_parent"
GEOM_NO_TIMEPOINT = "no_timepoint_geometry"
GEOMETRY_STATUSES = (GEOM_ASSIGNED, GEOM_PARTIAL, GEOM_MISSING, GEOM_AMBIGUOUS,
                     GEOM_NO_TIMEPOINT)

# Definite exclusion reasons.
REASON_OVERLAP = "overlapping_droplet_geometry"
REASON_ENRICHMENT = "enrichment"
REASON_SOLIDITY = "solidity"
REASON_ONE_PIXEL = "one_pixel_component"
# Reasons a component cannot be judged (in addition to the GEOM_* states).
REASON_NO_REFERENCE = "unassessable_cytoplasm"

COMPONENT_KEYS = ["t", "z", "label"]
ID_KEYS = ["t", "nucleus_3d_id"]

PAIR_COLUMNS = ["t", "z", "parent_a", "parent_b", "overlap_fraction",
                "observed_pair", "source_a", "source_b"]
FLAG_COLUMNS = ["t", "parent", "max_observed_overlap", "evidence_planes",
                "evidence_z", "partners", "reason"]

POPULATIONS = ("baseline", "accepted", "excluded", "unresolved",
               "accepted_plus_unresolved")
_SEP = " + "


# ── Settings ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PopulationSettings:
    """Every rule that decides the selected population, in one place.

    Deliberately separate from the notebook's PipelineConfig: that dataclass
    is hashed into the run ID, so adding a field there would point the
    notebook at a new, empty run directory and force neural inference again.
    """

    # APPROVED rule.
    overlap_tolerance: float = 0.05
    # EXISTING GATE thresholds (training-like).
    min_enrichment: float = 1.20
    min_solidity: float = 0.80
    # EXPLORATORY. "unresolved": a component whose best parent holds less than
    # the containment fraction used by assign_components stays unresolved.
    # "evaluate": judge it against that best parent like an assigned one.
    partial_parent_policy: str = "unresolved"
    # EXPLORATORY, off by default: drop components of exactly one pixel.
    exclude_one_pixel_components: bool = False

    STATUS_OF_FIELD = {
        "overlap_tolerance": "APPROVED",
        "min_enrichment": "existing gate",
        "min_solidity": "existing gate",
        "partial_parent_policy": "EXPLORATORY",
        "exclude_one_pixel_components": "EXPLORATORY",
    }

    def __post_init__(self):
        if not 0.0 < float(self.overlap_tolerance) < 1.0:
            raise ValueError("overlap_tolerance must lie strictly between 0 and 1")
        if self.partial_parent_policy not in ("unresolved", "evaluate"):
            raise ValueError("partial_parent_policy must be 'unresolved' or 'evaluate'")
        if not np.isfinite([self.min_enrichment, self.min_solidity]).all():
            raise ValueError("gate thresholds must be finite")

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> pd.DataFrame:
        """Table of each rule, its value, and whether it is approved."""
        values = self.to_dict()
        return pd.DataFrame(
            [dict(setting=k, value=values[k], standing=v)
             for k, v in self.STATUS_OF_FIELD.items()])


# ── Circle overlap ──────────────────────────────────────────────────────────
def circle_overlap_fraction(x1, y1, r1, x2, y2, r2) -> float:
    """Intersection area / area of the smaller FULL circle (analytic).

    Circles are taken un-eroded and are not clipped to the image. Returns 0
    for disjoint or externally tangent circles and 1 when one circle lies
    inside the other. Reference implementation; `droplet_overlap_pairs`
    evaluates the same expression in bulk.
    """
    if not np.isfinite([x1, y1, r1, x2, y2, r2]).all() or min(r1, r2) <= 0:
        raise ValueError("Circles must have finite coordinates and positive radii")
    d = float(np.hypot(x1 - x2, y1 - y2))
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return 1.0
    alpha = np.arccos(np.clip((d * d + r1 * r1 - r2 * r2) / (2 * d * r1), -1, 1))
    beta = np.arccos(np.clip((d * d + r2 * r2 - r1 * r1) / (2 * d * r2), -1, 1))
    term = max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    lens = r1 * r1 * alpha + r2 * r2 * beta - 0.5 * np.sqrt(term)
    return float(np.clip(lens / (np.pi * min(r1, r2) ** 2), 0, 1))


def _pairwise_overlap(cx: np.ndarray, cy: np.ndarray, r: np.ndarray):
    """Vectorised circle_overlap_fraction for every pair i < j on one plane."""
    i, j = np.triu_indices(len(r), k=1)
    d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
    r1, r2 = r[i], r[j]
    fraction = np.zeros(len(d), dtype=float)
    contained = d <= np.abs(r1 - r2)
    lens = ~contained & (d < r1 + r2)
    fraction[contained] = 1.0
    if lens.any():
        dd, a, b = d[lens], r1[lens], r2[lens]
        alpha = np.arccos(np.clip((dd * dd + a * a - b * b) / (2 * dd * a), -1, 1))
        beta = np.arccos(np.clip((dd * dd + b * b - a * a) / (2 * dd * b), -1, 1))
        term = np.maximum(0.0, (-dd + a + b) * (dd + a - b) * (dd - a + b) * (dd + a + b))
        area = a * a * alpha + b * b * beta - 0.5 * np.sqrt(term)
        fraction[lens] = np.clip(area / (np.pi * np.minimum(a, b) ** 2), 0, 1)
    return i, j, fraction


def droplet_overlap_pairs(circles: pd.DataFrame) -> pd.DataFrame:
    """Every pair of circles on the same (t, z) plane with non-zero overlap.

    `circles` needs t, z, parent, cx, cy, radius_px, source. Pairs involving a
    sphere-predicted circle are reported with observed_pair=False so they can
    be inspected, but they never trigger exclusion.
    """
    need = {"t", "z", "parent", "cx", "cy", "radius_px", "source"}
    missing = need - set(circles.columns)
    if missing:
        raise KeyError(f"circles is missing columns: {sorted(missing)}")
    if circles.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    if circles.duplicated(["t", "z", "parent"]).any():
        raise ValueError("Duplicate circle keys (t, z, parent)")
    numeric = circles[["cx", "cy", "radius_px"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric[:, 2] <= 0).any():
        raise ValueError("Circles must have finite coordinates and positive radii")
    unknown = set(circles.source) - {OBSERVED, PREDICTED}
    if unknown:
        raise ValueError(f"Unknown circle source(s): {sorted(unknown)}")

    frames = []
    for (t, z), plane in circles.groupby(["t", "z"], sort=True):
        if len(plane) < 2:
            continue
        plane = plane.sort_values("parent")
        parent = plane.parent.to_numpy()
        source = plane.source.to_numpy()
        i, j, fraction = _pairwise_overlap(
            plane.cx.to_numpy(float), plane.cy.to_numpy(float),
            plane.radius_px.to_numpy(float))
        keep = fraction > 0
        if not keep.any():
            continue
        i, j = i[keep], j[keep]
        frames.append(pd.DataFrame({
            "t": int(t), "z": int(z),
            "parent_a": parent[i].astype(int), "parent_b": parent[j].astype(int),
            "overlap_fraction": fraction[keep],
            "observed_pair": (source[i] == OBSERVED) & (source[j] == OBSERVED),
            "source_a": source[i], "source_b": source[j]}))
    if not frames:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    return pd.concat(frames, ignore_index=True)[PAIR_COLUMNS]


def overlap_hits(pairs: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Pairs that trigger the rule: both observed, overlap STRICTLY above tolerance."""
    if not 0.0 < float(tolerance) < 1.0:
        raise ValueError("tolerance must lie strictly between 0 and 1")
    if pairs.empty:
        return pairs.iloc[0:0].copy()
    observed = pairs.observed_pair.astype(bool)
    return pairs[observed & (pairs.overlap_fraction.astype(float) > float(tolerance))].copy()


def flag_overlapping_droplets(pairs: pd.DataFrame, tolerance: float = 0.05) -> pd.DataFrame:
    """One row per flagged (t, parent). Both members of every hit are flagged.

    A flag belongs to the fitted inventory identity within one timepoint. It
    applies to that droplet on every Z plane of that timepoint and says
    nothing about any other timepoint.
    """
    hits = overlap_hits(pairs, tolerance)
    if hits.empty:
        return pd.DataFrame(columns=FLAG_COLUMNS)
    long = pd.concat([
        hits[["t", "z", "parent_a", "parent_b", "overlap_fraction"]].rename(
            columns={"parent_a": "parent", "parent_b": "partner"}),
        hits[["t", "z", "parent_b", "parent_a", "overlap_fraction"]].rename(
            columns={"parent_b": "parent", "parent_a": "partner"}),
    ], ignore_index=True)
    rows = []
    for (t, parent), g in long.groupby(["t", "parent"], sort=True):
        rows.append(dict(
            t=int(t), parent=int(parent),
            max_observed_overlap=float(g.overlap_fraction.max()),
            evidence_planes=int(g.z.nunique()),
            evidence_z=";".join(str(int(z)) for z in sorted(g.z.unique())),
            partners=";".join(str(int(p)) for p in sorted(g.partner.unique())),
            reason=REASON_OVERLAP))
    return pd.DataFrame(rows, columns=FLAG_COLUMNS)


# ── Component decisions ─────────────────────────────────────────────────────
_DECISION_INPUTS = ["t", "z", "label", "nucleus_3d_id", "area_pixels",
                    "geometry_status", "parent", "enrichment", "solidity",
                    "reference_ok"]


def decide_components(components: pd.DataFrame, flags: pd.DataFrame,
                      settings: PopulationSettings = PopulationSettings()) -> pd.DataFrame:
    """Add status and explicit reasons to each component. Input is not modified.

    Precedence: any definite failure -> excluded; otherwise anything that
    cannot be judged -> unresolved; otherwise accepted. Enrichment and the
    overlap rule need a resolved parent droplet, so they are only applied
    where the parent is resolved. Solidity is a property of the mask alone and
    is applied everywhere. A component is never moved to a different parent.
    """
    missing = set(_DECISION_INPUTS) - set(components.columns)
    if missing:
        raise KeyError(f"components is missing columns: {sorted(missing)}")
    if components.duplicated(COMPONENT_KEYS).any():
        raise ValueError("Duplicate component keys (t, z, label)")
    unknown = set(components.geometry_status) - set(GEOMETRY_STATUSES)
    if unknown:
        raise ValueError(f"Unknown geometry_status value(s): {sorted(unknown)}")

    d = components.copy()
    n = len(d)
    status_geom = d.geometry_status.to_numpy()
    resolved = status_geom == GEOM_ASSIGNED
    if settings.partial_parent_policy == "evaluate":
        resolved = resolved | (status_geom == GEOM_PARTIAL)

    flagged = set()
    if flags is not None and len(flags):
        flagged = set(zip(flags.t.astype(int), flags.parent.astype(int)))
    parent = d.parent.fillna(0).astype(int).to_numpy()
    t = d.t.astype(int).to_numpy()
    parent_flagged = np.fromiter(
        ((int(a), int(b)) in flagged for a, b in zip(t, parent)), bool, count=n)
    parent_flagged &= parent > 0

    solidity = pd.to_numeric(d.solidity, errors="coerce").to_numpy(float)
    enrichment = pd.to_numeric(d.enrichment, errors="coerce").to_numpy(float)
    reference_ok = d.reference_ok.map(lambda v: v is True or v == 1).to_numpy(bool)
    area = pd.to_numeric(d.area_pixels, errors="coerce").to_numpy(float)

    # Same tests as measure_segment_gate, applied to the stored measurements.
    fail_solidity = ~np.isfinite(solidity) | (solidity < settings.min_solidity)
    fail_enrichment = resolved & reference_ok & (enrichment < settings.min_enrichment)
    fail_overlap = resolved & parent_flagged
    fail_one_pixel = (area == 1) if settings.exclude_one_pixel_components else np.zeros(n, bool)
    no_reference = resolved & ~reference_ok

    exclusion, unresolved, no_overlap = [], [], []
    for k in range(n):
        ex = []
        if fail_enrichment[k]:
            ex.append(REASON_ENRICHMENT)
        if fail_solidity[k]:
            ex.append(REASON_SOLIDITY)
        if fail_one_pixel[k]:
            ex.append(REASON_ONE_PIXEL)
        no_overlap.append(bool(ex))
        if fail_overlap[k]:
            ex.append(REASON_OVERLAP)
        un = []
        if not resolved[k]:
            un.append(str(status_geom[k]))
        elif no_reference[k]:
            un.append(REASON_NO_REFERENCE)
        exclusion.append(_SEP.join(ex))
        unresolved.append(_SEP.join(un))

    has_exclusion = np.array([bool(x) for x in exclusion], bool)
    has_unknown = np.array([bool(x) for x in unresolved], bool)
    status = np.where(has_exclusion, STATUS_EXCLUDED,
                      np.where(has_unknown, STATUS_UNRESOLVED, STATUS_ACCEPTED))
    without_overlap = np.where(np.array(no_overlap, bool), STATUS_EXCLUDED,
                               np.where(has_unknown, STATUS_UNRESOLVED, STATUS_ACCEPTED))

    d["assignment_resolved"] = resolved
    d["parent_flagged"] = parent_flagged
    d["exclusion_reasons"] = exclusion
    d["unresolved_reasons"] = unresolved
    d["status"] = status
    d["reason"] = np.where(has_exclusion, d.exclusion_reasons,
                           np.where(has_unknown, d.unresolved_reasons, STATUS_ACCEPTED))
    d["status_without_overlap"] = without_overlap
    d["removed_by_overlap"] = (without_overlap == STATUS_ACCEPTED) & (status != STATUS_ACCEPTED)
    return d


def summarize_linked_ids(decisions: pd.DataFrame) -> pd.DataFrame:
    """One row per (t, linked nuclear ID).

    An ID is accepted when at least one of its components is accepted. With no
    accepted component it is unresolved if any component could not be judged,
    and excluded only when every component definitely failed. `lost_to_overlap`
    separates losing a whole ID to the overlap rule from losing some of its
    components (`partially_removed`).
    """
    columns = ID_KEYS + [
        "review_id", "status", "reasons", "n_components", "n_accepted",
        "n_excluded", "n_unresolved", "n_planes", "n_accepted_planes",
        "components_removed_by_overlap", "lost_to_overlap", "partially_removed",
        "has_overlap_flagged_component", "multiple_instances_same_z",
        "max_area_pixels_all", "max_area_pixels_accepted"]
    if decisions.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (t, nid), g in decisions.groupby(ID_KEYS, sort=True):
        accepted = g[g.status == STATUS_ACCEPTED]
        n_acc = len(accepted)
        n_exc = int((g.status == STATUS_EXCLUDED).sum())
        n_unr = int((g.status == STATUS_UNRESOLVED).sum())
        status = (STATUS_ACCEPTED if n_acc else
                  STATUS_UNRESOLVED if n_unr else STATUS_EXCLUDED)
        reasons = sorted({part for text in g.loc[g.status != STATUS_ACCEPTED, "reason"]
                          for part in str(text).split(_SEP) if part})
        rows.append(dict(
            t=int(t), nucleus_3d_id=int(nid), review_id="T%d_N%d" % (t, nid),
            status=status, reasons=_SEP.join(reasons),
            n_components=len(g), n_accepted=n_acc, n_excluded=n_exc, n_unresolved=n_unr,
            n_planes=int(g.z.nunique()), n_accepted_planes=int(accepted.z.nunique()),
            components_removed_by_overlap=int(g.removed_by_overlap.sum()),
            lost_to_overlap=bool((g.status_without_overlap == STATUS_ACCEPTED).any() and not n_acc),
            partially_removed=bool(n_acc and n_acc < len(g)),
            has_overlap_flagged_component=bool(g.parent_flagged.any()),
            multiple_instances_same_z=bool((g.groupby("z").size() > 1).any()),
            max_area_pixels_all=float(g.area_pixels.max()),
            max_area_pixels_accepted=float(accepted.area_pixels.max()) if n_acc else np.nan))
    return pd.DataFrame(rows, columns=columns)


def overlap_impact(decisions: pd.DataFrame, pairs: pd.DataFrame, flags: pd.DataFrame,
                   tolerance: float) -> pd.DataFrame:
    """Per-timepoint effect of the overlap rule on components and linked IDs.

    "baseline" here means the population with every other rule applied and the
    overlap rule switched off. Counts are linked IDs, not independently
    confirmed biological nuclei.
    """
    hits = overlap_hits(pairs, tolerance)
    rows = []
    for t, g in decisions.groupby("t", sort=True):
        before = g.status_without_overlap == STATUS_ACCEPTED
        after = g.status == STATUS_ACCEPTED
        ids = (g.assign(_before=before, _after=after)
               .groupby("nucleus_3d_id")[["_before", "_after"]].any())
        hits_t = hits[hits.t == t] if len(hits) else hits
        rows.append(dict(
            t=int(t),
            droplets_flagged=int((flags.t == t).sum()) if len(flags) else 0,
            overlapping_observed_pairs=int(len(hits_t[["parent_a", "parent_b"]].drop_duplicates())),
            baseline_components=int(before.sum()),
            remaining_components=int(after.sum()),
            components_removed=int(g.removed_by_overlap.sum()),
            baseline_linked_ids=int(ids._before.sum()),
            remaining_linked_ids=int(ids._after.sum()),
            linked_ids_lost=int((ids._before & ~ids._after).sum()),
            unresolved_assignments=int(g.geometry_status.isin(
                [GEOM_MISSING, GEOM_AMBIGUOUS, GEOM_PARTIAL, GEOM_NO_TIMEPOINT]).sum()),
            unknown_gate_decisions=int(g.geometry_status.isin(
                [GEOM_MISSING, GEOM_AMBIGUOUS, GEOM_NO_TIMEPOINT]).sum())))
    return pd.DataFrame(rows)


def status_summary(decisions: pd.DataFrame, linked_ids: pd.DataFrame) -> pd.DataFrame:
    """Per-timepoint counts of components and linked IDs in each status."""
    rows = []
    for t in sorted(set(decisions.t.astype(int))):
        c = decisions[decisions.t == t]
        i = linked_ids[linked_ids.t == t]
        row = dict(t=t, components=len(c), linked_ids=len(i))
        for s in STATUSES:
            row[f"components_{s}"] = int((c.status == s).sum())
        for s in STATUSES:
            row[f"linked_ids_{s}"] = int((i.status == s).sum())
        row["linked_ids_partially_removed"] = int(i.partially_removed.sum())
        row["linked_ids_branched"] = int(i.multiple_instances_same_z.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def reason_counts(decisions: pd.DataFrame) -> pd.DataFrame:
    """Component counts per timepoint, status and individual reason."""
    d = decisions[decisions.status != STATUS_ACCEPTED]
    if d.empty:
        return pd.DataFrame(columns=["t", "status", "reason", "components"])
    long = d.assign(reason=d.reason.str.split(_SEP, regex=False)).explode("reason")
    return (long.groupby(["t", "status", "reason"]).size()
            .rename("components").reset_index())


# ── Building, tagging, saving ───────────────────────────────────────────────
def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def population_tag(settings: PopulationSettings, upstream: Optional[Mapping] = None) -> str:
    """Short stable name for one (rules, upstream measurements) combination."""
    payload = dict(settings=settings.to_dict(), upstream=dict(upstream or {}))
    return "pop-" + hashlib.sha256(_canonical(payload).encode()).hexdigest()[:10]


def build_population(components: pd.DataFrame, circles: pd.DataFrame,
                     settings: PopulationSettings = PopulationSettings(),
                     upstream: Optional[Mapping] = None) -> dict:
    """Apply every enabled rule and return the population as a dict of tables.

    Keys: components (all, with status and reasons), accepted, excluded,
    unresolved, linked_ids, circles, pairs, flags, impact, summary, reasons,
    settings, upstream, tag. The inputs are not modified.
    """
    pairs = droplet_overlap_pairs(circles)
    flags = flag_overlapping_droplets(pairs, settings.overlap_tolerance)
    decisions = decide_components(components, flags, settings)
    linked = summarize_linked_ids(decisions)
    return dict(
        components=decisions,
        accepted=decisions[decisions.status == STATUS_ACCEPTED].copy(),
        excluded=decisions[decisions.status == STATUS_EXCLUDED].copy(),
        unresolved=decisions[decisions.status == STATUS_UNRESOLVED].copy(),
        linked_ids=linked, circles=circles.copy(), pairs=pairs, flags=flags,
        impact=overlap_impact(decisions, pairs, flags, settings.overlap_tolerance),
        summary=status_summary(decisions, linked),
        reasons=reason_counts(decisions),
        settings=settings.to_dict(), upstream=dict(upstream or {}),
        tag=population_tag(settings, upstream))


_TABLES = ("components", "accepted", "excluded", "unresolved", "linked_ids",
           "circles", "pairs", "flags", "impact", "summary", "reasons")
_FILE_OF = {"components": "component_decisions", "accepted": "accepted_components",
            "excluded": "excluded_components", "unresolved": "unresolved_components",
            "linked_ids": "linked_id_decisions", "circles": "droplet_circles",
            "pairs": "overlapping_circle_pairs", "flags": "flagged_droplets",
            "impact": "overlap_impact_by_t", "summary": "status_summary_by_t",
            "reasons": "reason_counts"}


def _atomic_write(path: Path, writer) -> None:
    tmp = path.with_name(path.name + ".tmp")
    writer(tmp)
    os.replace(tmp, path)


def save_population(population: Mapping, root: Path) -> Path:
    """Write the population under root/<tag>/ and return that directory.

    Only this directory is written. The pipeline's baseline pickles
    (plane_objects, grouped_z_objects, best_z_nuclei, tracked_nuclei, ...) and
    every mask file are never opened for writing. Re-saving the same tag
    replaces the files in place.
    """
    out = Path(root) / str(population["tag"])
    out.mkdir(parents=True, exist_ok=True)
    for key in _TABLES:
        table = population[key]
        stem = _FILE_OF[key]
        _atomic_write(out / f"{stem}.pkl", lambda p, table=table: table.to_pickle(p))
        _atomic_write(out / f"{stem}.csv", lambda p, table=table: table.to_csv(p, index=False))
    manifest = dict(
        tag=population["tag"], settings=population["settings"],
        upstream=population["upstream"],
        saved_at=datetime.now().isoformat(timespec="seconds"),
        rows={key: int(len(population[key])) for key in _TABLES},
        definition=dict(
            overlap_metric="analytic intersection / smaller full un-eroded circle area; no image clipping",
            overlap_trigger="both circles observed_inlier, any evaluated plane, strict > tolerance",
            overlap_propagation="both droplet identities across Z within the timepoint; never across timepoints",
            unresolved="not judged; neither accepted nor a confirmed artifact",
            not_enforced=["Z-linking unchanged", "multi-nucleus droplets not excluded"]))
    _atomic_write(out / "population_manifest.json",
                  lambda p: p.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf8"))
    return out


def list_saved_populations(root: Path) -> pd.DataFrame:
    """Populations saved under `root`, newest first (tag, time, rules, row counts)."""
    rows = []
    for manifest_path in sorted(Path(root).glob("pop-*/population_manifest.json")):
        m = json.loads(manifest_path.read_text(encoding="utf8"))
        rows.append(dict(tag=m["tag"], saved_at=m["saved_at"], directory=manifest_path.parent,
                         **{f"rule_{k}": v for k, v in m["settings"].items()},
                         accepted=m["rows"]["accepted"], excluded=m["rows"]["excluded"],
                         unresolved=m["rows"]["unresolved"]))
    table = pd.DataFrame(rows)
    return table.sort_values("saved_at", ascending=False).reset_index(drop=True) if len(table) else table


def load_population(directory: Path) -> dict:
    """Reload a population written by save_population."""
    directory = Path(directory)
    manifest = json.loads((directory / "population_manifest.json").read_text(encoding="utf8"))
    population = {key: pd.read_pickle(directory / f"{_FILE_OF[key]}.pkl") for key in _TABLES}
    population.update(settings=manifest["settings"], upstream=manifest["upstream"],
                      tag=manifest["tag"], output_dir=directory)
    return population


# ── Using the population downstream ─────────────────────────────────────────
def _wanted_statuses(which: str) -> tuple:
    if which not in POPULATIONS:
        raise ValueError(f"which must be one of {POPULATIONS}, not {which!r}")
    return {"accepted": (STATUS_ACCEPTED,), "excluded": (STATUS_EXCLUDED,),
            "unresolved": (STATUS_UNRESOLVED,),
            "accepted_plus_unresolved": (STATUS_ACCEPTED, STATUS_UNRESOLVED),
            "baseline": STATUSES}[which]


def select_population(table: pd.DataFrame, population: Mapping, which: str = "accepted",
                      key_lookup: Optional[pd.DataFrame] = None,
                      on_missing: str = "raise") -> pd.DataFrame:
    """Return the rows of a baseline table that belong to one population.

    The table is matched on its own component, (t, z, label). A best-Z row is
    therefore kept or dropped on the decision for the component it actually
    measures; no other Z plane is substituted. Tables without `label` (the
    radial sweep) are matched through `key_lookup`, a table carrying
    t, track_id, z, label (tracked_df).

    which = "baseline" returns an unfiltered copy. "accepted_plus_unresolved"
    exists so that including unjudged components is always a visible choice.
    Rows with no decision (a timepoint the stage was not run on) raise unless
    on_missing="drop". The input table is never modified.
    """
    wanted = _wanted_statuses(which)
    if on_missing not in ("raise", "drop"):
        raise ValueError("on_missing must be 'raise' or 'drop'")
    if which == "baseline":
        return table.copy()
    if table.empty:
        return table.copy()
    decisions = population["components"][COMPONENT_KEYS + ["status"]]

    if set(COMPONENT_KEYS) <= set(table.columns):
        merged = table.merge(decisions, on=COMPONENT_KEYS, how="left", validate="many_to_one")
        status = merged["status"].to_numpy(object)
    elif {"t", "track_id"} <= set(table.columns):
        if key_lookup is None:
            raise ValueError("This table has no (t, z, label); pass key_lookup=tracked_df")
        keys = ["t", "z", "track_id"] if "z" in table.columns else ["t", "track_id"]
        lookup = key_lookup[list(dict.fromkeys(keys + COMPONENT_KEYS))].merge(
            decisions, on=COMPONENT_KEYS, how="left", validate="many_to_one")
        # A key that maps to several components is kept only if they all agree.
        per_key = lookup.groupby(keys)["status"].agg(
            lambda s: s.iloc[0] if s.notna().all() and s.nunique() == 1 else
            (np.nan if s.isna().any() else "mixed")).rename("status").reset_index()
        merged = table.merge(per_key, on=keys, how="left", validate="many_to_one")
        status = merged["status"].to_numpy(object)
    else:
        raise KeyError("Table needs (t, z, label) or (t, track_id) to be matched to the population")

    # Positional masks: a left merge keeps the table's row order, and the
    # table's own index may not be unique.
    undecided = pd.isna(status)
    if undecided.any() and on_missing == "raise":
        times = sorted(set(table["t"].to_numpy()[undecided].astype(int)))
        raise ValueError(
            f"{int(undecided.sum())} row(s) have no population decision (timepoints {times}). "
            "Run the population stage on those timepoints, or pass on_missing='drop' "
            "to leave them out explicitly.")
    keep = np.isin(status, wanted)
    out = table[keep].copy()
    out["population_status"] = status[keep]
    return out


def population_as_gate(population: Mapping, pixel_size_um: float, which: str = "accepted",
                       legacy_gate: Optional[Mapping] = None) -> dict:
    """Present a population in the dict shape the section-38 audits expect.

    `gate_pass` is membership of the requested population. Enrichment, solidity
    and parent come from the fitted-droplet measurement. If the original
    model-droplet gate is supplied, its plane-local `parent_droplet` is carried
    over for the envelope audit, which draws the model droplet mask; the fitted
    parent stays available as `fitted_parent`.
    """
    wanted = _wanted_statuses(which)
    # The audits merge this table with the linked object table on (t, z, label)
    # and take nucleus_3d_id from there, exactly as with the original gate.
    d = population["components"].drop(columns="nucleus_3d_id")
    d["gate_pass"] = d.status.isin(wanted).to_numpy(bool)
    d["gate_reason"] = d.reason
    d["area_um2"] = d.area_pixels.astype(float) * float(pixel_size_um) ** 2
    d["fitted_parent"] = d.parent
    d["parent_droplet"] = d.parent
    if legacy_gate is not None:
        legacy = legacy_gate["all_instances"][COMPONENT_KEYS + ["parent_droplet"]]
        d = d.drop(columns="parent_droplet").merge(
            legacy, on=COMPONENT_KEYS, how="left", validate="one_to_one")
        if d.parent_droplet.isna().any():
            raise ValueError("Original gate does not cover every population component")
        d["parent_droplet"] = d.parent_droplet.astype(int)
    settings = dict(population["settings"])
    return dict(all_instances=d, accepted=d[d.gate_pass].copy(), rejected=d[~d.gate_pass].copy(),
                settings=settings, population=which, tag=population["tag"],
                output_dir=population.get("output_dir"))


_REVIEW_ID = re.compile(r"^T(\d+)_N(\d+)$")


def reviewed_outcomes(population: Mapping, review_labels: Mapping[str, str]) -> pd.DataFrame:
    """Where each manually reviewed linked ID ended up.

    `review_labels` maps "T<t>_N<linked id>" to a label such as "artifact" or
    "real nucleus". Linked IDs belong to one segmentation run: labels from
    another run are meaningless here. This is a report only and never feeds
    back into the decisions.
    """
    ids = population["linked_ids"].set_index("review_id")
    rows = []
    for review_id, label in review_labels.items():
        if not _REVIEW_ID.match(str(review_id)):
            raise ValueError(f"Review ID {review_id!r} is not of the form T<t>_N<id>")
        if review_id in ids.index:
            r = ids.loc[review_id]
            rows.append(dict(review_id=review_id, review_label=label, found=True,
                             status=r.status, reasons=r.reasons,
                             n_components=int(r.n_components), n_accepted=int(r.n_accepted),
                             n_excluded=int(r.n_excluded), n_unresolved=int(r.n_unresolved),
                             lost_to_overlap=bool(r.lost_to_overlap)))
        else:
            rows.append(dict(review_id=review_id, review_label=label, found=False,
                             status="not_in_population", reasons="", n_components=0,
                             n_accepted=0, n_excluded=0, n_unresolved=0, lost_to_overlap=False))
    return pd.DataFrame(rows, columns=["review_id", "review_label", "found", "status", "reasons",
                                       "n_components", "n_accepted", "n_excluded",
                                       "n_unresolved", "lost_to_overlap"])


def compare_tolerances(components: pd.DataFrame, circles: pd.DataFrame,
                       tolerances: Iterable[float],
                       settings: PopulationSettings = PopulationSettings()) -> pd.DataFrame:
    """Overlap impact at several tolerances. Diagnostic only; saves nothing."""
    pairs = droplet_overlap_pairs(circles)
    frames = []
    for tolerance in sorted(set(map(float, tolerances))):
        trial = PopulationSettings(**{**settings.to_dict(), "overlap_tolerance": tolerance})
        flags = flag_overlapping_droplets(pairs, tolerance)
        decisions = decide_components(components, flags, trial)
        impact = overlap_impact(decisions, pairs, flags, tolerance)
        impact.insert(0, "tolerance", tolerance)
        frames.append(impact)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
