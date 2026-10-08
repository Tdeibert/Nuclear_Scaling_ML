"""Synthetic run for the droplet-population tests. Builds only temporary files.

A small TZCYX image, a matching nuclear instance stack, a linked object table
and a miniature "training notebook" are written to a temporary directory. The
notebook defines every function `load_training_geometry` looks for; its
geometry is read from a truth file next to the image, so each test knows
exactly which circle is observed, which is predicted, and how much any two
overlap. The real fitting code is covered separately by the parity test, which
runs the actual training notebook when it is available.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tifffile

PIXEL_UM = 0.5
GEOMETRY_Z_STEP_UM = 6.0
EROSION_PX = 2
N_T, N_Z, HEIGHT, WIDTH = 3, 6, 220, 420
MEMBRANE, NLS, NPC = 0, 1, 2
MIN_Z = 1                       # nuclear target-plane floor of the synthetic run
BACKGROUND, CYTOPLASM, BRIGHT, DIM = 100, 1000, 3000, 1100

# Droplets per timepoint. Parent ID = position in the list + 1.
#   1 + 2   overlap 6.8 % at Z3 and 4.3 % at Z2/Z4 (T0 only): the approved rule
#   3       isolated
#   4 + 5   overlap heavily, but 5 is observed only on Z0, below the evaluated
#           planes, so on every evaluated plane the pair is predicted-only
#   6       isolated; a nucleus straddles its wall
#   7       small; a nucleus nearly fills it, leaving no usable reference ring
_ALL_Z = list(range(N_Z))
TRUTH = {
    0: [dict(cx=60, cy=60, R_um=20, z_eq=3, observed_z=_ALL_Z),
        dict(cx=128, cy=60, R_um=20, z_eq=3, observed_z=_ALL_Z),
        dict(cx=60, cy=160, R_um=20, z_eq=3, observed_z=_ALL_Z),
        dict(cx=250, cy=60, R_um=20, z_eq=3, observed_z=_ALL_Z),
        dict(cx=300, cy=60, R_um=20, z_eq=3, observed_z=[0]),
        dict(cx=360, cy=160, R_um=20, z_eq=3, observed_z=_ALL_Z),
        dict(cx=190, cy=110, R_um=7, z_eq=3, observed_z=[2, 3, 4])],
    # Same parent IDs 1 and 2 as T0, now well apart: a T0 flag must not follow them.
    1: [dict(cx=60, cy=60, R_um=20, z_eq=3, observed_z=_ALL_Z),
        dict(cx=160, cy=60, R_um=20, z_eq=3, observed_z=_ALL_Z)],
    2: [dict(cx=60, cy=60, R_um=20, z_eq=3, observed_z=_ALL_Z)],
}

# Nuclear components. `nid` is the linked ID; `expect` is the default-settings
# outcome as (geometry_status, status, reason).
_OVERLAP = "overlapping_droplet_geometry"
NUCLEI = [
    # T0 -- ID 1: every component in flagged droplet 1 -> whole ID lost to overlap.
    *[dict(t=0, z=z, nid=1, shape="disk", cx=60, cy=60, r=8, value=BRIGHT,
           expect=("assigned", "excluded", _OVERLAP)) for z in (1, 2, 3, 4, 5)],
    # ID 2: in flagged droplet 2, including planes with no overlap evidence.
    *[dict(t=0, z=z, nid=2, shape="disk", cx=142, cy=60, r=8, value=BRIGHT,
           expect=("assigned", "excluded", _OVERLAP)) for z in (2, 3, 4)],
    # ID 3: a clean nucleus in isolated droplet 3.
    *[dict(t=0, z=z, nid=3, shape="disk", cx=50, cy=150, r=8, value=BRIGHT,
           expect=("assigned", "accepted", "accepted")) for z in (1, 2, 3, 4, 5)],
    # ID 4: dim cytoplasm artifact -> fails enrichment.
    dict(t=0, z=3, nid=4, shape="disk", cx=80, cy=170, r=6, value=DIM,
         expect=("assigned", "excluded", "enrichment")),
    # ID 5: bright but C-shaped -> fails solidity.
    dict(t=0, z=3, nid=5, shape="c", cx=75, cy=140, r=7, value=BRIGHT,
         expect=("assigned", "excluded", "solidity")),
    # ID 6: a single bright pixel -> accepted unless the one-pixel option is on.
    dict(t=0, z=3, nid=6, shape="pixel", cx=60, cy=185, r=0, value=BRIGHT,
         expect=("assigned", "accepted", "accepted")),
    # ID 7: one component in flagged droplet 1, one in clean droplet 3 ->
    # the ID survives with one component removed.
    dict(t=0, z=2, nid=7, shape="disk", cx=45, cy=45, r=6, value=BRIGHT,
         expect=("assigned", "excluded", _OVERLAP)),
    dict(t=0, z=3, nid=7, shape="disk", cx=40, cy=175, r=6, value=BRIGHT,
         expect=("assigned", "accepted", "accepted")),
    # ID 8: inside both droplet 4 and droplet 5 -> ambiguous, stays unresolved.
    dict(t=0, z=3, nid=8, shape="disk", cx=275, cy=60, r=6, value=BRIGHT,
         expect=("ambiguous_parent", "unresolved", "ambiguous_parent")),
    # ID 9: outside every droplet -> missing parent, stays unresolved.
    dict(t=0, z=3, nid=9, shape="disk", cx=200, cy=180, r=6, value=BRIGHT,
         expect=("missing_parent", "unresolved", "missing_parent")),
    # ID 10: half across droplet 6's wall -> partial parent.
    dict(t=0, z=3, nid=10, shape="disk", cx=398, cy=160, r=8, value=BRIGHT,
         expect=("partial_parent", "unresolved", "partial_parent")),
    # ID 11: in droplet 4 only. Droplet 4 overlaps predicted-only droplet 5,
    # which must not trigger the rule.
    dict(t=0, z=3, nid=11, shape="disk", cx=235, cy=45, r=6, value=BRIGHT,
         expect=("assigned", "accepted", "accepted")),
    # ID 12: in droplet 5 only; its support is a sphere prediction.
    dict(t=0, z=3, nid=12, shape="disk", cx=320, cy=75, r=6, value=BRIGHT,
         expect=("assigned", "accepted", "accepted")),
    # ID 13: nearly fills small droplet 7 -> no reference ring to measure.
    dict(t=0, z=3, nid=13, shape="disk", cx=190, cy=110, r=10.5, value=BRIGHT,
         expect=("assigned", "unresolved", "unassessable_cytoplasm")),
    # T1 -- the same parent IDs, no overlap.
    *[dict(t=1, z=z, nid=20, shape="disk", cx=60, cy=60, r=8, value=BRIGHT,
           expect=("assigned", "accepted", "accepted")) for z in (2, 3, 4)],
    *[dict(t=1, z=z, nid=21, shape="disk", cx=160, cy=60, r=8, value=BRIGHT,
           expect=("assigned", "accepted", "accepted")) for z in (2, 3, 4)],
    # T2
    *[dict(t=2, z=z, nid=30, shape="disk", cx=60, cy=60, r=8, value=BRIGHT,
           expect=("assigned", "accepted", "accepted")) for z in (2, 3, 4)],
]

# The miniature training notebook. Function bodies for the sphere/circle policy
# follow the real ones; fitting is replaced by a lookup in the truth file.
_FAKE_GEOMETRY_CELL = '''
def extract_plane(hyperstack, t, z, c):
    import json, pathlib
    truth = json.loads(pathlib.Path(str(hyperstack.filename) + ".truth.json").read_text())
    return truth.get(str(t), [])

def detect_droplets_npc_watershed(npc_plane, cfg=cfg, compact=False):
    return [dict(label=i + 1, centroid=(d["cy"], d["cx"]),
                 area=3.141592653589793 * (d["R_um"] / cfg.pixel_size_um) ** 2)
            for i, d in enumerate(npc_plane)]

def sphere_radius_px(g, z, cfg=cfg):
    dz = float(cfg.geometry_z_step_um)
    r2 = g["R_um"] ** 2 - (dz * (z - g["z_eq"])) ** 2
    return float(np.sqrt(r2) / cfg.pixel_size_um) if r2 > 0 else None

def predicted_circle(g, z, cfg=cfg):
    radius = sphere_radius_px(g, z, cfg)
    usable = [zz for zz in g["prof"] if zz not in g["sphere_outliers"]]
    if radius is None or not usable:
        return None
    nearest = min(usable, key=lambda zz: (abs(zz - z), zz))
    cx, cy, _ = g["prof"][nearest]
    return float(cx), float(cy), radius

def p2_circle(g, z, cfg=cfg):
    if z in g["prof"] and z not in g["sphere_outliers"]:
        return "fitted", tuple(map(float, g["prof"][z]))
    c = predicted_circle(g, z, cfg)
    return None if c is None else ("predicted", c)

def compute_droplet_geometry(hs, t, inv, cfg=cfg, zs=None, progress=True):
    geometry = {}
    for did, d in enumerate(extract_plane(hs, t, 0, 0)):
        g = dict(R_um=float(d["R_um"]), z_eq=float(d["z_eq"]), sphere_outliers=[], prof={})
        for z in d["observed_z"]:
            radius = sphere_radius_px(g, int(z), cfg)
            if radius is not None:
                g["prof"][int(z)] = (float(d["cx"]), float(d["cy"]), radius)
        geometry[did] = g
    return geometry

def clip_histogram(*a, **k): raise NotImplementedError
def _circularity(*a, **k): raise NotImplementedError
def _circle_from_3(*a, **k): raise NotImplementedError
def _fit_circle_kasa(*a, **k): raise NotImplementedError
def fit_circle_ransac(*a, **k): raise NotImplementedError
def smooth_npc_plane(*a, **k): raise NotImplementedError
def wall_points_on_smoothed(*a, **k): raise NotImplementedError
def fit_droplet_circle_on_smoothed(*a, **k): raise NotImplementedError
def _sphere_lsq(*a, **k): raise NotImplementedError
def fit_sphere_profile(*a, **k): raise NotImplementedError
def fit_sphere_ransac(*a, **k): raise NotImplementedError
'''

_FAKE_CONFIG_CELL = f'''
class PipelineConfig:
    inventory_ref_z: int = 2
    geometry_z_step_um: float = {GEOMETRY_Z_STEP_UM}
    erosion_px: int = {EROSION_PX}
    training_only_field: int = 99
    _GEOM_FIELDS = ("inventory_ref_z", "geometry_z_step_um")
'''


def write_training_notebook(path: Path, geometry_cell: str = _FAKE_GEOMETRY_CELL) -> Path:
    cells = [
        dict(cell_type="markdown", metadata={}, source=["# miniature training notebook\n"]),
        dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
             source=_FAKE_CONFIG_CELL.splitlines(keepends=True)),
        dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
             source=geometry_cell.splitlines(keepends=True)),
        # A cell that must never run: the loader only lifts named definitions.
        dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
             source=["raise RuntimeError('training cell executed')\n"]),
    ]
    path.write_text(json.dumps(dict(cells=cells, metadata={}, nbformat=4, nbformat_minor=5)),
                    encoding="utf8")
    return path


def section_radius_px(droplet: dict, z: int):
    r2 = droplet["R_um"] ** 2 - (GEOMETRY_Z_STEP_UM * (z - droplet["z_eq"])) ** 2
    return float(np.sqrt(r2) / PIXEL_UM) if r2 > 0 else None


def _component_mask(spec: dict) -> np.ndarray:
    yy, xx = np.ogrid[:HEIGHT, :WIDTH]
    if spec["shape"] == "pixel":
        mask = np.zeros((HEIGHT, WIDTH), bool)
        mask[int(spec["cy"]), int(spec["cx"])] = True
        return mask
    disk = (xx - spec["cx"]) ** 2 + (yy - spec["cy"]) ** 2 <= spec["r"] ** 2
    if spec["shape"] == "c":
        # Remove a wide notch so the convex hull is much larger than the mask.
        notch = (np.abs(yy - spec["cy"]) <= 3) & (xx >= spec["cx"] - 1)
        hole = (xx - spec["cx"]) ** 2 + (yy - spec["cy"]) ** 2 <= (spec["r"] - 4) ** 2
        return disk & ~notch & ~hole
    return disk


def build_run(root: Path, truth=None, nuclei=None) -> SimpleNamespace:
    """Write the synthetic run under `root` and return its paths and tables."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    truth = TRUTH if truth is None else truth
    nuclei = NUCLEI if nuclei is None else nuclei

    image = np.full((N_T, N_Z, 3, HEIGHT, WIDTH), BACKGROUND, np.uint16)
    labels = np.zeros((N_T, N_Z, HEIGHT, WIDTH), np.uint16)
    yy, xx = np.ogrid[:HEIGHT, :WIDTH]
    for t, droplets in truth.items():
        for z in range(N_Z):
            for droplet in droplets:
                radius = section_radius_px(droplet, z)
                if radius is not None:
                    disk = (xx - droplet["cx"]) ** 2 + (yy - droplet["cy"]) ** 2 <= radius ** 2
                    image[t, z, NLS][disk] = CYTOPLASM
                    image[t, z, NPC][disk] = CYTOPLASM
    rows, next_label = [], {}
    for spec in nuclei:
        t, z = spec["t"], spec["z"]
        mask = _component_mask(spec)
        if labels[t, z][mask].any():
            raise AssertionError("synthetic nuclei overlap")
        label = next_label.get((t, z), 0) + 1
        next_label[(t, z)] = label
        labels[t, z][mask] = label
        image[t, z, NLS][mask] = spec["value"]
        ys, xs = np.nonzero(mask)
        rows.append(dict(t=t, z=z, label=label, nucleus_3d_id=spec["nid"],
                         area_px=int(mask.sum()), centroid_x_px=float(xs.mean()),
                         centroid_y_px=float(ys.mean()),
                         expect_geometry=spec["expect"][0], expect_status=spec["expect"][1],
                         expect_reason=spec["expect"][2]))
    expected = pd.DataFrame(rows)

    image_path = root / "raw.tif"
    labels_path = root / "nucleus_instance_hyperstack.tif"
    tifffile.imwrite(image_path, image, photometric="minisblack", metadata={"axes": "TZCYX"})
    tifffile.imwrite(labels_path, labels, photometric="minisblack", metadata={"axes": "TZYX"})
    Path(str(image_path) + ".truth.json").write_text(
        json.dumps({str(t): d for t, d in truth.items()}), encoding="utf8")
    notebook = write_training_notebook(root / "training.ipynb")

    grouped = expected[["t", "z", "label", "nucleus_3d_id", "area_px",
                        "centroid_x_px", "centroid_y_px"]].copy()
    # Best Z = the largest component of each linked ID, lowest Z on ties.
    best = (grouped.sort_values(["area_px", "z"], ascending=[False, True])
            .drop_duplicates(["t", "nucleus_3d_id"]).sort_values(["t", "nucleus_3d_id"])
            .reset_index(drop=True))
    tracked = best.copy()
    tracked["track_id"] = np.arange(1, len(tracked) + 1)
    tracked["true_time_min"] = tracked.t * 6.0
    radial = pd.DataFrame([dict(t=r.t, z=r.z, track_id=r.track_id, theta_index=k, intensity=1.0)
                           for r in tracked.itertuples() for k in range(4)])
    index = pd.DataFrame([dict(t=t, z=z, included=True) for t in range(N_T) for z in range(N_Z)])
    return SimpleNamespace(
        root=root, image_path=image_path, labels_path=labels_path, training_notebook=notebook,
        grouped=grouped, expected=expected, best_z=best, tracked=tracked, radial=radial,
        segmentation_index=index, truth=truth, pixel_size_um=PIXEL_UM,
        nuclear_channel_index=NLS, npc_channel_index=NPC, min_z=MIN_Z)
