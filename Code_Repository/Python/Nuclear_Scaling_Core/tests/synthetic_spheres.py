"""Image-like synthetic run for exercising the REAL training geometry code.

Droplets are spheres sectioned at the training geometry Z step, drawn as bright
filled discs in the NPC channel so that the training watershed inventory, wall
finder, RANSAC circle fit and sphere fit all have something genuine to work on.
Nuclei are scattered at random (fixed seed), so the run contains assigned,
partial, ambiguous and missing parents without any of them being scripted.

Used only by the parity test. Everything is written to a temporary directory.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tifffile
from skimage import measure

PIXEL_UM = 0.1625
GEOMETRY_Z_STEP_UM = 2.18      # must equal the training notebook's value
N_T, N_Z, HEIGHT, WIDTH = 3, 20, 620, 1040
MEMBRANE, NLS, NPC = 0, 1, 2
MIN_Z = 6
R_UM, Z_EQ = 16.0, 13.0

# (cx, cy) in pixels. Pairs at T0: 1+2 overlap ~10 %, 3+4 overlap ~2 %; 5-7 isolated.
_T0 = [(130, 130), (286, 130), (520, 130), (704, 130), (900, 140), (160, 450), (480, 460)]
# T1: the first pair is pulled apart; everything else stays.
_T1 = [(120, 130), (330, 130), (560, 130), (790, 130), (900, 400), (160, 450), (480, 460)]
# T2: a different pair (6+7) overlaps instead.
_T2 = [(130, 130), (360, 130), (600, 130), (850, 130), (900, 420), (300, 450), (452, 450)]
DROPLETS = {0: _T0, 1: _T1, 2: _T2}


def section_radius_px(z: int):
    r2 = R_UM ** 2 - (GEOMETRY_Z_STEP_UM * (z - Z_EQ)) ** 2
    return float(np.sqrt(r2) / PIXEL_UM) if r2 > 0 else None


def build_run(root: Path, seed: int = 7) -> SimpleNamespace:
    root = Path(root)
    for name in ("seg", "obj", "qc", "masks"):
        (root / name).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.ogrid[:HEIGHT, :WIDTH]
    image = rng.normal(100, 5, (N_T, N_Z, 3, HEIGHT, WIDTH))
    labels = np.zeros((N_T, N_Z, HEIGHT, WIDTH), np.int32)
    droplets = np.zeros((N_T, N_Z, HEIGHT, WIDTH), np.int32)
    rows, next_id = [], 1
    for t in range(N_T):
        inside_any = np.zeros((N_Z, HEIGHT, WIDTH), bool)
        for z in range(N_Z):
            radius = section_radius_px(z)
            if radius is None:
                continue
            for cx, cy in DROPLETS[t]:
                inside_any[z] |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            image[t, z, NPC][inside_any[z]] += 900
            image[t, z, NLS][inside_any[z]] += 500
            # Model-style droplets: connected components of the droplet mask.
            droplets[t, z] = measure.label(inside_any[z], connectivity=2)
        # Random nuclei: linked across 2-5 consecutive planes, mixed brightness.
        next_label = {z: 1 for z in range(N_Z)}
        # One small nucleus in the lens of each strongly overlapping pair, so
        # an ambiguous parent is guaranteed to occur.
        scripted = {0: [(208.0, 130.0, 5.0)], 2: [(376.0, 450.0, 5.0)]}.get(t, [])
        for attempt in range(160 + len(scripted)):
            if attempt < len(scripted):
                cx, cy, radius = scripted[attempt]
            elif rng.random() < 0.8:    # mostly in or near a droplet, some anywhere
                dx, dy = DROPLETS[t][int(rng.integers(len(DROPLETS[t])))]
                angle, reach = rng.uniform(0, 2 * np.pi), rng.uniform(0, 105)
                cx, cy = dx + reach * np.cos(angle), dy + reach * np.sin(angle)
            else:
                cx, cy = rng.uniform(20, WIDTH - 20), rng.uniform(20, HEIGHT - 20)
            if attempt >= len(scripted):
                radius = rng.uniform(4, 16)
            z0 = int(rng.integers(MIN_Z, N_Z - 2))
            planes = range(z0, min(N_Z, z0 + int(rng.integers(2, 6))))
            value = rng.choice([1500.0, 1500.0, 1500.0, 60.0])   # some dim "artifacts"
            if attempt < len(scripted):
                planes, value = range(12, 15), 1500.0
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            grow = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius + 3) ** 2
            if any(labels[t, z][grow].any() for z in planes):
                continue
            for z in planes:
                label = next_label[z]
                next_label[z] += 1
                labels[t, z][mask] = label
                image[t, z, NLS][mask] += value
                ys, xs = np.nonzero(mask)
                rows.append(dict(t=t, z=z, label=label, nucleus_3d_id=next_id,
                                 area_px=int(mask.sum()), centroid_x_px=float(xs.mean()),
                                 centroid_y_px=float(ys.mean()), class_name="nucleus"))
            next_id += 1
    image = np.clip(image, 0, 65535).astype(np.uint16)

    paths = dict(image=root / "raw.tif", labels=root / "masks/nucleus_instance_hyperstack.tif",
                 droplets=root / "masks/droplet_instance_hyperstack.tif",
                 index=root / "seg/segmentation_index.pkl")
    tifffile.imwrite(paths["image"], image, photometric="minisblack")
    tifffile.imwrite(paths["labels"], labels, photometric="minisblack")
    tifffile.imwrite(paths["droplets"], droplets, photometric="minisblack")
    pd.DataFrame([dict(t=t, z=z, included=True) for t in range(N_T) for z in range(N_Z)]
                 ).to_pickle(paths["index"])
    grouped = pd.DataFrame(rows)
    grouped["nucleus_area_um2"] = grouped.area_px * PIXEL_UM ** 2
    grouped.to_pickle(root / "obj/plane_objects.pkl")
    grouped.to_pickle(root / "obj/grouped_z_objects.pkl")

    # The attributes the reviewed audit scripts read from the notebook's cfg.
    cfg = SimpleNamespace(
        input_image_path=paths["image"], nucleus_instance_hyperstack_path=paths["labels"],
        droplet_instance_hyperstack_path=paths["droplets"], segmentation_index_path=paths["index"],
        obj_dir=root / "obj", qc_dir=root / "qc", seg_dir=root / "seg",
        pixel_size_um=PIXEL_UM, nuclear_channel_index=NLS, npc_channel_index=NPC,
        membrane_channel_index=MEMBRANE, focus_min_z=MIN_Z)
    return SimpleNamespace(root=root, cfg=cfg, grouped=grouped, paths=paths)
