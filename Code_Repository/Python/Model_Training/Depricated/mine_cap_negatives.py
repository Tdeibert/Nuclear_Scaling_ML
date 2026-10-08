"""
mine_cap_negatives.py — hard-negative mining for droplet-cap artifacts.

The problem
-----------
A droplet cap (a plane sliced near the top/bottom of a droplet) is a small,
round, filled disc — geometrically almost identical to a nucleus in a single
Z-plane. The 2D U-Net has no Z context and calls these "nucleus". This has
persisted across model versions because parameter tuning cannot fix a gap in
training coverage: if patches were only drawn from in-focus equatorial planes,
the model has never seen a cap.

Why the labels here are trustworthy
-----------------------------------
The existing labels come from the v7 classical pipeline, which almost certainly
also calls caps nuclei — so labelling these patches with it would teach the
same error. Instead, labels are derived from GEOMETRY:

    A nucleus of radius r sits inside a droplet of radius R. It can only
    intersect planes within +/- r of the droplet's equatorial plane.

Measured nucleus/droplet area ratio ~0.28 => r/R ~ 0.53. So for any plane with

    u = |z - z_equator| * z_step / R  >  U_SAFE  (default 0.75, conservative)

that droplet contains NO nucleus and NO nuclear envelope in that plane. The
label is therefore known a priori:

    nucleus = 0, NPC = 0, droplet = droplet mask, background = everything else

No classical segmentation is involved, so no upstream error propagates in.

What it produces
----------------
    <out>/images/cap_neg_t{t}_z{z}_d{droplet}_{i}.npy   (P, P, 3) float32
    <out>/labels/cap_neg_t{t}_z{z}_d{droplet}_{i}.npy   (P, P, 4) uint8
    <out>/manifest.csv                                  provenance per patch

Also runs an optional BASELINE CHECK: applies the current model to the mined
patches and reports how much nucleus signal it hallucinates on planes where a
nucleus is geometrically impossible. That quantifies the problem before
retraining and gives a number to measure improvement against.

Usage
-----
    python mine_cap_negatives.py --run /path/to/Runs/<run_id> \
        --out /path/to/Inputs/Training_Data/cap_negatives_v1 \
        --model /path/to/Inputs/Models/unet_nuclear_scaling_v9_best.keras

NOTE: untested against real data. Verify the label encoding matches what the
v9 training script expects before training on this (see LABEL FORMAT below).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage import measure

# ── label channel convention (matches the 4-class scheme) ────────────────
BACKGROUND, DROPLET, NPC, NUCLEUS = 0, 1, 2, 3
N_CLASSES = 4

# Geometric safety margin. r/R measured at ~0.53; 0.75 leaves generous headroom
# for non-spherical nuclei, off-centre nuclei, and equatorial-plane error.
U_SAFE_DEFAULT = 0.75

MIN_DROPLET_PX = 2000        # ignore specks in the droplet instance mask


# =============================================================================
# Geometry
# =============================================================================

def droplet_equators(drop_path: Path, pixel_size_um: float,
                     min_area_px: int = MIN_DROPLET_PX) -> pd.DataFrame:
    """For each (t, droplet_id): the Z of maximum cross-section (the equator)
    and the equatorial radius R in microns.

    Memmapped — the instance hyperstack is far too large for RAM.
    """
    drop = tifffile.memmap(str(drop_path), mode="r")     # (T, Z, Y, X)
    rows = []
    for t in range(drop.shape[0]):
        for z in range(drop.shape[1]):
            plane = np.asarray(drop[t, z])
            if plane.max() == 0:
                continue
            for r in measure.regionprops(plane):
                if r.area >= min_area_px:
                    rows.append((t, z, r.label, r.area,
                                 r.centroid[0], r.centroid[1]))
    dz = pd.DataFrame(rows, columns=["t", "z", "droplet_id", "area_px",
                                     "cy", "cx"])
    if dz.empty:
        raise RuntimeError("no droplets found — check droplet instance hyperstack")

    eq = (dz.loc[dz.groupby(["t", "droplet_id"])["area_px"].idxmax()]
            .rename(columns={"z": "z_eq", "area_px": "area_eq_px"})
            [["t", "droplet_id", "z_eq", "area_eq_px"]])
    eq["R_um"] = np.sqrt(eq.area_eq_px / np.pi) * pixel_size_um
    return dz.merge(eq, on=["t", "droplet_id"], how="left")


def mark_safe_negatives(dz: pd.DataFrame, z_step_um: float,
                        u_safe: float = U_SAFE_DEFAULT) -> pd.DataFrame:
    """u = normalised distance from the droplet equator (0 = equator, 1 = pole).
    u > u_safe  =>  no nucleus can intersect this plane of this droplet."""
    dz = dz.copy()
    dz["u"] = (dz.z - dz.z_eq).abs() * z_step_um / dz.R_um
    dz["safe_negative"] = dz.u > u_safe
    return dz


# =============================================================================
# Patch extraction
# =============================================================================

def _crop(arr: np.ndarray, cy: int, cx: int, p: int):
    """Centred crop with reflect padding at the FOV edge. Returns (patch, ok)."""
    h, w = arr.shape[-2:]
    half = p // 2
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half
    py0, py1 = max(0, -y0), max(0, y1 - h)
    px0, px1 = max(0, -x0), max(0, x1 - w)
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    sub = arr[..., y0:y1, x0:x1]
    if py0 or py1 or px0 or px1:
        pad = [(0, 0)] * (sub.ndim - 2) + [(py0, py1), (px0, px1)]
        sub = np.pad(sub, pad, mode="reflect")
    return sub, sub.shape[-2:] == (p, p)


def mine(run_dir: Path, out_dir: Path, patch: int = 512,
         u_safe: float = U_SAFE_DEFAULT, max_patches: int | None = None,
         min_caps_per_patch: int = 2, seed: int = 0) -> pd.DataFrame:

    cfg = json.loads((run_dir / "config.json").read_text())
    px = float(cfg.get("pixel_size_um", 0.2167))
    zs = float(cfg.get("z_step_um", 1.0))
    print(f"pixel_size_um={px}  z_step_um={zs}  u_safe={u_safe}")

    masks = run_dir / "mask_tifs"
    drop_p = masks / "droplet_instance_hyperstack.tif"
    img_p = (run_dir.parent.parent / "Inputs" / "Raw_Images"
             / cfg.get("input_image_name", ""))
    for p in (drop_p, img_p):
        if not p.exists():
            raise FileNotFoundError(p)

    dz = mark_safe_negatives(droplet_equators(drop_p, px), zs, u_safe)
    safe = dz[dz.safe_negative]
    print(f"{len(safe)} safe-negative (t, z, droplet) instances "
          f"out of {len(dz)} total droplet cross-sections")
    if safe.empty:
        raise RuntimeError("no safe negatives — lower --u-safe or check geometry")

    drop = tifffile.memmap(str(drop_p), mode="r")
    img = tifffile.memmap(str(img_p), mode="r")          # (T, Z, C, Y, X)

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    recs = []

    # Group by plane: one plane may hold several cap cross-sections, and a
    # patch containing several caps is worth more than one containing a single
    # cap surrounded by background.
    for (t, z), grp in safe.groupby(["t", "z"]):
        dplane = np.asarray(drop[int(t), int(z)])
        iplane = np.asarray(img[int(t), int(z)]).astype(np.float32)   # (C, Y, X)
        safe_ids = set(grp.droplet_id.astype(int))

        # Candidate centres: cap centroids, shuffled so we do not always take
        # the same corner of the FOV.
        cands = grp[["cy", "cx", "droplet_id"]].to_numpy()
        rng.shuffle(cands)

        for cy, cx, did in cands:
            cy, cx, did = int(cy), int(cx), int(did)

            lab_crop, ok1 = _crop(dplane, cy, cx, patch)
            img_crop, ok2 = _crop(iplane, cy, cx, patch)
            if not (ok1 and ok2):
                continue

            # Every droplet appearing in this patch must itself be a safe
            # negative. If any droplet in view is near its own equator, a real
            # nucleus could be present and the all-zero nucleus label would be
            # WRONG — so discard the patch rather than risk a bad label.
            present = set(np.unique(lab_crop)) - {0}
            if not present.issubset(safe_ids):
                continue
            if len(present) < min_caps_per_patch:
                continue

            # ── LABEL FORMAT ────────────────────────────────────────────
            # (P, P, 4) uint8, independent binary channels (the v15+ model
            # uses independent sigmoids, not softmax). Verify this matches
            # the v9 training script before use.
            y = np.zeros((patch, patch, N_CLASSES), dtype=np.uint8)
            droplet_mask = (lab_crop > 0)
            y[..., DROPLET] = droplet_mask
            y[..., BACKGROUND] = ~droplet_mask
            # NPC and NUCLEUS stay 0 — guaranteed by geometry.

            x = np.transpose(img_crop, (1, 2, 0))          # (P, P, C)

            stem = f"cap_neg_t{int(t):03d}_z{int(z):03d}_d{did:04d}"
            np.save(out_dir / "images" / f"{stem}.npy", x.astype(np.float32))
            np.save(out_dir / "labels" / f"{stem}.npy", y)

            recs.append(dict(
                stem=stem, t=int(t), z=int(z), droplet_id=did,
                cy=cy, cx=cx, n_droplets_in_patch=len(present),
                u=float(grp.loc[grp.droplet_id == did, "u"].iloc[0]),
                z_eq=int(grp.loc[grp.droplet_id == did, "z_eq"].iloc[0]),
                R_um=float(grp.loc[grp.droplet_id == did, "R_um"].iloc[0]),
                droplet_frac=float(droplet_mask.mean()),
            ))
            if max_patches and len(recs) >= max_patches:
                break
        if max_patches and len(recs) >= max_patches:
            break

    man = pd.DataFrame(recs)
    man.to_csv(out_dir / "manifest.csv", index=False)
    print(f"\nwrote {len(man)} patches to {out_dir}")
    if not man.empty:
        print("\nper timepoint:")
        print(man.groupby("t").size())
        print(f"\nu range: {man.u.min():.2f}–{man.u.max():.2f}  "
              f"(all > {u_safe}, so nucleus-free by geometry)")
        print(f"mean droplet coverage per patch: {man.droplet_frac.mean():.1%}")
    return man


# =============================================================================
# Baseline: how badly does the CURRENT model fail on these?
# =============================================================================

def baseline_check(out_dir: Path, model_path: Path, nucleus_threshold: float = 0.5,
                   n: int = 40) -> None:
    """Apply the current model to the mined patches. Every nucleus prediction
    here is a false positive by construction, so this is a clean error rate —
    and the number to beat after retraining."""
    import tensorflow as tf

    man = pd.read_csv(out_dir / "manifest.csv")
    if man.empty:
        print("no patches to check")
        return
    sel = man.sample(min(n, len(man)), random_state=0)

    model = tf.keras.models.load_model(model_path, compile=False)
    fracs, maxes = [], []
    for stem in sel.stem:
        x = np.load(out_dir / "images" / f"{stem}.npy")[None, ...]
        p = model(x, training=False).numpy()[0, ..., NUCLEUS]
        fracs.append(float((p > nucleus_threshold).mean()))
        maxes.append(float(p.max()))

    fracs, maxes = np.array(fracs), np.array(maxes)
    print(f"\nBASELINE on {len(sel)} geometrically nucleus-free patches:")
    print(f"  patches with ANY nucleus pixel : {(fracs > 0).mean():.1%}")
    print(f"  mean false-positive area frac  : {fracs.mean():.4f}")
    print(f"  median max nucleus probability : {np.median(maxes):.3f}")
    print("  (all of this is error — a nucleus cannot exist in these planes)")


# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", type=Path, default=None,
                    help="if given, run the baseline false-positive check")
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--u-safe", type=float, default=U_SAFE_DEFAULT)
    ap.add_argument("--max-patches", type=int, default=None)
    ap.add_argument("--min-caps", type=int, default=2)
    args = ap.parse_args()

    mine(args.run, args.out, patch=args.patch, u_safe=args.u_safe,
         max_patches=args.max_patches, min_caps_per_patch=args.min_caps)

    if args.model:
        baseline_check(args.out, args.model)


if __name__ == "__main__":
    main()
