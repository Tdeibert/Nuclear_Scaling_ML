# V8 Development Issues

Generated after v7 training run evaluation — May 29, 2026

---

## Issue #1 — Revised 4-class label scheme (High Priority)

Current class 1 (droplet interior) has holes punched in it by NPC and nucleus assignments,
creating a patchy heterogeneous prediction. Biologically incorrect and makes droplet boundary
learning harder.

**Proposed fix:**
- Class 1 redefined as entire filled droplet volume — nucleus and NPC pixels included underneath
- Classes 2 and 3 overlay on top of class 1 in the label map but do not erase it from ground truth
- `make_4class_label` rewritten to fill complete droplet circle first, then assign NPC and nucleus on top
- Model learns "everything inside this boundary is at minimum class 1" — correct biological definition
- Droplet center calculation becomes reliable from a solid predicted circle

**Label scheme:**

| Class | Label | Definition |
|-------|-------|------------|
| 0 | Background | Outside all droplets |
| 1 | Droplet interior | Entire filled droplet volume (base layer, never erased) |
| 2 | NPC ring | NPC signal overlaid on droplet interior |
| 3 | Nucleus interior | NLS-defined ROI overlaid on droplet interior |

---

## Issue #2 — Late timepoint droplet segmentation failure (High Priority)

Droplet counts collapse from ~177 at t=0 to ~19 at t=8/9 despite ~100 real nuclei present.
Segmentation parameters calibrated on T7 do not generalize to fully assembled T8/T9 NPC signal
(complete bright rings vs diffuse boundary signal at earlier timepoints).

**Fix options:**
- Recalibrate segmentation parameters on T8/T9 planes using diagnostic notebook
- Per-timepoint parameter sets in config for early/mid/late assembly stages
- Evaluate membrane channel as alternative droplet boundary signal at late timepoints
  where NPC signal morphology changes dramatically

---

## Issue #3 — Parallel timepoint patch generation (High Priority — Must Have)

Current sequential timepoint loop takes 6+ hours on a single core. Each timepoint is
completely independent and embarrassingly parallel. Must utilize HPC resources fully.

**Implementation plan using `ProcessPoolExecutor`:**
- Each worker receives: timepoint index, path to TIFF file (each worker opens its own memmap),
  config object, worker-specific RNG seed (`cfg.seed + t`), patch ID offset (`t × max_patches_per_timepoint`)
- Each worker returns: summary rows and patch count
- Main process spawns workers up to `min(n_timepoints, available_cores)`, collects and merges
  summary rows, writes combined CSV
- Sentinel files written per worker — resume capability preserved

**SLURM script changes required:**
```bash
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
```

**Target:** Total patch generation time reduced from 6+ hours to ~60-90 minutes
(time of slowest single timepoint).

**Key design decisions:**
- Each worker opens its own `tiff.memmap()` — avoids shared memory complexity, safe since read-only
- Patch filenames prefixed with timepoint index — no collision possible
- Config must be serializable — dataclass is already pickle-safe
- Sentinel files written per worker — resume still works

---

## Issue #4 — Early timepoint channel routing (Medium Priority)

t=0 and t=1 routing to `nucleus_laplacian` instead of expected `npc_ring`.
`nucleus_score_floor` threshold is too low, picking up background NLS texture as genuine
nuclear signal at early timepoints where no nuclear import has occurred.

**Fix:** Raise `nucleus_score_floor` from current value until t=0/1 correctly route to `npc_ring`.

---

## Issue #5 — `min_label_fraction` too aggressive (Medium Priority)

Heavily weighted late timepoints producing far fewer patches than expected.
t=5 with weight=6 produced only 84 patches from 54 droplets. The filter is rejecting
patch attempts at exactly the timepoints with highest biological value.

**Fix:** Lower `min_label_fraction` threshold or investigate patch center placement logic
to ensure patch windows reliably capture label content.

---

## Implementation Order for V8

1. **Fix label scheme (Issue #1)** — changes ground truth generation, affects all downstream training
2. **Fix late timepoint segmentation (Issue #2)** — recalibrate on T8/T9 before generating new patches
3. **Implement parallel patch generation (Issue #3)** — then regenerate full patch dataset
4. **Tune channel routing and label fraction (Issues #4 and #5)** — during patch generation QC

---

## V7 Baseline Metrics (to beat in V8)

| Metric | Train | Validation |
|--------|-------|------------|
| Mean Dice | ~0.62 | ~0.47 |
| NPC Dice | ~0.47 | ~0.32 |
| Nucleus Dice | ~0.48 | ~0.47 |
| Categorical Accuracy | ~82% | ~71% |

**Known limitations of v7 baseline:**
- NPC class nearly absent in predictions — directly caused by late timepoint underrepresentation
- Droplet predictions patchy/heterogeneous — caused by label scheme hole-punching
- Train/val gap significant — insufficient late timepoint patches
