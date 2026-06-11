# Nuclear_Scaling_ML — Claude Code Context

## Project Overview
Xenopus cell-free extract microfluidic droplet imaging pipeline for nuclear assembly analysis.
Goal: segmentation → patch labeling → U-Net training → N/C ratio quantification.
GitHub: Tdeibert/Nuclear_Scaling_ML

## Environment
- Conda env: `ml_env_tf_2.15` (Python 3.10, TF 2.15 — do NOT upgrade)
- CUDA 12.2 / cuDNN 8.9 / numpy pinned to 1.26.4
- HPC: Cheaha (UAB), partition `amperenodes`, user `tdeibert`
- VSCode tunnel: `star-forge` (VS Code pinned to 1.98.2)
- Standard srun: `--partition=amperenodes --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=4:00:00`

## Conda / Python Commands
```bash
conda activate ml_env_tf_2.15
jupyter lab --no-browser --port=8888      # launch notebook on compute node
pip install <pkg> --break-system-packages  # if pip installs are ever needed outside conda
```

## Project Structure
```
Nuclear_Scaling_ML/
├── notebooks/
│   ├── Label_Generation_QC_v1.ipynb      # ACTIVE — label rebuild from scratch
│   └── Nuclear_Segmentation_v8.1.ipynb   # training notebook (stale v8 labels — do not use outputs)
├── scripts/
│   └── build_dirs.py                     # directory scaffold utility
├── configs/                              # YAML/JSON pipeline configs
├── data/
│   ├── raw/                              # original .tif hyperstacks (read-only)
│   ├── patches/                          # extracted image patches
│   └── labels/                          # generated label masks
├── models/                              # saved model weights
├── outputs/                             # inference results, QC figures
└── docs/                                # session notes, ad-hoc references
```

## Experimental System (read before touching segmentation logic)
- **Imaging:** RAN008 oil-interface droplets, Xenopus extract + fluorescent probes
- **Channels:** Ch0=Membrane (dim NE probe), Ch1=NLS (nuclear import reporter), Ch2=NPC
- **Spatial hierarchy:** background → droplet interior → nucleus (NE boundary) → NLS interior
- **4-class labels:** 0=background, 1=droplet interior, 2=NPC on NE, 3=nucleus interior (NLS+)
- **Pixel size:** 0.108 µm/px (50,000 px² ≈ 583 µm²)
- **Assembly progression:** t=0–2 early (NPC puncta only), t=3–6 mid (partial ring), t=7–9 late (complete ring, high NLS)

## Segmentation Parameters (v7, calibrated on control_extract_1.1.tif)
- **Droplet:** percentile-clip NPC p1–p80, local adaptive block=301, blur_sigma=8, offset=-0.05, area 200–1500 µm², circ≥0.70
- **NPC:** mean + 2.0×std across full droplet interior (no erosion zone)
- **Nucleus:** per-droplet Otsu, circ≥0.4, max_droplet_fraction≤0.50

## Known Issues / Active Work
- v8 training used stale v7.0 labels instead of v8.1 — all v8 inference results unreliable
- `Label_Generation_QC_v1.ipynb` is a ground-up label rebuild to fix three failures:
  1. Watershed separation failing at dense late timepoints (t=7–9)
  2. NPC labels misplaced at droplet wall instead of nuclear envelope
  3. Nucleus labels essentially absent from most recent patch set
- All outputs must produce a **binary mask hyperstack TIFF** for ROI generation (do not measure directly)

## Code Style & Workflow Preferences
- Anton writes code independently — provide guidance, logic checks, and feedback; do not paste complete solutions unless explicitly asked
- Watch for: assignment vs. comparison operators (`=` vs `==`), indentation scope errors
- Parallelism: `joblib.Parallel` with loky backend; pass file paths to workers, not memmaps
- Python is the analysis engine; R/tidyverse is downstream visualization only
- Prefer `pathlib.Path` over `os.path`; use dataclasses for structured config objects

## What NOT to Do
- Do not suggest upgrading TF above 2.15 — it was unstable on Cheaha
- Do not add NPC erosion zones — this was a v6 bug, intentionally removed in v7
- Do not measure nuclei directly from label masks — always generate ROIs and apply to raw images
- Do not add generic comments like `# This loops over items` — keep comments meaningful
