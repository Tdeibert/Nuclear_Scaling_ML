"Machine Learning Pipeline Development" 
# Nuclear Scaling ML Pipeline

A modular microscopy analysis pipeline for **nuclear segmentation, ROI extraction, and quantitative analysis** using machine learning and image processing.

This project is designed for **large-scale microscopy datasets**, including multi-channel, multi-Z, and time-lapse imaging, with compatibility for HPC environments.

---

## 🧬 Project Overview

This pipeline performs:

1. **Image Handling**
   - ND2 → TIFF conversion
   - Hyperstack concatenation
   - Multi-dimensional image support (C, Z, T)

2. **Segmentation (U-Net)**
   - Nuclear classification
   - Probability map generation
   - Binary mask output

3. **ROI Extraction**
   - Identification of individual nuclei
   - Filtering based on size, circularity, and proximity

4. **Quantification**
   - Nuclear area (µm²)
   - N/C ratio calculations
   - Time-resolved measurements

---

## 🧠 Design Philosophy

- **Modular**: Each step is isolated (IO, segmentation, ROI, measurements)
- **Reproducible**: Config-driven workflows
- **Scalable**: Designed for HPC + large datasets
- **Debuggable**: Notebook-friendly but not notebook-dependent

---

## 📁 Repository Structure

```
Nuclear_Scaling/
├── src/                  importable modules (nsdb, nsplots, radial_surface,
│                         experiment_config, paths_config, project_paths)
├── scripts/              runnable entry points
│   ├── build_db.py       rebuild the database from a CSV export
│   ├── database_tools/   schema creation + import utilities (self-contained)
│   ├── scaffold/         build_dirs.py + structure.{json,py,yaml,yml}
│   └── training/         acquisition_metadata.py, mine_cap_negatives.py, rigs/
├── sql/                  schema_v2.sql, views.sql
├── configs/              experiment JSON + environments/*.yml
├── notebooks/
│   ├── analysis/         Nuclear_Scaling_Analysis, Large_FOV_* diagnostics
│   ├── segmentation/     Gold_Standard_*, Label_Generation_QC, Large_FOV v17-v18.1
│   ├── model_training/   vulcan_training_1.1
│   ├── utilities/        loaders, converters, plotting.py
│   ├── refactors/        in-flight rewrites (gitignored)
│   └── deprecated/       superseded versions, by area
├── R/                    tidyverse downstream visualisation
├── data/                 (gitignored)
│   ├── raw/              original .tif hyperstacks -- read-only
│   ├── derived/          parquet radial profiles etc.
│   └── db/               nuclear_scaling.db, nuclear_scaling_test.db
│       └── exports/      CSV staging for the importer -- temporary
├── models/               saved weights (Vulcan_1.0.keras)
├── outputs/figures/      generated figures
└── docs/                 talks, references
```


---
🖥️ HPC Usage

Designed for:

GPU acceleration (U-Net inference)
Batch processing
Large memory datasets

Typical workflow:

Transfer scripts + configs to cluster
Run segmentation jobs
Pull results locally for analysis

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Tdeibert/Nuclear_Scaling_ML.git
cd Nuclear_Scaling_ML

conda env create -f environment.yml
conda activate nuclear_scaling
