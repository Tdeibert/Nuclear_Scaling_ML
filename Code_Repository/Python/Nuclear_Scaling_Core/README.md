# Nuclear_Scaling_Core

Installable Python package holding the project's shared modules
(`nsdb`, `experiment_config`, `project_paths`, `paths_config`, `nsplots`, `radial_surface`).

## Decision record (2026-09-22)

**Decision:** the shared modules formerly in `src/` are packaged as `nuclear_scaling`
and installed in editable mode into each conda environment, instead of being loaded
with `sys.path.insert(...)` from each notebook.

**Why:** every notebook located `src/` with a path relative to its own location.
The September 2026 restructure moved notebooks and scripts, and those imports broke
(`build_db.py` pointed at a `src/` folder that no longer existed). An installed package
is found by Python from any folder, so future reorganizations cannot break imports.

**Alternatives considered:** keeping the modules in one folder and updating each
`sys.path` line. Rejected: it fixes today's breakage but leaves every notebook
dependent on its own location.

## Install (once per environment, on each machine)

```bash
cd ~/Projects/Nuclear_Scaling/Code_Repository/Python/Nuclear_Scaling_Core
pip install -e . --no-deps
```

- `-e` (editable): Python imports the files in this folder directly, so edits take
  effect on the next kernel restart; no reinstall needed.
- `--no-deps`: pip installs nothing else. The environment already provides numpy,
  pandas, etc.; this protects pinned stacks such as `ml_tf_2.15`.
- Environments to install into: `starforge` (local analysis and database work) and
  `ml_tf_2.15` (Cheaha pipeline and training), plus any other environment whose
  notebooks import these modules.
- Check: `python -c "import nuclear_scaling, sys; print(nuclear_scaling.__file__)"`
  should print a path inside this folder.
- Reinstall is only needed if this folder moves or `pyproject.toml` changes.

## Rules

- Notebooks and scripts import with `from nuclear_scaling import nsdb` or
  `from nuclear_scaling.experiment_config import ExperimentConfig`.
  No `sys.path.insert` lines for these modules anywhere.
- Modules inside the package import each other relatively: `from . import nsdb`.
- New shared module: add a `.py` file to `nuclear_scaling/`. No reinstall needed.
- Retiring a module: move it to `Deprecated/Python/Nuclear_Scaling_Core/` following
  the deprecation conventions in `Repository_Structure.json`.
- Bump `version` in `pyproject.toml` when a change breaks existing callers.

## Decision record (2026-10-08): selected-population code lives here

**Decision:** the droplet geometry and the rules that decide the selected analysis
population are two modules of this package, `droplet_geometry.py` and
`droplet_population.py`, with a test suite in `tests/`. The pipeline notebook
(`Image_Segmentation/Large_FOV_Nuclear_Pipeline_v18.1.ipynb`, sections 35c to 35g)
imports them and keeps only configuration, stage calls and editable QC / plotting cells.

**Why:** the same logic existed twice, embedded in notebook cells and saved as scripts in
`Model_Training` (`ransac_droplet_audit.py`, `droplet_overlap_audit.py`,
`segmented_artifact_gate.py`), and one notebook cell reached the scripts through
`sys.path.insert`. There was no package for this workflow, it is infrastructure rather
than analysis, and a rule that removes data needs tests that outlive one session.

**What is where**

- `droplet_population.py`: numpy / pandas only. Circle overlap, the approved 5 %
  whole-droplet rule, accepted / excluded / unresolved decisions, `select_population`
  for downstream tables. Its docstring states which rules are approved, which are the
  existing gate, and which are exploratory.
- `droplet_geometry.py`: needs scipy, scikit-image, tifffile (all in `ml_tf_2.15`).
  Loads the training notebook's geometry functions without running the notebook,
  computes and caches geometry per timepoint, reassigns components to fitted supports,
  and holds the single copy of the enrichment / solidity ring formula.
- Settings are a separate `PopulationSettings` object, not fields of the notebook's
  `PipelineConfig`: that dataclass is hashed into the run ID, so a new field would move
  the run directory and force neural inference again.

**Tests** (`unittest`, because `ml_tf_2.15` has no pytest; they also run under pytest):

```bash
cd ~/Projects/Nuclear_Scaling/Code_Repository/Python/Nuclear_Scaling_Core
python -m unittest discover -s tests -v
```

They write only to temporary directories. `test_parity_with_reviewed_audit` runs the
original `Model_Training` scripts and the new modules on the same synthetic images with
the real training geometry and requires identical results; it takes about a minute and
is skipped if `Model_Training` is not present.

**Left in place on purpose:** the three `Model_Training` scripts above. They are the
reference the parity test compares against and they produced the numbers the 5 %
tolerance was approved on. Retire them only together with that test.

**No reinstall needed:** an editable install picks up new modules. If
`from nuclear_scaling import droplet_geometry` fails in a kernel, the package was never
installed in that environment; run the install command above once.
