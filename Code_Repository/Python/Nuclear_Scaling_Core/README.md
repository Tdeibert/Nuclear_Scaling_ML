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
