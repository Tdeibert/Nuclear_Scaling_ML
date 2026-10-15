# Notebook support files and cleanup — 2026-10-15

Active notebooks: `vulcan_training_2_5_3.ipynb` and
`../Image_Segmentation/Large_FOV_Nuclear_Pipeline_v18.1.ipynb`.

## Archived, not destroyed

Recoverable archive (outside the repository):
`/home/tdeibert/Projects/Nuclear_Scaling/repo_cleanup_archive/20261015_143510`.
Its `manifest.json` records original paths, archive paths, sizes, and SHA256.
Every copy was checksum-verified before removal from the working tree.

- 15 `.pre_*.ipynb` snapshots from Model_Training and Image_Segmentation.
- Three historical notebook editors: `append_survivor_probability_cell.py`,
  `restore_v18_1_audits.py`, and `refactor_v18_1_vulcan.py`. These are not runtime
  dependencies; do not rerun them against the current refactored notebook.
- Six duplicate definition files: `nucleus_area_audit.py`, `post_training_audit.py`,
  `survivor_audit.py`, `survivor_component_audit.py`, `survivor_envelope_audit.py`,
  and `survivor_probability_audit.py`. Every top-level function/class in these
  files was AST-identical to a definition already in the active notebooks.
  No Python/notebook import of these modules was found in the Python tree.

No notebook behavior was changed by this cleanup. Earlier uncommitted
checkpoint-comparison and segmentation-timer work was preserved.

## Keep: runtime support, tests, and unresolved legacy code

Hard-negative review, correction, and workflow modules are live dependencies,
not disposable patch scripts. Keep their tests. Keep checkpoint comparison,
empty-plane supervision audit, launch/resume scripts, batch tooling, core
geometry/population modules, and training support modules.

Other audit/contract source files are retained where definitions differ from
the notebooks or have been superseded by core modules; absence of an import
alone does not establish that their functionality is safe to discard.
Historical versions under `Depricated` are not timestamped backups and remain.

Pending: `../Image_Segmentation/v18_2_repair_writeback.py` contains a repair
write-back change not present in the current notebook (`merge_repaired_planes`
is absent). It was NOT applied or deleted: integrating it changes scientific
measurement/export behavior and needs a separate compatibility review.

To restore a cleanup file, copy its archived version back to the original path
recorded in the manifest, checking for newer work before overwriting anything.
