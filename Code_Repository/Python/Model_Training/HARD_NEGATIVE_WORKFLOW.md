# Reviewed hard negatives for the next Vulcan model

This workflow does not change original pools/masks or start training.

## 1. Review in v18.1

The new opt-in bottom section nominates previously reviewed IDs but exports ALL
their component masks with blank approvals. Inspect previews and review.csv.
Set decision=artifact_negative and reviewer=your name only when the ENTIRE exact
center-plane mask is non-nuclear. Use real, uncertain, or skip otherwise.
Mixed masks must be skipped; arbitrary edited-region import is not implemented.
Adjacent-plane previews project the center mask for context, not annotation.
Never infer negative truth from stitching exclusion, multiplicity, or T0 alone.

## 2. Build in the training kernel

Set HARD_NEGATIVE_REVIEW_DIR in the new bottom section. Use the original training
configuration and original validated p3_rows(cfg). The builder reuses existing
extract_input_stack and ensure_norm_stats: five-plane input and global normalization.
Only approved pixels get nuclear interior/edge/equatorial target 0 and weight 1.
Other pixels/heads remain unknown and zero-weighted; original positive patches stay.
Conflicts with any supplied gold nuclear-positive label abort instead of overriding it.
This does not repair contaminated positive labels in an old pool.

The complete XY crop must avoid the spatial holdout plus margin; temporal holdout
is also excluded. Withheld tiles are reported, not silently added to validation.
Large masks are tiled. Small masks remain sparse; weight/sampling balance needs later
evaluation. Files are hashed; incomplete or modified pools cannot be loaded.

## 3. Opt into a NEW training run

The training opt-in cell makes a separate cfg with a new model name and pool fingerprint.
The baseline cfg still resolves the original gold/classical pool paths. A p3_split
wrapper appends hard negatives to training only; validation is unchanged and existing
timepoint balancing still applies. No training is launched automatically.

Restarting the kernel or redefining p3_split removes the hook: re-enable before training.
Never resume an old model directory. Retain config and hard-negative pool provenance
with the new model. Evaluate independent held-out artifacts and real early nuclei.

Candidate IDs are run-specific. Same raw image/calibration required. Gold conflict
checks cannot protect annotations absent from the supplied validated gold pool.
