# Read-only artifact supervision trace — 2026-10-15

Candidate: `T0_Z6_L212`, review ID `T0_N212`.
Run: `Vulcan_2.5.4_hardneg_hn2392da6c18e0_iezqw_a0cf3b2a13`.
Committed review/pools:
`/data/user/tdeibert/Nuclear_Scaling/Inputs/Training_Data/reviewed_training_commit_20261013_121152_980805`.
Comparison: `control_extract_1.1__cfg-77ad828f/qc/checkpoint_comparison_20261015_160109_686387`.

## Verified from saved arrays and current training definitions

- Reviewed mask: 84,835 pixels, bounding box y660:989, x5042:5390.
- Comparison reference contains all 84,835 pixels, not a different fragment.
- Dedicated patch: `t000_z006_d000212_y00916_x05298_hard_p00000002`.
- Image, label, and weight SHA256 values match the committed manifest.
- Its supervised mask exactly equals the reviewed mask placed in the training
  crop. Heads 4, 5, and 6 have target zero and weight one throughout that mask.
  Other pixels in this negative patch are ignored, not labeled negative.
- Executing the notebook's `p3_load_numpy` preserves the nuclear target zero and
  weight one in the packed target. No training was started.
- Eight overlapping, training-eligible gold patches at T0/Z6 contain no
  positive-weight nuclear-interior positives on the reviewed mask. All their
  overlapping pixels are supervised negatives. One covers the entire mask.
  No gold correction entry exists for this candidate; no conflicting positive
  supervision was found in these same-plane training patches.
- Production preprocessing of the same crop produces an input exactly equal
  to the saved training array: max/mean absolute difference zero across all
  15 channels. This verifies channel/context order and normalization for this
  crop, not equivalence between different crop positions or augmented inputs.

## Sampling reconstruction

Using saved pool metadata, the notebook's split predicates and timepoint
stratifier, sorted patch stems, and seed 42:

- Baseline training rows: 28,747, agreeing with the run log.
- Added dedicated hard-negative rows: 124; total 28,871, agreeing with run metadata.
- Epoch presentations: 3,755 per timepoint x 9 = 33,795.
- Dedicated hard-negative presentations: 179 (~0.53%).
- This candidate's dedicated patch: two presentations per epoch.
- Eight overlapping gold patches also contribute negative supervision; the
  dedicated-patch count is NOT the complete exposure of this artifact.

These are reconstructed planned presentations, not recorded batch-level
observations. No historical batch/augmentation trace was available. The
stratified list is built before model.fit; epoch shuffling changes its order.

## Loss inspection and limits

The current loss includes weighted sigmoid BCE on these negative pixels.
For a false positive, BCE supplies a nonzero gradient toward lower logits.
Dice and focal Tversky on an all-negative target are nearly saturated at high
false-positive probability; their smoothing-based gradients are much smaller.
This is a code-level observation, not a measured explanation of this run.
Class weights apply to each head's combined loss, not only positive pixels.

The comparison still predicts 100% of this reference as nucleus for all three
checkpoints. A missing/zero-weight target or input-normalization mismatch is
not supported for this example. Adequacy of exposure, augmentation effects,
crop-context effects, and optimization remain unresolved. Do not generalize
this single-candidate audit to all reviewed negatives or all Z planes.

## Next bounded test (not executed)

First compare checkpoint predictions on the exact saved training crop versus
the full-image-grid crop, with augmentation disabled. Then, if needed, use an
isolated model copy for a small learning test containing this negative and
reviewed real positives, tracking negative suppression and positive retention.
Do not alter production checkpoints, annotations, or pools. Passing such a
test demonstrates learnability, not generalization or a validated final fix.
