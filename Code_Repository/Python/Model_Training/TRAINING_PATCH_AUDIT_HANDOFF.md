# Vulcan training-patch artifact audit — handoff

Prepared 2026-10-21 (America/Chicago). Paths and pool counts checked on disk.

## Objective and user requirements

Investigate whether artifacts are present as supervised nuclear positives in
the actual training patches and could explain persistent downstream artifacts.
This is an investigation, not an assumption that contaminated labels are the cause.
Build an intuitive notebook workflow with:

1. One contained audit/setup cell: load the effective training data, validate
   provenance, summarize supervision, nominate candidates, and open/resume a review.
2. One interactive decision/resolver cell: inspect images and exact target masks,
   classify each flagged region, save progress, then explicitly commit decisions.

The user works through a remote Cheaha/Jupyter connection. Do not require managing
image folders or editing CSVs manually. Include progress bars, elapsed timers,
and useful ETA for long operations. Support a fresh-kernel recovery entry point;
do not require running the entire training or segmentation notebook.

Do not launch training, inference, background jobs, change existing segmentation,
or silently rebuild pools. New reviewed corrections require an explicit commit.
Use new versioned outputs and a new model identity for any later training.

## Working code and environment

Repository Python root:
`/home/tdeibert/Projects/Nuclear_Scaling/Code_Repository/Python`

- Training notebook: `Model_Training/vulcan_training_2_5_3.ipynb`
  (filename retained even though the latest model is 2.5.4).
- Segmentation notebook: `Image_Segmentation/Large_FOV_Nuclear_Pipeline_v18.1.ipynb`.
- Python: `/home/tdeibert/.conda/envs/ml_tf_2.15/bin/python`.
- Reusable geometry/ROI code: `Nuclear_Scaling_Core/nuclear_scaling/`.
- Read current repository instructions and git status; preserve uncommitted work.
- Do not import/execute an entire notebook to obtain a few definitions: some cells
  trigger generation/training/export. Prefer explicit definition-only loading or
  reusable helpers with a lightweight audit-only configuration.

## Exact data paths

### Raw image and original manual annotations

Raw image:
`/data/user/tdeibert/Nuclear_Scaling/Inputs/Raw_Images/Condition/Control/control_extract_1.1.tif`

Shape used by this pipeline: `(T,Z,C,Y,X) = (10,20,3,3889,5732)`.
Raw channels: membrane=0, NLS=1, NPC=2. Pixel size=0.1625 µm;
training/segmentation Z step=2.0 µm. Geometry fitting separately uses 2.18 µm;
do not conflate these. Nuclear target Z floor is **6 inclusive**.

Original ROI directory:
`/data/user/tdeibert/Nuclear_Scaling/Inputs/Raw_Images/Condition/Control/Regions_of_Interest`

- `NucleiRoiSet.zip`
- `DropletRoiSet.zip`
- `NPCRoiSet.zip`

Use exact paths and SHA256 from the baseline pool manifest, not a guessed sibling
archive. ImageJ ROI T/Z positions are explicitly 1-based; readers convert to zero-based.
Original nucleus ROIs have no persistent nucleus identity across Z; the existing
`nuclear_scaling.roi_zcluster.assign_z` reconstructs columns and flags uncertain ones.
Do not treat its inferred links as manual truth.

### Baseline gold pool — 42,101 stored patches

`/data/user/tdeibert/Nuclear_Scaling/Inputs/Training_Data/reviewed_patches_Vulcan_2.5.3_phase2`

Contents useful for auditing:

- `manifest.json`: completion status, generation/gold hashes, ROI paths/digests,
  configuration, label/weight layouts, and patch count.
- `patch_metadata.csv`: stem, t, z, droplet, cy, cx, sample_class, sample_weight.
- `images/img_<stem>.npy`: normalized training inputs.
- `labels/lab_<stem>.npy`: original stored targets.
- `weights/wgt_<stem>.npy`: per-head supervision weights.
- `qc/norm_stats.json`: saved normalization statistics.
- `candidate_qc.csv`: generation candidate QC.
- `droplet_roi_flags.csv`, `npc_roi_flags.csv`.
- `z_cluster_flags.csv`, `z_cluster_summary.json`.
- `NPCRoiSet_orphans.zip`: inspect as annotation context, not automatic negatives.
- `manifest.pre_hash_migration.json`: historical provenance only; not active config.

Do not assume a patch's sample_class describes every object in its crop. A crop
anchored on an empty droplet can include genuine nuclei in neighboring droplets.
Likewise, a positive in an `empty_focal_plane` patch is not automatically wrong.

### Committed review, gold corrections, and added negatives

Root:
`/data/user/tdeibert/Nuclear_Scaling/Inputs/Training_Data/reviewed_training_commit_20261013_121152_980805`

- `manifest.json`: complete transaction; authoritative paths to the three children.
- `review/review.csv`: decisions/reviewer/notes and available failure-type metadata.
- `review/candidates.json`: exact candidate IDs, T/Z, bounding boxes, source
  instance IDs, mask files and SHA256.
- `review/masks/*.npy`: reviewed exact masks, in bounding-box coordinates.
- `gold_corrections/manifest.json`: **185** corrected gold patches; original label/
  weight paths and digests, corrected paths/digests, mask paths, candidate IDs,
  corrected pixel counts, removed positives, withheld records.
- `gold_corrections/labels/`, `weights/`, `masks/`.
- `hard_negatives/manifest.json`: **124** additional patches, source reviews,
  training signature, correction provenance, per-file hashes and patch records.
- `hard_negatives/images/`, `labels/`, `weights/`.
- `hard_negatives/patch_metadata.csv`, `heldout_candidates.csv`.

Committed decision counts: artifact_negative=175, real=70, skip=11, uncertain=6.
These are review decisions, not 175 additional training patches; masks can be
tiled or withheld. Do not automatically reinterpret uncertain/skip decisions.

Source review before the versioned commit:
`/data/user/tdeibert/Nuclear_Scaling/Runs/control_extract_1.1__cfg-9c4d8e9e/qc/hard_negative_combined_corrected_20261009`

Audit the **effective** labels: original pool rows -> apply `gold_corrections`
overrides by stem -> original split -> append eligible hard-negative rows. Show
original and effective labels side-by-side. Original positives already corrected
are historical findings, not necessarily current training contamination.

### Model/run provenance and actual split

Training run:
`/home/tdeibert/Projects/Nuclear_Scaling/Models/runs/Vulcan_2.5.4_hardneg_hn2392da6c18e0_iezqw_a0cf3b2a13`

- `config.json`: actual run configuration (many values serialized as strings).
- `phase3_run.json`: pool provenance, active mask heads, class weights, split sizes.
- `qc/history_s42.csv`: training history.
- `resume/s42/state.json`: stopped after epoch 18; best epoch 4.
- `resume/s42/ckpt-18.index` and matching `.data-*`: model/optimizer checkpoint.
- `Vulcan_2.5.4_hardneg_hn2392da6c18e0_iezqw_s42_best.keras`: best epoch-4 model.

Best-model SHA256:
`ec93d8bd4fd4731deb019a4e3d2c9cbf73adadeafb3277e7df0c2a617feb0532`.

Old model: `/data/user/tdeibert/Nuclear_Scaling/Inputs/Models/Vulcan_2.5.3.keras`.

Observed split: 28,747 baseline training rows + 124 hard negatives = 28,871;
spatial validation=4,526; temporal holdout=3,767. Actual baseline census was all
gold-source rows despite `label_source='roi+classical'`. Do not assume an old
classical directory participates; reconstruct the run's reader/deduplication.

Spatial holdout tile=(1,2), tile size=2048 px, overlap=205 px, margin=256 px;
temporal holdout T=5. Use existing split functions rather than approximating the
rules. Added negatives use the stricter **whole-crop** holdout intersection guard.
Validation and temporal holdout must not be corrected or reused as training.
If heldout annotations appear wrong, quarantine/report them separately; do not
tune labels to the model and continue calling the result independent validation.

### Segmentation, reviewed artifacts, and existing audit evidence

Current segmentation run:
`/data/user/tdeibert/Nuclear_Scaling/Runs/control_extract_1.1__cfg-77ad828f`

- `pipeline_config.json`, `config.json`, `manifest.json`.
- `objects/grouped_z_objects.pkl`: T/Z/instance labels, linked nuclear IDs,
  bounding boxes, areas, centroids, probability summaries.
- `objects/plane_objects.pkl`, `objects/best_z_nuclei.pkl`.
- `mask_tifs/nucleus_instance_hyperstack.tif`.
- `mask_tifs/droplet_instance_hyperstack.tif`.
- `segmentation/vulcan_all_heads.tif` and `.json`: full-head probabilities;
  consult metadata and do not apply sigmoid again.
- `tracking/best_z_nuclei_timed.pkl`: actual acquisition time in minutes.
- `analysis/selected_population/populations/pop-89f574a946/`: population manifest,
  `component_decisions.pkl`, accepted/excluded/unresolved tables, geometry/overlap QC.
  Excluded/unresolved status is NOT an artifact label.
- `qc/checkpoint_comparison_20261015_160109_686387/`: `cases.csv`, `metrics.csv`,
  manifest, reference masks/raw previews, and three-model probability `.npz` files.

Old segmentation/review source:
`/data/user/tdeibert/Nuclear_Scaling/Runs/control_extract_1.1__cfg-9c4d8e9e`.
Candidate labels/linked IDs are run-specific; join by verified image coordinates,
T/Z, and mask provenance, never by assuming the same integer label across runs.

Existing investigation notes in Model_Training:

- `ARTIFACT_TRACE_T0_Z6_L212.md`: completed end-to-end negative supervision trace.
- `HARD_NEGATIVE_WORKFLOW.md`: original workflow; later interactive/correction
  extensions live in the Python modules listed below.
- `REPOSITORY_CLEANUP.md`: removed patchers/backups and recovery locations.
- `RESUME_VULCAN_HARDNEG.md`: training resume context; do not launch it for this audit.

## Input/label semantics — essential

- Input shape `(512,512,15)`: five adjacent Z planes, **plane-major [NLS,NPC,Mem]**.
  Center-plane channels are indices 6,7,8. Context is clipped at stack boundaries.
- Inputs are globally normalized per timepoint, not per-crop auto-contrast.
  Show raw and normalized views with explicit display scaling. Do not use
  independently contrast-stretched pictures to quantify enrichment.
- Labels `(512,512,10)`; weights `(512,512,9)`.
- Storage indices: 0 background, 1 droplet interior, 2 droplet edge, 3 NPC,
  4 nucleus interior, 5 nucleus edge, 6 nucleus equatorial, 7 abnormal nucleus,
  8 z_offset, 9 droplet_source (metadata, not a loss head).
- Mask targets 0/1; 255 means unannotated. Weight zero means no supervision.
- Z-offset storage scale is 10 units/µm; 255 is sentinel, not a distance.
- Active model output has seven mask logits + Z-offset; do not confuse output
  channel 7 with stored abnormal-nucleus channel 7.
- Signed crop centers occur (e.g. x=-1). Parse signed coordinates; distinguish
  padded/out-of-image pixels and preserve actual extraction/padding conventions.

## What is already known (avoid repeating or overstating)

`T0_Z6_L212` is a confirmed reviewed artifact, 84,835 pixels, bbox
y660:989/x5042:5390. Dedicated negative stem:
`t000_z006_d000212_y00916_x05298_hard_p00000002`.

Verified: exact reviewed mask gets targets 0, weights 1 in nuclear heads 4/5/6;
file hashes match; packed loader preserves it; input matches production
preprocessing exactly. Eight overlapping T0/Z6 training gold patches also label
the overlapping nuclear-interior pixels negative. No same-plane positive conflict
was found for this candidate. It still predicts nuclear-positive in all three
comparison checkpoints. Thus contaminated positives are not yet a demonstrated
explanation for this example, and auditing other patches remains worthwhile.

Reconstructed epoch list: 33,795 presentations; 179 dedicated hard-negative
presentations (~0.53%); this dedicated patch appears twice. Overlapping gold patches
also supply negative exposure. These are planned/reconstructed counts, not saved
batch-level observations or proof of effective exposure after augmentation.

Area versus integrated-NLS selection mostly agrees. Gold-matched spatial-holdout
survivors show good late-time overlap but some early errors and positive area bias.
The latest paired validation had 89 eligible gold columns, 74 matches, 63
unambiguous primary pairs. This excludes unmatched/ambiguous/filtered objects;
it does not establish overall precision or solve the artifact problem.

## Audit implementation plan

### 1. Effective supervision census and candidate nomination

- Verify manifests, raw identity, ROI/model hashes, file triplets, shapes,
  finite inputs, sentinel/weight consistency, and split membership.
- Index by global (image identity,T,Z,XY region), not merely patch stem.
- Compare original versus corrected labels and count remaining supervised
  nuclear-positive connected regions by timepoint/class/source/split.
- Rank candidates using explainable signals: overlap with confirmed artifact
  masks, inconsistent positive/negative labels across overlapping crops,
  large droplet-filling positives, wall/stitching proximity, weak NLS enrichment,
  weak NPC spatial support, odd connected components, and Z inconsistencies.
- These signals are review priorities, NOT automatic ground truth. T0, weak NPC,
  small size, non-roundness, and multinuclear droplets cannot alone prove artifact.
- Include a stratified random sample of unflagged positives and negatives across
  timepoints; otherwise the audit cannot characterize contamination prevalence.
- Scan all co-located training crops for conflicts, including corrected overrides;
  retain provenance and distinguish actual weight-positive conflicts from ignored
  stored positives. Flag them separately because the builder checks stored positives.
- Inspect augmentation separately after static labels: reproduce transforms with
  a fixed seed, show before/after, verify label/weight/image alignment and padding.
  Do not assume zoom augmentation caused errors without evidence.

### 2. Interactive resolver

Display for each candidate:

- raw central NLS/NPC/membrane, normalized input, neighboring Z planes;
- exact candidate contour and editable/reviewed-region extent;
- original labels, effective corrected labels, and per-head supervision weights;
- parent droplet/ROI context and original gold outline if present;
- why flagged, training/holdout status, coordinates, stable ID, affected patch stems;
- previous review decision and whether a correction already exists.

Decisions: **keep real nuclear positive**, **confirmed artifact negative**,
**mixed/needs local redraw**, **uncertain**, **skip**. Require reviewer and allow
failure-type notes (cytoplasm, empty droplet/plane, wall, stitching, annotation
spillover, unknown mechanism, etc.). Unknown failure mechanism does not negate
a confident exact-mask artifact decision. Do not default unreviewed items to real.

Support next/previous, jump/filter, progress count, undo/edit, and autosaved
versioned review state. Restarting/rerunning the UI must resume rather than erase.
Preview adjacent planes for context only; do not transfer a center-plane label
to neighboring Z planes without review.

Prefer exact connected-region review. Never zero a whole patch because one part
is artifact. Existing resolver cannot safely handle arbitrary partial redraws:
if local editing is not implemented, mixed masks must remain unresolved/skipped.

### 3. Explicit commit

- Require all candidates to have a decision (uncertain/skip are valid completed
  reviews), but apply changes only to explicitly approved exact regions.
- Preview affected original/effective patches and pixel counts before commit.
- Propagate the approved global mask consistently to every overlapping eligible
  training crop. Do not correct only the displayed crop and leave contradictions.
- Do not overwrite the existing 185-patch correction pool; compose new overlays
  on top of current effective labels and preserve the full provenance chain.
- Negative decisions set reviewed nuclear-family pixels to 0 with explicit
  weights. Keep other heads/regions unchanged. Handle Z-offset/equatorial
  consistency explicitly; masked-out regression must never become an invented target.
- Restoring/adding real positives requires reviewed boundaries and consistent
  edge/equatorial/Z targets; do not fabricate those from a binary button decision.
- Preserve validation and heldout files byte-for-byte. Write withheld/conflict
  reports. If contradictory review decisions overlap, stop and resolve first.
- Commit into a new timestamped directory with staging/incomplete status,
  immutable source hashes, masks, decisions, overrides and summary, then mark
  complete only after validation. Idempotent resume/retry; no duplicate commits.
- Separate commit from prepare-new-model and from RUN_TRAIN. No automatic training.

## Existing code to reuse and extend

Under Model_Training:

- `hard_negative_workflow.py`: `Workflow`, `audit_cell`, `decision_cell`,
  explicit commit and separate prepare-training pattern. This is the current
  interactive workflow reference, not permission to overwrite old commits.
- `hard_negative_notebook_review.py`: `ReviewSession`, inline review widgets.
- `hard_negative_conflict_audit.py`: `audit_conflicts`, `plot_conflicts`.
- `gold_artifact_corrections.py`: `build_corrections`, `corrected_rows`,
  `enable_corrections`; sparse versioned training-only overrides.
- `reviewed_hard_negatives.py`: manifest checks, holdout_reason, exact negative
  targets, row loading and new-run split hook.
- `empty_plane_supervision_audit.py`: exposure/label-weight audit reference.
- `mine_diverse_hard_negatives.py`: diversity/candidate nomination reference.
- `gold_selection_validation.py`: original ROI rasterization/provenance checks
  and fresh-kernel recovery pattern; no need to rerun model inference.
- Training notebook F1/F2/G2: authoritative split, sampling, packed-loader,
  augmentation, loss definitions. Functions include p3_rows, p3_split,
  p3_stratify_by_timepoint, p3_load_numpy, p3_augment_numpy, p3_make_loss.

Existing tests: `test_reviewed_hard_negatives.py`, `test_gold_artifact_corrections.py`,
`test_hard_negative_workflow.py`, `test_diverse_hard_negatives.py`,
`test_vulcan_pipeline_contract.py`, `test_gold_selection_validation.py`.

## Acceptance checks for the new workflow

- Fresh-kernel audit/resolver works without upstream generation cells.
- A known corrected positive is displayed as corrected, not reflagged as active.
- A confirmed artifact spanning several crops is consistently corrected in all
  eligible training views; heldout views stay untouched and are reported.
- Mixed/uncertain/skipped regions produce no automatic negative correction.
- Unrelated genuine positives and non-nuclear heads remain byte-identical.
- Weight/sentinel/head indexing, signed centers, padding, T/Z and raw mapping tested.
- Interrupted review and commit resume safely; hashes reject stale or changed inputs.
- Synthetic tests verify conflicting resolutions abort and validation is unchanged.
- Handoff outputs: notebook audit + resolver cells, reusable helpers/tests,
  candidate census and review progress, explicit versioned commit only after user
  approval. Do not add another collection of one-off notebook patchers/backups.
