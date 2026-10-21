# Training-patch artifact audit

Audits supervised nuclear positives in the **effective** training patches of
`Vulcan_2.5.4_hardneg_hn2392da6c18e0_iezqw_a0cf3b2a13`, reviews flagged regions, and
corrects only explicitly approved exact regions on an explicit commit.

Implements `TRAINING_PATCH_AUDIT_HANDOFF.md`. Does not launch training or inference,
start background jobs, change the segmentation run, or rebuild any pool.

## Files

| file | role |
|---|---|
| `Training_Patch_Audit.ipynb` | the notebook: one audit/setup cell, one resolver cell, plus read-only augmentation inspection and a self-check |
| `vulcan_audit_recovery.py` | fresh-kernel config recovery and effective-label assembly |
| `training_patch_audit.py` | census, global region index, signals, nomination, augmentation check |
| `training_patch_resolver.py` | review state, candidate display, widgets, explicit commit |
| `test_training_patch_audit.py` | synthetic end-to-end suite; `python test_training_patch_audit.py` prints PASS/FAIL |

## Loading the notebook's definitions without running a stage

`vulcan_audit_recovery` reads `vulcan_training_2_5_3.ipynb` and compiles selected cells
through an **AST filter** that keeps only definitions, imports and constant assignments.
Every other top-level statement is dropped and reported, so a generation, training or
export trigger cannot fire as a side effect of loading a reader. Lambda bodies are not
treated as side effects, which is why the head table survives the filter.

Cells are selected by content marker in notebook order (`DEFINITION_MARKERS`), so a
later cell legitimately supersedes an earlier definition exactly as a top-to-bottom run
would. A marker that stops matching exactly one cell is a hard error: fix the markers
rather than letting the audit silently load different code. TensorFlow is deliberately
not imported; `p3_dataset` is the only loaded function that needs it and the audit never
builds a tf.data pipeline.

## Config recovery

`recover_config(run_dir)` rebuilds the run's own `PipelineConfig` from its saved
`config.json`, then refuses to continue unless:

- every cached field round-trips to the same string the training stage wrote;
- the live configuration still hashes to **this run directory's own name**;
- `cfg.reviewed_root` is the gold pool `phase3_run.json` recorded;
- the raw image exists and the storage/head layout is the one the audit assumes.

Path roots are excluded from `config.json` by `stale_guard`, so they are re-derived from
the recorded gold-pool root and the run directory's position, and the derived layout is
checked by name. Normalization statistics are **loaded, never computed** — computing
them rescans the raw stack.

## Effective labels

    original gold pool rows
      -> gold_corrections overrides applied by stem      (185 patches)
      -> the run's own p3_split
      -> reviewed hard-negative rows appended to TRAINING only   (124 patches)

The production readers do the work: `gold_artifact_corrections.enable_corrections`
wraps the pool reader as the training hook did, the notebook's own `p3_split` runs on the
corrected rows, and `reviewed_hard_negatives.load_reviewed_negative_rows` validates the
additions. Split sizes are compared with `phase3_run.json` and any mismatch aborts.
Original and effective labels are kept side by side, so an already-corrected positive
reads as a historical finding rather than current contamination.

## Two census passes

A full-pixel census of every effective patch reads labels and weights only (~12 MB per
patch). It is parallel (`joblib` loky, paths passed to workers), shows rate and elapsed
time, and is **cached and resumable per (source, timepoint) shard**; a shard is reused
only when every input file still has the same path, size and mtime. Image-derived
signals are measured in a second pass, for the review set only, from the stored
globally-normalized inputs. `audit.SIGNAL_TIERS` records which tier each column
came from; `negative_supervision_px == -1` means the exact scan did not run.

### Global indexing

Patch-local components are merged into global `(t, z, XY)` regions by union-find over
8-adjacency in image coordinates, because the same physical object appears in several
overlapping crops under different stems. Region ids are content-addressed and stable
across reruns. Out-of-image padding is excluded from regions and counted separately;
signed crop centres (e.g. `x=-1`) are parsed, not rejected.

### Nuclear-mask semantics

A pixel is a supervised nuclear positive when a nuclear object head (4 interior,
5 edge) says 1 with weight. It is supervised **not** nuclear only when some nuclear
object head carries weight there and no nuclear object head calls it a nucleus —
testing the heads independently would mark every nucleus interior negative, because an
interior pixel is legitimately 0 on the edge head. `stored` ignores weight, matching the
hard-negative builder's guard, so an unweighted stored positive still blocks a negative
and is reported separately as `ignored_stored_positive`.

## Review and commit

Decisions: **real** (no change), **confirmed artifact negative**, **mixed/needs local
redraw** (recorded, left unresolved — local partial redraw is not implemented),
**uncertain**, **skip**. An unreviewed candidate is never treated as real. State
autosaves to its own versioned folder keyed by a candidate-set signature; reopening the
same set resumes, and a changed set refuses to reuse the old decisions.

A commit requires every candidate to have a decision and a reviewer, aborts on
contradictory overlapping decisions, and writes a **new** timestamped directory with
`staging` → `complete` status. For each approved region it composes a sparse overlay on
top of the current effective labels for **every** eligible training crop the region
touches, so no crop is left contradicting another. Nuclear heads 4/5/6 get target 0 with
an explicit weight over exactly those pixels; the Z-offset there is set to `UNANNOTATED`
with weight 0, so a masked-out regression never becomes an invented target. All other
heads, the `droplet_source` channel and all other pixels are untouched. Holdout and
seam crops are reported in `withheld_regions.csv` and `holdout_only_regions.csv` and
left byte-identical. Source digests are re-verified before the write and again after, and
the commit is idempotent.

Enabling an overlay for training is **not** wired up. `audit_corrected_rows` is the
reader a later run would use; it re-verifies the effective-label digests and the overlay
digests and refuses holdout rows. Any such run needs a new model identity and new
versioned outputs, and must carry this commit's provenance with it.

## Note on the existing correction pool

The census counts `z_supervised_outside_nucleus_px`: pixels where the Z-offset head
still carries a live regression target although the mask heads no longer call the pixel a
nucleus. `gold_artifact_corrections.build_corrections` writes heads 4/5/6 only, so the
existing 185-patch pool leaves such targets behind. The audit reports them; the new
commit path does not create them. The existing pool is not modified.
