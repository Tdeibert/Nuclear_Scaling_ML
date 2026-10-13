# Resume the reviewed-hard-negative run

Saved on October 13, 2026: epoch **8** complete, best validation nuclear quality
**0.9417250156402588** at epoch 4, early-stopping wait 4, training not finished.
The checkpoint includes the optimizer. The next epoch is epoch 9.

On the next Cheaha allocation, open `vulcan_training_2_5_3.ipynb`. Load the
original baseline configuration and helper definitions through G4 with generation
and training disabled. Do not rebuild patches or rerun the review/commit UI.
The original baseline model name must be `Vulcan_2.5.3`.

Run this recovery cell once in the fresh kernel:

```python
from pathlib import Path
import copy
import json
import sys

code_dir = Path('/home/tdeibert/Projects/Nuclear_Scaling/Code_Repository/Python/Model_Training')
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))
import reviewed_hard_negatives as hn
import gold_artifact_corrections as gold_fix

commit_dir = Path('/data/user/tdeibert/Nuclear_Scaling/Inputs/Training_Data/reviewed_training_commit_20261013_121152_980805')
commit = json.loads((commit_dir / 'manifest.json').read_text())
assert commit['status'] == 'complete'
assert not globals().get('_HN_SPLIT_ENABLED')
assert not globals().get('_GOLD_CORRECTION_ENABLED')
assert cfg.model_name == 'Vulcan_2.5.3', 'Load the original baseline cfg first'

BASELINE_TRAINING_CFG = cfg
gold_fix.enable_corrections(globals(), BASELINE_TRAINING_CFG, commit['correction_pool'])
cfg = copy.copy(BASELINE_TRAINING_CFG)
cfg.__dict__.pop('_frozen_run_hash', None)
cfg.model_name = 'Vulcan_2.5.4_hardneg'
hn.enable_hard_negative_training(
    globals(), cfg, commit['negative_pool'], baseline_cfg=BASELINE_TRAINING_CFG
)
assert cfg is p3_split.prepared_cfg
assert cfg.model_name == 'Vulcan_2.5.4_hardneg_hn2392da6c18e0'
expected_run = Path('/home/tdeibert/Projects/Nuclear_Scaling/Models/runs/Vulcan_2.5.4_hardneg_hn2392da6c18e0_iezqw_a0cf3b2a13')
assert cfg.run_dir.resolve() == expected_run.resolve(), 'Configuration differs; do not start a different run'
resume_dir = expected_run / 'resume' / 's42'
assert (resume_dir / 'checkpoint').exists()
print('Resume state:', json.loads((resume_dir / 'state.json').read_text()))
RUN_TRAIN = False
```

Pool validation can take time on shared storage. Once recovery finishes, set
`RUN_TRAIN = True` and execute G5 only. It should say `resuming after epoch 8`
(or a later saved epoch). Do not rerun configuration/F1 after recovery.
For a bounded allocation, G5 can call `p3_train(cfg, time_budget_s=...)`; this
stops at an epoch boundary, so leave at least one epoch of allocation margin.

## Persistent artifacts

- Commit above: 175 artifacts, 70 real, 11 skipped, 6 uncertain.
- Corrected gold pool: 185 patches; original files preserved.
- Hard-negative pool: 124 patches, 51 tiles withheld.
- Checkpoints and review/pool arrays stay on shared Cheaha storage; they are not
  stored in GitHub. The repository stores the notebook and helper code.
- Notebook backup files (`*.pre_*.ipynb`) are retained locally, outside this commit.

## Follow-up

Add progress bars, elapsed time, and useful stage/ETA reporting to long-running
audit, commit, pool verification, and model-preparation operations. The current
workflow still has quiet validation stages. An epoch currently takes around
28 minutes and contains 16,898 training batches.
