"""Train (or resume) Vulcan 3.0 on a SLURM allocation without opening the notebook.

Executes the cells of vulcan_training_3_0.ipynb tagged ``vulcan-def`` in notebook
order (definitions + A5 setup, every RUN_* flag forced False), builds the corrected
split through p3_effective_split, and calls p3_train with a time budget that leaves
an epoch of margin. Resuming is automatic: p3_train picks up the run's own
checkpoint, and refuses if the correction layers changed since the run started.

    python -u run_vulcan_3_0.py --job-id "$SLURM_JOB_ID"            # train / resume
    python -u run_vulcan_3_0.py --job-id "$SLURM_JOB_ID" --check-only   # verify only

Do not run notebook G5 concurrently with this runner (a lock file enforces it).
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = Path(os.environ.get('VULCAN_NOTEBOOK', HERE / 'vulcan_training_3_0.ipynb')).resolve()  # per-run frozen copy
DEF_TAG = 'vulcan-def'
MARGIN_S = 45 * 60          # one ~28 min epoch + validation + checkpoint
EXIT_INCOMPLETE = 75        # budget spent before training finished; resume in a new job


def remaining_seconds(job):
    value = subprocess.check_output(['squeue', '-h', '-j', job, '-o', '%L'], text=True).strip()
    days = 0
    if '-' in value:
        day, value = value.split('-', 1)
        days = int(day)
    parts = list(map(int, value.split(':')))
    if len(parts) == 2:
        parts = [0] + parts
    if len(parts) != 3:
        raise ValueError(f'Cannot parse remaining time {value!r}')
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def load_definitions():
    cells = json.loads(NOTEBOOK.read_text())['cells']
    selected = [c for c in cells if c['cell_type'] == 'code'
                and DEF_TAG in c.get('metadata', {}).get('tags', [])]
    if not selected:
        raise RuntimeError(f'No cells tagged {DEF_TAG!r} in {NOTEBOOK}')
    print('notebook:', NOTEBOOK, flush=True)
    ns = {'__name__': '__main__', '__file__': str(NOTEBOOK)}
    for c in selected:
        source = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
        print(f"-- cell {c.get('id')}: {source.strip().splitlines()[1 if source.startswith('# ===') else 0][:70]}",
              flush=True)
        exec(compile(source, f"notebook:{c.get('id')}", 'exec'), ns)
        for key in [k for k in ns if k.startswith('RUN_')]:
            ns[key] = False                     # never let a stage fire as a side effect
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job-id', required=True)
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.chdir(HERE)

    ns = load_definitions()
    cfg = ns['cfg']
    if not ns['tf'].config.list_physical_devices('GPU') and not args.check_only:
        raise RuntimeError('No GPU available')

    cfg.run_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = (cfg.run_dir.parent / f'{cfg.run_dir.name}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    ns['p3_effective_split'](cfg)              # verifies every correction layer now
    print('run_dir:', cfg.run_dir, flush=True)
    state = cfg.run_dir / 'resume' / f's{cfg.seeds[0]}' / 'state.json'
    print('resume state:', json.loads(state.read_text()) if state.exists() else 'fresh run', flush=True)
    if args.check_only:
        return

    budget = remaining_seconds(args.job_id) - MARGIN_S
    if budget < 30 * 60:
        print('Not enough allocation time left for an epoch plus margin; deferring', flush=True)
        sys.exit(EXIT_INCOMPLETE)
    print(f'training budget {budget / 3600:.2f} h ({MARGIN_S // 60} min margin)', flush=True)
    result = ns['p3_train'](cfg, time_budget_s=budget)
    print('training returned:', result['status'], flush=True)
    if result['status'] != 'complete':
        sys.exit(EXIT_INCOMPLETE)      # tells the sbatch script to queue the next job


if __name__ == '__main__':
    main()
