"""Run vulcan_training_2_5.ipynb headlessly, up to and including training.

Executes the notebook's CODE cells in order into one namespace -- the same
thing the kernel does when you Run All -- and stops after the cell that calls
`train_ensemble`. Everything after that (plot_history, inference,
post-processing, evaluate_inference) is interactive work, not batch work.

Why exec the .ipynb rather than nbconvert to a .py: the notebook stays the
single source of truth. No generated copy to drift out of sync, and no risk of
training a different version of the code than the one you edited.

Usage:
    python run_notebook_training.py [--dry-run] [--notebook PATH]

    --dry-run   execute everything EXCEPT the training cell, then stop. Use it
                to prove imports, config, patch discovery, the tf.data pipeline
                and the model build all work before burning a GPU allocation.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_NB = REPO / "notebooks" / "model_training" / "vulcan_training_2_5.ipynb"

ap = argparse.ArgumentParser()
ap.add_argument("--notebook", type=Path, default=DEFAULT_NB)
ap.add_argument("--dry-run", action="store_true",
                help="stop before the training cell")
ap.add_argument("--with-inference", action="store_true",
                help="also run the cells after training (inference, "
                     "post-processing, evaluate_inference). Off by default: on "
                     "2026-09-14 job 40170827 spent 1.3 h running these against "
                     "a model whose loss had gone NaN at epoch 2, producing "
                     "Section 18 numbers that meant nothing. Check the history "
                     "CSV first, then run these interactively.")
args = ap.parse_args()

# Headless: no display, and plt.show() must not block or try to draw.
import matplotlib
matplotlib.use("Agg")

# Cell 2's _resolve_script_dir() walks up from cwd to find scripts/training.
os.chdir(REPO)

cells = json.loads(args.notebook.read_text())["cells"]
code = [(i, "".join(c["source"])) for i, c in enumerate(cells) if c["cell_type"] == "code"]

# The training cell is identified by content, not index, so this keeps working
# when cells are added or reordered.
TRAIN_MARKER = "train_ensemble(cfg)"
train_idx = [i for i, src in code if TRAIN_MARKER in src and not src.lstrip().startswith("def ")]
if not train_idx:
    sys.exit(f"could not find a cell calling {TRAIN_MARKER!r} in {args.notebook}")
train_cell = train_idx[-1]

print(f"notebook : {args.notebook}")
print(f"cwd      : {Path.cwd()}")
print(f"python   : {sys.executable}")
print(f"cells    : {len(code)} code cells; training is cell {train_cell}")
print(f"mode     : {'DRY RUN (stops before training)' if args.dry_run else 'full training'}")
print(f"SLURM    : job={os.environ.get('SLURM_JOB_ID','-')} "
      f"node={os.environ.get('SLURMD_NODENAME','-')} "
      f"cpus={os.environ.get('SLURM_CPUS_PER_TASK','-')}", flush=True)

G = {"__name__": "__nb__", "__file__": str(args.notebook)}
_patched = False
t_start = time.time()

for idx, src in code:
    if idx == train_cell and args.dry_run:
        # Still compile it, so a syntax error in the training cell surfaces here
        # rather than after the batch job has queued and started.
        compile(src, f"<cell {idx}>", "exec")
        print(f"\n--- dry run: training cell {idx} compiles; stopping before it ---",
              flush=True)
        break
    head = next((l for l in src.split("\n") if l.strip() and not l.strip().startswith("#")), "")
    print(f"\n=== cell {idx} ({time.time()-t_start:6.1f}s) | {head[:70]}", flush=True)
    exec(compile(src, f"<cell {idx}>", "exec"), G)

    if idx == train_cell and not args.with_inference:
        print(f"\n--- training cell {idx} done; stopping (pass --with-inference "
              f"to continue into Section 16-18) ---", flush=True)
        break

    if not _patched and "tf" in G:
        # Per-step progress bars are unreadable in a SLURM log (6746 steps/epoch).
        # verbose=2 gives one line per epoch. plt.show() is a no-op headless.
        tf = G["tf"]
        _orig_fit = tf.keras.Model.fit

        def _fit(self, *a, **k):
            k.setdefault("verbose", 2)
            return _orig_fit(self, *a, **k)

        tf.keras.Model.fit = _fit
        if "plt" in G:
            G["plt"].show = lambda *a, **k: None
        _patched = True

print(f"\n=== done in {(time.time()-t_start)/60:.1f} min ===")
if not args.dry_run:
    cfg = G.get("cfg")
    if cfg is not None:
        print(f"run_dir  : {cfg.run_dir}")
        print(f"weights  : {cfg.model_path(cfg.seeds[0], 'best')}")
