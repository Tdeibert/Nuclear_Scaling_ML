#!/usr/bin/env python
"""Vulcan 2.5.3 batch runner for SLURM: classical patches, training, segmentation test.

The notebook stays the single source of truth. This script executes every code cell of
vulcan_training_2_5_3.ipynb in order with all RUN_* stage flags forced to False, so only
definitions and setup run, then calls the requested stages explicitly:

  check      print configuration, pool status and GPUs; writes nothing.
  classical  recompute E1 (t=2, t=9) and E2, then p2_accept_checks() must pass; needs
             --reviewed-checks (you reviewed E1/E2 for this configuration interactively).
             The gold pool must be complete and built under the current configuration.
             Then p2_build("classical"). NOT resumable: if a job dies mid-build, move the
             incomplete training_patches_* folder aside before retrying.
  train      p3_train with a time budget. Resumes automatically from the run directory.
             When the budget is spent it stops at an epoch boundary and exits with code 75
             so the sbatch wrapper resubmits itself.
  segtest    p3_segmentation_test on the best checkpoints (holdout tile).

--scratch copies the patch pools to node-local NVMe (/local/$USER/$SLURM_JOB_ID by
default) before train/segtest and points cfg.out_root there. Cheaha RC strongly
recommends this on amperenodes: GPFS cannot feed an A100. Falls back to GPFS if the
pools do not fit. The sbatch wrapper deletes the scratch copy when the job ends.

Exit codes: 0 = requested stages finished; 75 = training incomplete (resubmit);
1 = error. A JSON summary per job is written to --log-dir (default: the submit dir).

Usage (from Python/Model_Training on a compute node or in an sbatch script):
  python vulcan_2_5_3_batch.py --stages check
  python vulcan_2_5_3_batch.py --stages classical --reviewed-checks
  python vulcan_2_5_3_batch.py --stages train
  python vulcan_2_5_3_batch.py --stages segtest --segtest-t 2,5,9
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

EXIT_INCOMPLETE = 75
HERE = Path(__file__).resolve().parent
DEFAULT_NOTEBOOK = HERE / "vulcan_training_2_5_3.ipynb"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Notebook definitions
# ---------------------------------------------------------------------------
def _strip_magics(src):
    """Drop IPython-only lines (%magic, !shell); the notebook has none today."""
    return "\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("%", "!")))


def load_definitions(nb_path, after_cell=None):
    """Execute every code cell into one namespace with all RUN_* flags False.

    The flags are forced off right after the A5 cell defines them, so every later
    stage cell is skipped. after_cell(ns, src) is an optional hook (used by tests).
    """
    nb_path = Path(nb_path).resolve()
    cells = ["".join(c["source"]) for c in json.loads(nb_path.read_text(encoding="utf8"))["cells"]
             if c["cell_type"] == "code"]
    ns = {"__name__": "__vulcan_batch__", "__file__": str(nb_path)}
    try:
        import IPython.display  # noqa: F401  (cells import display from here)
    except ImportError:
        import types
        ipd = types.ModuleType("IPython.display"); ipd.display = lambda o: print(o)
        ip = types.ModuleType("IPython"); ip.display = ipd
        sys.modules.update({"IPython": ip, "IPython.display": ipd})
    forced = False
    for i, src in enumerate(cells):
        exec(compile(_strip_magics(src), f"{nb_path.name}:cell{i}", "exec"), ns)
        if not forced and "RUN_GOLD_IMPORT" in ns:
            for k in [k for k in ns if k.startswith("RUN_")]:
                ns[k] = False
            forced = True
        if after_cell is not None:
            after_cell(ns, src)
    if not forced:
        raise RuntimeError("no A5 run-control cell (RUN_GOLD_IMPORT) found in the notebook")
    return ns


def install_figure_saver(ns, fig_dir):
    """plt.show() inside notebook code saves the figure instead."""
    import matplotlib.pyplot as plt
    fig_dir = Path(fig_dir); counter = {"n": 0}

    def _save(*a, **k):
        fig_dir.mkdir(parents=True, exist_ok=True)
        for num in plt.get_fignums():
            counter["n"] += 1
            plt.figure(num).savefig(fig_dir / f"fig_{counter['n']:03d}.png", dpi=90, bbox_inches="tight")
        plt.close("all")
    plt.show = _save
    ns["plt"] = plt


# ---------------------------------------------------------------------------
# Time budget
# ---------------------------------------------------------------------------
def slurm_remaining_s():
    """Seconds left in this SLURM job from squeue's %L, or None."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        out = subprocess.run(["squeue", "-h", "-j", job, "-o", "%L"], capture_output=True,
                             text=True, timeout=30).stdout.strip()
    except Exception:
        return None
    if not out or out in ("UNLIMITED", "NOT_SET", "INVALID"):
        return None
    days = 0
    if "-" in out:
        d, out = out.split("-", 1); days = int(d)
    parts = [int(p) for p in out.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return ((days * 24 + h) * 60 + m) * 60 + s


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_to_scratch(ns, args):
    """Copy the patch pools to node-local scratch and point cfg.out_root at the copy."""
    cfg = ns["cfg"]
    base = Path(args.scratch_dir or f"/local/{os.environ.get('USER', 'user')}/{os.environ.get('SLURM_JOB_ID', 'local')}")
    pools = [p for p in (cfg.reviewed_root, cfg.training_root) if (p / "manifest.json").exists()]
    if not pools:
        log("scratch: no complete pools to stage; training reads GPFS"); return
    need = sum(f.stat().st_size for p in pools for f in p.rglob("*") if f.is_file())
    base.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(base).free
    if need > 0.9 * free:
        log(f"scratch: pools need {need / 1e9:.0f} GB but {base} has {free / 1e9:.0f} GB free; reading GPFS")
        return
    t0 = time.time()
    for p in pools:
        dst = base / p.name
        if not dst.exists():
            shutil.copytree(p, dst)
        n_src = sum(1 for f in p.rglob("*") if f.is_file()); n_dst = sum(1 for f in dst.rglob("*") if f.is_file())
        if n_src != n_dst:
            raise RuntimeError(f"scratch copy of {p.name} incomplete: {n_dst}/{n_src} files")
    log(f"scratch: staged {need / 1e9:.1f} GB ({len(pools)} pools) to {base} in {(time.time() - t0) / 60:.1f} min")
    cfg.out_root = base


def stage_check(ns, args, summary):
    cfg = ns["cfg"]; tf = ns["tf"]
    gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    pools = {}
    for name, root in (("gold", cfg.reviewed_root), ("classical", cfg.training_root)):
        m = root / "manifest.json"
        if m.exists():
            man = json.loads(m.read_text(encoding="utf8"))
            pools[name] = dict(root=str(root), status=man.get("status"), n_patches=man.get("n_patches"),
                               hashes_current=(man.get("gen_hash") == cfg.gen_hash and man.get("gold_hash") == cfg.gold_hash))
        else:
            pools[name] = dict(root=str(root), status="absent")
    info = dict(gpus=gpus, model=cfg.model_name, algorithm=cfg.algorithm_version, gen_hash=cfg.gen_hash,
                gold_hash=cfg.gold_hash, run_dir=str(cfg.run_dir), label_source=cfg.label_source,
                epochs=cfg.epochs, seeds=list(cfg.seeds), pools=pools)
    log("check:\n" + json.dumps(info, indent=2))
    summary["check"] = info


def stage_classical(ns, args, summary):
    cfg = ns["cfg"]
    if not args.reviewed_checks:
        raise SystemExit("classical: pass --reviewed-checks to confirm you reviewed E1/E2 for this configuration")
    gold = cfg.reviewed_root / "manifest.json"
    if not gold.exists():
        raise RuntimeError(f"gold pool missing: {cfg.reviewed_root}")
    man = json.loads(gold.read_text(encoding="utf8"))
    if man.get("status") != "complete":
        raise RuntimeError(f"gold pool {cfg.reviewed_root} is not complete")
    if man.get("gen_hash") != cfg.gen_hash or man.get("gold_hash") != cfg.gold_hash:
        raise RuntimeError("gold pool was built under a different configuration than the notebook now "
                           "describes; the classical pool must match it (training would refuse the mix)")
    hs = ns["load_memmap_tiff"](cfg.image_file)
    log(f"E1 geometry QC for t={tuple(ns['E1_TIMEPOINTS'])}")
    ns["E1_RESULTS"] = {t: ns["geometry_qc"](t, hs) for t in ns["E1_TIMEPOINTS"]}
    ns["E1_CONFIG_HASH"] = cfg._hash_of(cfg._GEOM_FIELDS + ("pixel_size_um",))
    e1 = ns["pd"].concat([df for df, _ in ns["E1_RESULTS"].values()], ignore_index=True)
    log("E1 worst spheres:\n" + e1.sort_values("sphere_rms_um", ascending=False).head(10).to_string())
    del hs
    log("E2 normalisation check")
    ns["E2_RESULT"] = ns["norm_stats_check"]()
    ns["E2_CONFIG_HASH"] = cfg._hash_of(cfg._INPUT_FIELDS)
    ns["p2_accept_checks"](cfg)
    times = None if not args.timepoints else [int(t) for t in args.timepoints.split(",")]
    log(f"building classical pool -> {cfg.training_root}")
    t0 = time.time()
    m = ns["p2_build"]("classical", times, cfg)
    summary["classical"] = dict(root=str(cfg.training_root), n_patches=m.get("n_patches"),
                                status=m.get("status"), minutes=round((time.time() - t0) / 60, 1),
                                e1_rms_p95=float(e1.sphere_rms_um.quantile(0.95)))
    log(f"classical pool {m.get('status')}: {m.get('n_patches')} patches")


def stage_train(ns, args, summary, job_start):
    cfg = ns["cfg"]; tf = ns["tf"]
    if not tf.config.list_physical_devices("GPU") and not args.allow_cpu:
        raise RuntimeError("no GPU visible; request --gres=gpu:1 (or pass --allow-cpu)")
    remaining = slurm_remaining_s()
    if args.time_budget_min is not None:
        budget = args.time_budget_min * 60 - (time.time() - job_start)
    elif remaining is not None:
        budget = remaining - args.margin_min * 60
    else:
        budget = None
    log(f"training: budget {'unlimited' if budget is None else f'{budget / 60:.0f} min'} "
        f"(margin {args.margin_min} min for the last epoch and checkpoint)")
    if budget is not None and budget <= 0:
        raise RuntimeError("no time left for training in this job")
    res = ns["p3_train"](cfg, time_budget_s=budget)
    for s in res["states"]:
        try:
            ns["plot_history"](ns["load_history"](s))
        except Exception as e:  # plots must never fail the run
            log(f"history plot for seed {s} skipped: {e}")
    summary["train"] = dict(status=res["status"], run_dir=str(cfg.run_dir),
                            states={str(k): v for k, v in res["states"].items()})
    log(f"training status: {res['status']}")
    return res["status"]


def stage_segtest(ns, args, summary):
    cfg = ns["cfg"]
    missing = [s for s in cfg.seeds if not cfg.model_path(s, "best").exists()]
    if missing:
        raise RuntimeError(f"no best checkpoint for seeds {missing} in {cfg.run_dir}")
    t_range = None if not args.segtest_t else [int(t) for t in args.segtest_t.split(",")]
    z_range = None if not args.segtest_z else [int(z) for z in args.segtest_z.split(",")]
    res = ns["p3_segmentation_test"](cfg, which="best", t_range=t_range, z_range=z_range)
    summary["segtest"] = dict(qc_dir=str(cfg.qc_dir), n_detections=int(len(res["detections"])),
                              gates=res["report"].get("gates", {}).get("gates"))


# ---------------------------------------------------------------------------
def main(argv=None, after_cell=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--notebook", default=str(DEFAULT_NOTEBOOK))
    p.add_argument("--stages", default="check", help="comma list of check,classical,train,segtest")
    p.add_argument("--reviewed-checks", action="store_true",
                   help="you reviewed E1/E2 for this configuration (required for 'classical')")
    p.add_argument("--timepoints", default=None, help="classical: comma list (default: all)")
    p.add_argument("--time-budget-min", type=float, default=None,
                   help="train: minutes for this job (default: SLURM time left minus --margin-min)")
    p.add_argument("--margin-min", type=float, default=30.0,
                   help="train: reserve for the epoch in flight + checkpoint (> one epoch)")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--scratch", action="store_true", help="stage pools to node-local scratch for train/segtest")
    p.add_argument("--scratch-dir", default=None, help="default /local/$USER/$SLURM_JOB_ID")
    p.add_argument("--segtest-t", default=None); p.add_argument("--segtest-z", default=None)
    p.add_argument("--log-dir", default=os.environ.get("SLURM_SUBMIT_DIR", os.getcwd()))
    args = p.parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = set(stages) - {"check", "classical", "train", "segtest"}
    if bad:
        p.error(f"unknown stages {sorted(bad)}")
    job_start = time.time()
    job = os.environ.get("SLURM_JOB_ID", f"local{int(job_start * 1000)}")
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(job=job, host=socket.gethostname(), stages=stages, argv=sys.argv[1:],
                   resubmit=int(os.environ.get("VULCAN_RESUBMIT_COUNT", 0)), started=time.ctime(job_start))
    code = 0
    try:
        log(f"loading notebook definitions from {args.notebook}")
        ns = load_definitions(args.notebook, after_cell=after_cell)
        install_figure_saver(ns, log_dir / f"vulcan_{job}_figures")
        staged = False
        for stage in stages:
            log(f"=== stage {stage} ===")
            if args.scratch and not staged and stage in ("train", "segtest"):
                stage_to_scratch(ns, args); staged = True
            if stage == "check":
                stage_check(ns, args, summary)
            elif stage == "classical":
                stage_classical(ns, args, summary)
            elif stage == "train":
                if stage_train(ns, args, summary, job_start) != "complete":
                    code = EXIT_INCOMPLETE
                    break           # later stages need a finished model
            elif stage == "segtest":
                stage_segtest(ns, args, summary)
    except SystemExit as e:
        summary["error"] = str(e); code = 1; log(str(e))
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"; summary["traceback"] = traceback.format_exc()
        code = 1; log(summary["traceback"])
    summary["finished"] = time.ctime(); summary["exit_code"] = code
    (log_dir / f"vulcan_{job}_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"exit {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
