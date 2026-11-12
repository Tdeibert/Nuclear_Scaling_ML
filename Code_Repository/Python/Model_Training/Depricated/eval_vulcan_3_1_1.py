"""Evaluate Vulcan 3.1.1 against 3.0 on a SLURM allocation (no notebook needed).

Stages, each skipped if its output already exists (so a resubmit resumes):
  1-3. H4 segmentation test for 3.0 e13, 3.1.1 best (e15) and 3.1.1 final (e55), each in its
       own folder under /data/.../Evaluations/vulcan_3_1_1_vs_3_0/ (checkpoint copy, CSVs and
       its OWN probability maps -- H4 always writes maps to the training run's qc, so they are
       moved after each run; otherwise the next H4 would overwrite them)
  4.   Pixel + object overlap vs hand-traced gold for all three
       -> Evaluations/vulcan_3_1_1_vs_3_0/model_compare.csv (+ printed table)

Loads the notebook's `vulcan-def` cells exactly like run_vulcan_3_0.py (every RUN_* flag
forced False), so H4 and post-processing are the notebook's own code. Original checkpoints
are only read (copied into the evaluation folders); nothing in the training runs changes,
apart from H4's maps passing briefly through the 3.1.1 run's qc folder.

    python -u eval_vulcan_3_1_1.py                 # run / resume all stages
    python -u eval_vulcan_3_1_1.py --check-only    # verify paths and checkpoints, run nothing
"""
import argparse
import fcntl
import json
import os
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = Path(os.environ.get('VULCAN_NOTEBOOK', HERE / 'vulcan_training_3_1_1.ipynb')).resolve()
DEF_TAG = 'vulcan-def'
REF_RUN = 'Vulcan_3.0_iezqw_b3caa96b89'           # 3.0 reference run (its e13 best checkpoint is re-tested)


def load_definitions():
    cells = json.loads(NOTEBOOK.read_text())['cells']
    selected = [c for c in cells if c['cell_type'] == 'code' and DEF_TAG in c.get('metadata', {}).get('tags', [])]
    if not selected:
        raise RuntimeError(f'No cells tagged {DEF_TAG!r} in {NOTEBOOK}')
    print('notebook:', NOTEBOOK, flush=True)
    ns = {'__name__': '__main__', '__file__': str(NOTEBOOK)}
    for c in selected:
        src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
        exec(compile(src, f"notebook:{c.get('id')}", 'exec'), ns)
        for k in [k for k in ns if k.startswith('RUN_')]:
            ns[k] = False                                  # never let a stage fire as a side effect
    return ns


# Executed inside the notebook namespace so it uses the notebook's own readers/post-processing.
OVERLAP_SRC = r'''
def overlap_summary(run_dir, label):
    rd = Path(run_dir); meta = json.loads((rd / "qc" / "segtest_prob.json").read_text())
    box = tuple(meta["box"]); y0, x0, y1, x1 = box; H, W = y1 - y0, x1 - x0
    ch = {n: i for i, n in enumerate(meta["channels"])}
    mm = tiff.memmap(str(rd / "qc" / "segtest_prob.tif")); hs = load_memmap_tiff(cfg.image_file)
    gp = p3_gold_in_box((-1, -1, hs.shape[-2] + 1, hs.shape[-1] + 1), meta["t_range"], meta["z_range"], cfg)
    objs, seg_px, gold_px, both_px = [], 0, 0, 0
    for ti, t in enumerate(meta["t_range"]):
        for zi, z in enumerate(meta["z_range"]):
            pc = np.asarray(mm[ti, zi, ch["nucleus_interior"]], np.float32)
            pe = np.asarray(mm[ti, zi, ch["nucleus_edge"]], np.float32) if "nucleus_edge" in ch else np.zeros_like(pc)
            nls = read_crop(hs, t, z, cfg.nucleus_channel_idx, box).astype(np.float32)
            lab = watershed_nuclei(nls, pc, pe, cfg) if cfg.use_watershed_postproc else measure.label(pc > cfg.mask_threshold)
            gold = np.zeros((H, W), np.int32)
            for k, ((a, b, c, d), m) in enumerate(gp.get((t, z), []), start=1):
                ya, xa, yb, xb = max(a, y0), max(b, x0), min(c, y1), min(d, x1)
                if ya < yb and xa < xb:
                    gold[ya - y0:yb - y0, xa - x0:xb - x0][m[ya - a:yb - a, xa - b:xb - b]] = k
            s_, g_ = lab > 0, gold > 0
            seg_px += int(s_.sum()); gold_px += int(g_.sum()); both_px += int((s_ & g_).sum())
            for r in measure.regionprops(lab):
                g = gold[r.slice][lab[r.slice] == r.label]; g = g[g > 0]
                best = max([(g == k).sum() / (r.area + (gold == k).sum() - (g == k).sum()) for k in np.unique(g)],
                           default=0.0)
                objs.append(dict(t=t, area_um2=r.area * cfg.pixel_size_um ** 2, iou=best))
        print(f"  {label}: t{t} done", flush=True)
    o = pd.DataFrame(objs)
    o["class"] = pd.cut(o.iou, [-1, 1e-9, 0.5, 0.9, 1.01], labels=["none", "minor", "partial", "near-exact"], right=False)
    pct = o["class"].value_counts(normalize=True).reindex(["none", "minor", "partial", "near-exact"]) * 100
    area = o.groupby("class", observed=False).area_um2.sum(); area = 100 * area / area.sum()
    return pd.Series({"% seg px in gold": 100 * both_px / seg_px, "% gold px covered": 100 * both_px / gold_px,
                      "objects": len(o), **{f"% obj {k}": v for k, v in pct.items()},
                      **{f"% area {k}": v for k, v in area.reindex(pct.index).items()}}, name=label)
'''


def h4_done(eval_dir):
    qc = Path(eval_dir) / 'qc'
    return all((qc / f).exists() for f in ('segtest_report.json', 'segtest_prob.json', 'segtest_prob.tif'))


def stage(title):
    print(f'\n=== {title} | {time.strftime("%H:%M:%S")} ===', flush=True)


def run_h4(ns, cfg, checkpoint, eval_dir):
    """H4 for one checkpoint in its own folder. H4's probability maps are always written to
    cfg.qc_dir (the training run's qc), whatever run_dir says; move them next to this model's
    CSVs so the next H4 cannot overwrite them."""
    eval_dir = Path(eval_dir); (eval_dir / 'qc').mkdir(parents=True, exist_ok=True)
    target = eval_dir / cfg.model_path(cfg.seeds[0], 'best').name      # H4 loads 'best' by name
    if not target.exists():
        shutil.copy2(checkpoint, target)
    (eval_dir / 'qc' / 'checkpoint_source.txt').write_text(str(checkpoint) + '\n')
    if h4_done(eval_dir):
        print('already done; skipping', flush=True)
        return
    ns['p3_segmentation_test'](cfg, which='best', run_dir=eval_dir)
    for f in ('segtest_prob.tif', 'segtest_prob.json'):
        src = cfg.qc_dir / f
        if not src.exists():
            raise FileNotFoundError(f'H4 did not write {src}')
        shutil.move(str(src), str(eval_dir / 'qc' / f))
    print('maps ->', eval_dir / 'qc', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.chdir(HERE)

    ns = load_definitions()
    cfg = ns['cfg']
    state = json.loads((cfg.run_dir / 'resume' / f's{cfg.seeds[0]}' / 'state.json').read_text())
    ref_run = cfg.model_root / 'runs' / REF_RUN
    eval_root = Path(str(cfg.data_root)) / 'Evaluations' / 'vulcan_3_1_1_vs_3_0'
    models = [   # (label, checkpoint, eval folder)
        ('3.0 e13 (best)', ref_run / 'Vulcan_3.0_iezqw_s42_best.keras', eval_root / 'v3_0_e13'),
        (f"3.1.1 e{state.get('best_epoch')} (best)", cfg.model_path(cfg.seeds[0], 'best'), eval_root / 'v3_1_1_best'),
        (f"3.1.1 e{state.get('epoch')} (final)", cfg.model_path(cfg.seeds[0], 'final'), eval_root / 'v3_1_1_final'),
    ]
    print('run_dir   :', cfg.run_dir)
    print('state     :', state)
    print('eval_root :', eval_root)
    for label, ckpt, _ in models:
        print(('ok        :' if ckpt.exists() else 'MISSING   :'), label, ckpt, flush=True)
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
    if not state.get('done'):
        raise RuntimeError('3.1.1 training is not finished; evaluate after it completes')
    if args.check_only:
        return
    if not ns['tf'].config.list_physical_devices('GPU'):
        raise RuntimeError('No GPU available (H4 inference needs one)')

    eval_root.mkdir(parents=True, exist_ok=True)
    lock = (eval_root / 'eval.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)        # one evaluation at a time

    for i, (label, ckpt, d) in enumerate(models, 1):
        stage(f'{i}/{len(models) + 1} H4: {label}')
        run_h4(ns, cfg, ckpt, d)

    stage(f'{len(models) + 1}/{len(models) + 1} overlap comparison vs hand-traced gold')
    exec(OVERLAP_SRC, ns)
    pd = ns['pd']
    table = pd.concat([ns['overlap_summary'](d, label) for label, _, d in models], axis=1).round(1)
    out = eval_root / 'model_compare.csv'
    table.to_csv(out)
    pd.set_option('display.width', 200)
    print('\n' + table.to_string(), flush=True)
    print('\nsaved ->', out, flush=True)


if __name__ == '__main__':
    main()
