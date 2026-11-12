"""Full-frame segmentation with Vulcan 3.1.1 final (e55), one timepoint at a time.

Uses the notebook's own inference + post-processing (p3_run_inference, p3_extract_detections),
loaded from the `vulcan-def` cells exactly like eval_vulcan_3_1_1.py (all RUN_* forced False).
Gold is passed in so each detection carries its gold match (gold is complete over the frame).

Per timepoint t (skipped if its CSV exists, so a resubmit resumes):
    <OUT>/detections/det_t{t:02d}.csv
    <OUT>/maps/full_t{t:02d}_prob.tif/.json   (deleted afterwards unless --keep-maps)
Then all timepoints are concatenated -> <OUT>/full_detections.csv, with true_time_min added
(2x3 tiles, serpentine on odd rows, 6 min/frame, 1 min/tile -- control_extract_1.1.json).

    python -u segment_full_3_1_1.py               # run / resume
    python -u segment_full_3_1_1.py --check-only  # paths, checkpoint, map size; runs nothing
    python -u segment_full_3_1_1.py --keep-maps   # keep probability maps (~GBs per t)
"""
import argparse, fcntl, os, shutil, time
from pathlib import Path
import eval_vulcan_3_1_1 as ev                        # reuse its notebook loader

CKPT_WHICH = 'final'                                 # 3.1.1 e55
TILE_ROWS, TILE_COLS, MIN_PER_FRAME, MIN_PER_TILE = 2, 3, 6, 1


def true_time_min(t, cy, cx, H, W):
    r = (cy * TILE_ROWS // H).clip(0, TILE_ROWS - 1).astype(int)
    c = (cx * TILE_COLS // W).clip(0, TILE_COLS - 1).astype(int)
    c = c.where(r % 2 == 0, TILE_COLS - 1 - c)        # serpentine on odd rows
    return t * MIN_PER_FRAME + (r * TILE_COLS + c) * MIN_PER_TILE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--keep-maps', action='store_true')
    args = ap.parse_args()
    os.environ.setdefault('MPLBACKEND', 'Agg'); os.chdir(ev.HERE)
    ns = ev.load_definitions(); cfg = ns['cfg']; pd = ns['pd']; np = ns['np']

    out = Path(str(cfg.data_root)) / 'Evaluations' / 'full_frame_v3_1_1_e55'
    ckpt = cfg.model_path(cfg.seeds[0], CKPT_WHICH)
    hs = ns['load_memmap_tiff'](cfg.image_file); n_t, n_z, _, H, W = hs.shape
    box = (0, 0, H, W)
    z_range = [z for z in range(n_z) if ns['z_in_focus_range'](z, n_z, cfg)]
    layout = ns['p3_layout'](cfg)
    n_ch = len(layout['mask_heads']) + (1 if layout['has_z'] else 0) + 1
    print('checkpoint:', ckpt, 'OK' if ckpt.exists() else 'MISSING')
    print(f'frame {H}x{W}, t 0..{n_t-1}, z_range {z_range[0]}..{z_range[-1]} ({len(z_range)} planes)')
    print(f'map per timepoint ~{len(z_range)*n_ch*H*W*2/1e9:.1f} GB (float16) | out: {out}', flush=True)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    if args.check_only:
        return
    if not ns['tf'].config.list_physical_devices('GPU'):
        raise RuntimeError('No GPU available')

    (out / 'detections').mkdir(parents=True, exist_ok=True); (out / 'maps').mkdir(exist_ok=True)
    lock = (out / 'run.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (out / 'checkpoint_source.txt').write_text(str(ckpt) + '\n')

    members = ns['p3_load_members'](cfg, CKPT_WHICH, None)
    ns['p3_use_pool_norm_stats'](cfg)
    for t in range(n_t):
        csv = out / 'detections' / f'det_t{t:02d}.csv'
        if csv.exists():
            print(f't{t}: done, skipping', flush=True); continue
        print(f'\n=== t{t} | {time.strftime("%H:%M:%S")} ===', flush=True)
        tag = f'full_t{t:02d}'
        prob = ns['p3_run_inference'](members, layout, cfg, box=box, t_range=[t], z_range=z_range, tag=tag)
        for suf in ('.tif', '.json'):                 # H4 quirk: maps land in cfg.qc_dir; move them
            shutil.move(str(cfg.qc_dir / f'{tag}_prob{suf}'), str(out / 'maps' / f'{tag}_prob{suf}'))
        prob = out / 'maps' / f'{tag}_prob.tif'
        gold = ns['p3_gold_in_box'](box, [t], z_range, cfg)
        det, _ = ns['p3_extract_detections'](prob, cfg, gold=gold)
        det.to_csv(csv.with_suffix('.tmp'), index=False); csv.with_suffix('.tmp').rename(csv)
        print(f't{t}: {len(det)} detection rows', flush=True)
        if not args.keep_maps:
            for suf in ('.tif', '.json'):
                (out / 'maps' / f'{tag}_prob{suf}').unlink(missing_ok=True)

    parts = [pd.read_csv(p) for p in sorted((out / 'detections').glob('det_t*.csv'))]
    full = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    full['true_time_min'] = true_time_min(full.t, full.cy, full.cx, H, W)
    full.to_csv(out / 'full_detections.csv', index=False)
    print('\nsaved ->', out / 'full_detections.csv', f'({len(full)} rows)', flush=True)


if __name__ == '__main__':
    main()
