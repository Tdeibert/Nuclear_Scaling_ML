"""Read-only comparison of approved artifact masks with stored gold positives."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from reviewed_hard_negatives import _digest, _identity


def audit_conflicts(review_dir, cfg, gold_rows):
    root = Path(review_dir).resolve()
    manifest = json.loads((root / 'candidates.json').read_text())
    if manifest['raw_image'] != _identity(cfg.image_file):
        raise ValueError('Review and training image differ')
    if manifest['pixel_size_um'] != float(cfg.pixel_size_um):
        raise ValueError('Pixel calibration differs')
    review = pd.read_csv(root / 'review.csv').fillna('')
    records = {r['candidate_id']: r for r in manifest['candidates']}
    if review.candidate_id.duplicated().any() or set(review.candidate_id) != set(records):
        raise ValueError('Review candidate set differs from manifest')
    gold = {}
    for row in gold_rows:
        if row['source'] == 'gold':
            gold.setdefault((int(row['t']), int(row['z'])), []).append(row)
    if not gold:
        raise ValueError('Validated baseline gold rows required')
    pairs = []; objects = {}; cache = {}
    for row in review[review.decision.eq('artifact_negative')].itertuples():
        if not str(row.reviewer).strip():
            raise ValueError('Approved negative lacks reviewer')
        rec = records[row.candidate_id]
        mp = (root / rec['mask_file']).resolve()
        if root not in mp.parents or _digest(mp) != rec['mask_sha256']:
            raise ValueError('Mask changed: ' + row.candidate_id)
        mask = np.load(mp, allow_pickle=False).astype(bool)
        a, b, c, d = [int(rec[k]) for k in ('y0', 'x0', 'y1', 'x1')]
        if mask.shape != (c-a, d-b) or int(mask.sum()) != rec['area_pixels']:
            raise ValueError('Invalid mask geometry')
        positive_union = np.zeros_like(mask)
        local_pairs = []
        for grow in gold.get((rec['t'], rec['z']), []):
            key = str(grow['lab'])
            if key not in cache:
                cache[key] = np.load(key, mmap_mode='r', allow_pickle=False)
            lab = cache[key]; h, w = lab.shape[:2]
            ya, xb = int(grow['cy'])-h//2, int(grow['cx'])-w//2
            y0, x0, y1, x1 = max(a, ya), max(b, xb), min(c, ya+h), min(d, xb+w)
            if y0 >= y1 or x0 >= x1:
                continue
            pos = np.any(lab[y0-ya:y1-ya, x0-xb:x1-xb, 4:7] == 1, axis=-1)
            overlap = pos & mask[y0-a:y1-a, x0-b:x1-b]
            if not overlap.any():
                continue
            positive_union[y0-a:y1-a, x0-b:x1-b] |= pos
            item = dict(candidate_id=row.candidate_id, review_id=rec['review_id'],
                        t=rec['t'], z=rec['z'], gold_stem=grow['stem'],
                        overlap_pixels=int(overlap.sum()), mask_pixels=int(mask.sum()),
                        failure_type=getattr(row, 'failure_type', ''), gold_label_path=key)
            pairs.append(item); local_pairs.append(item)
        if local_pairs:
            objects[row.candidate_id] = dict(record=rec, mask=mask, positive_union=positive_union,
                overlap_pixels=int((mask & positive_union).sum()), pairs=local_pairs)
        cache.clear()  # keep audit memory bounded across candidates
    columns = ['candidate_id', 'review_id', 't', 'z', 'gold_stem', 'overlap_pixels',
               'mask_pixels', 'failure_type', 'gold_label_path']
    table = pd.DataFrame(pairs, columns=columns)
    summary = pd.DataFrame([dict(candidate_id=cid, t=obj['record']['t'], z=obj['record']['z'],
        mask_pixels=int(obj['mask'].sum()), overlapping_gold_patches=len(obj['pairs']),
        overlap_pixels=obj['overlap_pixels'], overlap_fraction=obj['overlap_pixels']/obj['mask'].sum())
        for cid, obj in objects.items()])
    print(f'{len(objects)} conflicting approved components; {len(table)} candidate/gold patch pairs.')
    print('Read-only: no reviews, pools, or gold labels changed. Overlaps include zero-weight gold positives, matching the builder.')
    return dict(pairs=table, summary=summary, objects=objects, review_dir=root)


def plot_conflicts(audit, cfg, start=0, count=6, candidate_ids=None):
    ids = list(audit['objects']) if candidate_ids is None else list(candidate_ids)
    chosen = ids[start:start+count]
    if not chosen:
        print('No conflicts on this page.'); return
    raw = tifffile.memmap(cfg.image_file, mode='r')
    for cid in chosen:
        obj = audit['objects'][cid]; rec = obj['record']
        a, b, c, d = [int(rec[k]) for k in ('y0', 'x0', 'y1', 'x1')]
        pad = max(12, int(np.ceil(3/cfg.pixel_size_um)))
        ya, xb, yc, xd = max(0,a-pad), max(0,b-pad), min(raw.shape[-2],c+pad), min(raw.shape[-1],d+pad)
        negative = np.zeros((yc-ya,xd-xb), bool); positive = np.zeros_like(negative)
        negative[a-ya:c-ya,b-xb:d-xb] = obj['mask']
        # Show all gold-positive pixels in the displayed crop, not just the overlap box.
        for pair in obj['pairs']:
            lab = np.load(pair['gold_label_path'], mmap_mode='r', allow_pickle=False)
            # Recover patch location from the canonical signed-coordinate stem.
            import re
            m = re.search(r'_y(-?\d+)_x(-?\d+)', pair['gold_stem'])
            if m is None: raise ValueError('Unparseable gold stem')
            cy,cx = map(int,m.groups()); h,w = lab.shape[:2]; gy,gx = cy-h//2,cx-w//2
            y0,x0,y1,x1 = max(ya,gy),max(xb,gx),min(yc,gy+h),min(xd,gx+w)
            if y0<y1 and x0<x1:
                positive[y0-ya:y1-ya,x0-xb:x1-xb] |= np.any(lab[y0-gy:y1-gy,x0-gx:x1-gx,4:7]==1,axis=-1)
        fig, axes = plt.subplots(1,3,figsize=(15,5))
        for ax,channel,title in zip(axes,[cfg.nucleus_channel_idx,cfg.nucleus_channel_idx,cfg.npc_channel_idx],
                                   ['NLS raw','NLS: cyan artifact / magenta gold','NPC: cyan artifact / magenta gold']):
            im = np.asarray(raw[rec['t'],rec['z'],channel,ya:yc,xb:xd])
            lo,hi = np.percentile(im,[1,99.8]); ax.imshow(im,cmap='gray',vmin=lo,vmax=max(hi,lo+1e-6))
            if ax is not axes[0]:
                for region,color in [(negative,'cyan'),(positive,'magenta')]:
                    if region.any():
                        ax.contour(np.arange(-1,region.shape[1]+1),np.arange(-1,region.shape[0]+1),
                                   np.pad(region,1),levels=[.5],colors=[color],linewidths=.8)
            ax.set_title(title); ax.axis('off')
        fig.suptitle(f'{cid} / {rec["review_id"]}: {obj["overlap_pixels"]} conflicting pixels '
                     f'({obj["overlap_pixels"]/obj["mask"].sum():.1%} of artifact mask)\n'
                     + 'Gold: ' + ', '.join(p['gold_stem'] for p in obj['pairs']))
        fig.tight_layout(); plt.show(); plt.close(fig)
