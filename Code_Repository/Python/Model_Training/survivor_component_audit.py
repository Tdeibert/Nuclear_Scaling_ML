"""Inspect saved survivor components without changing masks or linking."""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from IPython.display import display


def component_overlap_metrics(first, second):
    a, b = int(first.sum()), int(second.sum())
    intersection = int(np.count_nonzero(first & second))
    return dict(overlap_pixels=intersection,
                overlap_over_smaller=intersection/min(a, b) if min(a, b) else np.nan,
                iou=intersection/(a+b-intersection) if a+b else np.nan,
                next_over_current_area=b/a if a else np.nan)


def audit_survivor_components(cfg, probability_audit, review_ids=None, save=True):
    """Default to manually labeled artifacts. Additional real controls may be supplied.

    Adjacent-Z pair overlaps are geometric evidence, not a reconstruction of link
    decisions: production links against previous ID unions, not individual pairs.
    """
    planes = probability_audit['planes'].copy()
    summary = probability_audit['summary'].copy()
    if review_ids is None:
        review_ids = summary.loc[summary.review_label.eq('artifact'), 'review_id'].tolist()
    review_ids = list(dict.fromkeys(review_ids))
    if not review_ids:
        raise ValueError('No manually labeled artifacts. Supply review_ids or rerun the labeled probability audit.')
    missing = set(review_ids) - set(summary.review_id)
    if missing:
        raise ValueError('Unknown review IDs: %s' % sorted(missing))
    if planes.duplicated(['t', 'z', 'label']).any():
        raise ValueError('Duplicate component identities')
    labels = tifffile.memmap(cfg.nucleus_instance_hyperstack_path, mode='r')
    raw = tifffile.memmap(cfg.input_image_path, mode='r')
    if raw.shape[:2] + raw.shape[-2:] != labels.shape:
        raise ValueError('Image and mask shapes differ')
    index = pd.read_pickle(cfg.segmentation_index_path)
    valid = set(map(tuple, index.loc[index.included.astype(str).str.lower().isin(['true', '1']), ['t', 'z']].to_numpy()))
    planes = planes[planes.review_id.isin(review_ids)].copy()
    if not planes.gate_pass.isin([True, False]).all():
        raise ValueError('Invalid gate decisions')
    planes['contributes_to_score'] = planes.gate_pass & np.isfinite(planes.core_interior_median)
    planes['score_exclusion_reason'] = np.where(~planes.gate_pass, 'gate rejected',
        np.where(planes.contributes_to_score, 'included', 'no finite core probability'))
    planes['component_area_um2'] = planes.mask_pixels * cfg.pixel_size_um**2
    planes['components_same_z'] = planes.groupby(['review_id', 'z']).label.transform('size')
    out = None
    if save:
        out = Path(cfg.qc_dir) / ('survivor_components_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out.mkdir(parents=True, exist_ok=False)
    links, features = [], []

    def finish(fig, name):
        fig.tight_layout()
        if out is not None:
            fig.savefig(out / (name + '.png'), dpi=160, bbox_inches='tight')
        plt.show(); plt.close(fig)

    for rid in review_ids:
        g = planes[planes.review_id.eq(rid)].sort_values(['z', 'label'])
        if g.empty or g.t.nunique() != 1:
            raise ValueError('Missing or ambiguous component rows for ' + rid)
        t = int(g.t.iloc[0])
        bounds = []
        for r in g.itertuples():
            if (t, int(r.z)) not in valid or r.z < cfg.focus_min_z:
                raise ValueError('Component references an excluded plane')
            yy, xx = np.nonzero(labels[t, int(r.z)] == int(r.label))
            if len(yy) != r.mask_pixels:
                raise ValueError('Saved masks changed since probability audit: ' + rid)
            bounds.append((yy.min(), xx.min(), yy.max()+1, xx.max()+1))
        pad = max(8, int(np.ceil(3/cfg.pixel_size_um)))
        a = max(0, min(x[0] for x in bounds)-pad)
        b = max(0, min(x[1] for x in bounds)-pad)
        c = min(labels.shape[-2], max(x[2] for x in bounds)+pad)
        d = min(labels.shape[-1], max(x[3] for x in bounds)+pad)
        zs = [z for z in range(max(cfg.focus_min_z, int(g.z.min())-1),
                              min(labels.shape[1], int(g.z.max())+2)) if (t, z) in valid]
        masks = {(int(r.z), int(r.label)): labels[t, int(r.z), a:c, b:d] == int(r.label)
                 for r in g.itertuples()}
        for r in g.itertuples():
            # Include zero-overlap pairs to reveal discontinuous/branched ID membership.
            for s in g[g.z.eq(r.z+1)].itertuples():
                links.append(dict(review_id=rid, t=t, z=int(r.z), label=int(r.label),
                    next_z=int(s.z), next_label=int(s.label),
                    current_gate_pass=bool(r.gate_pass), next_gate_pass=bool(s.gate_pass),
                    **component_overlap_metrics(masks[(int(r.z), int(r.label))],
                                                masks[(int(s.z), int(s.label))])))
        scores = g.loc[g.contributes_to_score, 'core_interior_median']
        recomputed = float(scores.median()) if len(scores) else np.nan
        recorded = float(summary.set_index('review_id').loc[rid, 'core_probability_score'])
        if not np.isclose(recomputed, recorded, equal_nan=True):
            raise ValueError('Probability score could not be reproduced for ' + rid)
        features.append(dict(review_id=rid, t=t, components=len(g),
            passing_components=int(g.gate_pass.sum()), score_contributors=len(scores),
            reproduced_score=recomputed, min_pixels=int(g.mask_pixels.min()),
            max_pixels=int(g.mask_pixels.max()), branches=bool(g.components_same_z.gt(1).any())))
        print('\n', rid, '| reproduced probability score:', recomputed)
        display(g[['review_id', 'z', 'label', 'mask_pixels', 'component_area_um2',
                   'core_pixels', 'gate_pass', 'gate_reason', 'enrichment', 'solidity',
                   'core_interior_median', 'contributes_to_score', 'score_exclusion_reason',
                   'components_same_z']])
        crops = [np.asarray(raw[t, z, cfg.nuclear_channel_index, a:c, b:d]) for z in zs]
        # Sample only for display scaling; metrics above use full-resolution masks.
        sampled = np.concatenate([x[::4, ::4].ravel() for x in crops])
        lo, hi = np.percentile(sampled, [1, 99.8]); hi = max(hi, lo+1e-6)
        for start in range(0, len(zs), 8):
            page = zs[start:start+8]
            fig, axs = plt.subplots(int(np.ceil(len(page)/4)), 4,
                                    figsize=(18, 4.5*np.ceil(len(page)/4)), squeeze=False)
            for ax, z in zip(axs.ravel(), page):
                ax.imshow(crops[zs.index(z)], cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
                sub = g[g.z.eq(z)]
                for r in sub.itertuples():
                    mask = masks[(int(z), int(r.label))]
                    color = 'cyan' if r.gate_pass else 'red'
                    ax.contour(np.arange(-1, mask.shape[1]+1), np.arange(-1, mask.shape[0]+1),
                               np.pad(mask, 1), levels=[.5], colors=[color], linewidths=.8)
                    yy, xx = np.nonzero(mask)
                    ax.text(xx.mean(), yy.mean(), '%d%s (%dpx)' %
                            (r.label, '*' if r.contributes_to_score else '', r.mask_pixels),
                            color='yellow', fontsize=8)
                    if r.mask_pixels <= 4:
                        ax.scatter(xx, yy, s=80, facecolors='none', edgecolors=color)
                ax.set_title('Z%d | %d linked components' % (z, len(sub)))
                ax.set_xlim(-.5, d-b-.5); ax.set_ylim(c-a-.5, -.5); ax.axis('off')
            for ax in axs.ravel()[len(page):]: ax.axis('off')
            fig.suptitle(rid + ' | cyan: gate pass; red: reject; *: contributes to score\n'
                         'Fixed XY window/intensity scale. Circles highlight <=4px masks (not their size).')
            finish(fig, rid + '_z_page_%02d' % (start//8+1))
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        for passed, color in [(True, 'teal'), (False, 'tomato')]:
            sub = g[g.gate_pass.eq(passed)]
            axs[0].scatter(sub.z, sub.mask_pixels, color=color, label='pass' if passed else 'reject')
        axs[0].set_yscale('log'); axs[0].set_ylabel('Component area (pixels; log scale)'); axs[0].legend()
        axs[1].scatter(g.z, g.core_interior_median, color='gray', label='all finite core scores')
        contributing = g[g.contributes_to_score]
        axs[1].scatter(contributing.z, contributing.core_interior_median, marker='*', s=100,
                       color='teal', label='score contributors')
        if np.isfinite(recomputed): axs[1].axhline(recomputed, ls='--', color='teal')
        axs[1].set_ylim(0, 1); axs[1].set_ylabel('Core probability'); axs[1].legend(fontsize=8)
        for ax in axs: ax.set_xlabel('Z'); ax.grid(alpha=.2)
        fig.suptitle(rid); finish(fig, rid + '_components')
    overlaps = pd.DataFrame(links, columns=['review_id', 't', 'z', 'label', 'next_z', 'next_label',
        'current_gate_pass', 'next_gate_pass', 'overlap_pixels', 'overlap_over_smaller', 'iou',
        'next_over_current_area'])
    features = pd.DataFrame(features)
    display(features); display(overlaps)
    print('Pair overlaps are not inferred link edges; production compares previous ID unions.')
    print('No area cutoff, probability cutoff, relinking, or multi-nucleus exclusion applied.')
    if out is not None:
        planes.to_csv(out / 'component_decisions.csv', index=False)
        overlaps.to_csv(out / 'adjacent_z_pair_overlaps.csv', index=False)
        features.to_csv(out / 'score_reproduction.csv', index=False)
        (out / 'settings.json').write_text(json.dumps(dict(review_ids=review_ids,
            probability_audit=str(probability_audit.get('output_dir')),
            masks=str(cfg.nucleus_instance_hyperstack_path), pixel_size_um=cfg.pixel_size_um,
            overlap_scope='same-ID component pairs on adjacent Z; not historical link edges'), indent=2), encoding='utf8')
        print('Saved:', out)
    return dict(components=planes, overlaps=overlaps, summary=features, output_dir=out)
