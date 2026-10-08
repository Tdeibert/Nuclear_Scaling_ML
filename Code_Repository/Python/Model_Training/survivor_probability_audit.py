"""Read-only probability diagnostics for the existing survivor population."""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from skimage import measure
from scipy.ndimage import binary_erosion
from IPython.display import display


def survivor_probability_metrics(probabilities, mask, core_px=2):
    """Heads: interior, edge, equatorial. Already probabilities, not logits."""
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (3,) + mask.shape or not mask.any():
        raise ValueError('Probability/mask shape mismatch or empty instance')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Expected finite decoded probabilities in [0, 1]')
    if core_px < 1:
        raise ValueError('core_px must be positive')
    # Explicit zero padding at the crop edge prevents artificial core retention.
    core = binary_erosion(mask, iterations=core_px, border_value=0)
    boundary = mask & ~binary_erosion(mask, border_value=0)
    result = dict(mask_pixels=int(mask.sum()), core_pixels=int(core.sum()),
                  interior_median=float(np.median(p[0][mask])),
                  interior_p10=float(np.quantile(p[0][mask], .1)),
                  core_interior_median=float(np.median(p[0][core])) if core.any() else np.nan,
                  boundary_edge_median=float(np.median(p[1][boundary])),
                  equatorial_median=float(np.median(p[2][mask])))
    for threshold in (.5, .7, .8, .9, .95):
        result['interior_fraction_ge_%s' % str(threshold).replace('.', '_')] = float(
            np.mean(p[0][mask] >= threshold))
    return result


def audit_survivor_probabilities(cfg, survivors, core_px=2, review_labels=None, save=True):
    """Measure all survivors in the prior audit; plot its selected examples.

    review_labels maps review_id to artifact / real nucleus / multi-nucleus droplet /
    uncertain. No label is inferred from timepoint. No masks or decisions are changed.
    """
    root = Path(cfg.seg_dir)
    meta = json.loads((root / 'vulcan_all_heads.json').read_text(encoding='utf8'))
    names = meta['channels']
    wanted = ['nucleus_interior', 'nucleus_edge', 'nucleus_equatorial']
    if meta.get('axes') != 'TZCYX' or any(x not in names for x in wanted):
        raise ValueError('Unsupported full-head metadata')
    if meta.get('model_sha256') != cfg.model_sha256:
        raise ValueError('Saved probability model hash does not match cfg')
    if json.dumps(meta.get('config'), sort_keys=True) != json.dumps(cfg.to_run_signature(), sort_keys=True):
        raise ValueError('Saved probability configuration does not match cfg')
    heads = tifffile.memmap(root / 'vulcan_all_heads.tif', mode='r')
    labels = tifffile.memmap(cfg.nucleus_instance_hyperstack_path, mode='r')
    raw = tifffile.memmap(cfg.input_image_path, mode='r')
    if heads.shape != labels.shape[:2] + (len(names),) + labels.shape[-2:]:
        raise ValueError('Probability/instance stack mismatch')
    if raw.shape[:2] + raw.shape[-2:] != labels.shape:
        raise ValueError('Raw/instance stack mismatch')
    planes = survivors['planes'].copy()
    features = survivors['features']
    # Prior audit planes also contains fully rejected IDs; remove those here.
    planes = planes.merge(features[['t', 'nucleus_3d_id', 'review_id']],
                          on=['t', 'nucleus_3d_id'], validate='many_to_one')
    if planes.empty or planes.duplicated(['t', 'z', 'label']).any():
        raise ValueError('No survivors or duplicate instance keys')
    eligible = pd.read_pickle(cfg.segmentation_index_path)
    included = eligible.included.astype(str).str.lower().isin(['true', '1'])
    valid = set(map(tuple, eligible.loc[included, ['t', 'z']].to_numpy()))
    channels = [names.index(x) for x in wanted]
    rows = []
    for (t, z), group in planes.groupby(['t', 'z']):
        t, z = int(t), int(z)
        if (t, z) not in valid or z not in meta['included_z'] or z < cfg.focus_min_z:
            raise ValueError('Survivor references an excluded or unfinished plane')
        lab = np.asarray(labels[t, z])
        boxes = {r.label: r.bbox for r in measure.regionprops(lab)}
        for row in group.itertuples():
            if row.label not in boxes:
                raise ValueError('Survivor label missing from saved segmentation')
            a, b, c, d = boxes[row.label]
            mask = lab[a:c, b:d] == row.label
            p = np.stack([heads[t, z, k, a:c, b:d] for k in channels])
            metrics = survivor_probability_metrics(p, mask, core_px)
            rows.append(dict(t=t, z=z, label=row.label, **metrics))
    measurements = planes.merge(pd.DataFrame(rows), on=['t', 'z', 'label'], validate='one_to_one')
    # Diagnostic score: median of per-component core medians over passing components.
    # Branched IDs are retained but explicitly flagged in the review table.
    scores = measurements[measurements.gate_pass].groupby('review_id').agg(
        core_probability_score=('core_interior_median', 'median'),
        interior_probability_score=('interior_median', 'median'),
        measured_components=('label', 'size'),
        components_with_core=('core_interior_median', 'count'))
    summary = features.merge(scores, on='review_id', validate='one_to_one')
    review_labels = {} if review_labels is None else dict(review_labels)
    unknown = set(review_labels) - set(summary.review_id)
    if unknown:
        raise ValueError('Review IDs not in this audit: %s' % sorted(unknown))
    summary['review_label'] = summary.review_id.map(review_labels).fillna('unreviewed')
    display(summary[['review_id', 'core_probability_score', 'interior_probability_score',
                     'components_with_core', 'measured_components', 'multiple_instances_same_z', 'review_label']])
    out = None
    if save:
        out = Path(cfg.qc_dir) / ('survivor_probability_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out.mkdir(parents=True, exist_ok=False)

    def finish(fig, name):
        fig.tight_layout()
        if out is not None:
            fig.savefig(out / (name + '.png'), dpi=140, bbox_inches='tight')
        plt.show(); plt.close(fig)

    selected = survivors['selected'].review_id.tolist()
    for rid in selected:
        g = measurements[measurements.review_id == rid].sort_values(['z', 'label'])
        if g.empty:
            continue
        # Largest passing component, not an automatically assumed equatorial plane.
        peak = g[g.gate_pass].sort_values('area_um2').iloc[-1]
        t, z, label = int(peak.t), int(peak.z), int(peak.label)
        yy, xx = np.nonzero(labels[t, z] == label)
        pad = max(8, int(np.ceil(3 / cfg.pixel_size_um)))
        a, b = max(0, yy.min()-pad), max(0, xx.min()-pad)
        c, d = min(labels.shape[-2], yy.max()+pad+1), min(labels.shape[-1], xx.max()+pad+1)
        mask = labels[t, z, a:c, b:d] == label
        image = np.asarray(raw[t, z, cfg.nuclear_channel_index, a:c, b:d])
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        lo, hi = np.percentile(image, [1, 99.8])
        axs[0].imshow(image, cmap='gray', vmin=lo, vmax=max(hi, lo+1))
        axs[0].set_title('Raw NLS')
        for ax, k, title in zip(axs[1:], channels, wanted):
            im = ax.imshow(heads[t, z, k, a:c, b:d], vmin=0, vmax=1, cmap='viridis')
            ax.set_title(title); fig.colorbar(im, ax=ax, fraction=.046)
        for ax in axs:
            ax.contour(np.arange(-1, mask.shape[1]+1), np.arange(-1, mask.shape[0]+1),
                       np.pad(mask, 1), levels=[.5], colors=['cyan'])
            ax.set_xlim(-.5, mask.shape[1]-.5); ax.set_ylim(mask.shape[0]-.5, -.5); ax.axis('off')
        fig.suptitle('%s Z%d | largest passing component | core P=%.3f | E=%.2f S=%.2f' %
                     (rid, z, peak.core_interior_median, peak.enrichment, peak.solidity))
        finish(fig, rid + '_maps')
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for col in ['interior_median', 'core_interior_median', 'boundary_edge_median', 'equatorial_median']:
            axs[0].plot(g.z, g[col], 'o', label=col)
        axs[0].set_ylim(0, 1); axs[0].legend(fontsize=7); axs[0].set_ylabel('Probability')
        for passed, color in [(True, 'teal'), (False, 'tomato')]:
            sub = g[g.gate_pass == passed]
            axs[1].scatter(sub.z, sub.enrichment, color=color, label='gate pass' if passed else 'gate reject')
            axs[2].scatter(sub.z, sub.solidity, color=color)
        axs[1].set_ylabel('Enrichment'); axs[1].legend(); axs[2].set_ylabel('Solidity')
        for ax in axs:
            ax.set_xlabel('Z plane'); ax.grid(alpha=.2)
        fig.suptitle(rid + ' | all linked components; no probability filter applied')
        finish(fig, rid + '_profiles')
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, group in summary.groupby('review_label'):
        values = np.sort(group.core_probability_score.dropna().to_numpy())
        if len(values):
            ax.step(values, np.arange(1, len(values)+1)/len(values), where='post',
                    label='%s (n=%d)' % (label, len(values)))
    ax.set(xlabel='Median core probability score', ylabel='Cumulative fraction',
           xlim=(0, 1), ylim=(0, 1), title='Manual review categories; not inferred from T')
    if ax.lines: ax.legend()
    finish(fig, 'review_probability_distributions')
    threshold_rows = []
    for label, group in summary.groupby('review_label'):
        finite = group.core_probability_score.dropna()
        for threshold in (.5, .6, .7, .8, .9, .95):
            threshold_rows.append(dict(review_label=label, threshold=threshold, n_total=len(group),
                n_assessable=len(finite), n_unassessable=len(group)-len(finite),
                fraction_below=float((finite < threshold).mean()) if len(finite) else np.nan))
    thresholds = pd.DataFrame(threshold_rows)
    display(thresholds)
    print('Below-threshold fractions are hypothetical, not applied. Only manual labels support artifact-removal/real-loss estimates.')
    print('Equatorial confidence varies with Z. Empty cores are NaN, never automatic rejection.')
    if out is not None:
        measurements.to_csv(out / 'component_probabilities.csv', index=False)
        summary.to_csv(out / 'survivor_probability_review.csv', index=False)
        thresholds.to_csv(out / 'hypothetical_thresholds.csv', index=False)
        (out / 'settings.json').write_text(json.dumps(dict(core_px=core_px,
            heads_path=str(root / 'vulcan_all_heads.tif'), model_sha256=meta['model_sha256'],
            selected_ids=selected, score='median core probability over passing components',
            source_audit=str(survivors.get('output_dir')), review_labels=review_labels), indent=2), encoding='utf8')
        print('Saved:', out)
    return dict(planes=measurements, summary=summary, thresholds=thresholds, output_dir=out)
