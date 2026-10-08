"""One-pixel sensitivity test and raw-channel evidence; never rewrites masks."""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import tifffile
from IPython.display import display


def test_one_pixel_exclusion(gate, grouped, pixel_size_um):
    data = gate['all_instances'].copy()
    keys = ['t', 'z', 'label']
    # area_um2 was measured directly from the saved instance mask by the gate.
    pixels = data.area_um2.to_numpy() / pixel_size_um**2
    if not np.allclose(pixels, np.rint(pixels), atol=1e-5, rtol=0):
        raise ValueError('Gate areas disagree with the current pixel calibration')
    data['area_pixels'] = np.rint(pixels).astype(int)
    data['test_gate_pass'] = data.gate_pass & data.area_pixels.gt(1)
    data['one_pixel_removed'] = data.gate_pass & ~data.test_gate_pass
    ids = grouped[keys + ['nucleus_3d_id']]
    linked = ids.merge(data, on=keys, validate='one_to_one', how='outer', indicator=True)
    if not linked['_merge'].eq('both').all():
        raise ValueError('Grouped objects and gate must cover identical instance keys')
    tracks = linked.groupby(['t', 'nucleus_3d_id']).agg(
        before=('gate_pass', 'any'), after=('test_gate_pass', 'any')).reset_index()
    tracks['lost'] = tracks.before & ~tracks.after
    counts = data.groupby('t').agg(before_components=('gate_pass', 'sum'),
        after_components=('test_gate_pass', 'sum'), removed_one_pixel=('one_pixel_removed', 'sum'))
    counts = counts.join(tracks.groupby('t').agg(before_ids=('before', 'sum'),
        after_ids=('after', 'sum'), lost_ids=('lost', 'sum')))
    return data, tracks, counts


def envelope_measurements(npc, nucleus, parent, other_nuclei, pixel_size_um):
    # The caller pads the crop beyond these distances; parent limits reference support.
    inside = distance_transform_edt(nucleus) * pixel_size_um
    outside = distance_transform_edt(~nucleus) * pixel_size_um
    boundary = (nucleus & (inside <= .5)) | (~nucleus & parent & ~other_nuclei & (outside <= .5))
    reference = parent & ~nucleus & ~other_nuclei & (outside >= 1.) & (outside <= 2.)
    b, r = npc[boundary], npc[reference]
    valid = len(b) > 0 and len(r) >= 50 and np.isfinite(b).all() and np.isfinite(r).all()
    bmean = float(b.mean()) if len(b) else np.nan
    rmean = float(r.mean()) if len(r) else np.nan
    ratio = bmean/rmean if valid and rmean > 0 else np.nan
    return dict(npc_boundary_mean=bmean, npc_reference_mean=rmean,
                npc_boundary_reference_ratio=ratio, npc_boundary_pixels=len(b),
                npc_reference_pixels=len(r), npc_reference_ok=bool(valid and rmean > 0)), boundary, reference


def audit_survivor_envelopes(cfg, gate, grouped, probability_audit, save=True):
    candidate, tracks, counts = test_one_pixel_exclusion(gate, grouped, cfg.pixel_size_um)
    print('Hypothetical one-pixel exclusion across the gate dataset; production decisions unchanged:')
    display(counts)
    summary = probability_audit['summary']
    reviewed = summary[summary.review_label.isin(['artifact', 'real nucleus'])].copy()
    if reviewed.empty:
        raise ValueError('Run the manually labeled probability audit first')
    reviewed = reviewed.merge(tracks, on=['t', 'nucleus_3d_id'], validate='one_to_one')
    display(reviewed.groupby('review_label').agg(reviewed=('review_id', 'size'),
        removed_by_one_pixel_test=('lost', 'sum')))
    raw = tifffile.memmap(cfg.input_image_path, mode='r')
    nuclei = tifffile.memmap(cfg.nucleus_instance_hyperstack_path, mode='r')
    droplets = tifffile.memmap(cfg.droplet_instance_hyperstack_path, mode='r')
    if raw.shape[:2] + raw.shape[-2:] != nuclei.shape or droplets.shape != nuclei.shape:
        raise ValueError('Raw/nuclear/droplet stack mismatch')
    measured = probability_audit['planes'].merge(
        candidate[['t', 'z', 'label', 'test_gate_pass']], on=['t', 'z', 'label'],
        validate='one_to_one', how='left')
    if measured.test_gate_pass.isna().any():
        raise ValueError('Probability audit references missing gate instances')
    out = None
    if save:
        out = Path(cfg.qc_dir)/('survivor_envelope_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out.mkdir(parents=True, exist_ok=False)
    rows = []
    for review in reviewed[reviewed.after].itertuples():
        g = measured[measured.review_id.eq(review.review_id) & measured.test_gate_pass]
        peak_index = g.area_um2.idxmax()
        for ix, row in g.iterrows():
            t, z, label = int(row.t), int(row.z), int(row.label)
            plane = np.asarray(nuclei[t, z]); drop = np.asarray(droplets[t, z])
            yy, xx = np.nonzero(plane == label)
            if len(yy) != row.mask_pixels:
                raise ValueError('Masks changed since probability audit')
            values = drop[yy, xx]; values = values[values > 0]
            parent_id = int(np.bincount(values).argmax()) if len(values) else 0
            if parent_id != int(row.parent_droplet):
                raise ValueError('Parent droplet changed since gate audit')
            parent_area = int(np.count_nonzero(drop == parent_id)) if parent_id else 0
            pad = max(8, int(np.ceil(3/cfg.pixel_size_um)))
            a,b = max(0, yy.min()-pad), max(0, xx.min()-pad)
            c,d = min(plane.shape[0], yy.max()+pad+1), min(plane.shape[1], xx.max()+pad+1)
            mask = plane[a:c,b:d] == label
            parent = (drop[a:c,b:d] == parent_id) if parent_id else np.zeros(mask.shape, bool)
            npc = np.asarray(raw[t,z,cfg.npc_channel_index,a:c,b:d], dtype=float)
            other = (plane[a:c,b:d] > 0) & ~mask
            metrics, band, reference = envelope_measurements(npc, mask, parent, other, cfg.pixel_size_um)
            overlap = int(np.count_nonzero(parent & mask))
            rec = dict(review_id=review.review_id, review_label=review.review_label,
                t=t, z=z, label=label, area_pixels=len(yy), parent_droplet=parent_id,
                parent_area_pixels=parent_area, parent_overlap_fraction=overlap/len(yy),
                nucleus_to_parent_area=len(yy)/parent_area if parent_area else np.nan,
                parent_occupied_fraction=overlap/parent_area if parent_area else np.nan,
                image_border=bool(yy.min()==0 or xx.min()==0 or yy.max()==plane.shape[0]-1 or xx.max()==plane.shape[1]-1),
                **metrics)
            rows.append(rec)
            if ix != peak_index:
                continue
            fig, axs = plt.subplots(1,3,figsize=(16,5))
            nls = np.asarray(raw[t,z,cfg.nuclear_channel_index,a:c,b:d])
            for ax, image, title in [(axs[0],nls,'Raw NLS'),(axs[1],npc,'Raw NPC'),(axs[2],npc,'NPC measurement regions')]:
                lo,hi=np.percentile(image,[1,99.8])
                ax.imshow(image,cmap='gray',vmin=lo,vmax=max(hi,lo+1e-6),interpolation='nearest')
                ax.contour(mask,levels=[.5],colors=['cyan'])
                if parent.any() and not parent.all(): ax.contour(parent,levels=[.5],colors=['yellow'])
                ax.set_title(title); ax.axis('off')
            for region,color in [(band,'magenta'),(reference,'lime')]:
                if region.any() and not region.all(): axs[2].contour(region,levels=[.5],colors=[color],linewidths=.6)
            fig.suptitle('%s | %s | Z%d | nuclear/parent area=%.3f | NPC ratio=%.3f\n'
                'Largest remaining passing component; cyan=nucleus, yellow=parent, magenta=boundary band, green=reference\n'
                'Channels scaled separately per crop; use raw-intensity metrics for comparison' %
                (review.review_id,review.review_label,z,rec['nucleus_to_parent_area'],rec['npc_boundary_reference_ratio']))
            fig.tight_layout()
            if out: fig.savefig(out/(review.review_id+'_raw_envelope.png'),dpi=150,bbox_inches='tight')
            plt.show(); plt.close(fig)
    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        display(metrics.groupby(['review_id','review_label']).agg(
            components=('label','size'), median_npc_ratio=('npc_boundary_reference_ratio','median'),
            assessable_npc_components=('npc_reference_ok','sum'),
            max_nucleus_parent_ratio=('nucleus_to_parent_area','max')))
        fig,ax=plt.subplots(figsize=(8,5))
        for category,g in metrics.groupby('review_label'):
            ax.scatter(g.nucleus_to_parent_area,g.npc_boundary_reference_ratio,label=category,alpha=.6)
        ax.set(xlabel='Nuclear component area / parent droplet area',ylabel='Raw NPC boundary / local reference',
               title='Remaining passing components; exploratory, no threshold applied')
        ax.legend();fig.tight_layout()
        if out: fig.savefig(out/'envelope_comparison.png',dpi=150,bbox_inches='tight')
        plt.show();plt.close(fig)
    if out:
        candidate.to_csv(out/'hypothetical_component_decisions.csv',index=False)
        tracks.to_csv(out/'hypothetical_track_survival.csv',index=False)
        counts.to_csv(out/'one_pixel_counts_by_t.csv')
        reviewed.to_csv(out/'reviewed_one_pixel_effect.csv',index=False)
        metrics.to_csv(out/'raw_envelope_metrics.csv',index=False)
        (out/'settings.json').write_text(json.dumps(dict(pixel_size_um=cfg.pixel_size_um,
            test='remove gate-passing components with exactly one pixel; no relinking',
            npc_boundary_um=.5, npc_reference_um=[1,2], min_reference_pixels=50,
            gate_source=str(gate.get('output_dir')),probability_source=str(probability_audit.get('output_dir')),
            masks=str(cfg.nucleus_instance_hyperstack_path)),indent=2),encoding='utf8')
        print('Saved:',out)
    print('NPC ratios are exploratory, not envelope-closure measurements. Parent labels are plane-local.')
    print('Original gate and masks unchanged; no multi-nucleus droplet exclusion applied.')
    return dict(candidate_components=candidate,track_survival=tracks,counts=counts,
                reviewed_effect=reviewed,envelope_metrics=metrics,output_dir=out)
