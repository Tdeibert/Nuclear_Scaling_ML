"""Diverse all-timepoint review proposals, never automatic biological negatives."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from reviewed_hard_negatives import export_artifact_candidates, _identity


def select_diverse_candidates(grouped, population, pixel_size_um, *, excluded_ids=(),
                               per_timepoint=20, seed=42):
    if per_timepoint<1: raise ValueError('per_timepoint must be positive')
    keys=['t','z','label']
    comps=population['components'].copy()
    if set(grouped.t)!=set(comps.t):
        raise ValueError('Population must cover every timepoint in grouped objects; run all-timepoint population stage first.')
    d=grouped[keys+['nucleus_3d_id','centroid_x_px','centroid_y_px']].merge(
        comps.drop(columns='nucleus_3d_id'),on=keys,how='outer',validate='one_to_one',indicator=True)
    if not d['_merge'].eq('both').all():raise ValueError('Population and grouped component keys differ')
    d['review_id']=['T%d_N%d'%(t,n) for t,n in zip(d.t,d.nucleus_3d_id)]
    # Candidate categories describe measurements, not ground-truth failure types.
    counts=d.groupby(['t','nucleus_3d_id']).z.transform('nunique')
    d['single_plane_detection']=counts.eq(1)
    d['tiny_component']=d.area_pixels.le(4)
    d['large_component']=d.area_pixels.mul(pixel_size_um**2).gt(400)
    d['low_enrichment']=d.enrichment.notna() & d.enrichment.lt(1.2)
    d['unresolved_parent']=~d.geometry_status.eq('assigned')
    flags=set(zip(population['flags'].t,population['flags'].parent))
    d['overlapping_droplets']=[(t,p) in flags for t,p in zip(d.t,d.parent)]
    d['accepted_control']=d.status.eq('accepted')
    circle=population['circles'][['t','z','parent','cx','cy','radius_px']]
    d=d.merge(circle,on=['t','z','parent'],how='left',validate='many_to_one')
    distance=np.hypot(d.centroid_x_px-d.cx,d.centroid_y_px-d.cy)
    d['near_droplet_edge']=distance.div(d.radius_px).ge(.8) & d.radius_px.notna()
    cats=['overlapping_droplets','near_droplet_edge','low_enrichment','single_plane_detection',
          'tiny_component','large_component','unresolved_parent','accepted_control']
    d['candidate_signals']=[';'.join(c for c in cats if bool(row[c])) for _,row in d.iterrows()]
    d=d[~d.review_id.isin(set(excluded_ids))].copy()
    rng=np.random.default_rng(seed);selected=[]
    for t,g in d.groupby('t',sort=True):
        # Shuffle separately within each category; round-robin prevents one signal dominating.
        queues={c:list(rng.permutation(g.index[g[c]].to_numpy())) for c in cats}
        used_ids=set();used_parents=set();chosen=[]
        while len(chosen)<per_timepoint:
            added=False
            for cat in cats:
                while queues[cat]:
                    ix=queues[cat].pop();r=d.loc[ix]
                    parent=int(r.parent) if pd.notna(r.parent) and r.parent>0 else None
                    if r.review_id in used_ids or (parent is not None and parent in used_parents):continue
                    selected.append({**r.to_dict(),'selection_category':cat})
                    chosen.append(ix);used_ids.add(r.review_id)
                    if parent is not None:used_parents.add(parent)
                    added=True;break
                if len(chosen)>=per_timepoint:break
            if not added:break
    result=pd.DataFrame(selected)
    if result.empty:raise ValueError('No new candidates; relax selection or check previous reviews')
    return result.drop(columns=['_merge'])


def export_diverse_review(cfg, grouped, population, output_dir, *, previous_reviews=(),
                          per_timepoint=20, seed=42):
    excluded=set()
    for folder in previous_reviews:
        m=json.loads((Path(folder)/'candidates.json').read_text())
        if m['raw_image']!=_identity(cfg.input_image_path) or m['source_masks']!=_identity(cfg.nucleus_instance_hyperstack_path):
            raise ValueError('Previous review belongs to different/changed image or segmentation masks')
        excluded.update(r['review_id'] for r in m['candidates'])
    selected=select_diverse_candidates(grouped,population,cfg.pixel_size_um,
        excluded_ids=excluded,per_timepoint=per_timepoint,seed=seed)
    print('Proposed components by timepoint/category (not artifact labels):')
    print(pd.crosstab(selected.t,selected.selection_category).to_string())
    # Export only selected exact components, not every Z of each selected ID.
    subset=grouped.merge(selected[['t','z','label']],on=['t','z','label'],validate='one_to_one')
    out=export_artifact_candidates(cfg,subset,selected.review_id.tolist(),output_dir)
    selected.to_csv(out/'mining_selection.csv',index=False)
    review=pd.read_csv(out/'review.csv',keep_default_na=False)
    review=review.merge(selected[['t','z','label','selection_category','candidate_signals']],
        on=['t','z','label'],validate='one_to_one')
    review['failure_type']=''
    review.to_csv(out/'review.csv',index=False)
    (out/'mining_settings.json').write_text(json.dumps(dict(per_timepoint=per_timepoint,seed=seed,
        population_tag=population.get('tag'),previous_reviews=list(map(str,previous_reviews)),
        selection='one component per linked ID and at most one per resolved parent per T; round-robin signals',
        interpretation='candidate categories are not ground truth; empty droplet/plane require visual confirmation'),indent=2),encoding='utf8')
    print('Review center mask; assign failure_type manually. Empty droplets/planes cannot be inferred from model detections alone.')
    return out
