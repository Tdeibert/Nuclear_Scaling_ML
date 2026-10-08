"""Hypothetical whole-droplet exclusion from overlapping observed RANSAC circles."""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from IPython.display import display


def circle_overlap_fraction(x1, y1, r1, x2, y2, r2):
    """Analytic intersection / smaller FULL circle area (un-eroded, unclipped)."""
    if not np.isfinite([x1,y1,r1,x2,y2,r2]).all() or min(r1,r2)<=0:
        raise ValueError('Circles must have finite coordinates and positive radii')
    d=float(np.hypot(x1-x2,y1-y2))
    if d>=r1+r2:return 0.
    if d<=abs(r1-r2):return 1.
    alpha=np.arccos(np.clip((d*d+r1*r1-r2*r2)/(2*d*r1),-1,1))
    beta=np.arccos(np.clip((d*d+r2*r2-r1*r1)/(2*d*r2),-1,1))
    term=max(0.,(-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2))
    return float(np.clip((r1*r1*alpha+r2*r2*beta-.5*np.sqrt(term))/(np.pi*min(r1,r2)**2),0,1))


def droplet_overlap_pairs(circles):
    if circles.duplicated(['t','z','parent']).any():raise ValueError('Duplicate circle keys')
    rows=[]
    for (t,z),group in circles.groupby(['t','z']):
        items=list(group.itertuples())
        for i,a in enumerate(items):
            for b in items[i+1:]:
                fraction=circle_overlap_fraction(a.cx,a.cy,a.radius_px,b.cx,b.cy,b.radius_px)
                if fraction<=0:continue
                rows.append(dict(t=int(t),z=int(z),parent_a=int(a.parent),parent_b=int(b.parent),
                    overlap_fraction=fraction,observed_pair=a.source=='observed_inlier' and b.source=='observed_inlier',
                    source_a=a.source,source_b=b.source))
    return pd.DataFrame(rows,columns=['t','z','parent_a','parent_b','overlap_fraction','observed_pair','source_a','source_b'])


def audit_droplet_overlaps(cfg, ransac_audit, grouped, thresholds=(.05,.10,.20),
                           examples_per_threshold=6, save=True):
    """Any observed pair above tolerance flags both (T, droplet ID) across Z.

    Component impact is relative to contained_gate_pass from the RANSAC comparison,
    NOT the production population. Missing/ambiguous assignments remain unresolved.
    No predicted-only overlap triggers exclusion, no relinking or mask edits occur.
    """
    thresholds=sorted(set(map(float,thresholds)))
    if not thresholds or any(not 0<x<1 for x in thresholds):raise ValueError('Use thresholds strictly between 0 and 1')
    if examples_per_threshold<1:raise ValueError('examples_per_threshold must be positive')
    circles=ransac_audit['circles'].copy()
    if circles.empty:raise ValueError('No RANSAC circles to audit')
    pairs=droplet_overlap_pairs(circles)
    keys=['t','z','label']
    comparison=ransac_audit['comparison'].copy()
    data=comparison.merge(grouped[keys+['nucleus_3d_id']],on=keys,how='left',validate='one_to_one',indicator=True)
    if not data['_merge'].eq('both').all():raise ValueError('RANSAC components do not match grouped objects')
    data=data.drop(columns='_merge')
    # A flag follows the fitted inventory identity, not frame-local model labels.
    data['baseline_pass']=data.contained_gate_pass.astype('boolean')
    raw=tifffile.memmap(cfg.input_image_path,mode='r')
    out=None
    if save:
        out=Path(cfg.qc_dir)/('droplet_overlap_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out.mkdir(parents=True,exist_ok=False)
    all_flags=[]; decisions=[]; summaries=[]; reviews=[]
    lookup=circles.set_index(['t','z','parent'])
    for threshold in thresholds:
        hits=pairs[pairs.observed_pair.eq(True)&pairs.overlap_fraction.gt(threshold)]
        flagged=set()
        for p in hits.itertuples():flagged.update([(p.t,p.parent_a),(p.t,p.parent_b)])
        for t,parent in sorted(flagged):
            evidence=hits[(hits.t==t)&((hits.parent_a==parent)|(hits.parent_b==parent))]
            all_flags.append(dict(threshold=threshold,t=t,parent=parent,
                max_observed_overlap=float(evidence.overlap_fraction.max()),
                evidence_planes=int(evidence.z.nunique()),reason='overlapping_droplet_geometry'))
        trial=data.copy();trial['threshold']=threshold
        trial['flagged_assigned_parent']=[(int(t),int(parent)) in flagged for t,parent in zip(trial.t,trial.parent)]
        trial['assignment_unresolved']=trial.geometry_status.isin(['missing_parent','ambiguous_parent','partial_parent'])
        # Unknown geometry remains unknown, even when another hypothesis is flagged.
        trial['hypothetical_pass']=trial.baseline_pass.copy()
        trial.loc[trial.flagged_assigned_parent & ~trial.assignment_unresolved,'hypothetical_pass']=False
        trial['overlap_removed']=trial.baseline_pass.fillna(False)&~trial.hypothetical_pass.fillna(False)
        trial['overlap_reason']=np.where(trial.flagged_assigned_parent,'overlapping_droplet_geometry','not_flagged')
        decisions.append(trial)
        for t,g in trial.groupby('t'):
            before=g.baseline_pass.fillna(False)
            after=g.hypothetical_pass.fillna(False)
            ids=g.assign(_before=before,_after=after).groupby('nucleus_3d_id')[['_before','_after']].any()
            summaries.append(dict(threshold=threshold,t=int(t),droplets_flagged=sum(tt==t for tt,_ in flagged),
                overlapping_observed_pairs=len(hits[hits.t.eq(t)][['parent_a','parent_b']].drop_duplicates()),
                baseline_components=int(before.sum()),remaining_components=int(after.sum()),
                components_removed=int(g.overlap_removed.sum()),baseline_nuclear_ids=int(ids._before.sum()),
                remaining_nuclear_ids=int(ids._after.sum()),nuclear_ids_lost=int((ids._before&~ids._after).sum()),
                unresolved_assignments=int(g.assignment_unresolved.sum()),
                unknown_gate_decisions=int(g.baseline_pass.isna().sum())))
        # One strongest plane per pair; evenly spaced overlap ranks rather than only worst cases.
        unique=hits.sort_values('overlap_fraction').drop_duplicates(['t','parent_a','parent_b'],keep='last')
        indices=np.unique(np.rint(np.linspace(0,len(unique)-1,min(examples_per_threshold,len(unique)))).astype(int)) if len(unique) else []
        for p in unique.iloc[indices].itertuples():
            a=lookup.loc[(p.t,p.z,p.parent_a)];b=lookup.loc[(p.t,p.z,p.parent_b)]
            x0=max(0,int(np.floor(min(a.cx-a.radius_px,b.cx-b.radius_px)))-10)
            y0=max(0,int(np.floor(min(a.cy-a.radius_px,b.cy-b.radius_px)))-10)
            x1=min(raw.shape[-1],int(np.ceil(max(a.cx+a.radius_px,b.cx+b.radius_px)))+11)
            y1=min(raw.shape[-2],int(np.ceil(max(a.cy+a.radius_px,b.cy+b.radius_px)))+11)
            if x1<=x0 or y1<=y0:continue
            fig,axs=plt.subplots(1,2,figsize=(12,5))
            for ax,ch,title in [(axs[0],cfg.npc_channel_index,'Raw NPC'),(axs[1],cfg.nuclear_channel_index,'Raw NLS')]:
                crop=np.asarray(raw[p.t,p.z,ch,y0:y1,x0:x1]);lo,hi=np.percentile(crop,[1,99.8])
                ax.imshow(crop,cmap='gray',vmin=lo,vmax=max(hi,lo+1e-6),extent=(x0-.5,x1-.5,y1-.5,y0-.5))
                for c,parent,color in [(a,p.parent_a,'cyan'),(b,p.parent_b,'magenta')]:
                    ax.add_patch(plt.Circle((c.cx,c.cy),c.radius_px,fill=False,color=color,lw=1.5))
                    ax.text(c.cx,c.cy,str(parent),color=color)
                ax.set(xlim=(x0-.5,x1-.5),ylim=(y1-.5,y0-.5),title=title);ax.axis('off')
            fig.suptitle('T%d Z%d | droplets %d + %d | overlap %.1f%% > %.0f%%\nBoth IDs flagged across evaluated Z; stitching cause requires visual review'%
                (p.t,p.z,p.parent_a,p.parent_b,100*p.overlap_fraction,100*threshold))
            fig.tight_layout()
            name='tol_%g_T%d_Z%d_D%d_D%d'%(threshold,p.t,p.z,p.parent_a,p.parent_b)
            if out:fig.savefig(out/(name+'.png'),dpi=150,bbox_inches='tight')
            plt.show();plt.close(fig)
            reviews.append(dict(threshold=threshold,t=p.t,z=p.z,parent_a=p.parent_a,parent_b=p.parent_b,
                overlap_fraction=p.overlap_fraction,review_label='',notes=''))
    summary=pd.DataFrame(summaries);decision=pd.concat(decisions,ignore_index=True)
    flags=pd.DataFrame(all_flags,columns=['threshold','t','parent','max_observed_overlap','evidence_planes','reason'])
    display(summary)
    print('Evidence uses available RANSAC audit planes only; thresholds are strict > comparisons.')
    print('Predicted-pair overlaps are reported but never trigger exclusion. Unresolved assignments are not reassigned.')
    print('Counts are linked IDs, not confirmed biological nuclei. No production tables or masks changed.')
    if out:
        pairs.to_csv(out/'all_overlapping_circle_pairs.csv',index=False)
        flags.to_csv(out/'flagged_droplets_by_tolerance.csv',index=False)
        decision.to_csv(out/'hypothetical_component_decisions.csv',index=False)
        summary.to_csv(out/'tolerance_impact.csv',index=False)
        pd.DataFrame(reviews).to_csv(out/'pair_review_sheet.csv',index=False)
        (out/'settings.json').write_text(json.dumps(dict(thresholds=thresholds,
            metric='analytic intersection / smaller full un-eroded circle area; no image clipping',
            trigger='both circles observed_inlier, any evaluated plane, strict > tolerance',
            propagation='both droplet identities across evaluated Z within T',baseline='contained_gate_pass',
            source=str(ransac_audit.get('output_dir')),timepoints=sorted(map(int,circles.t.unique()))),indent=2),encoding='utf8')
        print('Saved:',out)
    return dict(summary=summary,pairs=pairs,flags=flags,decisions=decision,output_dir=out)
