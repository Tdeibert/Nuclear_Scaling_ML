"""Drop-in survivor review after ARTIFACT_GATE and grouped_z_df.

T0 and T1/T2 are comparison groups, not truth labels. No new rejection rules.
"""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from IPython.display import display


def _survivor_run(zs):
    zs=sorted(set(map(int,zs)))
    longest=current=0; previous=None
    for z in zs:
        current=current+1 if previous is not None and z==previous+1 else 1
        longest=max(longest,current);previous=z
    return longest


def survivor_tables(grouped, gate, times):
    keys=['t','z','label']
    ids=grouped[keys+['nucleus_3d_id']].copy()
    if ids.duplicated(keys).any() or gate.duplicated(keys).any():
        raise ValueError('Duplicate instance keys; cannot assign review identities safely')
    planes=ids.merge(gate,on=keys,how='left',validate='one_to_one',indicator=True)
    if not planes['_merge'].eq('both').all():
        raise ValueError('Grouped instances do not match ARTIFACT_GATE; rerun the gate for this dataset')
    planes=planes.drop(columns='_merge')
    if not planes.gate_pass.isin([True,False]).all():raise ValueError('gate_pass must contain booleans')
    planes=planes[planes.t.isin(times)].copy()
    summaries=[]
    for (t,nid),g in planes.groupby(['t','nucleus_3d_id']):
        passed=g[g.gate_pass]
        if passed.empty:continue
        # A linked ID may branch; show summed area per Z and flag that ambiguity.
        areas=g.groupby('z').area_um2.sum().sort_index()
        kept=passed.groupby('z').area_um2.sum().sort_index()
        peak=int(kept.idxmax());all_peak=int(areas.idxmax())
        summaries.append(dict(t=int(t),nucleus_3d_id=int(nid),
            review_id='T%d_N%d'%(t,nid),detected_planes=int(g.z.nunique()),
            passing_planes=int(passed.z.nunique()),consecutive_passing=_survivor_run(passed.z),
            z_lo=int(g.z.min()),z_hi=int(g.z.max()),peak_z=peak,
            max_area_before_um2=float(areas.max()),max_area_after_um2=float(kept.max()),
            enrichment_median=float(passed.enrichment.median()),
            solidity_median=float(passed.solidity.median()),
            peak_at_detected_edge=all_peak in (int(areas.index.min()),int(areas.index.max())),
            multiple_instances_same_z=bool((g.groupby('z').size()>1).any()),
            parent_overlap_median=float(passed.parent_overlap_fraction.median())))
    return planes,pd.DataFrame(summaries)


def audit_survivors(cfg,grouped,gate,image=None,times=(0,1,2),per_t=6,save=True):
    if per_t<1:raise ValueError('per_t must be positive')
    planes,summary=survivor_tables(grouped,gate['all_instances'],times)
    if summary.empty:raise ValueError('No survivors in the requested timepoints')
    if image is None:image=tifffile.memmap(str(cfg.input_image_path),mode='r')
    labels=tifffile.memmap(str(cfg.nucleus_instance_hyperstack_path),mode='r')
    if labels.shape!=image.shape[:2]+image.shape[-2:]:raise ValueError('Image/instance-stack mismatch')
    eligible=pd.read_pickle(cfg.segmentation_index_path)
    included=eligible.included.astype(str).str.lower().isin(['true','1'])
    eligible=eligible[included & (eligible.z>=cfg.focus_min_z)]
    selected=[]
    for t in times:
        g=summary[summary.t==t].sort_values(['max_area_after_um2','nucleus_3d_id'])
        if g.empty:print('No survivors at T%d'%t);continue
        ix=np.unique(np.rint(np.linspace(0,len(g)-1,min(per_t,len(g)))).astype(int))
        selected.append(g.iloc[ix])
    review=pd.concat(selected,ignore_index=True)
    output=None
    if save:
        output=cfg.qc_dir/('survivor_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        output.mkdir(parents=True,exist_ok=False)
    print('All survivors: comparisons are not manually confirmed truth labels.')
    display(summary.groupby('t').agg(nuclei=('review_id','size'),
        area_median=('max_area_after_um2','median'),passing_planes_median=('passing_planes','median'),
        consecutive_median=('consecutive_passing','median'),enrichment_median=('enrichment_median','median'),
        solidity_median=('solidity_median','median'),branched_ids=('multiple_instances_same_z','sum')).round(3))
    display(review.round(3))
    settings=gate['settings']
    for r in review.itertuples():
        g=planes[(planes.t==r.t)&(planes.nucleus_3d_id==r.nucleus_3d_id)]
        allowed=set(eligible[eligible.t==r.t].z.astype(int))
        zs=[z for z in range(max(cfg.focus_min_z,r.z_lo-1),min(image.shape[1],r.z_hi+2)) if z in allowed]
        if not zs:raise ValueError('Track has no eligible planes')
        # Fixed XY window encompasses the complete track on every displayed Z.
        bounds=[]
        for z,sub in g.groupby('z'):
            mask=np.isin(labels[r.t,int(z)],sub.label.to_numpy())
            yy,xx=np.nonzero(mask)
            if not len(yy):raise ValueError('Instance label missing from saved mask')
            bounds.append((yy.min(),xx.min(),yy.max()+1,xx.max()+1))
        pad=max(8,int(np.ceil(3/cfg.pixel_size_um)))
        a=max(0,min(x[0] for x in bounds)-pad);b=max(0,min(x[1] for x in bounds)-pad)
        c=min(labels.shape[-2],max(x[2] for x in bounds)+pad);d=min(labels.shape[-1],max(x[3] for x in bounds)+pad)
        crops=np.stack([np.asarray(image[r.t,z,cfg.nuclear_channel_index,a:c,b:d],np.float32) for z in zs])
        lo,hi=np.percentile(crops,[1,99.8]);hi=max(hi,lo+1e-6)
        fig,axs=plt.subplots(1,3,figsize=(14,3.5))
        all_area=g.groupby('z').area_um2.sum().reindex(zs)
        pass_area=g[g.gate_pass].groupby('z').area_um2.sum().reindex(zs)
        axs[0].plot(zs,all_area,'o-',color='gray',label='all detected')
        axs[0].plot(zs,pass_area,'o-',color='teal',label='passing only')
        axs[0].set_ylabel('Summed instance area (µm²)');axs[0].legend()
        for ax,col,threshold in [(axs[1],'enrichment',settings['min_enrichment']),
                                 (axs[2],'solidity',settings['min_solidity'])]:
            for accepted,color in [(True,'teal'),(False,'tomato')]:
                s=g[g.gate_pass==accepted];ax.scatter(s.z,s[col],c=color,s=28)
            ax.axhline(threshold,c='black',ls='--');ax.set_ylabel(col)
        for ax in axs:ax.set_xlabel('Z plane');ax.set_xticks(zs);ax.grid(alpha=.2)
        fig.suptitle('%s | passing %d/%d planes | longest run %d | branched ID: %s'%
            (r.review_id,r.passing_planes,r.detected_planes,r.consecutive_passing,r.multiple_instances_same_z))
        fig.tight_layout()
        if output:fig.savefig(output/(r.review_id+'_profiles.png'),dpi=140,bbox_inches='tight')
        plt.show();plt.close(fig)
        fig,axs=plt.subplots(int(np.ceil(len(zs)/5)),5,figsize=(16,3.4*np.ceil(len(zs)/5)),squeeze=False)
        for ax,z,crop in zip(axs.ravel(),zs,crops):
            ax.imshow(crop,cmap='gray',vmin=lo,vmax=hi)
            sub=g[g.z==z];local=np.asarray(labels[r.t,z,a:c,b:d])
            for row in sub.itertuples():
                mask=local==row.label
                ax.contour(mask,levels=[.5],colors=['cyan' if row.gate_pass else 'red'],linewidths=1)
            text='no linked detection' if sub.empty else 'pass %d / %d'%(int(sub.gate_pass.sum()),len(sub))
            ax.set_title('Z%d: %s'%(z,text),fontsize=9);ax.axis('off')
        for ax in axs.ravel()[len(zs):]:ax.axis('off')
        fig.suptitle(r.review_id+' | raw NLS, common intensity scale | cyan: pass; red: reject\n'
                     'Missing outlines mean no linked detection, not confirmed biological absence')
        fig.tight_layout()
        if output:fig.savefig(output/(r.review_id+'_z_stack.png'),dpi=140,bbox_inches='tight')
        plt.show();plt.close(fig)
    if output:
        summary.to_csv(output/'all_survivor_features.csv',index=False)
        planes.to_csv(output/'plane_measurements.csv',index=False)
        review=review.copy();review['review_label']='';review['notes']=''
        review.to_csv(output/'review_sheet.csv',index=False)
        (output/'settings.json').write_text(json.dumps(dict(times=list(times),per_t=per_t,
            selection='evenly spaced ranks of maximum passing area within each timepoint',
            gate=settings,source=str(cfg.nucleus_instance_hyperstack_path)),indent=2),encoding='utf8')
        print('Review labels: artifact / real nucleus / segmentation error / uncertain. Saved:',output)
    return dict(features=summary,planes=planes,selected=review,output_dir=output)


if all(k in globals() for k in ('cfg','grouped_z_df','ARTIFACT_GATE')):
    SURVIVOR_AUDIT=audit_survivors(cfg,grouped_z_df,ARTIFACT_GATE,image=globals().get('img_5d'))
