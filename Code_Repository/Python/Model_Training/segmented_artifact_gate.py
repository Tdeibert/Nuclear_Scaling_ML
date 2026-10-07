"""Drop-in enrichment/solidity gate for the v18.1 segmented instance dataset.

Execute after segmentation; creates filtered TABLES, not replacement mask files.
Training measurement formulas are preserved. Support comes from segmented
droplet interiors instead of training's fitted/eroded droplet circles.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology
from IPython.display import display


@dataclass(frozen=True)
class ArtifactGateSettings:
    min_enrichment: float = 1.20
    min_solidity: float = 0.80
    ring_gap_um: float = 0.5
    ring_width_um: float = 2.0
    min_ring_pixels: int = 50


def measure_segment_gate(image, candidate, support, raw_foreground, pixel_size_um,
                         settings=ArtifactGateSettings(), solidity=None):
    """Training formula; masks share a crop padded by the outer ring radius."""
    gap=max(1,int(np.ceil(settings.ring_gap_um/pixel_size_um)))
    outer=gap+max(1,int(np.ceil(settings.ring_width_um/pixel_size_um)))
    ring=(morphology.binary_dilation(candidate,morphology.disk(outer)) &
          ~morphology.binary_dilation(candidate,morphology.disk(gap)) &
          support & ~raw_foreground)
    nr=int(ring.sum())
    inside=float(np.mean(image[candidate],dtype=np.float64)) if candidate.any() else np.nan
    outside=float(np.mean(image[ring],dtype=np.float64)) if nr else np.nan
    valid=nr>=settings.min_ring_pixels and np.isfinite(outside) and outside>0 and np.isfinite(inside)
    enrichment=inside/outside if valid else np.nan
    if solidity is None:
        regions=measure.regionprops(candidate.astype(np.uint8))
        solidity=float(regions[0].solidity) if regions else np.nan
    reasons=[]
    if not valid: reasons.append('unassessable_cytoplasm')
    elif enrichment<settings.min_enrichment: reasons.append('enrichment')
    if not np.isfinite(solidity) or solidity<settings.min_solidity: reasons.append('solidity')
    return dict(enrichment=enrichment,solidity=solidity,nucleus_mean=inside,
                cytoplasm_mean=outside,ring_pixels=nr,reference_ok=bool(valid),
                gate_pass=not reasons,gate_reason='keep' if not reasons else ' + '.join(reasons)),ring


def audit_segmented_artifacts(cfg, image=None, objects=None, best_z=None,
                             settings=ArtifactGateSettings(), save=True, examples_per_class=4):
    if cfg.pixel_size_um<=0: raise ValueError('Positive pixel size is required')
    if image is None: image=tifffile.memmap(str(cfg.input_image_path),mode='r')
    nuclei=tifffile.memmap(str(cfg.nucleus_instance_hyperstack_path),mode='r')
    droplets=tifffile.memmap(str(cfg.droplet_instance_hyperstack_path),mode='r')
    if nuclei.shape!=droplets.shape or nuclei.shape!=image.shape[:2]+image.shape[-2:]:
        raise ValueError('Raw image and instance-stack shapes disagree')
    index=pd.read_pickle(cfg.segmentation_index_path)
    included=index['included'].astype(str).str.lower().isin(['true','1'])
    planes=index[included & (index.z>=cfg.focus_min_z)].sort_values(['t','z'])
    if planes.duplicated(['t','z']).any(): raise ValueError('Duplicate plane index')
    if objects is None:
        p=cfg.obj_dir/'plane_objects.pkl'
        if p.exists(): objects=pd.read_pickle(p)
    if best_z is None:
        p=cfg.obj_dir/'best_z_nuclei.pkl'
        if p.exists(): best_z=pd.read_pickle(p)
    gap=max(1,int(np.ceil(settings.ring_gap_um/cfg.pixel_size_um)))
    outer=gap+max(1,int(np.ceil(settings.ring_width_um/cfg.pixel_size_um)))
    rows=[]; examples={True:[],False:[]}; counts={True:0,False:0}
    rng=np.random.default_rng(42)
    for t, sub in planes.groupby('t',sort=True):
        for z in sub.z:
            t,z=int(t),int(z)
            lab=np.asarray(nuclei[t,z]); drop=np.asarray(droplets[t,z])
            raw=np.asarray(image[t,z,cfg.nuclear_channel_index])
            H,W=lab.shape
            drop_regions={r.label:r for r in measure.regionprops(drop)}
            thresholds={}
            for r in measure.regionprops(lab):
                a,b,c,d=r.bbox
                local=lab[a:c,b:d]==r.label
                ids=drop[a:c,b:d][local]; valid_ids=ids[ids>0]
                parent=int(np.bincount(valid_ids).argmax()) if valid_ids.size else 0
                if parent and parent not in thresholds:
                    dr=drop_regions[parent]; sl=dr.slice
                    values=raw[sl][drop[sl]==parent]
                    thresholds[parent]=(float(filters.threshold_otsu(values))
                        if values.size and values.max()>values.min() else np.nan)
                threshold=thresholds.get(parent,np.nan)
                a,b,c,d=max(0,a-outer),max(0,b-outer),min(H,c+outer),min(W,d+outer)
                sl=(slice(a,c),slice(b,d))
                candidate=lab[sl]==r.label
                support=(drop[sl]==parent) if parent else np.zeros(candidate.shape,bool)
                crop=raw[sl]
                foreground=(crop>threshold)&support if np.isfinite(threshold) else np.zeros_like(support)
                rec,ring=measure_segment_gate(crop,candidate,support,foreground,cfg.pixel_size_um,
                                               settings,float(r.solidity))
                if not parent:
                    rec['gate_pass']=False;rec['gate_reason']='no_parent_droplet'
                rec.update(t=t,z=z,label=int(r.label),parent_droplet=parent,
                    area_um2=float(r.area*cfg.pixel_size_um**2),
                    parent_overlap_fraction=float((ids==parent).sum()/r.area) if parent else 0.,
                    otsu_threshold=threshold,centroid_y_px=float(r.centroid[0]),centroid_x_px=float(r.centroid[1]))
                rows.append(rec)
                key=rec['gate_pass']
                # Deterministic reservoir sampling spans all times/planes without keeping full images.
                counts[key]+=1
                if len(examples[key])<examples_per_class:
                    examples[key].append((rec,crop.copy(),candidate.copy(),ring.copy()))
                elif examples_per_class:
                    slot=int(rng.integers(counts[key]))
                    if slot<examples_per_class:
                        examples[key][slot]=(rec,crop.copy(),candidate.copy(),ring.copy())
        print('Artifact audit T%d complete: %d cumulative plane instances' % (t,len(rows)),flush=True)
    result=pd.DataFrame(rows)
    if result.empty: raise ValueError('No segmented instances on included planes')
    summary=result.groupby('t').agg(total=('gate_pass','size'),accepted=('gate_pass','sum'),
        unassessable=('reference_ok',lambda s:int((~s).sum())))
    summary['rejected']=summary.total-summary.accepted
    summary['rejected_fraction']=summary.rejected/summary.total
    display(summary.round(3))
    display(pd.crosstab(result.t,result.gate_reason))
    def attach(table):
        if table is None: return None
        if 'repair_status' in table and table.repair_status.eq('repaired').any():
            raise ValueError('Supply unrepaired rows: this audit measures the saved model instances')
        cols=['t','z','label','enrichment','solidity','ring_pixels','reference_ok','gate_pass','gate_reason']
        merged=table.merge(result[cols],on=['t','z','label'],how='left',validate='many_to_one',indicator=True)
        if not merged['_merge'].eq('both').all():
            raise ValueError('Some supplied object rows do not match the audited plane instances')
        return merged.drop(columns='_merge')
    annotated=attach(objects)
    best=attach(best_z)
    accepted=result[result.gate_pass].copy()
    rejected=result[~result.gate_pass].copy()
    filtered_objects=annotated[annotated.gate_pass].copy() if annotated is not None else accepted.copy()
    filtered_best=best[best.gate_pass].copy() if best is not None else None
    if best is not None:
        print('Selected best-Z nuclei (no fallback to another Z):')
        display(best.groupby('t').gate_pass.agg(['size','sum','mean']).rename(
            columns={'size':'total','sum':'accepted','mean':'accepted_fraction'}).round(3))
    fig,axs=plt.subplots(1,2,figsize=(13,4))
    summary[['accepted','rejected']].plot.bar(stacked=True,ax=axs[0],color=['teal','tomato'])
    axs[0].set_title('All segmented plane instances');axs[0].set_ylabel('Count')
    for passed,color in [(True,'teal'),(False,'tomato')]:
        s=result[result.gate_pass==passed]
        axs[1].scatter(s.enrichment,s.solidity,s=8,alpha=.3,c=color,label='accepted' if passed else 'rejected')
    axs[1].axvline(settings.min_enrichment,c='black',ls='--')
    axs[1].axhline(settings.min_solidity,c='black',ls='--')
    axs[1].set(xlabel='Raw NLS enrichment',ylabel='Solidity',title='Gate measurements (invalid references omitted)')
    axs[1].legend();plt.tight_layout();plt.show()
    samples=examples[False]+examples[True]
    if samples:
        fig,axs=plt.subplots(int(np.ceil(len(samples)/2)),2,figsize=(12,4*np.ceil(len(samples)/2)))
        for ax,(r,crop,mask,ring) in zip(np.atleast_1d(axs).ravel(),samples):
            lo,hi=np.percentile(crop,[1,99.8]);ax.imshow(crop,cmap='gray',vmin=lo,vmax=max(hi,lo+1))
            ax.contour(mask,levels=[.5],colors=['cyan' if r['gate_pass'] else 'red'])
            if ring.any():ax.contour(ring,levels=[.5],colors=['yellow'],linewidths=.5)
            ax.set_title('T%d Z%d label %d: %s\nE=%.2f S=%.2f ring=%d' %
                (r['t'],r['z'],r['label'],r['gate_reason'],r['enrichment'],r['solidity'],r['ring_pixels']))
            ax.axis('off')
        for ax in np.atleast_1d(axs).ravel()[len(samples):]:ax.axis('off')
        plt.tight_layout();plt.show()
    output=None
    if save:
        output=cfg.qc_dir/('enrichment_solidity_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        output.mkdir(parents=True,exist_ok=False)
        result.to_csv(output/'all_instances.csv',index=False)
        summary.to_csv(output/'summary_by_t.csv')
        filtered_objects.to_pickle(output/'accepted_plane_objects.pkl')
        if filtered_best is not None:filtered_best.to_pickle(output/'accepted_best_z.pkl')
        (output/'settings.json').write_text(json.dumps(dict(settings=asdict(settings),
            pixel_size_um=cfg.pixel_size_um,support='segmented droplet interior',
            candidate_source=str(cfg.nucleus_instance_hyperstack_path),
            reference_exclusion='raw Otsu foreground',best_z_policy='filter selected row; no reselection'),indent=2),encoding='utf8')
        print('Saved audit and filtered tables:',output)
    print('Original masks/tables retained. Unassessable references are not proof of an artifact.')
    return dict(all_instances=result,accepted=accepted,rejected=rejected,summary=summary,
        annotated_objects=annotated,filtered_objects=filtered_objects,annotated_best_z=best,
        filtered_best_z=filtered_best,settings=asdict(settings),output_dir=output)


if 'cfg' in globals():
    ARTIFACT_GATE=audit_segmented_artifacts(cfg,image=globals().get('img_5d'),
        objects=globals().get('objects_df'),best_z=globals().get('best_z_df'))
