"""Read-only gold validation of area/intensity selection; no model execution."""
import ast
import hashlib
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
import roifile
import cv2
from scipy.optimize import curve_fit, linear_sum_assignment, OptimizeWarning
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / 'Nuclear_Scaling_Core'
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
from nuclear_scaling.roi_zcluster import assign_z

DEFAULT_TRAIN_RUN = Path('/home/tdeibert/Projects/Nuclear_Scaling/Models/runs/Vulcan_2.5.4_hardneg_hn2392da6c18e0_iezqw_a0cf3b2a13')
DEFAULT_COMMIT = Path('/data/user/tdeibert/Nuclear_Scaling/Inputs/Training_Data/reviewed_training_commit_20261013_121152_980805')

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()

def recover_and_run(
    run_dir='/data/user/tdeibert/Nuclear_Scaling/Runs/control_extract_1.1__cfg-77ad828f',
    population_tag='pop-89f574a946', match_iou=.10,
):
    """Fresh-kernel entry point; no notebook globals, TensorFlow, or output writes.

    A private lightweight configuration is used only by this audit. It must not
    be substituted for the notebook's full PipelineConfig in other stages.
    """
    import threading
    import time
    root=Path(run_dir).resolve()
    pop_root=root/'analysis/selected_population/populations'/population_tag
    if pop_root.resolve().parent != (root/'analysis/selected_population/populations').resolve():
        raise ValueError('Population tag must name a direct population folder.')
    paths=[root/'pipeline_config.json',root/'config.json',
           pop_root/'population_manifest.json',pop_root/'component_decisions.pkl',
           root/'objects/grouped_z_objects.pkl',root/'tracking/best_z_nuclei_timed.pkl']
    for p in paths:
        if not p.is_file():raise FileNotFoundError(p)
    settings=json.loads(paths[0].read_text())
    signature=json.loads(paths[1].read_text())
    for key,value in signature.items():
        if key in settings and settings[key]!=value:
            raise ValueError('Saved configuration/signature mismatch: '+key)
    manifest=json.loads((pop_root/'population_manifest.json').read_text())
    if manifest['tag']!=population_tag:raise ValueError('Population tag mismatch.')
    image=Path(manifest['upstream']['image']).resolve()
    masks=Path(manifest['upstream']['nuclear_masks']).resolve()
    if masks != root/'mask_tifs/nucleus_instance_hyperstack.tif':
        raise ValueError('Population points to a different segmentation run.')
    for p in [image,masks]:
        if not p.is_file():raise FileNotFoundError(p)
    stat=image.stat()
    if settings['input_fingerprint']!=f'{image}:{stat.st_size}:{stat.st_mtime_ns}':
        raise ValueError('Raw image changed since the saved segmentation.')
    cfg=SimpleNamespace(**settings)
    cfg.obj_dir=root/'objects';cfg.track_dir=root/'tracking'
    cfg.input_image_path=image;cfg.nucleus_instance_hyperstack_path=masks
    cfg.paths=SimpleNamespace(run_dir=root)
    population={'tag':population_tag,'components':pd.read_pickle(pop_root/'component_decisions.pkl')}
    if len(population['components'])!=manifest['rows']['components']:
        raise ValueError('Saved population row count differs from manifest.')
    print('Recovering audit only:',root,'\nPopulation:',population_tag,flush=True)
    started=time.monotonic();stop=threading.Event()
    bar=tqdm(total=None,desc='Recover and validate gold selection',bar_format='{desc} | elapsed {elapsed}')
    def heartbeat():
        while not stop.wait(1):bar.refresh()
    worker=threading.Thread(target=heartbeat,daemon=True);worker.start()
    try:
        result=run(cfg,population,match_iou=match_iou)
    finally:
        stop.set();worker.join();bar.close()
    print(f'Completed in {time.monotonic()-started:.1f}s. No training, inference, or data writes.')
    return result

def inside(box, region):
    return box[0] > region[0] and box[1] > region[1] and box[2] < region[2] and box[3] < region[3]

def raster(poly, shape):
    a,b = np.floor(poly[:,1].min()).astype(int),np.floor(poly[:,0].min()).astype(int)
    c,d = int(np.ceil(poly[:,1].max()))+1,int(np.ceil(poly[:,0].max()))+1
    if a < 0 or b < 0 or c > shape[0] or d > shape[1]:
        raise ValueError('Gold ROI extends outside image; do not silently clip.')
    mask = np.zeros((c-a,d-b),np.uint8)
    cv2.fillPoly(mask,[np.round(poly-[b,a]).astype(np.int32)],1)
    return (a,b,c,d),mask.astype(bool)

def saturation(t, a0, gain, k):
    return a0 + gain * (-np.expm1(-k*np.asarray(t)))

def fit_growth(times, areas):
    """Fit medians at distinct acquisition times; do not weight by object count."""
    med = pd.DataFrame({'time':times,'area':areas}).dropna().groupby('time').area.median()
    if len(med) < 6 or med.index.max() <= med.index.min():
        return dict(status='insufficient distinct times',curve=None)
    t0=float(med.index.min()); x=med.index.to_numpy(float)-t0; y=med.to_numpy(float)
    upper=max(1000.,float(y.max())*100)
    best=None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',OptimizeWarning)
        for rate in (.005,.02,.1):
            try:
                p,cov=curve_fit(saturation,x,y,p0=[max(y[0],1),max(y.max()-y[0],1),rate],
                    bounds=([0,0,1e-6],[upper,upper,10]),maxfev=20000)
                error=float(np.sum((saturation(x,*p)-y)**2))
                if best is None or error<best[0]:best=(error,p,cov)
            except (RuntimeError,ValueError):pass
    if best is None:return dict(status='fit failed',curve=None)
    error,p,cov=best; a0,gain,k=p; plateau=a0+gain
    var=float(cov[0,0]+cov[1,1]+2*cov[0,1]);se=np.sqrt(max(var,0)) if np.isfinite(var) else np.inf
    # Local covariance is diagnostic only, not a biological confidence interval.
    constrained=bool(np.isfinite(se) and se < .5*max(plateau,1) and k*x.max()>1
                     and max(a0,gain)<upper*.99 and gain>1)
    xx=np.linspace(med.index.min(),med.index.max(),200)
    return dict(status='plateau locally constrained' if constrained else 'plateau poorly constrained',
                a0_at_first_time=float(a0),plateau_um2=float(plateau),k_per_min=float(k),
                plateau_local_se=float(se),time_origin_min=t0,median_rmse_um2=float(np.sqrt(error/len(x))),
                curve=(xx,saturation(xx-t0,*p)))

def run(config, population, train_run=DEFAULT_TRAIN_RUN, commit=DEFAULT_COMMIT, match_iou=.1):
    """Spatial holdout only. Matching uses all eligible gold planes, before selection."""
    train_run=Path(train_run);commit=Path(commit)
    tc=json.loads((train_run/'config.json').read_text())
    for k,v in tc.items():
        try:tc[k]=ast.literal_eval(v)
        except (ValueError,SyntaxError):pass
    train_cfg=SimpleNamespace(**tc)
    if config.pixel_size_um!=train_cfg.pixel_size_um:
        raise ValueError('Training/segmentation pixel calibration mismatch.')
    record=json.loads((train_run/'phase3_run.json').read_text())
    pool=Path(record['pools']['gold']['root']);manifest=json.loads((pool/'manifest.json').read_text())
    roi_path=Path(manifest['roi_paths']['nuc'])
    if digest(roi_path)!=manifest['roi_sha256']['nuc']:
        raise ValueError('Gold ROI archive changed since patch generation.')
    expected=list(train_run.glob('*_s42_best.keras'))
    if len(expected)!=1 or Path(config.model_name).resolve()!=expected[0].resolve():
        raise ValueError('This provenance check requires the configured epoch-4 model from the supplied training run.')
    if digest(expected[0])!=config.model_sha256:
        raise ValueError('Configured model digest does not match the training checkpoint.')
    cm=json.loads((commit/'manifest.json').read_text());review=Path(cm['review'])
    negmeta=json.loads((Path(cm['negative_pool'])/'manifest.json').read_text())
    raw_path=Path(config.input_image_path).resolve();stat=raw_path.stat()
    if negmeta['raw_image']!=[str(raw_path),stat.st_size,stat.st_mtime_ns]:
        raise ValueError('Training review and segmentation raw-image provenance differ.')
    if '_hn'+digest(Path(cm['negative_pool'])/'manifest.json')[:12] not in train_cfg.model_name:
        raise ValueError('Hard-negative commit does not match this model.')
    raw=tifffile.memmap(raw_path,mode='r')
    masks=tifffile.memmap(config.nucleus_instance_hyperstack_path,mode='r')
    if masks.shape!=raw.shape[:2]+raw.shape[-2:]:raise ValueError('Mask/raw shape mismatch.')
    H,W=raw.shape[-2:];step=train_cfg.mosaic_tile_px-train_cfg.mosaic_overlap_px
    ty,tx=train_cfg.holdout_tile;m=train_cfg.holdout_margin_px
    region=(ty*step+m,tx*step+m,min(H,ty*step+train_cfg.mosaic_tile_px-m),
            min(W,tx*step+train_cfg.mosaic_tile_px-m))
    print('Spatial holdout interior (y0,x0,y1,x1):',region,'; no area cutoff.')
    items=[]
    for obj in tqdm(roifile.roiread(str(roi_path)),desc='Read original gold outlines',unit='ROI'):
        if not obj.t_position or not obj.z_position:raise ValueError('ROI missing explicit T/Z.')
        t,z=int(obj.t_position)-1,int(obj.z_position)-1
        if not (0<=t<raw.shape[0] and 0<=z<raw.shape[1]):raise ValueError('ROI T/Z out of bounds.')
        poly=np.asarray(obj.coordinates(),float)
        if poly.ndim!=2 or poly.shape[1]!=2 or len(poly)<3 or not np.isfinite(poly).all():raise ValueError('Malformed gold polygon.')
        box,mask=raster(poly,(H,W))
        items.append(dict(t=t,z=z,poly=poly,loc=(box,mask),area=int(mask.sum()),name=obj.name))
    # Linking uses gold only; do not borrow predicted nuclear identities.
    zc=assign_z(items,train_cfg)
    rois=zc.table.copy();rois['gold_id']=rois.nucleus_id
    rois['raster_area_px']=[items[int(i)]['area'] for i in rois.roi_index]
    reviewed=pd.read_csv(review/'review.csv').fillna('')
    rejected=set(reviewed.loc[reviewed.decision.eq('artifact_negative'),'candidate_id'])
    negrecords=[r for r in json.loads((review/'candidates.json').read_text())['candidates'] if r['candidate_id'] in rejected]
    # Exclude disputed gold columns conservatively using bounding-box intersections.
    def disputed(g):
        for row in g.itertuples():
            box=items[row.roi_index]['loc'][0]
            for r in negrecords:
                if r['t']==row.t and r['z']==row.z and max(box[0],r['y0'])<min(box[2],r['y1']) and max(box[1],r['x0'])<min(box[3],r['x1']):return True
        return False
    lower=max(int(config.focus_min_z),int(train_cfg.z_floor))
    upper=raw.shape[1]-1 if config.focus_max_z is None else min(config.focus_max_z,raw.shape[1]-1)
    gold_records=[];eligible_ids=[]
    for gid,g in rois.groupby('gold_id'):
        col=zc.columns.set_index('nucleus_id').loc[gid]
        if not all(inside(items[r.roi_index]['loc'][0],region) for r in g.itertuples()):continue
        e=g[g.z.between(lower,upper)];reason=[]
        if not bool(col.validated):reason.append(str(col.reasons))
        if disputed(g):reason.append('reviewed_artifact_conflict')
        if e.empty:reason.append('no_eligible_Z')
        else:
            peak=e.sort_values(['raster_area_px','z'],ascending=[False,True]).iloc[0]
            if len(e)<3 or int(peak.z) in (int(e.z.min()),int(e.z.max())):reason.append('unbracketed_raster_peak')
            if int(g.raster_area_px.max())>int(e.raster_area_px.max()):reason.append('larger_gold_outside_eligible_Z')
        ok=not reason
        gold_records.append(dict(t=int(g.t.iloc[0]),gold_id=gid,valid_for_max=ok,
                                 reasons=';'.join(reason),n_annotated_planes=len(e)))
        if ok:eligible_ids.append(gid)
    gold_inventory=pd.DataFrame(gold_records)
    if not eligible_ids:raise ValueError('No holdout gold columns with bracketed maxima.')
    gold=rois[rois.gold_id.isin(eligible_ids)&rois.z.between(lower,upper)]
    keys=['t','z','label'];ids=['t','nucleus_3d_id']
    original=pd.read_pickle(config.obj_dir/'grouped_z_objects.pkl');dec=population['components']
    original=original[original.t.isin(dec.t.unique())]
    if original.duplicated(keys).any() or dec.duplicated(keys).any():raise ValueError('Duplicate component keys.')
    rows=original.merge(dec[keys+['status']],on=keys,how='outer',validate='one_to_one',indicator=True)
    if not rows['_merge'].eq('both').all():raise ValueError('Population/run mismatch.')
    rows=rows[rows.status.eq('accepted')&rows.z.between(lower,upper)].copy()
    rows['inside_holdout']=[inside((r.bbox_min_row,r.bbox_min_col,r.bbox_max_row,r.bbox_max_col),region) for r in rows.itertuples()]
    good=rows.groupby(ids).inside_holdout.all();keep=good[good].reset_index()[ids]
    rows=rows.merge(keep,on=ids,validate='many_to_one')
    measured=[]
    for r in tqdm(rows.itertuples(),total=len(rows),desc='Measure accepted holdout masks',unit='component'):
        a,b,c,d=map(int,(r.bbox_min_row,r.bbox_min_col,r.bbox_max_row,r.bbox_max_col))
        mask=masks[int(r.t),int(r.z),a:c,b:d]==r.label;n=int(mask.sum())
        if n!=int(r.area_px) or n==0:raise ValueError('Saved mask/table area mismatch.')
        val=np.asarray(raw[int(r.t),int(r.z),config.nuclear_channel_index,a:c,b:d])[mask]
        if not np.isfinite(val).all():raise ValueError('Nonfinite NLS.')
        measured.append(dict(t=int(r.t),z=int(r.z),label=int(r.label),nucleus_3d_id=r.nucleus_3d_id,
                             area_px=n,area_um2=n*config.pixel_size_um**2,integrated_nls=float(np.sum(val,dtype=np.float64))))
    components=pd.DataFrame(measured)
    if components.empty:raise ValueError('No accepted holdout predictions.')
    lookup={(r.t,r.z,r.label):r for r in components.itertuples()}
    overlaps=[]
    for r in tqdm(gold.itertuples(),total=len(gold),desc='Gold/prediction overlap',unit='ROI'):
        (a,b,c,d),gm=items[r.roi_index]['loc'];vals,counts=np.unique(masks[r.t,r.z,a:c,b:d][gm],return_counts=True)
        for label,inter in zip(vals,counts):
            pred=lookup.get((r.t,r.z,int(label)))
            if pred is None:continue
            iou=inter/(r.raster_area_px+pred.area_px-inter)
            overlaps.append(dict(t=r.t,z=r.z,gold_id=r.gold_id,nucleus_3d_id=pred.nucleus_3d_id,
                                 label=int(label),iou=float(iou),inter=int(inter),gold_area_px=int(r.raster_area_px)))
    ov=pd.DataFrame(overlaps)
    if ov.empty:raise ValueError('No overlap between holdout gold and accepted predictions.')
    scores=ov.groupby(['t','gold_id','nucleus_3d_id']).iou.max().reset_index()
    matches=[]
    for t,s in scores.groupby('t'):
        matrix=s.pivot(index='gold_id',columns='nucleus_3d_id',values='iou').fillna(0)
        # Maximize number of eligible pairs first, then IoU; selection-independent.
        quality=matrix.to_numpy();valid=quality>=match_iou
        rr,cc=linear_sum_assignment(-(valid*(min(quality.shape)+1)+quality*valid))
        for i,j in zip(rr,cc):
            if not valid[i,j]:continue
            matches.append(dict(t=t,gold_id=matrix.index[i],nucleus_3d_id=matrix.columns[j],
                                matching_iou=quality[i,j],ambiguous=bool(valid[i].sum()>1 or valid[:,j].sum()>1)))
    match=pd.DataFrame(matches)
    if match.empty:raise ValueError('No matches at configured association threshold.')
    # Flag and retain ambiguous matches in coverage, but exclude from primary errors.
    match['primary']=~match.ambiguous
    selections=[]
    timing=pd.read_pickle(config.track_dir/'best_z_nuclei_timed.pkl')[ids+['true_time_min']].drop_duplicates()
    for method,metric in [('maximum_area','area_um2'),('maximum_integrated_nls','integrated_nls')]:
        pick=components.sort_values(ids+[metric,'z','label'],ascending=[True,True,False,True,True]).drop_duplicates(ids)
        pick=pick.merge(match,on=ids,validate='one_to_one').merge(timing,on=ids,how='left',validate='one_to_one')
        pick['method']=method;selections.append(pick)
    selected=pd.concat(selections,ignore_index=True)
    if not np.isfinite(selected.true_time_min).all():raise ValueError('Missing acquisition timing.')
    max_gold=gold.groupby('gold_id').raster_area_px.max()*config.pixel_size_um**2
    selected['gold_max_um2']=selected.gold_id.map(max_gold)
    selected['signed_error_um2']=selected.area_um2-selected.gold_max_um2
    selected['absolute_error_um2']=selected.signed_error_um2.abs()
    selected['relative_error_pct']=100*selected.signed_error_um2/selected.gold_max_um2
    selected['absolute_relative_error_pct']=selected.relative_error_pct.abs()
    same={(r.gold_id,r.z):int(r.raster_area_px) for r in gold.itertuples()}
    inters={(r.gold_id,r.z,r.label):r.inter for r in ov.itertuples()}
    selected['gold_same_plane_um2']=[same.get((r.gold_id,r.z),np.nan)*config.pixel_size_um**2 for r in selected.itertuples()]
    selected['same_plane_iou']=[inters.get((r.gold_id,r.z,r.label),0)/(same[(r.gold_id,r.z)]+r.area_px-inters.get((r.gold_id,r.z,r.label),0)) if (r.gold_id,r.z) in same else np.nan for r in selected.itertuples()]
    selected['same_plane_dice']=[2*inters.get((r.gold_id,r.z,r.label),0)/(same[(r.gold_id,r.z)]+r.area_px) if (r.gold_id,r.z) in same else np.nan for r in selected.itertuples()]
    selected['same_plane_error_um2']=selected.area_um2-selected.gold_same_plane_um2
    selected['same_plane_abs_error_um2']=selected.same_plane_error_um2.abs()
    coverage=[]
    for t in sorted(set(gold.t)|set(components.t)):
        ng=gold[gold.t.eq(t)].gold_id.nunique();npred=components[components.t.eq(t)].nucleus_3d_id.nunique();mt=match[match.t.eq(t)]
        coverage.append(dict(t=t,gold_valid=ng,predicted_accepted=npred,matched=len(mt),
                             unmatched_gold=ng-len(mt),unmatched_predictions=npred-len(mt),ambiguous_matches=int(mt.ambiguous.sum())))
    primary=selected[selected.primary]
    summary=primary.groupby(['t','method']).agg(n=('gold_id','size'),bias_um2=('signed_error_um2','mean'),
        mae_um2=('absolute_error_um2','mean'),median_abs_relative_error_pct=('absolute_relative_error_pct','median'),
        median_same_plane_iou=('same_plane_iou','median'),same_plane_assessable=('same_plane_iou','count'),
        same_plane_mae_um2=('same_plane_abs_error_um2','mean'))
    fits={}
    if not primary.empty:
        for method,g in primary.groupby('method'):fits[method]=fit_growth(g.true_time_min,g.area_um2)
        g=primary[primary.method.eq('maximum_area')];fits['gold']=fit_growth(g.true_time_min,g.gold_max_um2)
    return dict(selected=selected,components=components,matches=match,coverage=pd.DataFrame(coverage),
                gold_inventory=gold_inventory,summary=summary,fits=fits,roi_path=str(roi_path),
                roi_sha256=manifest['roi_sha256']['nuc'],region=region,population_tag=population['tag'],
                match_iou=match_iou,run_dir=str(config.paths.run_dir))

def plot(result):
    from IPython.display import display
    print('Coverage (unmatched predictions are NOT automatically false positives):')
    display(result['coverage']);print('Gold column coverage/ambiguity exclusions:')
    display(result['gold_inventory'].groupby(['valid_for_max','reasons']).size().rename('columns'))
    display(result['summary'])
    d=result['selected'];d=d[d.primary]
    if d.empty:print('No unambiguous paired matches; inspect match table.');return
    paired=d.pivot(index=['t','gold_id'],columns='method',values='absolute_error_um2')
    delta=paired.maximum_integrated_nls-paired.maximum_area
    print(f'Paired absolute error: intensity better in {(delta < -1e-8).sum()}, '
          f'area better in {(delta > 1e-8).sum()}, tied in {(delta.abs() <= 1e-8).sum()} '
          f'of {len(delta)} matches; mean intensity-minus-area absolute error {delta.mean():.2f} µm².')
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    for method,color in [('maximum_area','teal'),('maximum_integrated_nls','darkorange')]:
        g=d[d.method.eq(method)]
        axes[0,0].scatter(g.gold_max_um2,g.area_um2,s=16,alpha=.5,color=color,label=method)
        axes[0,1].scatter(g.true_time_min,g.signed_error_um2,s=16,alpha=.4,color=color,label=method)
        axes[1,0].scatter(g.true_time_min,g.area_um2,s=12,alpha=.2,color=color)
        axes[1,1].scatter(g.true_time_min,g.same_plane_iou,s=16,alpha=.4,color=color,label=method)
    g=d[d.method.eq('maximum_area')];axes[1,0].scatter(g.true_time_min,g.gold_max_um2,s=14,alpha=.35,color='black',label='Gold observed maximum')
    for name,color in [('maximum_area','teal'),('maximum_integrated_nls','darkorange'),('gold','black')]:
        f=result['fits'].get(name,{})
        if f.get('curve') is not None:
            x,y=f['curve'];axes[1,0].plot(x,y,color=color,ls='--' if 'poorly' in f['status'] else '-',label=name+' — '+f['status'])
    limit=max(d.gold_max_um2.max(),d.area_um2.max());axes[0,0].plot([0,limit],[0,limit],'k--')
    axes[0,0].set(xlabel='Gold observed maximum area (µm²)',ylabel='Selected area (µm²)')
    axes[0,1].axhline(0,color='black');axes[0,1].set(xlabel='Acquisition time (minutes)',ylabel='Selected minus gold maximum (µm²)')
    axes[1,0].set(xlabel='Acquisition time (minutes)',ylabel='Area (µm²)',title='Saturating fits to time-specific medians; shared matches')
    axes[1,1].set(xlabel='Acquisition time (minutes)',ylabel='Same-plane gold IoU',ylim=(0,1))
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Spatial holdout interior — paired gold validation; no size cutoff')
    fig.tight_layout();plt.show()
    display(pd.DataFrame({k:{a:b for a,b in v.items() if a!='curve'} for k,v in result['fits'].items()}).T)
    print('Gold maxima are observed, bracketed maxima, not proof of exhaustive Z annotation. '
          'Fits describe matched survivors, not individual longitudinal growth. '
          'Local fit uncertainty does not account for repeated nuclei or annotation uncertainty.')
