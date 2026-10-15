"""Matched-region comparison of old Vulcan, new best, and new resume weights.

No retraining, annotation changes, or production postprocessing. Reuses notebook
global normalization and its exact full-image patch grid, Hann blending, and TTA.
"""
import gc
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

NEW_RUN = Path('/home/tdeibert/Projects/Nuclear_Scaling/Models/runs/Vulcan_2.5.4_hardneg_hn2392da6c18e0_iezqw_a0cf3b2a13')
COMMIT = Path('/data/user/tdeibert/Nuclear_Scaling/Inputs/Training_Data/reviewed_training_commit_20261013_121152_980805')
OLD_MODEL = Path('/data/user/tdeibert/Nuclear_Scaling/Inputs/Models/Vulcan_2.5.3.keras')
NEW_BEST = NEW_RUN/'Vulcan_2.5.4_hardneg_hn2392da6c18e0_iezqw_s42_best.keras'
OLD_RUN = Path('/data/user/tdeibert/Nuclear_Scaling/Runs/control_extract_1.1__cfg-9c4d8e9e')
NEW_SEG_RUN = Path('/data/user/tdeibert/Nuclear_Scaling/Runs/control_extract_1.1__cfg-77ad828f')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def region_box(y,x,height,width,size=512):
    h,w=min(size,height),min(size,width)
    a=int(np.clip(round(y)-h//2,0,height-h));b=int(np.clip(round(x)-w//2,0,width-w))
    return a,b,a+h,b+w


def intersects(a,b):
    return max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])


def select_cases(cfg,img_5d,per_group=4):
    """Deterministic suggestions; unreviewed candidates are not assigned truth."""
    if Path(cfg.paths.run_dir).resolve()!=NEW_SEG_RUN.resolve():
        raise ValueError('Load the new comparison segmentation run cfg-77ad828f first')
    commit=json.loads((COMMIT/'manifest.json').read_text());review=Path(commit['review'])
    manifest=json.loads((review/'candidates.json').read_text())
    raw=Path(cfg.input_image_path).resolve();st=raw.stat()
    if manifest['raw_image']!=[str(raw),st.st_size,st.st_mtime_ns]:raise ValueError('Review/raw image provenance mismatch')
    records={r['candidate_id']:r for r in manifest['candidates']}
    table=pd.read_csv(review/'review.csv').fillna('')
    neg=json.loads((Path(commit['negative_pool'])/'manifest.json').read_text())
    used={r['candidate_id'] for r in neg['patches']}
    T,Z,C,H,W=img_5d.shape
    cases=[]
    def append_record(row,group):
        r=records[row.candidate_id];box=region_box((r['y0']+r['y1'])/2,(r['x0']+r['x1'])/2,H,W)
        cases.append(dict(case_id=group+'_'+r['candidate_id'],group=group,t=int(r['t']),z=int(r['z']),
            y0=box[0],x0=box[1],y1=box[2],x1=box[3],reference_kind='reviewed_component',
            reference_mask=str(review/r['mask_file']),reference_sha256=r['mask_sha256'],
            ref_y0=r['y0'],ref_x0=r['x0'],ref_y1=r['y1'],ref_x1=r['x1'],
            reference_label=0,reference_area_pixels=r['area_pixels'],failure_type=getattr(row,'failure_type',''),
            truth='artifact' if group=='trained_artifact' else 'reviewed_real_region'))
    def diverse(frame,n):
        # Take across timepoints first, then fill remaining slots. Favor visible masks.
        frame=frame.sort_values(['t','area_pixels'],ascending=[True,False])
        first=frame.drop_duplicates('t')
        if len(first)>n:first=first.iloc[np.linspace(0,len(first)-1,n).round().astype(int)]
        return pd.concat([first,frame.drop(first.index)]).head(n)
    for row in diverse(table[table.candidate_id.isin(used)&table.decision.eq('artifact_negative')],per_group).itertuples():
        append_record(row,'trained_artifact')
    real=table[table.decision.eq('real')]
    for row in diverse(real,per_group).itertuples():append_record(row,'reviewed_real')
    # Deliberately exclude regions touching ANY reviewed negative bounding box
    # at this timepoint (across Z), including withheld candidates.
    negative_boxes={}
    for cid in table.loc[table.decision.eq('artifact_negative'),'candidate_id']:
        r=records[cid];negative_boxes.setdefault(r['t'],[]).append(tuple(r[k] for k in ('y0','x0','y1','x1')))
    objects=pd.read_pickle(cfg.obj_dir/'grouped_z_objects.pkl')
    objects=objects.sort_values('area_px',ascending=False).drop_duplicates(['t','nucleus_3d_id'])
    chosen=[]
    # Favor the oversized T0 population, then large later detections.
    candidates=pd.concat([objects[objects.t.eq(0)],objects[objects.t.gt(0)]])
    for r in candidates.itertuples():
        box=region_box(r.centroid_y_px,r.centroid_x_px,H,W)
        if any(intersects(box,b) for b in negative_boxes.get(r.t,[])):continue
        if any(t==r.t and intersects(box,b) for t,b in chosen):continue
        if r.t==0 and sum(t==0 for t,_ in chosen)>=max(1,per_group//2):continue
        chosen.append((r.t,box))
        cases.append(dict(case_id=f'unreviewed_T{r.t}_N{r.nucleus_3d_id}_Z{r.z}',group='unreviewed_candidate',
            t=int(r.t),z=int(r.z),y0=box[0],x0=box[1],y1=box[2],x1=box[3],
            reference_kind='current_detection',reference_mask=str(cfg.nucleus_instance_hyperstack_path),
            reference_sha256='',reference_label=int(r.label),reference_area_pixels=int(r.area_px),
            ref_y0=0,ref_x0=0,ref_y1=0,ref_x1=0,failure_type='',truth='unreviewed'))
        if len(chosen)>=per_group:break
    result=pd.DataFrame(cases)
    print('Suggested regions:',result.groupby('group').size().to_dict())
    print('Unreviewed regions avoid reviewed negatives at the same T across Z; they may still occur in ordinary gold training patches.')
    print('Real-region masks are previously reviewed model components, not independent gold outlines.')
    return result


def reference_mask(case,shape,mask_stack=None):
    out=np.zeros(shape,bool);a,b,c,d=[int(case[k]) for k in ('y0','x0','y1','x1')]
    if case['reference_kind']=='current_detection':
        return np.asarray(mask_stack[int(case['t']),int(case['z']),a:c,b:d]==int(case['reference_label']))
    path=Path(case['reference_mask'])
    if digest(path)!=case['reference_sha256']:raise ValueError('Reviewed mask changed')
    mask=np.load(path,allow_pickle=False).astype(bool)
    ya,xb,yc,xd=[int(case[k]) for k in ('ref_y0','ref_x0','ref_y1','ref_x1')]
    y0,x0,y1,x1=max(a,ya),max(b,xb),min(c,yc),min(d,xd)
    if y0<y1 and x0<x1:out[y0-a:y1-a,x0-b:x1-b]=mask[y0-ya:y1-ya,x0-xb:x1-xb]
    return out


def predict_region(model,img,case,cfg,stats,namespace,progress=None):
    """Only infer full-grid tiles intersecting the ROI; same blend as full FOV."""
    a,b,c,d=[int(case[k]) for k in ('y0','x0','y1','x1')];t,z=int(case['t']),int(case['z'])
    H,W=img.shape[-2:];ps=cfg.patch_size
    if min(H,W)<ps:raise ValueError('Image smaller than model patch')
    starts=namespace['_generate_patch_starts']
    origins=[(y,x) for y in starts(H,ps,cfg.patch_stride) for x in starts(W,ps,cfg.patch_stride)
             if intersects((a,b,c,d),(y,x,y+ps,x+ps))]
    acc=np.zeros((c-a,d-b,8),np.float32);weight=np.zeros((c-a,d-b),np.float32)
    win=(np.hanning(ps)[:,None]*np.hanning(ps)[None,:]).astype(np.float32)+1e-6
    zs=namespace['context_plane_indices'](z,img.shape[1],cfg.n_z_context)
    channels=[cfg.nuclear_channel_index,cfg.npc_channel_index,cfg.membrane_channel_index]
    transforms=[(k,f) for k in range(4) for f in (False,True)] if cfg.tta else [(0,False)]
    for y,x in origins:
        tile=np.stack([np.asarray(img[t,zz,ch,y:y+ps,x:x+ps],np.float32) for zz in zs for ch in channels],axis=-1)
        tile=namespace['_preprocess_patch'](tile,stats)
        mean=np.zeros((ps,ps,8),np.float32)
        for k,flip in transforms:
            xb=np.rot90(np.flip(tile,1) if flip else tile,k)
            p=np.asarray(model(xb[None],training=False).numpy()[0],np.float32)
            if p.shape!=mean.shape or not np.isfinite(p).all():raise ValueError('Invalid model outputs')
            p[...,:7]=1/(1+np.exp(-np.clip(p[...,:7],-80,80)))
            p=np.rot90(p,-k);mean+=(np.flip(p,1) if flip else p)/len(transforms)
            if progress is not None:progress.update(1)
        y0,x0,y1,x1=max(a,y),max(b,x),min(c,y+ps),min(d,x+ps)
        local=(slice(y0-y,y1-y),slice(x0-x,x1-x));dest=(slice(y0-a,y1-a),slice(x0-b,x1-b))
        acc[dest]+=mean[local]*win[local][...,None];weight[dest]+=win[local]
    if not (weight>0).all():raise ValueError('ROI not fully covered')
    return acc/weight[...,None]


def load_comparison_model(name,tf):
    if name=='old_2_5_3':return tf.keras.models.load_model(OLD_MODEL,compile=False)
    model=tf.keras.models.load_model(NEW_BEST,compile=False)
    if name=='new_epoch18':
        state=json.loads((NEW_RUN/'resume/s42/state.json').read_text())
        if state['epoch']!=18:raise ValueError('Latest resume checkpoint is no longer epoch 18')
        prefix=NEW_RUN/'resume/s42/ckpt-18'
        status=tf.train.Checkpoint(model=model).restore(str(prefix))
        status.expect_partial()  # optimizer slots are deliberately not loaded for inference
        status.assert_existing_objects_matched()
    elif name!='new_best_epoch4':raise ValueError(name)
    return model


def run_comparison(namespace,cases,output_dir):
    cfg=namespace['cfg'];img=namespace['img_5d'];tf=namespace['tf'];root=Path(output_dir)
    if root.exists():raise FileExistsError('Choose a new comparison folder')
    if cases.empty or cases.case_id.duplicated().any():raise ValueError('Cases must be nonempty and uniquely named')
    if cfg.normalization_mode!='global_t':raise ValueError('Expected global_t normalization')
    for row in cases.itertuples():
        if not re.fullmatch(r'[A-Za-z0-9_-]+',row.case_id):raise ValueError('Invalid case ID')
        if row.z not in namespace['eligible_z_planes'](img.shape[1],cfg):raise ValueError('Target outside eligible Z planes')
    state=json.loads((NEW_RUN/'resume/s42/state.json').read_text())
    if state['epoch']!=18 or state['best_epoch']!=4:raise ValueError('Checkpoint epochs changed; update comparison labels')
    model_files=[OLD_MODEL,NEW_BEST,NEW_RUN/'resume/s42/ckpt-18.index']+list((NEW_RUN/'resume/s42').glob('ckpt-18.data-*'))
    if len(model_files)<4:raise FileNotFoundError('Epoch-18 checkpoint data missing')
    provenance={}
    for path in tqdm(model_files,desc='Fingerprint model files',unit='file'):provenance[str(path)]=digest(path)
    root.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',created=datetime.now().isoformat(),run=str(cfg.paths.run_dir),
        models=provenance,checkpoint_state=state,config=cfg.to_run_signature(),
        definition='Full-image tile-grid probabilities before production instance filtering; reference coverage is not gold recall.')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str));cases.to_csv(root/'cases.csv',index=False)
    stats={t:namespace['normalization_stats_for_timepoint'](img,int(t),cfg)
           for t in tqdm(sorted(cases.t.unique()),desc='Normalization',unit='timepoint')}
    import tifffile
    stack=tifffile.memmap(cfg.nucleus_instance_hyperstack_path,mode='r')
    for _,case in cases.iterrows():
        a,b,c,d=[int(case[k]) for k in ('y0','x0','y1','x1')]
        ref=reference_mask(case,(c-a,d-b),stack)
        np.savez_compressed(root/(case.case_id+'_reference.npz'),mask=ref,
            nls=img[int(case.t),int(case.z),cfg.nuclear_channel_index,a:c,b:d],
            npc=img[int(case.t),int(case.z),cfg.npc_channel_index,a:c,b:d])
    metrics=[]
    models=['old_2_5_3','new_best_epoch4','new_epoch18']
    H,W=img.shape[-2:];ps=cfg.patch_size;starts=namespace['_generate_patch_starts']
    total_tiles=sum(sum(intersects(tuple(int(case[k]) for k in ('y0','x0','y1','x1')),(y,x,y+ps,x+ps))
        for y in starts(H,ps,cfg.patch_stride) for x in starts(W,ps,cfg.patch_stride)) for _,case in cases.iterrows())
    try:
        with tqdm(total=3*total_tiles*(8 if cfg.tta else 1),desc='Matched inference',unit='tile/TTA') as progress:
            for name in models:
                with namespace['segmentation_timer']('Load '+name):model=load_comparison_model(name,tf)
                namespace['validate_vulcan_model'](model,cfg)
                for _,case in cases.iterrows():
                    progress.set_description(name+' / '+case.case_id)
                    heads=predict_region(model,img,case,cfg,stats[case.t],namespace,progress)
                    prob=heads[...,4];reference=np.load(root/(case.case_id+'_reference.npz'))['mask']
                    foreground=prob>cfg.nucleus_threshold
                    np.savez_compressed(root/(case.case_id+'_'+name+'.npz'),nucleus=prob,npc=heads[...,3],droplet=heads[...,1])
                    metrics.append(dict(case_id=case.case_id,group=case.group,truth=case.truth,model=name,
                        t=int(case.t),z=int(case.z),reference_pixels=int(reference.sum()),
                        reference_visible_fraction=float(reference.sum()/max(1,case.reference_area_pixels)),
                        reference_mean_probability=float(prob[reference].mean()) if reference.any() else np.nan,
                        reference_fraction_predicted=float(foreground[reference].mean()) if reference.any() else np.nan,
                        predicted_area_in_window_um2=float(foreground.sum()*cfg.pixel_size_um**2)))
                    pd.DataFrame(metrics).to_csv(root/'metrics.csv',index=False)
                del model;gc.collect()
        manifest['status']='complete'
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc));raise
    finally:(root/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    print('Saved comparison:',root)
    return root


def plot_case(root,case_id):
    root=Path(root);cases=pd.read_csv(root/'cases.csv');case=cases[cases.case_id.eq(case_id)].iloc[0]
    metrics=pd.read_csv(root/'metrics.csv');ref=np.load(root/(case_id+'_reference.npz'))
    fig,axes=plt.subplots(2,4,figsize=(18,9));nls=ref['nls'];lo,hi=np.percentile(nls,[1,99.8])
    axes[0,0].imshow(nls,cmap='gray',vmin=lo,vmax=max(lo+1e-6,hi));axes[0,0].set_title('Raw NLS; cyan = reference region')
    if ref['mask'].any():axes[0,0].contour(np.arange(-1,nls.shape[1]+1),np.arange(-1,nls.shape[0]+1),np.pad(ref['mask'],1),levels=[.5],colors=['cyan'])
    npc=ref['npc'];nl,nh=np.percentile(npc,[1,99.8]);axes[1,0].imshow(npc,cmap='gray',vmin=nl,vmax=max(nl+1e-6,nh));axes[1,0].set_title('Raw NPC')
    cfg=json.loads((root/'manifest.json').read_text())['config'];threshold=cfg['nucleus_threshold']
    for col,name in enumerate(['old_2_5_3','new_best_epoch4','new_epoch18'],1):
        p=np.load(root/(case_id+'_'+name+'.npz'))['nucleus'];m=metrics[(metrics.case_id==case_id)&(metrics.model==name)].iloc[0]
        axes[0,col].imshow(nls,cmap='gray',vmin=lo,vmax=max(lo+1e-6,hi))
        mask=p>threshold
        if mask.any():axes[0,col].contour(np.arange(-1,p.shape[1]+1),np.arange(-1,p.shape[0]+1),np.pad(mask,1),levels=[.5],colors=['yellow'],linewidths=.7)
        axes[0,col].set_title(name+'\nThresholded nuclear probability')
        im=axes[1,col].imshow(p,vmin=0,vmax=1,cmap='magma');axes[1,col].set_title(f'Reference coverage: {m.reference_fraction_predicted:.1%}\nMean P: {m.reference_mean_probability:.3f}')
    for ax in axes.flat:ax.axis('off')
    fig.colorbar(im,ax=list(axes[1,1:]),fraction=.025,pad=.02,label='Nuclear probability')
    fig.suptitle(f'{case_id} | {case.group} | T{case.t} Z{case.z}\nRaw predictions: no enrichment, solidity, overlap, or area filtering. Coverage is not gold recall.')
    plt.show();plt.close(fig)


def browse(root):
    import ipywidgets as w
    from IPython.display import display,clear_output
    root=Path(root);table=pd.read_csv(root/'cases.csv')
    selector=w.Dropdown(options=table.case_id.tolist(),description='Region:',layout=w.Layout(width='700px'));out=w.Output()
    def show(*_):
        with out:clear_output(wait=True);plot_case(root,selector.value)
    selector.observe(show,names='value');display(selector,out);show()
