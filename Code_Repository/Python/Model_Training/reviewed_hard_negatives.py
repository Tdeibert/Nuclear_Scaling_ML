"""Review exact v18.1 artifact components, build an isolated pool, opt into NEW training.

Never infer negative truth from an analysis exclusion. Candidate export is not approval.
Unknown pixels/other heads remain unsupervised. Existing pools/models are never edited.
"""
import hashlib
import json
from pathlib import Path
from datetime import datetime
from functools import wraps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile

HEADS = ['background','droplet_interior','droplet_edge','npc','nucleus_interior',
         'nucleus_edge','nucleus_equatorial','abnormal_nucleus','z_offset']


def _identity(path):
    p=Path(path).resolve();s=p.stat()
    return [str(p),s.st_size,s.st_mtime_ns]


def _digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=str),encoding='utf8')


def export_artifact_candidates(cfg, grouped, review_ids, output_dir):
    """Inference cfg; exports ALL components in nominated IDs with blank approvals."""
    output_dir=Path(output_dir)
    if output_dir.exists():raise FileExistsError('Choose a new review folder')
    keys=['t','z','label']
    if grouped.duplicated(keys).any():raise ValueError('Duplicate component keys')
    table=grouped.copy()
    table['review_id']=['T%d_N%d'%(t,n) for t,n in zip(table.t,table.nucleus_3d_id)]
    missing=set(review_ids)-set(table.review_id)
    if missing:raise ValueError('IDs absent from this run: %s'%sorted(missing))
    table=table[table.review_id.isin(review_ids)].sort_values(keys)
    if table.empty:raise ValueError('No candidate components')
    raw=tifffile.memmap(cfg.input_image_path,mode='r')
    masks=tifffile.memmap(cfg.nucleus_instance_hyperstack_path,mode='r')
    if raw.shape[:2]+raw.shape[-2:]!=masks.shape:raise ValueError('Image/mask mismatch')
    output_dir.mkdir(parents=True,exist_ok=False);(output_dir/'masks').mkdir();(output_dir/'previews').mkdir()
    records=[]
    for row in table.itertuples():
        t,z,label=int(row.t),int(row.z),int(row.label)
        yy,xx=np.nonzero(masks[t,z]==label)
        if not len(yy):raise ValueError('Missing candidate mask')
        a,b,c,d=int(yy.min()),int(xx.min()),int(yy.max()+1),int(xx.max()+1)
        mask=np.asarray(masks[t,z,a:c,b:d]==label,dtype=np.uint8)
        candidate_id='T%d_Z%d_L%d'%(t,z,label)
        mp=output_dir/'masks'/(candidate_id+'.npy');np.save(mp,mask,allow_pickle=False)
        records.append(dict(candidate_id=candidate_id,review_id=row.review_id,t=t,z=z,label=label,
            nucleus_3d_id=int(row.nucleus_3d_id),y0=a,x0=b,y1=c,x1=d,area_pixels=int(mask.sum()),
            mask_file=str(mp.relative_to(output_dir)),mask_sha256=_digest(mp)))
        pad=max(12,int(np.ceil(3/cfg.pixel_size_um)))
        ya,xb,yc,xd=max(0,a-pad),max(0,b-pad),min(raw.shape[-2],c+pad),min(raw.shape[-1],d+pad)
        view=np.zeros((yc-ya,xd-xb),bool);view[a-ya:c-ya,b-xb:d-xb]=mask
        fig,axs=plt.subplots(1,3,figsize=(14,5))
        for ax,offset in zip(axs,[-1,0,1]):
            zz=int(np.clip(z+offset,0,raw.shape[1]-1))
            image=np.asarray(raw[t,zz,cfg.nuclear_channel_index,ya:yc,xb:xd])
            lo,hi=np.percentile(image,[1,99.8]);ax.imshow(image,cmap='gray',vmin=lo,vmax=max(hi,lo+1e-6))
            ax.contour(np.arange(-1,view.shape[1]+1),np.arange(-1,view.shape[0]+1),np.pad(view,1),levels=[.5],colors=['cyan'])
            ax.set_title('Z%d%s'%(zz,' CENTER' if offset==0 else ' (center mask projected)'));ax.axis('off')
        fig.suptitle('%s / %s / %d pixels\nApprove only if the ENTIRE center-plane cyan region is non-nuclear'%(row.review_id,candidate_id,mask.sum()))
        fig.tight_layout();fig.savefig(output_dir/'previews'/(candidate_id+'.png'),dpi=140);plt.close(fig)
    manifest=dict(schema=1,status='candidates_only',raw_image=_identity(cfg.input_image_path),
        source_masks=_identity(cfg.nucleus_instance_hyperstack_path),pixel_size_um=float(cfg.pixel_size_um),
        channel_indices=dict(nls=int(cfg.nuclear_channel_index),npc=int(cfg.npc_channel_index),membrane=int(cfg.membrane_channel_index)),
        source_model=str(cfg.model_name),created=datetime.now().isoformat(),candidates=records)
    _json(output_dir/'candidates.json',manifest)
    review=pd.DataFrame(records);review['decision']='';review['reviewer']='';review['notes']=''
    review.to_csv(output_dir/'review.csv',index=False)
    print('Review:',output_dir/'review.csv','|',len(review),'components. No labels approved automatically.')
    print('Set decision=artifact_negative only for exact reviewed masks; use real/uncertain/skip otherwise. Enter reviewer.')
    return output_dir


def _training_signature(cfg):
    names=['pixel_size_um','z_step_um','z_floor','n_channels','n_z_context','patch_size',
        'nucleus_channel_idx','npc_channel_idx','membrane_channel_idx','normalization_mode',
        'norm_stats_stride','norm_stats_z_stride','holdout_tile','holdout_timepoint',
        'mosaic_tile_px','mosaic_overlap_px','holdout_margin_px']
    return json.loads(json.dumps({n:getattr(cfg,n) for n in names}))


def holdout_reason(t,cy,cx,cfg):
    """Conservative: entire input crop must avoid expanded spatial holdout rectangle."""
    if int(t)==int(cfg.holdout_timepoint):return 'temporal_holdout'
    step=cfg.mosaic_tile_px-cfg.mosaic_overlap_px
    r,c=cfg.holdout_tile;m=cfg.holdout_margin_px;half=cfg.patch_size//2
    ya,xa=r*step-m,c*step-m;yb,xb=r*step+cfg.mosaic_tile_px+m,c*step+cfg.mosaic_tile_px+m
    if cy+half>ya and cy-half<yb and cx+half>xa and cx-half<xb:return 'spatial_holdout_or_margin'
    return 'train'


def make_negative_targets(mask,weight=1.):
    if not np.isfinite(weight) or weight<=0:raise ValueError('Weight must be positive and finite')
    labels=np.full(mask.shape+(10,),255,np.uint8);labels[...,9]=0
    weights=np.zeros(mask.shape+(9,),np.float32)
    for k in [4,5,6]:
        labels[...,k][mask]=0;weights[...,k][mask]=weight
    return labels,weights


def _conflicts_with_gold(mask,box,t,z,gold_rows):
    a,b,c,d=box
    for row in gold_rows.get((t,z),[]):
        lab=np.load(row['lab'],mmap_mode='r',allow_pickle=False)
        h,w=lab.shape[:2];ya=int(row['cy'])-h//2;xb=int(row['cx'])-w//2
        y0,x0,y1,x1=max(a,ya),max(b,xb),min(c,ya+h),min(d,xb+w)
        if y0>=y1 or x0>=x1:continue
        # Protect ALL stored gold nuclear-family positives, even if their weight is zero.
        positive=np.any(lab[y0-ya:y1-ya,x0-xb:x1-xb,4:7]==1,axis=-1)
        if np.any(positive & mask[y0-a:y1-a,x0-b:x1-b]):return row['stem']
    return None


def build_reviewed_negative_pool(review_dir, output_dir, cfg, *, gold_rows,
                                extract_input_stack, ensure_norm_stats, weight=1.):
    """TRAINING cfg/callbacks. Only approved exact regions, new exclusive pool.

    gold_rows must come from the validated original p3_rows(cfg), before additions.
    Gold-positive conflicts abort: resolve annotations rather than override them.
    """
    review_dir,output_dir=Path(review_dir),Path(output_dir)
    if output_dir.exists():raise FileExistsError('Never overwrite a hard-negative pool')
    manifest=json.loads((review_dir/'candidates.json').read_text())
    if _identity(cfg.image_file)!=manifest['raw_image']:raise ValueError('Review and training raw image differ')
    if float(cfg.pixel_size_um)!=manifest['pixel_size_um']:raise ValueError('Pixel calibration differs')
    channels=dict(nls=int(cfg.nucleus_channel_idx),npc=int(cfg.npc_channel_idx),membrane=int(cfg.membrane_channel_idx))
    if channels!=manifest['channel_indices']:raise ValueError('Channel mapping differs')
    if cfg.n_channels!=3 or cfg.n_z_context!=5 or cfg.patch_size%2:raise ValueError('Expected five-plane, three-channel inputs and even patch size')
    if cfg.normalization_mode!='global_t':raise ValueError('Expected reviewed global_t normalization')
    review=pd.read_csv(review_dir/'review.csv').fillna('')
    if review.candidate_id.duplicated().any():raise ValueError('Duplicate review IDs')
    if not set(review.decision)<=set(['','artifact_negative','real','uncertain','skip']):raise ValueError('Unknown review decision')
    approved=review[review.decision.eq('artifact_negative')]
    if approved.empty:raise ValueError('No exact components approved')
    if approved.reviewer.astype(str).str.strip().eq('').any():raise ValueError('Approved rows require a reviewer')
    known={r['candidate_id']:r for r in manifest['candidates']}
    if set(review.candidate_id)!=set(known):raise ValueError('Review candidates changed; re-export')
    gold={}
    for r in gold_rows:
        if r['source']=='gold':gold.setdefault((int(r['t']),int(r['z'])),[]).append(r)
    if not gold:raise ValueError('Validated gold rows required for conflict checking')
    prepared=[]
    for row in approved.itertuples():
        rec=known[row.candidate_id];mp=review_dir/rec['mask_file']
        if _digest(mp)!=rec['mask_sha256']:raise ValueError('Reviewed mask changed; re-export/review')
        mask=np.load(mp,allow_pickle=False).astype(bool)
        box=tuple(rec[k] for k in ['y0','x0','y1','x1'])
        if mask.shape!=(box[2]-box[0],box[3]-box[1]) or int(mask.sum())!=rec['area_pixels']:raise ValueError('Invalid candidate mask geometry')
        if rec['z']<cfg.z_floor:raise ValueError('Reviewed target below nuclear Z floor')
        conflict=_conflicts_with_gold(mask,box,rec['t'],rec['z'],gold)
        if conflict:raise ValueError('%s conflicts with gold-positive patch %s; resolve annotations first'%(row.candidate_id,conflict))
        prepared.append((rec,mask))
    ensure_norm_stats(cfg)  # training's existing normalization implementation; no invented scaling
    raw=tifffile.memmap(cfg.image_file,mode='r')
    output_dir.mkdir(parents=True,exist_ok=False)
    for folder in ['images','labels','weights']:(output_dir/folder).mkdir()
    pool=dict(schema=1,status='building',source_review=str(review_dir.resolve()),
        review_sha256=_digest(review_dir/'review.csv'),candidates_sha256=_digest(review_dir/'candidates.json'),
        raw_image=manifest['raw_image'],training_signature=_training_signature(cfg),
        base_model_name=str(cfg.model_name),weight=float(weight),heads=HEADS,
        policy='explicit reviewed component negatives only; other pixels/heads unknown',patches=[])
    _json(output_dir/'manifest.json',pool)
    metadata=[];quarantine=[];ps=cfg.patch_size
    for rec,mask in prepared:
        a,b,c,d=[rec[k] for k in ['y0','x0','y1','x1']]
        # Nonoverlapping tiles cover each approved component, including large artifacts.
        for ya in range(a,c,ps):
            for xb in range(b,d,ps):
                local=np.zeros((ps,ps),bool);h,w=min(ps,c-ya),min(ps,d-xb)
                local[:h,:w]=mask[ya-a:ya-a+h,xb-b:xb-b+w]
                if not local.any():continue
                cy,cx=ya+ps//2,xb+ps//2;t,z=rec['t'],rec['z']
                reason=holdout_reason(t,cy,cx,cfg)
                if reason!='train':
                    quarantine.append(dict(candidate_id=rec['candidate_id'],t=t,z=z,cy=cy,cx=cx,reason=reason));continue
                x=extract_input_stack(raw,t,z,cy,cx,cfg=cfg)
                labels,weights=make_negative_targets(local,weight)
                if x.shape!=(ps,ps,15) or not np.isfinite(x).all() or (x<0).any() or (x>1).any():raise ValueError('Invalid normalized input')
                stem='t%03d_z%03d_d%06d_y%05d_x%05d_hard_p%08d'%(t,z,rec['nucleus_3d_id'],cy,cx,len(metadata))
                item=dict(stem=stem,t=t,z=z,droplet=rec['nucleus_3d_id'],cy=cy,cx=cx,
                    candidate_id=rec['candidate_id'],sample_class='reviewed_hard_negative',source='hard_negative',supervised_pixels=int(local.sum()))
                for folder,prefix,array,key in [('images','img',x,'img'),('labels','lab',labels,'lab'),('weights','wgt',weights,'wgt')]:
                    path=output_dir/folder/(prefix+'_'+stem+'.npy')
                    with path.open('xb') as f:np.save(f,array,allow_pickle=False)
                    item[key]=str(path.relative_to(output_dir));item[key+'_sha256']=_digest(path)
                metadata.append(item)
    if not metadata:raise ValueError('No training-safe patches; review output holdout policy before proceeding')
    pd.DataFrame(metadata).to_csv(output_dir/'patch_metadata.csv',index=False)
    pd.DataFrame(quarantine).to_csv(output_dir/'heldout_candidates.csv',index=False)
    pool['patches']=metadata;pool['status']='complete'
    _json(output_dir/'manifest.json',pool)
    print('Built',len(metadata),'hard-negative patches;',len(quarantine),'tiles withheld. Existing pools unchanged.')
    return output_dir


def load_reviewed_negative_rows(pool_dir,cfg):
    root=Path(pool_dir);m=json.loads((root/'manifest.json').read_text())
    if m.get('status')!='complete':raise ValueError('Incomplete hard-negative pool')
    if m['raw_image']!=_identity(cfg.image_file) or m['training_signature']!=_training_signature(cfg):raise ValueError('Hard-negative configuration/input mismatch')
    if m['heads']!=HEADS:raise ValueError('Head layout mismatch')
    rows=[]
    for saved in m['patches']:
        r=dict(saved)
        if holdout_reason(r['t'],r['cy'],r['cx'],cfg)!='train':raise ValueError('Holdout leakage detected')
        for key in ['img','lab','wgt']:
            p=(root/r[key]).resolve()
            if root.resolve() not in p.parents:raise ValueError('Invalid pool-relative path')
            if _digest(p)!=r[key+'_sha256']:raise ValueError('Hard-negative file changed: '+str(p))
            r[key]=p
        rows.append(r)
    return rows,m


def enable_hard_negative_training(namespace,cfg,pool_dir,*,baseline_cfg):
    """Explicit opt-in before p3_train; requires a fresh, differently named run.

    Includes a pool-content fingerprint in model_name BEFORE run hashing. Validation
    stays unchanged. Does not start training. Call again after kernel restart.
    """
    if getattr(cfg,'_frozen_run_hash',None):raise ValueError('Use a fresh training cfg; run hash is already frozen')
    rows,m=load_reviewed_negative_rows(pool_dir,cfg)
    if baseline_cfg is cfg:raise ValueError('Pass separate baseline and new-run configurations')
    if str(baseline_cfg.model_name)!=m['base_model_name']:raise ValueError('Baseline model configuration differs from pool provenance')
    if _training_signature(baseline_cfg)!=_training_signature(cfg):raise ValueError('Baseline and new run must use the same input/split contract')
    if cfg.model_name==m['base_model_name']:raise ValueError('Set a NEW model_name before enabling hard-negative training')
    if namespace.get('_HN_SPLIT_ENABLED'):raise ValueError('Hard-negative split already enabled in this kernel')
    base=namespace['p3_split'];tag=_digest(Path(pool_dir)/'manifest.json')[:12]
    cfg.model_name=str(cfg.model_name)+'_hn'+tag
    model_name=cfg.model_name
    split_signature=_training_signature(baseline_cfg)
    @wraps(base)
    def split(active_cfg=cfg,verbose=True):
        if active_cfg is not cfg:raise ValueError('Hard-negative hook belongs to a different cfg')
        if active_cfg.model_name!=model_name or _training_signature(baseline_cfg)!=split_signature:
            raise ValueError('Model name or baseline split configuration changed after enabling hard negatives')
        # Resolve the ORIGINAL gold/classical pool paths, not new model-name paths.
        train,val=base(baseline_cfg,verbose=verbose)
        additions,_=load_reviewed_negative_rows(pool_dir,active_cfg)
        train=list(train)+additions
        if 'SPLITS' in namespace:namespace['SPLITS']['train']=train
        if verbose:print('Added',len(additions),'reviewed hard negatives; validation unchanged; model:',active_cfg.model_name)
        return train,val
    namespace['p3_split']=split;namespace['_HN_SPLIT_ENABLED']=True
    print('Enabled hard-negative split for NEW model:',cfg.model_name)
    print('No training started. Keep existing real-positive patches and validation unchanged.')
