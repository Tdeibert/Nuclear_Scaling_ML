"""Versioned, sparse gold-label corrections; baseline files are never overwritten."""
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reviewed_hard_negatives import _digest, _identity, _training_signature, holdout_reason, _json
from hard_negative_conflict_audit import audit_conflicts


def build_corrections(review_dir, output_dir, cfg, gold_rows, train_rows, weight=1.):
    root = Path(output_dir).resolve()
    if root.exists(): raise FileExistsError('Choose a new correction folder')
    if not np.isfinite(weight) or weight <= 0: raise ValueError('Invalid weight')
    review_dir = Path(review_dir).resolve()
    review = pd.read_csv(review_dir/'review.csv').fillna('')
    audit = audit_conflicts(review_dir, cfg, gold_rows)
    train_keys = {(r['source'],r['stem']) for r in train_rows}
    originals = {r['stem']:r for r in gold_rows if r['source']=='gold'}
    edits = {}; withheld = []
    for cid, obj in audit['objects'].items():
        row = review[review.candidate_id.eq(cid)].iloc[0]
        rec = obj['record']; a,b,c,d = [rec[k] for k in ('y0','x0','y1','x1')]
        for pair in obj['pairs']:
            r = originals[pair['gold_stem']]
            if ('gold',r['stem']) not in train_keys or holdout_reason(r['t'],r['cy'],r['cx'],cfg)!='train':
                withheld.append(dict(candidate_id=cid,stem=r['stem'],reason='holdout/seam/full-crop guard'))
                continue
            if not (row.notes.startswith('Confirmed artifact; conflicting gold annotation requires correction.')
                    or row.get('conflict_resolution', '') == 'negative'):
                raise ValueError(cid+': explicitly resolve this conflict as Artifact in the resolver first')
            if rec['z'] < cfg.z_floor: raise ValueError('Target below Z floor')
            if r['stem'] not in edits:
                lab = np.load(r['lab'],allow_pickle=False)
                wgt = np.load(r['wgt'],allow_pickle=False)
                if lab.shape[-1]!=10 or wgt.shape!=lab.shape[:2]+(9,):
                    raise ValueError('Unexpected label/weight layout')
                edits[r['stem']] = dict(row=r,mask=np.zeros(lab.shape[:2],bool),candidates=[])
            edit = edits[r['stem']]; h,w = edit['mask'].shape
            ya,xb = int(r['cy'])-h//2,int(r['cx'])-w//2
            y0,x0,y1,x1 = max(a,ya),max(b,xb),min(c,ya+h),min(d,xb+w)
            edit['mask'][y0-ya:y1-ya,x0-xb:x1-xb] |= obj['mask'][y0-a:y1-a,x0-b:x1-b]
            edit['candidates'].append(cid)
    if not edits: raise ValueError('No explicitly confirmed, training-safe gold corrections')
    root.mkdir(parents=True,exist_ok=False)
    for folder in ('labels','weights','masks'): (root/folder).mkdir()
    manifest = dict(status='building',created=datetime.now().isoformat(),
        raw_image=_identity(cfg.image_file),training_signature=_training_signature(cfg),
        review_dir=str(review_dir),review_sha256=_digest(review_dir/'review.csv'),
        candidates_sha256=_digest(review_dir/'candidates.json'),weight=float(weight),
        policy='only explicitly resolved artifact masks; heads 4/5/6 only; training-safe patches only',
        patches=[],withheld=withheld)
    _json(root/'manifest.json',manifest)
    for stem,edit in edits.items():
        r=edit['row'];mask=edit['mask'];lab=np.load(r['lab'],allow_pickle=False);wgt=np.load(r['wgt'],allow_pickle=False)
        item=dict(stem=stem,candidate_ids=sorted(set(edit['candidates'])),corrected_pixels=int(mask.sum()),
                  removed_positive_pixels=int((mask & np.any(lab[...,4:7]==1,axis=-1)).sum()))
        for key in ('lab','wgt'):
            item['original_'+key]=str(Path(r[key]).resolve());item['original_'+key+'_sha256']=_digest(r[key])
        for k in (4,5,6): lab[...,k][mask]=0;wgt[...,k][mask]=weight
        for key,folder,array in [('lab','labels',lab),('wgt','weights',wgt),('mask','masks',mask)]:
            path=root/folder/(stem+'.npy')
            with path.open('xb') as stream: np.save(stream,array,allow_pickle=False)
            item[key]=str(path.relative_to(root));item[key+'_sha256']=_digest(path)
        manifest['patches'].append(item)
    manifest['status']='complete';_json(root/'manifest.json',manifest)
    print('Copied/corrected',len(edits),'gold patches;',len(withheld),'holdout/seam pairs untouched.')
    return root


def corrected_rows(pool_dir,cfg,rows):
    root=Path(pool_dir).resolve();m=json.loads((root/'manifest.json').read_text())
    if m['status']!='complete' or m['raw_image']!=_identity(cfg.image_file) or m['training_signature']!=_training_signature(cfg):
        raise ValueError('Correction pool incomplete or configuration changed')
    review=Path(m['review_dir'])
    if _digest(review/'review.csv')!=m['review_sha256'] or _digest(review/'candidates.json')!=m['candidates_sha256']:
        raise ValueError('Review changed after correction build; build a new version')
    changes={p['stem']:p for p in m['patches']};result=[];seen=set()
    for row in rows:
        r=dict(row)
        if r['source']=='gold' and r['stem'] in changes:
            item=changes[r['stem']];seen.add(r['stem'])
            if holdout_reason(r['t'],r['cy'],r['cx'],cfg)!='train': raise ValueError('Holdout correction refused')
            for key in ('lab','wgt'):
                if str(Path(r[key]).resolve())!=item['original_'+key] or _digest(r[key])!=item['original_'+key+'_sha256']:
                    raise ValueError('Baseline gold changed')
                path=(root/item[key]).resolve()
                if root not in path.parents or _digest(path)!=item[key+'_sha256']:raise ValueError('Correction file changed')
                r[key]=path
            r['gold_correction_manifest']=str(root/'manifest.json')
        result.append(r)
    if seen!=set(changes):raise ValueError('Correction pool does not match baseline rows')
    return result


def preview_corrections(pool_dir,rows,start=0,count=6):
    root=Path(pool_dir);m=json.loads((root/'manifest.json').read_text());by={r['stem']:r for r in rows if r['source']=='gold'}
    for item in m['patches'][start:start+count]:
        before=np.load(item['original_lab'],mmap_mode='r');after=np.load(root/item['lab'],mmap_mode='r');mask=np.load(root/item['mask'])
        fig,axs=plt.subplots(1,3,figsize=(15,5))
        # Display supervised targets directly; do not assume input-channel ordering.
        axs[0].imshow(np.any(before[...,4:7]==1,axis=-1),vmin=0,vmax=1);axs[0].set_title('Original nuclear targets')
        axs[1].imshow(np.any(after[...,4:7]==1,axis=-1),vmin=0,vmax=1);axs[1].set_title('Corrected nuclear targets')
        axs[2].imshow(mask,vmin=0,vmax=1);axs[2].set_title('Exact correction mask')
        for ax in axs:ax.axis('off')
        fig.suptitle(item['stem']+' | '+str(item['removed_positive_pixels'])+' positive pixels removed');fig.tight_layout();plt.show();plt.close(fig)


def enable_corrections(namespace,cfg,pool_dir):
    if namespace.get('_HN_SPLIT_ENABLED'):raise ValueError('Enable corrections BEFORE hard-negative training hook')
    if namespace.get('_GOLD_CORRECTION_ENABLED'):raise ValueError('Corrections already enabled; restart or restore F1 definitions first')
    base=namespace['p3_rows'];corrected_rows(pool_dir,cfg,base(cfg))
    def rows(active_cfg=cfg):return corrected_rows(pool_dir,active_cfg,base(active_cfg))
    namespace['p3_rows']=rows;namespace['_GOLD_CORRECTION_ENABLED']=True
    print('Gold correction reader enabled. Original files and validation remain unchanged. No training started.')
