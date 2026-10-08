"""Read saved training targets/weights; optional direct-patch model predictions.

Run in the TRAINING notebook's configured kernel, not with inference PipelineConfig.
No training, regeneration, or modification of labels. Sample classes describe patch
centres: an 'empty' sample may contain another droplet's nucleus elsewhere in its crop.
"""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


def nuclear_supervision_metrics(target, weight):
    if target.shape != weight.shape or not np.isin(target,[0,1,255]).all():
        raise ValueError('Expected matching binary/255 nuclear targets and weight arrays')
    if not np.isfinite(weight).all() or (weight<0).any():
        raise ValueError('Invalid supervision weights')
    pos=(target==1)&(weight>0);neg=(target==0)&(weight>0)
    ignored=~(pos|neg)
    return dict(positive_pixels=int(pos.sum()),negative_pixels=int(neg.sum()),
        ignored_fraction=float(ignored.mean()),negative_fraction=float(neg.mean()),
        positive_weight=float(weight[pos].sum(dtype=np.float64)),
        negative_weight=float(weight[neg].sum(dtype=np.float64)),
        negative_weight_per_patch_pixel=float(weight[neg].sum(dtype=np.float64)/target.size),
        positive_weight_per_patch_pixel=float(weight[pos].sum(dtype=np.float64)/target.size),
        unknown_with_nonzero_weight=int(((target==255)&(weight>0)).sum()),
        zero_label_zero_weight=int(((target==0)&(weight==0)).sum()),
        supervision_case='has_positive_target' if pos.any() else
            'negative_only_with_ignored_pixels' if neg.any() and ignored.any() else
            'fully_supervised_empty' if neg.all() else 'no_nuclear_supervision')


def audit_empty_plane_supervision(train_rows, val_rows, head_index, cfg, *,
        max_per_group=40, examples_per_class=2, model=None, model_nuclear_index=None,
        outputs_are_logits=True, save=True):
    """Sample per split/source/class/T; full split census and expected epoch exposure.

    model is optional and must be an explicitly chosen checkpoint. This compares
    direct patch predictions, without production TTA/watershed. Prediction metrics
    are pixel diagnostics, not per-droplet or instance false-positive rates.
    """
    if max_per_group is not None and max_per_group<1:raise ValueError('Invalid sample limit')
    if model is not None and model_nuclear_index is None:raise ValueError('Specify the active model nucleus output index')
    if cfg.n_channels!=3:raise ValueError('This audit expects saved NLS/NPC/membrane channel triplets')
    tables=[]
    for split,rows in [('train',train_rows),('validation',val_rows)]:
        table=pd.DataFrame(rows).copy();table['split']=split;tables.append(table)
    all_rows=pd.concat(tables,ignore_index=True)
    if all_rows.empty:raise ValueError('No split rows')
    if all_rows.duplicated(['split','source','stem']).any():raise ValueError('Pass original split rows, not resampled epoch rows')
    if set(map(str,train_rows and [r['img'] for r in train_rows] or [])) & set(map(str,[r['img'] for r in val_rows])):
        raise ValueError('Train/validation image paths overlap')
    group_cols=['split','source','sample_class','t']
    all_rows['expected_epoch_copies']=1.
    if cfg.stratify_by_timepoint:
        counts=all_rows[all_rows.split.eq('train')].groupby('t').size()
        chosen=all_rows.split.eq('train')
        all_rows.loc[chosen,'expected_epoch_copies']=all_rows.loc[chosen,'t'].map(counts.max()/counts)
    census=all_rows.groupby(group_cols).agg(unique_patches=('stem','size'),
        expected_epoch_patches=('expected_epoch_copies','sum')).reset_index()
    print('Full split census; expected epoch counts include timepoint balancing, not augmentation:')
    display(census.groupby(['split','source','sample_class'])[['unique_patches','expected_epoch_patches']].sum())
    out=None
    if save:
        out=Path(cfg.qc_dir)/('empty_plane_supervision_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(42);records=[];review=[];shown={}
    nuc=head_index['nucleus_interior'];edge=head_index['nucleus_edge']
    for key,group in all_rows.groupby(group_cols,sort=True):
        selected=group if max_per_group is None else group.iloc[np.sort(rng.choice(len(group),min(len(group),max_per_group),replace=False))]
        print('Auditing',key,':',len(selected),'of',len(group),'patches',flush=True)
        for r in selected.itertuples():
            lab=np.load(r.lab,allow_pickle=False,mmap_mode='r')
            weights=np.load(r.wgt,allow_pickle=False,mmap_mode='r')
            if lab.shape[:2]!=weights.shape[:2] or max(nuc,edge)>=min(lab.shape[-1],weights.shape[-1]):
                raise ValueError('Storage head/weight shape mismatch')
            y=np.asarray(lab[...,nuc]);w=np.asarray(weights[...,nuc],dtype=np.float32)
            metrics=nuclear_supervision_metrics(y,w)
            edge_metrics=nuclear_supervision_metrics(np.asarray(lab[...,edge]),np.asarray(weights[...,edge],dtype=np.float32))
            rec=dict(split=r.split,source=r.source,sample_class=r.sample_class,t=int(r.t),z=int(r.z),
                stem=r.stem,expected_epoch_copies=r.expected_epoch_copies,**metrics,
                edge_negative_fraction=edge_metrics['negative_fraction'],edge_ignored_fraction=edge_metrics['ignored_fraction'])
            plot_key=(r.split,r.sample_class)
            show=shown.get(plot_key,0)<examples_per_class
            pred=None
            if show or model is not None:
                x=np.load(r.img,allow_pickle=False).astype(np.float32)
                if x.shape[-1]!=cfg.n_z_context*cfg.n_channels:raise ValueError('Unexpected saved input context layout')
            if model is not None:
                center=cfg.n_z_context//2
                inp=x if cfg.use_2p5d_input else x[...,center*3:(center+1)*3]
                output=np.asarray(model(inp[None],training=False))
                if output.ndim!=4 or output.shape[1:3]!=y.shape or model_nuclear_index>=output.shape[-1]:
                    raise ValueError('Unexpected model output shape/index')
                pred=output[0,...,model_nuclear_index]
                if outputs_are_logits:pred=1/(1+np.exp(-np.clip(pred,-80,80)))
                if not np.isfinite(pred).all() or (pred<0).any() or (pred>1).any():raise ValueError('Invalid probabilities')
                negative=(y==0)&(w>0)
                rec['negative_pixel_predicted_fraction']=float((pred[negative]>.5).mean()) if negative.any() else np.nan
                rec['any_prediction_on_negative_pixels']=bool(((pred>.5)&negative).any()) if negative.any() else np.nan
                # A patch-wide empty truth claim requires every pixel to be supervised negative.
                rec['fully_empty_patch_any_prediction']=bool((pred>.5).any()) if negative.all() else np.nan
            records.append(rec)
            if show:
                shown[plot_key]=shown.get(plot_key,0)+1
                fig,axs=plt.subplots(2,max(cfg.n_z_context,4),figsize=(17,7),squeeze=False)
                for i in range(cfg.n_z_context):
                    axs[0,i].imshow(x[...,i*3],cmap='gray',vmin=0,vmax=1)
                    axs[0,i].set_title('NLS context offset %+d'%(i-cfg.n_z_context//2))
                axs[1,0].imshow(np.where(y==255,np.nan,y),vmin=0,vmax=1,cmap='viridis');axs[1,0].set_title('Center nucleus label; NaN=unknown')
                im=axs[1,1].imshow(w,vmin=0,cmap='magma');fig.colorbar(im,ax=axs[1,1]);axs[1,1].set_title('Center nuclear loss weights')
                axs[1,2].imshow((y==0)&(w>0),vmin=0,vmax=1);axs[1,2].set_title('Explicit negative supervision')
                if pred is not None:
                    axs[1,3].imshow(pred,vmin=0,vmax=1,cmap='viridis');axs[1,3].set_title('Model nucleus probability')
                for ax in axs.ravel():ax.axis('off')
                fig.suptitle('%s | %s | %s | %s\n%s'%(r.split,r.source,r.sample_class,metrics['supervision_case'],r.stem))
                fig.tight_layout()
                if out:fig.savefig(out/('%s_%s_%s.png'%(r.split,r.source,r.stem)),dpi=130,bbox_inches='tight')
                plt.show();plt.close(fig)
                review.append(dict(split=r.split,source=r.source,stem=r.stem,sample_class=r.sample_class,
                    center_really_empty='',neighbor_has_nucleus='',target_droplet_notes=''))
    patches=pd.DataFrame(records)
    means=patches.groupby(group_cols).agg(audited_patches=('stem','size'),
        mean_negative_fraction=('negative_fraction','mean'),mean_ignored_fraction=('ignored_fraction','mean'),
        mean_negative_weight_per_pixel=('negative_weight_per_patch_pixel','mean'),
        mean_positive_weight_per_pixel=('positive_weight_per_patch_pixel','mean'),
        patches_with_positive_targets=('positive_pixels',lambda s:int(s.gt(0).sum())),
        unknown_weight_errors=('unknown_with_nonzero_weight','sum')).reset_index().merge(census,on=group_cols,validate='one_to_one')
    means['estimated_epoch_negative_weight_per_pixel']=means.mean_negative_weight_per_pixel*means.expected_epoch_patches
    display(means)
    display(pd.crosstab([patches.split,patches.sample_class],patches.supervision_case))
    if model is not None:
        display(patches.groupby(['split','source','sample_class']).agg(
            n=('stem','size'),negative_pixel_prediction_fraction=('negative_pixel_predicted_fraction','mean'),
            fully_empty_patches=('fully_empty_patch_any_prediction','count'),
            fully_empty_patch_prediction_rate=('fully_empty_patch_any_prediction','mean')))
    print('Empty class describes the target droplet, not necessarily the whole crop. Positive pixels elsewhere are not automatically mislabels.')
    print('Neighbor context is visual evidence only: bright NLS is not automatically a nucleus. Complete the review sheet.')
    print('Weighted-pixel exposure is not the full nonlinear loss. This diagnoses the current pools/config, not historical training unless those match.')
    if out:
        census.to_csv(out/'full_split_census.csv',index=False);patches.to_csv(out/'sampled_patch_metrics.csv',index=False)
        means.to_csv(out/'supervision_by_split_class_time.csv',index=False);pd.DataFrame(review).to_csv(out/'context_review.csv',index=False)
        (out/'settings.json').write_text(json.dumps(dict(max_per_group=max_per_group,
            stratify_by_timepoint=bool(cfg.stratify_by_timepoint),head_index=head_index,
            model_evaluated=model is not None,model_name=getattr(model,'name',None),
            prediction_mode='direct patch, no augmentation/TTA/watershed',reviewed_root=str(cfg.reviewed_root),
            training_root=str(cfg.training_root)),indent=2),encoding='utf8')
        print('Saved:',out)
    return dict(census=census,patches=patches,summary=means,output_dir=out)
