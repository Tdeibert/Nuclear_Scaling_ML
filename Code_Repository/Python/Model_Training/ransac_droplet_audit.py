"""Training-derived droplet support audit. No inference or production-mask writes."""
import ast
import copy
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters, morphology, measure, segmentation
from skimage.morphology import h_maxima
from IPython.display import display


def load_training_geometry(training_notebook, cfg):
    """Execute only the named geometry definitions, never training notebook cells."""
    source = Path(training_notebook).read_text(encoding='utf8')
    functions, config = {}, None
    for cell in json.loads(source)['cells']:
        if cell['cell_type'] != 'code': continue
        text = ''.join(cell['source'])
        try: tree = ast.parse(text)
        except SyntaxError: continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef): functions[node.name] = ast.get_source_segment(text, node)
            if isinstance(node, ast.ClassDef) and node.name == 'PipelineConfig': config = node
    fields = {n.target.id: n.value for n in config.body if isinstance(n, ast.AnnAssign)}
    geom_fields = next(ast.literal_eval(n.value) for n in config.body
                      if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_GEOM_FIELDS' for t in n.targets))
    settings = {k: ast.literal_eval(fields[k]) for k in tuple(geom_fields) + ('erosion_px',)}
    settings.update(pixel_size_um=float(cfg.pixel_size_um), npc_channel_idx=int(cfg.npc_channel_index))
    gc = SimpleNamespace(**settings)
    gc.min_droplet_area_px = lambda: gc.min_droplet_area_um2 / gc.pixel_size_um**2
    names = ['extract_plane', 'clip_histogram', '_circularity', 'detect_droplets_npc_watershed',
             '_circle_from_3', '_fit_circle_kasa', 'fit_circle_ransac', 'smooth_npc_plane',
             'wall_points_on_smoothed', 'fit_droplet_circle_on_smoothed', '_sphere_lsq',
             'fit_sphere_profile', 'fit_sphere_ransac', 'compute_droplet_geometry',
             'sphere_radius_px', 'predicted_circle', 'p2_circle']
    selected = '\n\n'.join(functions[name] for name in names)
    namespace = dict(np=np, filters=filters, morphology=morphology, measure=measure,
                     segmentation=segmentation, ndimage=ndimage, h_maxima=h_maxima,
                     copy=copy, itertools=itertools, cfg=gc)
    exec(compile(selected, str(training_notebook) + ':geometry-only', 'exec'), namespace)
    return namespace, gc, settings, hashlib.sha256(selected.encode()).hexdigest()


def circle_crop(cx, cy, radius, shape):
    h,w = shape
    a,b = max(0,int(np.floor(cy-radius))), max(0,int(np.floor(cx-radius)))
    c,d = min(h,int(np.ceil(cy+radius))+1), min(w,int(np.ceil(cx+radius))+1)
    a,b,c,d = min(a,h),min(b,w),max(c,0),max(d,0)
    yy,xx = np.ogrid[a:c,b:d]
    return (a,b,c,d), (xx-cx)**2+(yy-cy)**2 <= radius**2


def fitted_plane_support(geometry, z, shape, gc, namespace):
    """Training p2_circle policy: valid observed circle, else sphere prediction."""
    circles = []
    for did,g in geometry.items():
        selected = namespace['p2_circle'](g,z,gc)
        if selected is None: continue
        kind,circ = selected
        if kind not in ('fitted','predicted'):
            raise ValueError('Unsupported training p2_circle contract')
        observed = kind == 'fitted'
        if circ[2] <= gc.erosion_px: continue
        cx,cy,r = map(float,circ)
        box,mask = circle_crop(cx,cy,r-gc.erosion_px,shape)
        if mask.any(): circles.append(dict(parent=int(did)+1, cx=cx,cy=cy,r=r,
            source='observed_inlier' if observed else 'sphere_predicted',box=box,mask=mask))
    return circles


def audit_ransac_droplets(cfg, gate, training_notebook, timepoints=(0,1,2),
                          cache_dir=None, min_parent_overlap=.85, save=True):
    """Separate comparison: geometry-only gate, plus explicitly reported containment test.

    No fitted support is not evidence of an artifact. Ambiguous/missing support is
    unassessable. Nuclear shapes and linked IDs are never changed. Cache is optional
    and validated against sources/config; partially completed timepoints are reusable.
    """
    from segmented_artifact_gate import ArtifactGateSettings, measure_segment_gate
    ns,gc,settings,code_hash = load_training_geometry(training_notebook,cfg)
    if not 0 < min_parent_overlap <= 1: raise ValueError('Invalid containment fraction')
    hs = tifffile.memmap(cfg.input_image_path, mode='r')
    labels = tifffile.memmap(cfg.nucleus_instance_hyperstack_path, mode='r')
    if hs.shape[:2]+hs.shape[-2:] != labels.shape: raise ValueError('Raw/mask shape mismatch')
    if not 0 <= gc.inventory_ref_z < hs.shape[1]: raise ValueError('Training inventory_ref_z outside stack')
    ts = sorted(set(map(int,timepoints)))
    if not ts or any(t < 0 or t >= hs.shape[0] for t in ts): raise ValueError('Invalid timepoints')
    old = gate['all_instances'].loc[lambda d:d.t.isin(ts)].copy()
    if set(old.t) != set(ts): raise ValueError('Requested timepoints absent from original gate')
    if old.duplicated(['t','z','label']).any(): raise ValueError('Duplicate gate keys')
    index = pd.read_pickle(cfg.segmentation_index_path)
    valid = set(map(tuple,index.loc[index.included.astype(str).str.lower().isin(['true','1']),['t','z']].to_numpy()))
    for t,z in old[['t','z']].drop_duplicates().itertuples(index=False,name=None):
        if (t,z) not in valid or z < cfg.focus_min_z: raise ValueError('Gate includes invalid nuclear plane')
    params = ArtifactGateSettings(**gate['settings'])
    st = Path(cfg.input_image_path).stat()
    signature = dict(settings=settings,code_sha256=code_hash,
                     image=[str(Path(cfg.input_image_path).resolve()),st.st_size,st.st_mtime_ns])
    out = None
    if save:
        out=Path(cfg.qc_dir)/('ransac_droplet_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out.mkdir(parents=True,exist_ok=False)
    cache = Path(cache_dir) if cache_dir is not None else out
    if cache is not None:
        manifest=cache/'geometry_signature.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != signature: raise ValueError('Geometry cache provenance mismatch')
        else:
            if cache.exists() and any(cache.iterdir()): raise ValueError('Cache lacks provenance; use a fresh directory')
            cache.mkdir(parents=True,exist_ok=True)
            manifest.write_text(json.dumps(signature,indent=2),encoding='utf8')
        print('Geometry cache (reusable after interruption):',cache,flush=True)
    rows,circle_rows,coverage=[],[],[]
    for t in ts:
        cp=cache/('geometry_t%03d.pkl'%t) if cache else None
        if cp is not None and cp.exists():
            saved=pd.read_pickle(cp); geometry=saved['geometry']; n_inventory=saved['n_inventory']
            print('Loaded validated geometry T%d'%t,flush=True)
        else:
            inventory=ns['detect_droplets_npc_watershed'](ns['extract_plane'](hs,t,gc.inventory_ref_z,gc.npc_channel_idx),gc,compact=True)
            n_inventory=len(inventory)
            if not inventory: raise RuntimeError('T%d: no inventory; refusing fallback to model droplets'%t)
            geometry=ns['compute_droplet_geometry'](hs,t,inventory,gc,progress=True)
            if cp is not None:
                tmp=cp.with_suffix('.tmp');pd.to_pickle(dict(geometry=geometry,n_inventory=n_inventory),tmp);tmp.replace(cp)
        coverage.append(dict(t=t,inventory=n_inventory,validated_geometry=len(geometry)))
        for z,group in old[old.t.eq(t)].groupby('z'):
            z=int(z);lab=np.asarray(labels[t,z]);raw=np.asarray(hs[t,z,cfg.nuclear_channel_index])
            circles=fitted_plane_support(geometry,z,lab.shape,gc,ns)
            assignments={}; support_cache={}
            for circle in circles:
                did=circle['parent'];a,b,c,d=circle['box'];mask=circle['mask']
                ids,counts=np.unique(lab[a:c,b:d][mask],return_counts=True)
                for label,count in zip(ids,counts):
                    if label: assignments.setdefault(int(label),[]).append((int(count),did))
                values=raw[a:c,b:d][mask]
                threshold=float(filters.threshold_otsu(values)) if values.max()>values.min() else np.nan
                support_cache[did]=(circle,threshold)
                circle_rows.append(dict(t=t,z=z,parent=did,cx=circle['cx'],cy=circle['cy'],
                    radius_px=circle['r'],support_pixels=int(mask.sum()),source=circle['source']))
            regions={r.label:r for r in measure.regionprops(lab)}
            for oldrow in group.itertuples():
                label=int(oldrow.label);region=regions.get(label)
                if region is None or not np.isclose(region.area*cfg.pixel_size_um**2,oldrow.area_um2):
                    raise ValueError('Original gate disagrees with saved nuclear masks')
                candidates=sorted(assignments.get(label,[]),reverse=True)
                eligible=[(n,did) for n,did in candidates if n/region.area>=min_parent_overlap]
                rec=dict(t=t,z=z,label=label,old_parent=oldrow.parent_droplet,
                    old_enrichment=oldrow.enrichment,old_gate_pass=bool(oldrow.gate_pass),
                    area_pixels=int(region.area),parent_candidates=len(candidates),
                    parent=0,parent_source='none',parent_overlap_fraction=0.,
                    geometry_status='missing_parent',geometry_gate_pass=None,contained_gate_pass=None,
                    enrichment=np.nan,solidity=float(region.solidity),gate_reason='unassessable geometry')
                # Any overlap with >1 circle is reported; >1 qualifying parents is ambiguous.
                tied = not eligible and len(candidates)>1 and candidates[0][0]==candidates[1][0]
                if candidates and len(eligible)<=1 and not tied:
                    n,did=eligible[0] if eligible else candidates[0]
                    circle,threshold=support_cache[did]
                    overlap=n/region.area
                    gap=max(1,int(np.ceil(params.ring_gap_um/cfg.pixel_size_um)))
                    pad=gap+max(1,int(np.ceil(params.ring_width_um/cfg.pixel_size_um)))
                    a,b,c,d=region.bbox;a,b,c,d=max(0,a-pad),max(0,b-pad),min(lab.shape[0],c+pad),min(lab.shape[1],d+pad)
                    yy,xx=np.ogrid[a:c,b:d]
                    support=(xx-circle['cx'])**2+(yy-circle['cy'])**2 <= (circle['r']-gc.erosion_px)**2
                    crop=raw[a:c,b:d];candidate=lab[a:c,b:d]==label
                    foreground=(crop>threshold)&support if np.isfinite(threshold) else np.zeros_like(support)
                    metric,_=measure_segment_gate(crop,candidate,support,foreground,cfg.pixel_size_um,params,float(region.solidity))
                    rec.update(metric)
                    rec.update(parent=did,parent_source=circle['source'],parent_overlap_fraction=overlap,
                        geometry_status='assigned' if overlap>=min_parent_overlap else 'partial_parent',
                        geometry_gate_pass=bool(metric['gate_pass']),
                        contained_gate_pass=bool(metric['gate_pass'] and overlap>=min_parent_overlap))
                elif len(eligible)>1 or tied: rec['geometry_status']='ambiguous_parent'
                rows.append(rec)
            print('Audited T%d Z%d: %d fitted/predicted supports, %d nuclei'%(t,z,len(circles),len(group)),flush=True)
            # Full-field raw NPC and fitted circles, not model droplet labels.
            if z==int(group.z.iloc[0]) and z==int(old[old.t.eq(t)].z.min()):
                fig,ax=plt.subplots(figsize=(12,8));im=np.asarray(hs[t,z,gc.npc_channel_idx]);lo,hi=np.percentile(im,[1,99.8])
                ax.imshow(im,cmap='gray',vmin=lo,vmax=max(hi,lo+1e-6))
                for circle in circles:
                    ax.add_patch(plt.Circle((circle['cx'],circle['cy']),circle['r'],fill=False,
                        color='cyan' if circle['source']=='observed_inlier' else 'orange',linewidth=.6))
                    ax.text(circle['cx'],circle['cy'],str(circle['parent']),color='yellow',fontsize=5)
                ax.set_title('T%d Z%d raw NPC: cyan observed fits; orange sphere predictions'%(t,z));ax.axis('off');fig.tight_layout()
                if out:fig.savefig(out/('geometry_T%d.png'%t),dpi=180,bbox_inches='tight')
                plt.show();plt.close(fig)
        if out:pd.DataFrame(rows).to_csv(out/'component_comparison_partial.csv',index=False)
    result=pd.DataFrame(rows)
    result['changed_assessable']=result.geometry_gate_pass.notna() & result.geometry_gate_pass.ne(result.old_gate_pass)
    display(pd.DataFrame(coverage));display(pd.crosstab(result.t,result.geometry_status))
    display(result.groupby('t').agg(old_pass=('old_gate_pass','sum'),
        geometry_assessable=('geometry_gate_pass','count'),geometry_pass=('geometry_gate_pass','sum'),
        containment_test_pass=('contained_gate_pass','sum'),changed_assessable=('changed_assessable','sum')))
    if out:
        result.to_csv(out/'component_comparison.csv',index=False)
        pd.DataFrame(circle_rows).to_csv(out/'droplet_circles.csv',index=False)
        pd.DataFrame(coverage).to_csv(out/'geometry_coverage.csv',index=False)
        (out/'comparison_settings.json').write_text(json.dumps(dict(timepoints=ts,min_parent_overlap=min_parent_overlap,
            geometry_signature=signature,gate_settings=gate['settings'],old_gate_source=str(gate.get('output_dir')),
            masks=str(cfg.nucleus_instance_hyperstack_path)),indent=2),encoding='utf8')
    print('No original masks or decisions changed. Missing/ambiguous geometry is unassessable, not confirmed artifact.')
    return dict(comparison=result,circles=pd.DataFrame(circle_rows),coverage=pd.DataFrame(coverage),
                output_dir=out,cache_dir=cache,settings=settings)
