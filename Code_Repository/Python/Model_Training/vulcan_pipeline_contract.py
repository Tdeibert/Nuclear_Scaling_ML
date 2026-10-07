"""Vulcan 2.5.3 inference contract, embedded in the v18.1 notebook by the refactor.

Functions intentionally use the notebook's imports and configuration.
"""

VULCAN_HEADS = ('background', 'droplet_interior', 'droplet_edge', 'npc',
                'nucleus_interior', 'nucleus_edge', 'nucleus_equatorial', 'z_offset')


def file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def eligible_z_planes(n_z, config):
    lo = max(0, int(config.focus_min_z))
    hi = n_z - 1 if config.focus_max_z is None else min(n_z - 1, config.focus_max_z)
    return list(range(lo, hi + 1))


def full_head_path(config):
    return config.seg_dir / 'vulcan_all_heads.tif'


def full_head_metadata_path(config):
    return config.seg_dir / 'vulcan_all_heads.json'


def validate_vulcan_model(model, config):
    if tuple(model.input_shape[1:]) != (config.patch_size, config.patch_size, 15):
        raise ValueError('Vulcan input must be (512, 512, 15): ' + str(model.input_shape))
    if int(model.output_shape[-1]) != 8:
        raise ValueError('Expected seven mask logits and one Z-offset output')
    masks = model.get_layer('masks_logits')
    zhead = model.get_layer('z_offset')
    if masks.filters != 7 or masks.activation.__name__ != 'linear':
        raise ValueError('Unexpected mask-head contract')
    if zhead.filters != 1 or zhead.activation.__name__ != 'softplus':
        raise ValueError('Unexpected Z-offset contract')
    if config.n_z_context != 5 or config.model_probability_output_indices != (0, 1, 3, 4):
        raise ValueError('Unsupported input context or semantic projection')


def predict_vulcan_plane(img_5d, t, z, model, config, channel_stats=None):
    """Streaming Hann-blended dihedral TTA, matching training H1 inference."""
    if z not in eligible_z_planes(img_5d.shape[1], config):
        raise ValueError('Target plane is outside the configured Z range')
    if channel_stats is None:
        channel_stats = normalization_stats_for_timepoint(img_5d, t, config)
    full = get_full_context_yxc(img_5d, t, z, config)
    if config.normalization_mode != 'global_t':
        raise ValueError('This validated model requires global_t normalization')
    full = _preprocess_patch(full, channel_stats)
    h, w = full.shape[:2]
    ps = config.patch_size
    if min(h, w) < ps:
        raise ValueError('Image must contain at least one full model patch')
    origins = [(y, x) for y in _generate_patch_starts(h, ps, config.patch_stride)
               for x in _generate_patch_starts(w, ps, config.patch_stride)]
    win = (np.hanning(ps)[:, None] * np.hanning(ps)[None, :]).astype(np.float32) + 1e-6
    acc = np.zeros((h, w, 8), np.float32)
    weight = np.zeros((h, w), np.float32)
    transforms = [(k, f) for k in range(4) for f in (False, True)] if config.tta else [(0, False)]
    for start in range(0, len(origins), config.batch_size):
        chunk = origins[start:start + config.batch_size]
        batch = np.stack([full[y:y+ps, x:x+ps] for y, x in chunk])
        mean = np.zeros(batch.shape[:3] + (8,), np.float32)
        for k, flip in transforms:
            xb = np.stack([np.rot90(np.flip(a, 1) if flip else a, k) for a in batch])
            p = np.asarray(model(xb, training=False).numpy(), np.float32)
            if p.shape != mean.shape or not np.isfinite(p).all():
                raise ValueError('Invalid or nonfinite model output')
            p[..., :7] = 1 / (1 + np.exp(-np.clip(p[..., :7], -80, 80)))
            for i, a in enumerate(p):
                a = np.rot90(a, -k)
                mean[i] += (np.flip(a, 1) if flip else a) / len(transforms)
        for (y, x), p in zip(chunk, mean):
            acc[y:y+ps, x:x+ps] += p * win[..., None]
            weight[y:y+ps, x:x+ps] += win
    return acc / np.maximum(weight[..., None], 1e-6)


def vulcan_instances(heads, nls, config, npc_raw=None, mem_raw=None, t_idx=None):
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    core, edge = heads[..., 4], heads[..., 5]
    seed = morphology.binary_erosion(core > config.watershed_core_thresh, morphology.disk(2))
    dist = ndi.distance_transform_edt(seed)
    comp = measure.label(seed)
    peaks = peak_local_max(dist, min_distance=config.watershed_min_marker_dist_px,
                           labels=comp, exclude_border=False)
    markers = np.zeros(seed.shape, np.int32)
    for i, (y, x) in enumerate(peaks, 1):
        markers[y, x] = i
    if not markers.any():
        markers = comp
    rng = ndi.maximum_filter(nls, config.sharpness_window_px) - ndi.minimum_filter(nls, config.sharpness_window_px)
    sharp = filters.sobel(nls) / (rng + 1e-3 * (nls.max() - nls.min() + 1e-9))
    lab = watershed(sharp + config.watershed_edge_weight * edge, markers,
                    mask=(core > config.nucleus_threshold) | (edge > config.nucleus_threshold))
    droplet = heads[..., 1] > config.droplet_threshold
    drops = measure.label(droplet, connectivity=2)
    # Optional legacy rejection is applied per instance; never merge watershed IDs.
    if config.require_stage2_gate:
        for r in measure.regionprops(lab):
            candidate = lab == r.label
            accepted, _, _, _, _ = apply_stage2_gate_to_plane(
                candidate, npc_raw, mem_raw, drops, config, t_idx=t_idx)
            if not accepted.any():
                lab[candidate] = 0
    if max(int(lab.max()), int(drops.max())) > np.iinfo(np.uint16).max:
        raise OverflowError('Instance count exceeds uint16; no truncated labels written')
    return lab.astype(np.uint16), drops.astype(np.uint16)


def run_segmentation_for_all_planes(img_5d, model, config, state=None,
                                    save_binary_masks=True, save_probability_tiff=True):
    """All eligible planes; fail-fast; timepoint resume; eight preserved heads."""
    if not save_binary_masks or not save_probability_tiff:
        raise ValueError('The reproducible baseline requires masks and probabilities')
    config.paths.assert_config_matches(config.to_run_signature())
    if file_sha256(config.model_path) != config.model_sha256:
        raise RuntimeError('Model changed since bootstrap; create a new run')
    T, Z, C, H, W = img_5d.shape
    if C != 3:
        raise ValueError('Expected microscopy axes TZCYX with three channels')
    zs = eligible_z_planes(Z, config)
    if not zs:
        raise ValueError('No eligible target planes')
    if config.use_focus_z_selection:
        raise ValueError('Baseline segments every eligible plane; disable focus window selection')
    done = set(state.completed_t if state is not None else [])
    if len(done) < T:
        validate_vulcan_model(model, config)
    stores = [
        open_hyperstack_memmap(config.segmentation_class_hyperstack_path,(T,Z,4,H,W),np.uint8,'TZCYX'),
        open_hyperstack_memmap(config.segmentation_label_hyperstack_path,(T,Z,H,W),np.uint8,'TZYX'),
        open_hyperstack_memmap(config.nucleus_instance_hyperstack_path,(T,Z,H,W),np.uint16,'TZYX'),
        open_hyperstack_memmap(config.droplet_instance_hyperstack_path,(T,Z,H,W),np.uint16,'TZYX'),
        open_hyperstack_memmap(config.segmentation_probability_hyperstack_path,(T,Z,4,H,W),np.float16,'TZCYX'),
        open_hyperstack_memmap(full_head_path(config),(T,Z,8,H,W),np.float16,'TZCYX')]
    classes, display_map, nuclei, droplets, canonical, all_heads = stores
    meta = dict(channels=list(VULCAN_HEADS), axes='TZCYX', z_offset_units='um',
                included_z=zs, model_sha256=config.model_sha256, config=config.to_run_signature())
    full_head_metadata_path(config).write_text(json.dumps(meta,indent=2),encoding='utf8')
    records, rows = [], []
    resume = config.seg_dir / '_resume'
    resume.mkdir(parents=True,exist_ok=True)
    try:
        for t in range(T):
            sidecar = resume / ('t%03d.pkl' % t)
            if t in done:
                saved = pd.read_pickle(sidecar)
                records.extend(saved['records']); rows.extend(saved['roi_rows'])
                continue
            tr, rr = [], []
            stats = normalization_stats_for_timepoint(img_5d,t,config)
            (resume / ('normalization_t%03d.json' % t)).write_text(json.dumps(stats),encoding='utf8')
            # Clear any unfinished timepoint so retry cannot retain stale planes.
            for mm in stores:
                mm[t] = 0
            for z in range(Z):
                path = config.seg_dir / ('nuclear_mask_t%03d_z%03d.npy' % (t,z))
                if z not in zs:
                    np.save(path,np.zeros((H,W),np.uint8))
                    tr.append(dict(t=t,z=z,included=False,reason='outside_z_range',mask_path=str(path)))
                    continue
                try:
                    heads = predict_vulcan_plane(img_5d,t,z,model,config,stats)
                    lab, drop = vulcan_instances(heads,np.asarray(img_5d[t,z,config.nuclear_channel_index],np.float32),
                        config,img_5d[t,z,config.npc_channel_index],img_5d[t,z,config.membrane_channel_index],t)
                except Exception as exc:
                    (config.seg_dir/'last_error.json').write_text(json.dumps(dict(t=t,z=z,error=repr(exc))),encoding='utf8')
                    raise RuntimeError('Segmentation failed at t=%d z=%d; timepoint not committed' % (t,z)) from exc
                all_heads[t,z] = np.moveaxis(heads,-1,0).astype(np.float16)
                canonical[t,z] = np.moveaxis(heads[..., [0,1,3,4]],-1,0).astype(np.float16)
                dm, nm, npc = drop>0, lab>0, heads[...,3]>config.npc_threshold
                classes[t,z] = np.stack([~dm,dm,npc,nm]).astype(np.uint8)
                display_map[t,z] = collapse_label_map(dm,npc,nm,config)
                nuclei[t,z], droplets[t,z] = lab,drop
                extra = {}
                for r in measure.regionprops(lab):
                    m = lab[r.slice] == r.label
                    extra[r.label] = dict(mean_p=float(heads[r.slice][...,4][m].mean()),
                        mean_z_offset=float(heads[r.slice][...,7][m].mean()),
                        mean_equatorial=float(heads[r.slice][...,6][m].mean()))
                rr.extend(regionprops_to_rows(lab,t,z,3,'nucleus',extra_by_label=extra))
                rr.extend(regionprops_to_rows(drop,t,z,1,'droplet'))
                np.save(path,nm.astype(np.uint8))
                tr.append(dict(t=t,z=z,included=True,reason='segmented',mask_path=str(path),
                    nucleus_pixels=int(nm.sum()),droplet_pixels=int(dm.sum())))
                print('t=%d z=%d: %d nuclei' % (t,z,len(extra)),flush=True)
            for mm in stores: mm.flush()
            tmp = sidecar.with_suffix('.tmp')
            pd.to_pickle(dict(records=tr,roi_rows=rr),tmp); os.replace(tmp,sidecar)
            records.extend(tr); rows.extend(rr); done.add(t)
            write_segmentation_checkpoint(config,status='in_progress',dims_tzyx=(T,Z,H,W),
                dtypes=_SEG_DTYPES,save_probability_tiff=True,completed_t=done)
            print('Timepoint %d committed' % t,flush=True)
        index = pd.DataFrame(records).sort_values(['t','z']).reset_index(drop=True)
        if len(index)!=T*Z or index.duplicated(['t','z']).any():
            raise RuntimeError('Incomplete or duplicated plane index')
        index.to_pickle(config.segmentation_index_path)
        pd.DataFrame(rows,columns=None if rows else
            ['t','z','label','class_name','class_id','area_px','centroid_x_px','centroid_y_px',
             'mean_p','mean_z_offset','mean_equatorial']).to_pickle(config.segmentation_roi_table_path)
        write_segmentation_checkpoint(config,status='complete',dims_tzyx=(T,Z,H,W),
            dtypes=_SEG_DTYPES,save_probability_tiff=True,completed_t=done)
        return index
    finally:
        for mm in stores: mm.flush()


def extract_objects_from_saved_masks(seg_index_df, config):
    """Use measured instance IDs, never relabel a binary watershed mask."""
    require_segmentation(config)
    roi = pd.read_pickle(config.segmentation_roi_table_path)
    result = roi[roi.class_name.eq('nucleus')].copy()
    result['nucleus_area_um2'] = result.area_px * config.pixel_size_um**2
    result.to_pickle(config.obj_dir/'plane_objects.pkl')
    return result


def group_nuclei_across_z(objects_df, config):
    """Adjacent-plane overlap links; disconnected planes start new tracks."""
    d = objects_df.copy()
    d['nucleus_3d_id'] = -1
    mm = tiff.memmap(str(config.nucleus_instance_hyperstack_path),mode='r')
    nxt = 1
    for t, group in d.groupby('t',sort=True):
        previous, prev_area, previous_z = None, {}, None
        for z, plane in group.groupby('z',sort=True):
            lab = np.asarray(mm[int(t),int(z)])
            current = np.zeros(lab.shape,np.int32)
            for ix,r in plane.iterrows():
                mask = lab==int(r.label)
                nid = None
                if previous is not None and int(z)==previous_z+1:
                    ov=previous[mask]; ov=ov[ov>0]
                    if ov.size:
                        candidate=int(np.bincount(ov).argmax())
                        if (ov==candidate).sum() > .3*min(r.area_px,prev_area[candidate]):
                            nid=candidate
                if nid is None: nid=nxt; nxt+=1
                d.at[ix,'nucleus_3d_id']=nid
                current[mask]=nid
            previous=current; previous_z=int(z)
            prev_area=dict(zip(*np.unique(current[current>0],return_counts=True)))
    d.to_pickle(config.obj_dir/'grouped_z_objects.pkl')
    return d


def select_best_z_per_nucleus(grouped_z_df, config):
    d=grouped_z_df.copy()
    if d.empty:
        d.to_pickle(config.obj_dir/'best_z_nuclei.pkl'); return d
    keys=['t','nucleus_3d_id']
    by_area=d.sort_values('area_px',ascending=False).drop_duplicates(keys)
    if config.best_z_mode=='argmin_z_offset':
        if not np.isfinite(d.mean_z_offset).all(): raise ValueError('Missing Z-offset measurements')
        best=d.sort_values(['mean_z_offset','area_px'],ascending=[True,False]).drop_duplicates(keys)
    elif config.best_z_mode=='max_area': best=by_area
    else: raise ValueError('Unknown best_z_mode')
    diagnostic=best[keys+['z','area_px']].merge(by_area[keys+['z','area_px']],on=keys,suffixes=('_selected','_max_area'))
    diagnostic.to_csv(config.qc_dir/'best_z_comparison.csv',index=False)
    best=best.sort_values(keys).reset_index(drop=True)
    best['n_planes']=best.set_index(keys).index.map(d.groupby(keys).z.nunique()).to_numpy()
    best.to_pickle(config.obj_dir/'best_z_nuclei.pkl')
    return best


def recover_nucleus_mask_from_plane(t_idx,z_idx,centroid_x_px,centroid_y_px,config):
    lab=tiff.memmap(str(config.nucleus_instance_hyperstack_path),mode='r')[t_idx,z_idx]
    regions=measure.regionprops(lab)
    if not regions: return np.zeros(lab.shape,bool)
    best=min(regions,key=lambda r:(r.centroid[1]-centroid_x_px)**2+(r.centroid[0]-centroid_y_px)**2)
    return lab==best.label


def recover_nucleus_mask_for_row(row,config):
    if config.repair_enabled:
        p=_repaired_mask_path(config,row.t,row.nucleus_3d_id)
        if p.exists(): return np.load(p).astype(bool)
    lab=tiff.memmap(str(config.nucleus_instance_hyperstack_path),mode='r')[int(row.t),int(row.z)]
    label=int(row.label)
    if not (lab==label).any(): raise ValueError('Instance missing for measured row')
    return lab==label


def segment_single_plane_with_overlap(img_5d,t_idx,z_idx,model,config,patch_size=512,stride=384):
    validate_vulcan_model(model,config)
    heads=predict_vulcan_plane(img_5d,t_idx,z_idx,model,config)
    lab,drop=vulcan_instances(heads,np.asarray(img_5d[t_idx,z_idx,config.nuclear_channel_index],np.float32),
        config,img_5d[t_idx,z_idx,config.npc_channel_index],img_5d[t_idx,z_idx,config.membrane_channel_index],t_idx)
    display=collapse_label_map(drop>0,heads[...,3]>config.npc_threshold,lab>0,config)
    return heads[...,[0,1,3,4]],display,lab>0,lab


def segment_all_planes_for_timepoint(img_5d,t_idx,keep_z,model,config):
    """Compatibility entry point using the same eight-head baseline decoder."""
    valid=set(eligible_z_planes(img_5d.shape[1],config))
    if any(z not in valid for z in keep_z):
        raise ValueError('Requested target below Z floor or above ceiling')
    return {z:segment_single_plane_with_overlap(img_5d,t_idx,z,model,config) for z in keep_z}


def run_reextraction_from_probability(config,save_binary_masks=True):
    raise RuntimeError('Legacy four-channel re-extraction is disabled for this baseline. '
                       'Set probability_hyperstack_input=None and run full inference.')
