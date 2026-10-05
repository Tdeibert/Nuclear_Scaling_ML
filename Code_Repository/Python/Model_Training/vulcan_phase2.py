"""Phase 2 notebook cell source. Embedded by build_vulcan_phase2_notebook.py.

Uses the notebook's A/B definitions; this file is not a standalone CLI.
"""
from skimage.draw import polygon as _p2_polygon
from scipy.spatial import ConvexHull
import zipfile


def p2_signature(cfg=cfg):
    return (cfg.gen_hash, cfg.gold_hash, str(cfg.image_file.resolve()),
            cfg.image_file.stat().st_size, cfg.image_file.stat().st_mtime_ns)


def p2_accept_checks(cfg=cfg):
    """Call explicitly ONLY after reviewing current E1 t2/t9 and E2 PASS."""
    if any(t not in globals().get('E1_RESULTS', {}) for t in (2, 9)):
        raise RuntimeError('Run E1 for t=2 and t=9 before accepting Phase 2 checks')
    result = globals().get('E2_RESULT')
    if result is None or result.empty or 'max_rel_diff' not in result:
        raise RuntimeError('E2 has no comparable normalization result')
    if ('status' in result and result.status.notna().any()) or not (
            np.isfinite(result.max_rel_diff).all() and (result.max_rel_diff < 1e-6).all()):
        raise RuntimeError('E2 did not pass')
    if globals().get('E1_CONFIG_HASH') != cfg._hash_of(cfg._GEOM_FIELDS + ('pixel_size_um',)):
        raise RuntimeError('E1 geometry is stale: rerun E1 under the current configuration')
    if globals().get('E2_CONFIG_HASH') != cfg._hash_of(cfg._INPUT_FIELDS):
        raise RuntimeError('E2 normalization is stale: rerun E2')
    globals()['P2_ACCEPTED_SIGNATURE'] = p2_signature(cfg)
    print('Phase 2 checks accepted for current configuration and image')


def p2_roi_local(poly, shape):
    H, W = shape
    y0 = max(0, int(np.floor(poly[:, 1].min())))
    x0 = max(0, int(np.floor(poly[:, 0].min())))
    y1 = min(H, int(np.ceil(poly[:, 1].max())) + 1)
    x1 = min(W, int(np.ceil(poly[:, 0].max())) + 1)
    if y1 <= y0 or x1 <= x0:
        raise ValueError('ROI lies entirely outside the image')
    mask = np.zeros((y1-y0, x1-x0), np.uint8)
    points = np.round(poly - [x0, y0]).astype(np.int32)
    if cv2 is not None:
        cv2.fillPoly(mask, [points], 1)
    else:
        rr, cc = _p2_polygon(points[:, 1], points[:, 0], shape=mask.shape)
        mask[rr, cc] = 1
    if not mask.any():
        raise ValueError('ROI rasterized to an empty mask')
    return (y0, x0, y1, x1), mask.astype(bool)


def p2_overlap(a, b):
    ab, am = a; bb, bm = b
    y0, x0 = max(ab[0], bb[0]), max(ab[1], bb[1])
    y1, x1 = min(ab[2], bb[2]), min(ab[3], bb[3])
    if y0 >= y1 or x0 >= x1:
        return 0
    return int((am[y0-ab[0]:y1-ab[0], x0-ab[1]:x1-ab[1]] &
                bm[y0-bb[0]:y1-bb[0], x0-bb[1]:x1-bb[1]]).sum())


def p2_intersects(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def p2_expand(local, pixels, hull=False):
    box, mask = local
    mask = np.pad(mask, pixels)
    if hull:
        mask = morphology.convex_hull_image(mask)
    if pixels:
        mask = morphology.binary_dilation(mask, morphology.disk(pixels))
    return (box[0]-pixels, box[1]-pixels, box[2]+pixels, box[3]+pixels), mask


def p2_read_rois(paths, shape, tz_shape, cfg=cfg):
    """Strict import: malformed/missing t/z fails instead of silently shifting labels."""
    rois = {kind: [] for kind in ('nuc', 'drop', 'npc')}
    for kind, path in paths.items():
        if path is None:
            continue
        for index, obj in enumerate(roifile.roiread(str(path))):
            poly = obj.coordinates()
            if poly is None or len(poly) < 3:
                raise ValueError(f'{kind} ROI {index}: not a polygon with >=3 points')
            if not obj.t_position or not obj.z_position:
                raise ValueError(f'{kind} ROI {index}: missing explicit ImageJ t/z positions')
            t, z = int(obj.t_position)-1, int(obj.z_position)-1
            if not (0 <= t < tz_shape[0] and 0 <= z < tz_shape[1]):
                raise ValueError(f'{kind} ROI {index}: t={t}, z={z} outside TIFF')
            poly = np.asarray(poly, float)
            if not np.isfinite(poly).all():
                raise ValueError(f'{kind} ROI {index}: nonfinite coordinates')
            local = p2_roi_local(poly, shape)
            rois[kind].append(dict(i=index, t=t, z=z, poly=poly, loc=local,
                                   area=int(local[1].sum()), name=obj.name,
                                   original=obj, cls='nucleus_interior' if kind == 'nuc' else kind))
    if not rois['nuc']:
        raise ValueError('Gold import requires the complete NucleiRoiSet.zip')
    return rois


def p2_classify_npc(rois, cfg=cfg):
    """Pair with same-plane nuclei, link orphan/blur ROIs through z, apply policy."""
    nuclei = {}
    for n in rois['nuc']:
        nuclei.setdefault((n['t'], n['z']), []).append(n)
    for p in rois['npc']:
        same = nuclei.get((p['t'], p['z']), [])
        paired = [n for n in same if p2_overlap(p['loc'], n['loc']) / n['area'] >= cfg.npc_roi_pair_frac]
        p['paired_ids'] = [n['i'] for n in paired]
        near = any(p2_overlap(p['loc'], n['loc']) / min(p['area'], n['area']) >= cfg.npc_roi_near_frac
                   for dz in range(-cfg.npc_roi_blur_dz, cfg.npc_roi_blur_dz+1) if dz
                   for n in nuclei.get((p['t'], p['z']+dz), []))
        p['policy'] = 'paired' if paired else ('blur' if near else 'orphan')
    remaining = [p for p in rois['npc'] if p['policy'] != 'paired']
    parent = {p['i']: p['i'] for p in remaining}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    by_tz = {}
    for p in remaining:
        by_tz.setdefault((p['t'], p['z']), []).append(p)
    for p in remaining:
        for q in by_tz.get((p['t'], p['z']+1), []):
            if p2_overlap(p['loc'], q['loc']) / min(p['area'], q['area']) >= cfg.npc_roi_near_frac:
                parent[find(p['i'])] = find(q['i'])
    groups = {}
    for p in remaining:
        groups.setdefault(find(p['i']), []).append(p)
    for oid, group in groups.items():
        policy = ('blur' if any(p['policy'] == 'blur' for p in group) else
                  'early_unknown' if group[0]['t'] <= cfg.npc_early_max_t else 'late_negative')
        for p in group:
            p['policy'], p['object_id'] = policy, oid
            if policy == 'early_unknown':
                p['unknown_local'] = p2_expand(p['loc'], int(np.ceil(
                    cfg.npc_orphan_dilate_um / cfg.pixel_size_um)), hull=True)
    return pd.DataFrame([dict(roi_index=p['i'], name=p['name'], t=p['t'], z=p['z'],
                              policy=p['policy'], object_id=p.get('object_id', -1),
                              paired_ids=';'.join(map(str, p['paired_ids']))) for p in rois['npc']],
                        columns=['roi_index', 'name', 't', 'z', 'policy', 'object_id', 'paired_ids'])


def p2_gold_circles(items, cfg=cfg):
    circles, flags = {}, []
    for it in items:
        points = it['poly']; reasons = []
        x, y = points.T
        area = .5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        try:
            hull = points[ConvexHull(points).vertices]
            diameter2 = ((hull[:, None] - hull[None, :])**2).sum(-1).max()
            roundness = 4*area / max(np.pi*diameter2, 1e-9)
        except Exception:
            roundness = 0
        if area * cfg.pixel_size_um**2 < cfg.roi_droplet_min_area_um2:
            reasons.append('small')
        if roundness < cfg.roi_droplet_min_roundness:
            reasons.append('roundness')
        fit = None if reasons else fit_circle_ransac(points, cfg.ransac_tol_px,
            cfg.ransac_n_iter, cfg.ransac_min_inlier_frac)
        if not reasons:
            if fit is None:
                reasons.append('fit_failed')
            else:
                cx, cy, r, inliers = fit
                prior = np.sqrt(area / np.pi)
                if not (cfg.fit_radius_min_frac*prior <= r <= cfg.fit_radius_max_frac*prior
                        and np.hypot(cx-x.mean(), cy-y.mean()) <= cfg.fit_centre_max_frac*prior
                        and inliers.mean() >= cfg.fit_min_final_inlier_frac and r > cfg.erosion_px):
                    reasons.append('fit_geometry')
        if not reasons:
            circles[it['i']] = tuple(map(float, fit[:3]))
        flags.append(dict(roi_index=it['i'], t=it['t'], z=it['z'],
                          accepted=not reasons, reason=';'.join(reasons) or 'keep'))
    return circles, pd.DataFrame(flags)


def p2_geometry(hs, t, cfg=cfg):
    stamp = cfg._hash_of(cfg._GEOM_FIELDS + ('pixel_size_um',))
    if globals().get('E1_CONFIG_HASH') == stamp and t in globals().get('E1_RESULTS', {}):
        return E1_RESULTS[t][1]
    inventory = detect_droplets_npc_watershed(extract_plane(hs, t, cfg.inventory_ref_z,
                                         cfg.npc_channel_idx), cfg=cfg, compact=True)
    return compute_droplet_geometry(hs, t, inventory, cfg=cfg)


def p2_circle(g, z, cfg=cfg):
    return (g['prof'][z] if z in g['prof'] and z not in g['sphere_outliers']
            else predicted_circle(g, z, cfg))


def p2_detect_plane(hs, t, z, circle, cfg=cfg):
    local = _compact_circle(*circle, hs.shape[-2:]); box, full = local
    yy, xx = np.ogrid[box[0]:box[2], box[1]:box[3]]
    cx, cy, radius = circle
    support = (xx-cx)**2 + (yy-cy)**2 <= max(radius-cfg.erosion_px, 0)**2
    image = read_crop(hs, t, z, cfg.nucleus_channel_idx, box)
    result = detect_nucleus(image, support, radius-cfg.erosion_px, cfg)
    # Include physical-gate and cleanup failures as uncertainty, not only the
    # enrichment/solidity failures reported by B5. Filled interiors remain unknown.
    rejected = ndimage.binary_fill_holes(result['raw_foreground']) & ~result['mask'] & support
    result['unannotated'] |= rejected
    npc = detect_npc_on_envelope(read_crop(hs, t, z, cfg.npc_channel_idx, box), result, support, cfg)
    return dict(box=box, circle=circle, support=support, result=result, npc=npc)


def p2_z_columns(items, cfg=cfg):
    if not items:
        return None
    zcfg = copy.copy(cfg)
    # Nucleus z-offset supervision intentionally uses the acquisition step.
    # Droplet sphere geometry remains independently calibrated at 2.18 um.
    zcfg.z_step_um = getattr(cfg, 'z_target_step_um', cfg.z_step_um)
    return assign_z(items, zcfg, verbose=False)


def p2_prepare(hs, timepoints, paths=None, cfg=cfg):
    """Read-only context. No generation directories or caches are written."""
    times = tuple(sorted(set(map(int, timepoints))))
    if not times or any(t < 0 or t >= hs.shape[0] for t in times):
        raise ValueError('Invalid or empty timepoint selection')
    if not (0<=cfg.empty_droplet_sample_fraction<=1 and 0<=cfg.empty_focal_sample_fraction<=1):
        raise ValueError('Negative sampling fractions must be between 0 and 1')
    if not (cfg.gold_roi_weight>cfg.classical_filler_weight>0 and cfg.negative_patch_weight>0):
        raise ValueError('Require gold weight > classical weight > 0 and positive negative-patch weight')
    if paths is None:
        paths = dict(nuc=ROI_ROOT/'NucleiRoiSet.zip', drop=ROI_ROOT/'DropletRoiSet.zip',
                     npc=ROI_ROOT/'NPCRoiSet.zip')
    rois = p2_read_rois(paths, hs.shape[-2:], hs.shape[:2], cfg)
    flags = p2_classify_npc(rois, cfg)
    circles, drop_flags = p2_gold_circles(rois['drop'], cfg)
    zcols = p2_z_columns(rois['nuc'], cfg)
    return dict(hs=hs, times=times, paths=paths, rois=rois, npc_flags=flags,
                gold_circles=circles, drop_flags=drop_flags, zcols=zcols,
                signature=p2_signature(cfg))


def p2_timepoint(context, t, cfg=cfg):
    """All inventory droplets; compact plane masks, no retained image crops."""
    hs = context['hs']; geometry = p2_geometry(hs, t, cfg)
    state = dict(t=t, geometry=geometry, planes={}, qc=[], occupied=set(), ambiguous=set())
    for did, g in geometry.items():
        # Union in XY across ALL acquisition planes; this assignment is independent
        # of z_floor and covers nuclei whose section extends beyond sphere caps.
        circles = [p2_circle(g, z, cfg) for z in range(hs.shape[1])]
        circles = [c for c in circles if c is not None and c[2] > 0]
        if not circles:
            continue
        footprints = [_compact_circle(*c, hs.shape[-2:]) for c in circles]
        matching = [n for n in context['rois']['nuc'] if n['t'] == t and
                    any(p2_overlap(n['loc'], footprint) > 0 for footprint in footprints)]
        if matching:
            state['occupied'].add(did)
        if any(p['t'] == t and p['policy'] == 'early_unknown' and
               any(p2_overlap(p['unknown_local'], footprint) > 0 for footprint in footprints)
               for p in context['rois']['npc']):
            state['ambiguous'].add(did)
        state.setdefault('gold_z', {})[did] = {n['z'] for n in matching}
        for z in range(hs.shape[1]):
            circle = p2_circle(g, z, cfg)
            if circle is None or circle[2] <= cfg.erosion_px:
                continue
            if z_in_focus_range(z, hs.shape[1], cfg):
                plane = p2_detect_plane(hs, t, z, circle, cfg)
                for record in plane['result']['records']:
                    state['qc'].append(dict(t=t, z=z, droplet=did, **record))
            else:
                plane = dict(box=_compact_circle(*circle, hs.shape[-2:])[0], circle=circle,
                             result=None, npc=None)
            state['planes'][(did, z)] = plane
    # Gold circles replace geometrically matched NPC circles, and add untracked
    # gold droplets. Matching uses centre distance plus radius compatibility.
    # Link accepted classical components by the same one-to-one, validated z
    # column method used for gold. Contours are only an identity/axial proxy;
    # the exact accepted pixel masks are retained for labels.
    classical_items=[]
    for (did,z),plane in state['planes'].items():
        if plane['result'] is None: continue
        labels=measure.label(plane['result']['mask'])
        for region in measure.regionprops(labels):
            mask=labels==region.label
            contours=measure.find_contours(np.pad(mask,1),.5)
            if not contours: continue
            contour=max(contours,key=len)-1
            poly=np.c_[contour[:,1]+plane['box'][1],contour[:,0]+plane['box'][0]]
            classical_items.append(dict(t=t,z=z,did=did,poly=poly,
                loc=(plane['box'],mask),cls='nucleus_interior'))
    # Link within droplets to prevent an adjacent droplet's component becoming
    # the same nucleus after a drift or segmentation failure.
    state['classical_zcols']={}
    for did in geometry:
        items=[it for it in classical_items if it['did']==did]
        state['classical_zcols'][did]=p2_z_columns(items,cfg)
    state['classical_items']=classical_items
    state['gold_overrides'] = {}
    for it in context['rois']['drop']:
        if it['t'] != t or it['i'] not in context['gold_circles']:
            continue
        circle = context['gold_circles'][it['i']]
        matches = [(np.hypot(p['circle'][0]-circle[0], p['circle'][1]-circle[1]), did)
                   for (did, z), p in state['planes'].items() if z == it['z']
                   and abs(p['circle'][2]-circle[2]) <= .5*circle[2]]
        best = min(matches) if matches else None
        did = best[1] if best and best[0] < .5*circle[2] else 100000+it['i']
        state['gold_overrides'][(did, it['z'])] = circle
    return state


def p2_edge(mask, cfg=cfg):
    a = max(1, cfg.edge_band_px//2); b = max(1, cfg.edge_band_px-a)
    return (morphology.binary_dilation(mask, morphology.disk(a)) &
            ~morphology.binary_erosion(mask, morphology.disk(b)))


def p2_compose(context, state, z, box, mode='gold', cfg=cfg):
    """Compose every object intersecting a padded window; never label crop padding."""
    if mode not in ('gold', 'classical'):
        raise ValueError(mode)
    y0, x0, y1, x1 = box; shape = (y1-y0, x1-x0)
    lab = np.full(shape+(N_LABEL_CHANNELS,), UNANNOTATED, np.uint8)
    lab[..., DROPLET_SOURCE_IDX] = DSRC_NONE
    weights = np.zeros(shape+(N_HEADS,), np.dtype(getattr(cfg, 'label_weight_dtype', 'float16')))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    H, W = context['hs'].shape[-2:]
    valid = np.broadcast_to((yy >= 0) & (yy < H) & (xx >= 0) & (xx < W), shape)
    t = state['t']
    # Bound work to nearby ROIs before allocating patch-sized masks.
    margin = max(cfg.edge_band_px+2, int(np.ceil(
        (cfg.npc_orphan_dilate_um+cfg.npc_shell_width_um)/cfg.pixel_size_um))+2)
    nearby_box = (y0-margin,x0-margin,y1+margin,x1+margin)
    rois = {kind:[it for it in items if it['t']==t and
                  (abs(it['z']-z)<=cfg.empty_plane_guard_dz if kind=='nuc' else it['z']==z)
                  and p2_intersects(it.get('unknown_local',it['loc'])[0],nearby_box)]
            for kind,items in context['rois'].items()}
    def put(head, where, value, weight):
        idx = HEAD_INDEX[head]; where = where & valid
        lab[..., idx][where] = value
        weights[..., idx][where] = weight if value != UNANNOTATED else 0
    def paint(local):
        return _compact_crop(local, box)
    def unknown(heads, mask):
        for h in heads:
            put(h, mask, UNANNOTATED, 0)
    nuc_heads = ('nucleus_interior', 'nucleus_edge', 'nucleus_equatorial', 'z_offset')
    drops = [(did, p['circle'], DSRC_NPC) for (did, zz), p in state['planes'].items() if zz == z]
    if mode == 'gold':
        drops = [(did, c, src) for did, c, src in drops if (did, z) not in state['gold_overrides']]
        drops += [(did, c, DSRC_ROI) for (did, zz), c in state['gold_overrides'].items() if zz == z]
    union_drop = np.zeros(shape, bool); union_inner = union_drop.copy(); union_edge = union_drop.copy()
    exterior = union_drop.copy(); exterior_source = np.zeros(shape,np.uint8)
    droplet_source = np.zeros(shape, np.uint8)
    # Paint classical first, ROI last; erode and derive edges per instance.
    for did, (cx, cy, r), source in sorted(drops, key=lambda item: item[2] == DSRC_ROI):
        if cx+r < x0 or cx-r >= x1 or cy+r < y0 or cy-r >= y1:
            continue
        distance = np.sqrt((xx-cx)**2 + (yy-cy)**2)
        full = distance <= r
        interior = distance <= max(0, r-cfg.erosion_px)
        edge = np.abs(distance-r) <= max(1, cfg.edge_band_px/2)
        outside = (distance > r+cfg.edge_band_px) & (distance <= r+cfg.edge_band_px+cfg.erosion_px)
        exterior |= outside; exterior_source[outside] = source
        union_drop |= full; union_inner |= interior; union_edge |= edge
        droplet_source[interior | edge] = source
    exterior &= ~union_drop & ~union_edge
    droplet_source[exterior] = exterior_source[exterior]
    known_drop = union_inner | union_edge | exterior
    # The uncertain eroded band and unmodelled field remain unknown for droplet
    # heads; missing geometry is never used to assert exterior background.
    for h, value in [('background', 0), ('droplet_interior', 0), ('droplet_edge', 0)]:
        put(h, known_drop, value, cfg.classical_filler_weight)
    put('droplet_interior', union_inner, 1, cfg.classical_filler_weight)
    put('droplet_edge', union_edge, 1, cfg.classical_filler_weight)
    put('background', exterior, 1, cfg.classical_filler_weight)
    roi_drop = (droplet_source == DSRC_ROI) & valid
    for h in DROPLET_HEADS:
        weights[..., HEAD_INDEX[h]][roi_drop] = cfg.gold_roi_weight
    lab[..., DROPLET_SOURCE_IDX][valid] = droplet_source[valid]

    if mode == 'classical' and t not in cfg.gold_complete_timepoints:
        for (did, zz), p in state['planes'].items():
            if zz != z or p['result'] is None or not p2_intersects(p['box'], box):
                continue
            result = p['result']; nucleus = paint((p['box'], result['mask']))
            support = paint((p['box'], p['support']))
            if result['mask'].any():
                put('nucleus_interior', support, 0, cfg.classical_filler_weight)
                put('nucleus_edge', support, 0, cfg.classical_filler_weight)
                put('nucleus_interior', nucleus, 1, cfg.classical_filler_weight)
                put('nucleus_edge', p2_edge(nucleus, cfg) & support, 1, cfg.classical_filler_weight)
            uncertain = paint((p['box'], result['unannotated']))
            unknown(nuc_heads, p2_expand_mask(uncertain, cfg.edge_band_px))
            npc = p['npc']; put('npc', support, 0, cfg.classical_filler_weight)
            put('npc', paint((p['box'], npc['mask'])), 1, cfg.classical_filler_weight)
            unknown(('npc',), paint((p['box'], npc['unannotated'])))
        for item in state.get('classical_items',[]):
            if item['z']!=z: continue
            columns=state['classical_zcols'].get(item['did'])
            if columns is None: continue
            offset,equatorial,supervised=columns.z_labels_for(
                t,z,float(item['poly'][:,1].mean()),float(item['poly'][:,0].mean()))
            if supervised and 0<=offset*Z_OFFSET_SCALE<=254:
                m=paint(item['loc']) & (lab[...,HEAD_INDEX['nucleus_interior']]==1)
                put('nucleus_equatorial',m,int(equatorial),cfg.classical_filler_weight)
                put('z_offset',m,int(round(offset*Z_OFFSET_SCALE)),cfg.classical_filler_weight)
        return lab, weights

    # Complete gold nucleus annotation gives frame-wide NLS negatives. Every
    # positive ROI on an eligible center plane is preserved outside droplet geometry.
    # All-z ROIs remain loaded for occupancy, column identity and boundary guards.
    gold = np.zeros(shape, bool)
    for n in rois['nuc']:
        if n['t'] == t and n['z'] == z:
            gold |= paint(n['loc'])
    for h in ('nucleus_interior', 'nucleus_edge', 'nucleus_equatorial'):
        put(h, valid, 0, cfg.gold_roi_weight)
    guards = np.zeros(shape, bool)
    for n in rois['nuc']:
        if n['t'] == t and 0 < abs(n['z']-z) <= cfg.empty_plane_guard_dz:
            guards |= paint(p2_expand(n['loc'], cfg.edge_band_px))
    early_unknown = np.zeros(shape, bool); npc_unknown = np.zeros(shape, bool)
    for p in rois['npc']:
        if p['t'] != t or p['z'] != z:
            continue
        if p['policy'] == 'early_unknown':
            early_unknown |= paint(p['unknown_local'])
        elif p['policy'] == 'blur':
            npc_unknown |= paint(p['loc'])
    unknown(nuc_heads, guards | early_unknown)
    # Exact manual interior wins over any overlapping uncertainty.
    put('nucleus_interior', gold, 1, cfg.gold_roi_weight)
    gold_edge = p2_edge(gold, cfg)
    put('nucleus_edge', gold_edge, 1, cfg.gold_roi_weight)
    put('nucleus_edge', gold & ~gold_edge, 0, cfg.gold_roi_weight)
    unknown(('nucleus_equatorial', 'z_offset'), gold)
    if cfg.roi_z_supervision and context['zcols'] is not None:
        for n in rois['nuc']:
            if n['t'] != t or n['z'] != z:
                continue
            offset, equatorial, supervised = context['zcols'].z_labels_for(
                t, z, float(n['poly'][:, 1].mean()), float(n['poly'][:, 0].mean()))
            if supervised and 0 <= offset*Z_OFFSET_SCALE <= 254:
                m = paint(n['loc'])
                put('nucleus_equatorial', m, int(equatorial), cfg.gold_roi_weight)
                put('z_offset', m, int(round(offset*Z_OFFSET_SCALE)), cfg.gold_roi_weight)

    # NPC negatives are weaker where no manual boundary exists. Calculate NPC for
    # each full nucleus ROI in a padded raw-channel crop, not at patch boundaries.
    put('npc', valid, 0, cfg.classical_filler_weight)
    for n in rois['nuc']:
        if n['t'] != t or n['z'] != z:
            continue
        pad = max(2, int(np.ceil(cfg.npc_shell_width_um/cfg.pixel_size_um))+2)
        expanded = p2_expand(n['loc'], pad)
        b = expanded[0]; b = (max(0,b[0]), max(0,b[1]), min(H,b[2]), min(W,b[3]))
        if not p2_intersects(b, box):
            continue
        nm = _compact_crop(n['loc'], b)
        raw = read_crop(context['hs'], t, z, cfg.npc_channel_idx, b)
        shell = nucleus_envelope_shell(nm, cfg)
        # Use the containing droplet's raw NPC distribution whenever available.
        values = None
        for (did, zz), plane in state['planes'].items():
            if zz == z and p2_overlap(n['loc'], _compact_circle(*plane['circle'], (H,W))) > n['area']/2:
                pb = plane['box']; pc = plane['circle']
                raw_drop = read_crop(context['hs'], t, z, cfg.npc_channel_idx, pb)
                py, px = np.ogrid[pb[0]:pb[2], pb[1]:pb[3]]
                m = (px-pc[0])**2+(py-pc[1])**2 <= max(pc[2]-cfg.erosion_px, 0)**2
                values = raw_drop[m]; break
        if values is None or not len(values) or not np.isfinite(values).all():
            unknown(('npc',), paint((b, shell)))
        else:
            calc = (raw > values.mean()+cfg.npc_std_mult*values.std()) & shell & ~nm
            put('npc', paint((b, calc)), 1, cfg.classical_filler_weight)
    # Manual outer outline minus ALL same-plane gold NLS interiors; no forced
    # fixed-width shell is imposed on a manual NPC annotation.
    manual_region = np.zeros(shape, bool); manual_positive = np.zeros(shape, bool)
    for p in rois['npc']:
        if p['t'] == t and p['z'] == z and p['policy'] == 'paired':
            outer = paint(p['loc']); manual_region |= outer; manual_positive |= outer & ~gold
    put('npc', manual_region, 0, cfg.gold_roi_weight)
    put('npc', manual_positive, 1, cfg.gold_roi_weight)
    for p in rois['npc']:
        if p['t'] == t and p['z'] == z and p['policy'] == 'late_negative':
            put('npc', paint(p['loc']) & ~manual_region, 0, cfg.gold_roi_weight)
    unknown(('npc',), (npc_unknown | early_unknown | guards) & ~manual_region)
    return lab, weights


def p2_expand_mask(mask, pixels):
    return morphology.binary_dilation(mask, morphology.disk(max(1, pixels))) if mask.any() else mask


def p2_samples(context, state, mode='gold', cfg=cfg):
    """Sample eligible centers (z >= z_floor), retaining all-z occupancy evidence."""
    H, W = context['hs'].shape[-2:]; t = state['t']; seen = set()
    candidates = []
    for (did, z), plane in sorted(state['planes'].items()):
        cx, cy, radius = plane['circle']
        gold_z = state.get('gold_z', {}).get(did, set())
        if mode == 'classical':
            if plane['result'] is None or not plane['result']['mask'].any():
                continue  # no detector-empty or extrapolated cap negatives
            kind = 'classical_positive'
        elif did in state['ambiguous'] and not gold_z:
            kind = 'ambiguous_plane'
        elif not gold_z:
            kind = 'empty_droplet'
        elif z in gold_z:
            kind = 'occupied_plane'
        elif min(gold_z) <= z <= max(gold_z) or min(abs(z-q) for q in gold_z) <= cfg.empty_plane_guard_dz:
            kind = 'ambiguous_plane'
        else:
            kind = 'empty_focal_plane'
        candidates.append((z, did, cy, cx, kind))
    if mode == 'gold':
        for n in context['rois']['nuc']:
            if n['t'] == t:
                candidates.append((n['z'], 200000+n['i'], n['poly'][:,1].mean(),
                                   n['poly'][:,0].mean(), 'gold_nucleus'))
        for (did, z), (cx, cy, r) in state['gold_overrides'].items():
            candidates.append((z, did, cy, cx, 'gold_droplet'))
    rng = np.random.default_rng(cfg.seed+t)
    for z, did, cy, cx, kind in candidates:
        # One common gate covers detector, gold-nucleus and manual-droplet centers,
        # including the preview, before sampling randomness or deduplication.
        if not z_in_focus_range(z, context['hs'].shape[1], cfg):
            continue
        # Negative sampling controls are independent of label meaning.
        probability = (cfg.empty_droplet_sample_fraction if kind == 'empty_droplet' else
                       cfg.empty_focal_sample_fraction if kind == 'empty_focal_plane' else 1.)
        if rng.random() > probability:
            continue
        cy, cx = int(round(cy)), int(round(cx))
        key = (z, cy, cx)
        if key in seen:
            continue
        seen.add(key)
        yield dict(t=t, z=z, droplet=did, cy=cy, cx=cx, sample_class=kind,
                   sample_weight=cfg.negative_patch_weight if kind.startswith('empty_') else 1.)


def p2_patch(context, state, sample, mode='gold', cfg=cfg):
    if not z_in_focus_range(sample['z'], context['hs'].shape[1], cfg):
        raise ValueError(f"Center z={sample['z']} is outside the training range "
                         f"[{cfg.z_floor}, {context['hs'].shape[1]-1}]")
    # Input neighbors retain the acquisition bounds, not the center-plane floor.
    ps = cfg.patch_size; cy, cx = sample['cy'], sample['cx']
    pad = max(cfg.edge_band_px+2, int(np.ceil(cfg.npc_orphan_dilate_um/cfg.pixel_size_um))+2)
    box = (cy-ps//2-pad, cx-ps//2-pad, cy-ps//2+ps+pad, cx-ps//2+ps+pad)
    lab, weight = p2_compose(context, state, sample['z'], box, mode, cfg)
    lab = lab[pad:pad+ps, pad:pad+ps].copy()
    weight = weight[pad:pad+ps, pad:pad+ps].copy()*sample['sample_weight']
    x = extract_input_stack(context['hs'], sample['t'], sample['z'], cy, cx, cfg=cfg)
    p2_validate_arrays(x, lab, weight, cfg)
    return x, lab, weight


def p2_validate_arrays(x, lab, weight, cfg=cfg):
    assert lab.shape == (cfg.patch_size, cfg.patch_size, N_LABEL_CHANNELS)
    assert weight.shape == lab.shape[:2]+(N_HEADS,)
    assert x.shape == lab.shape[:2]+(cfg.n_channels*cfg.n_z_context,)
    assert np.isfinite(x).all() and np.isfinite(weight).all() and (weight >= 0).all()
    assert (x >= 0).all() and (x <= 1).all()
    for name in HEAD_NAMES:
        if name != 'z_offset':
            assert np.isin(lab[..., HEAD_INDEX[name]], [0,1,255]).all(), name
    assert np.isin(lab[..., DROPLET_SOURCE_IDX], [0,1,2]).all()
    assert (weight[lab[...,:N_HEADS] == UNANNOTATED] == 0).all()


def p2_preview(timepoints=(2,9), per_class=2, cfg=cfg):
    """Read-only gold label/unknown/weight review, with class counts."""
    hs = load_memmap_tiff(cfg.image_file)
    if cfg.normalization_mode == 'global_t' and cfg.norm_stats is None:
        cfg.norm_stats = compute_global_norm_stats(cfg.image_file, cfg)
    if any(t not in cfg.gold_complete_timepoints for t in timepoints):
        raise ValueError('Gold preview requires declared complete timepoints')
    context = p2_prepare(hs, timepoints, cfg=cfg); counts=[]
    for t in context['times']:
        state = p2_timepoint(context, t, cfg); selected={}
        for sample in p2_samples(context, state, cfg=cfg):
            kind=sample['sample_class']; counts.append(sample)
            if selected.get(kind,0) >= per_class:
                continue
            selected[kind]=selected.get(kind,0)+1
            x, lab, weight = p2_patch(context,state,sample,cfg=cfg)
            fig, axes=plt.subplots(1,4,figsize=(15,4))
            axes[0].imshow(x[..., (cfg.n_z_context//2)*3],cmap='gray')
            axes[0].set_title(f"T{t} z{sample['z']} {kind}")
            for ax, head in zip(axes[1:],('nucleus_interior','npc','droplet_interior')):
                v=lab[...,HEAD_INDEX[head]]
                ax.imshow(np.where(v==255,2,v),vmin=0,vmax=2,
                          cmap=ListedColormap(['black','cyan','red']))
                ax.set_title(head+' (red=unknown)')
            for ax in axes: ax.axis('off')
            plt.tight_layout(); plt.show(); plt.close(fig)
    result=pd.DataFrame(counts)
    if not result.empty: print(result.groupby(['t','sample_class']).size().to_string())
    return result


def p2_file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def p2_build(mode='gold', timepoints=None, cfg=cfg):
    """Explicit write stage. Refuses existing pools; unfinished pools are never trained."""
    if globals().get('P2_ACCEPTED_SIGNATURE') != p2_signature(cfg):
        raise RuntimeError('Review E1/E2, then explicitly call p2_accept_checks()')
    if mode not in ('gold','classical'): raise ValueError(mode)
    hs=load_memmap_tiff(cfg.image_file)
    times=tuple(range(hs.shape[0])) if timepoints is None else tuple(timepoints)
    if mode=='gold' and any(t not in cfg.gold_complete_timepoints for t in times):
        raise ValueError('Gold negatives require complete annotation for every selected timepoint')
    context=p2_prepare(hs,times,cfg=cfg)
    root=cfg.reviewed_root if mode=='gold' else cfg.training_root
    # Never append to, overwrite, or silently reuse any earlier/partial pool.
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'{root} is not empty; choose a fresh pool path/version')
    if cfg.normalization_mode=='global_t':
        cfg.norm_stats=compute_global_norm_stats(cfg.image_file,cfg)
    root.mkdir(parents=True,exist_ok=True)
    for sub in ('images','labels','weights','qc'): (root/sub).mkdir(exist_ok=False)
    manifest=dict(status='building',mode=mode,gen_hash=cfg.gen_hash,gold_hash=cfg.gold_hash,
                  storage_channels=STORAGE_CHANNELS,weight_heads=HEAD_NAMES,
                  geometry_z_step_um=cfg.geometry_z_step_um,
                  z_target_step_um=getattr(cfg,'z_target_step_um',cfg.z_step_um),
                  weight_dtype=getattr(cfg,'label_weight_dtype','float16'),timepoints=list(times),
                  roi_sha256={k:p2_file_digest(p) for k,p in context['paths'].items() if p is not None},
                  roi_paths={k:str(Path(p).resolve()) for k,p in context['paths'].items() if p is not None},
                  config={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()
                          if not k.startswith('_') and k!='norm_stats'},
                  phase3_requires_weight_sidecars=True)
    manifest_path=root/'manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2,default=str),encoding='utf8')
    if cfg.norm_stats: save_norm_stats(cfg.norm_stats,root/'qc'/'norm_stats.json')
    context['npc_flags'].to_csv(root/'npc_roi_flags.csv',index=False)
    context['drop_flags'].to_csv(root/'droplet_roi_flags.csv',index=False)
    with zipfile.ZipFile(root/'NPCRoiSet_orphans.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in context['rois']['npc']:
            if p['policy']!='paired':
                archive.writestr(f"npc_{p['i']:06d}.roi",p['original'].tobytes())
    metadata=[]; qc=[]; n=0; classical_column_flags=[]
    for t in context['times']:
        state=p2_timepoint(context,t,cfg); qc.extend(state['qc'])
        if t not in cfg.gold_complete_timepoints:
            for did,columns in state['classical_zcols'].items():
                if columns is not None:
                    classical_column_flags.append(columns.columns.assign(droplet=did,source='classical'))
        for sample in p2_samples(context,state,mode,cfg):
            x,lab,weights=p2_patch(context,state,sample,mode,cfg)
            stem=(f"t{t:03d}_z{sample['z']:03d}_d{sample['droplet']:06d}_"
                  f"y{sample['cy']:05d}_x{sample['cx']:05d}_{'roi' if mode=='gold' else 'pos'}_p{n:08d}")
            # Individual files use exclusive creation; manifest remains building
            # until every triplet, metadata, and QC artifact has been checked.
            for folder,prefix,array in [('images','img',x),('labels','lab',lab),('weights','wgt',weights)]:
                with (root/folder/f'{prefix}_{stem}.npy').open('xb') as stream:
                    np.save(stream,array,allow_pickle=False)
            metadata.append(dict(stem=stem,**sample)); n+=1
        print(f'{mode}: t={t} complete; {n} patches',flush=True)
    pd.DataFrame(metadata).to_csv(root/'patch_metadata.csv',index=False)
    pd.DataFrame(qc).to_csv(root/'candidate_qc.csv',index=False)
    if n==0: raise RuntimeError('No patches emitted; pool left incomplete')
    gold_columns=(context['zcols'].columns[
        context['zcols'].columns.t.isin(context['times']) &
        context['zcols'].columns.t.isin(cfg.gold_complete_timepoints)].assign(source='gold')
        if cfg.roi_z_supervision and context['zcols'] is not None else pd.DataFrame())
    column_frames=([gold_columns] if not gold_columns.empty else [])+classical_column_flags
    all_columns=pd.concat(column_frames,ignore_index=True) if column_frames else pd.DataFrame()
    zsummary=dict(n_columns=len(all_columns),
                  n_validated=int(all_columns.validated.sum()) if len(all_columns) else 0,
                  z_target_step_um=getattr(cfg,'z_target_step_um',cfg.z_step_um),
                  geometry_z_step_um=cfg.geometry_z_step_um)
    (root/'z_cluster_summary.json').write_text(json.dumps(zsummary,indent=2),encoding='utf8')
    all_columns.to_csv(root/'z_cluster_flags.csv',index=False)
    p2_pool_files(root,require_complete=False)
    manifest.update(status='complete',n_patches=n)
    manifest_path.write_text(json.dumps(manifest,indent=2,default=str),encoding='utf8')
    cfg._z_label_cache=None
    return manifest


def p2_pool_files(root, require_complete=True):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    if require_complete and manifest.get('status')!='complete':
        raise RuntimeError(f'{root}: incomplete Phase 2 pool')
    sets=[]
    for folder,prefix in [('images','img_'),('labels','lab_'),('weights','wgt_')]:
        sets.append({p.name[len(prefix):-4]:p for p in (root/folder).glob(prefix+'*.npy')})
    if not sets[0] or set(sets[0])!=set(sets[1]) or set(sets[0])!=set(sets[2]):
        raise RuntimeError('Image/label/weight triplets missing or orphaned')
    metadata=pd.read_csv(root/'patch_metadata.csv')
    if metadata.stem.duplicated().any() or set(metadata.stem)!=set(sets[0]):
        raise RuntimeError('Patch metadata does not match file triplets')
    return [(sets[0][k],sets[1][k],sets[2][k]) for k in sorted(sets[0])]


def list_phase2_patch_files(cfg=cfg):
    """Phase 3 MUST consume weights; stale or partial pools are refused."""
    roots=([cfg.reviewed_root] if cfg.label_source=='roi' else
           [cfg.reviewed_root,cfg.training_root])
    result=[]; seen=set()
    for root in roots:
        if not root.exists(): continue
        manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
        if manifest.get('gen_hash')!=cfg.gen_hash or manifest.get('gold_hash')!=cfg.gold_hash:
            raise RuntimeError(f'{root}: stale configuration hashes')
        for kind, path in manifest['roi_paths'].items():
            if p2_file_digest(path) != manifest['roi_sha256'][kind]:
                raise RuntimeError(f'{root}: {kind} ROI source changed; rebuild required')
        for triple in p2_pool_files(root):
            # Gold supersedes the same spatial sample regardless of droplet ID.
            stem=triple[0].stem
            match=re.search(r't(\d+)_z(\d+)_d\d+_y(-?\d+)_x(-?\d+)',stem)
            if match is None:
                raise RuntimeError(f'Unrecognized Phase 2 patch stem: {stem}')
            key=match.groups()
            if key not in seen: result.append(triple); seen.add(key)
    return result


def list_patch_files(cfg=cfg, include_reviewed=True):
    raise RuntimeError('Phase 2 requires per-head weight sidecars. Use list_phase2_patch_files(); '
                       'the Phase 3 loader/loss must consume all three arrays.')
