"""Census and candidate nomination for supervised nuclear positives in the
EFFECTIVE Vulcan training patches.

Read-only. Nothing here trains, runs inference, launches a background job, edits
an existing pool, or rebuilds patches.

Every nomination signal is a REVIEW PRIORITY, never ground truth. T0, weak NPC
support, small size, non-roundness and multinuclear droplets cannot by themselves
prove that a label is an artifact. A stratified random sample of unflagged
positives and of supervised negatives is always included, because a flag-only
list cannot measure how prevalent contamination is.

Two passes, so a 37k-patch pool is auditable interactively:
  1. census  — every effective patch's labels and weights (no images). Cached and
               resumable per (source, timepoint) shard. Emits compact masks of
               each supervised nuclear-positive connected component.
  2. signals — image-derived signals (NLS enrichment, NPC rim support) for the
               nominated candidates and the random sample only, where the stored
               globally-normalized inputs are actually read.
Pass 1 signals are label-derived and cover the whole pool; pass 2 signals cover
the review set. Which tier a column came from is recorded in SIGNAL_TIERS.
"""
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

import reviewed_hard_negatives as hn
import vulcan_audit_recovery as vr

NUCLEUS_OBJECT_HEADS = (4, 5)      # interior | edge = the annotated nucleus
EQUATORIAL_HEAD = 6
NPC_HEAD = 3
DROPLET_INTERIOR_HEAD = 1
DROPLET_EDGE_HEAD = 2
Z_OFFSET_HEAD = 8
EIGHT = np.ones((3, 3), np.uint8)

SIGNAL_TIERS = dict(
    confirmed_artifact_overlap_px='labels', negative_supervision_px='labels',
    unsupervised_stored_px='labels', droplet_fill_fraction='labels',
    wall_contact_px='labels', seam_distance_px='geometry', npc_label_px='labels',
    solidity='labels', euler_number='labels', z_neighbour_support='labels',
    area_um2='labels', out_of_image_px='labels',
    nls_enrichment='image', npc_rim_support='image', nls_region_median='image',
)


def _signed(value, width=6):
    return f'{int(value):+0{width}d}'


def nuclear_masks(lab, wgt):
    """Stored, supervised-positive and supervised-NOT-nuclear nuclear-object masks.

    A pixel counts as supervised NOT-nuclear only when some nuclear object head
    carries weight there AND no nuclear object head calls it a nucleus. Testing the
    heads independently would mark every nucleus interior negative, because an
    interior pixel is legitimately 0 on the edge head while being 1 on the interior
    head. `stored` ignores weight on purpose: the hard-negative builder guards on
    STORED positives, so an unweighted stored positive still blocks a negative.
    """
    stored = np.zeros(lab.shape[:2], bool)
    supervised = np.zeros(lab.shape[:2], bool)
    weighted = np.zeros(lab.shape[:2], bool)
    for k in NUCLEUS_OBJECT_HEADS:
        head, weight = lab[..., k], wgt[..., k]
        stored |= head == 1
        supervised |= (head == 1) & (weight > 0)
        weighted |= weight > 0
    return stored, supervised, weighted & ~stored


# ---------------------------------------------------------------- census pass

def _census_one(task):
    """One effective patch: scalar census plus its supervised-positive components.

    Runs in a worker process and receives only paths and scalars, never memmaps.
    """
    (stem, source, t, z, cy, cx, sample_class, split, corrected,
     lab_path, wgt_path, patch_size, height, width) = task
    lab = np.load(lab_path, allow_pickle=False)
    wgt = np.load(wgt_path, allow_pickle=False)
    problems = []
    if lab.shape != (patch_size, patch_size, 10) or lab.dtype != np.uint8:
        problems.append(f'label layout {lab.shape}/{lab.dtype}')
    if wgt.shape != (patch_size, patch_size, 9):
        problems.append(f'weight layout {wgt.shape}')
    if problems:
        raise ValueError(f'{stem}: ' + '; '.join(problems))
    weights = wgt.astype(np.float32)
    if not np.isfinite(weights).all() or (weights < 0).any():
        problems.append('non-finite or negative weight')

    # Sentinel consistency: an UNANNOTATED target must never carry loss weight.
    supervised_sentinel = int(sum(int(((lab[..., k] == 255) & (weights[..., k] > 0)).sum())
                                  for k in range(9)))
    if supervised_sentinel:
        problems.append(f'{supervised_sentinel} supervised UNANNOTATED pixels')

    stored, supervised, negative = nuclear_masks(lab, weights)
    equatorial = (lab[..., EQUATORIAL_HEAD] == 1) & (weights[..., EQUATORIAL_HEAD] > 0)
    npc = (lab[..., NPC_HEAD] == 1) & (weights[..., NPC_HEAD] > 0)
    droplet = (lab[..., DROPLET_INTERIOR_HEAD] == 1) & (weights[..., DROPLET_INTERIOR_HEAD] > 0)
    wall = (lab[..., DROPLET_EDGE_HEAD] == 1) & (weights[..., DROPLET_EDGE_HEAD] > 0)
    z_supervised = weights[..., Z_OFFSET_HEAD] > 0
    z_real = lab[..., Z_OFFSET_HEAD] != 255
    if (z_supervised & ~z_real).any():
        problems.append('z_offset weight on a sentinel target')
    # A supervised z-offset outside the nuclear positive is a regression target at
    # a pixel the mask heads do not call a nucleus; counted, never corrected here.
    z_outside = int((z_supervised & ~stored).sum())

    origin_y, origin_x = int(cy) - patch_size // 2, int(cx) - patch_size // 2
    inside = np.zeros(lab.shape[:2], bool)
    y_lo, y_hi = max(0, -origin_y), min(patch_size, height - origin_y)
    x_lo, x_hi = max(0, -origin_x), min(patch_size, width - origin_x)
    if y_lo < y_hi and x_lo < x_hi:
        inside[y_lo:y_hi, x_lo:x_hi] = True

    scalars = dict(
        stem=stem, source=source, t=int(t), z=int(z), cy=int(cy), cx=int(cx),
        sample_class=sample_class, split=split, corrected=bool(corrected),
        origin_y=origin_y, origin_x=origin_x,
        stored_nuclear_px=int(stored.sum()), supervised_nuclear_px=int(supervised.sum()),
        unsupervised_stored_px=int((stored & ~supervised).sum()),
        supervised_negative_px=int(negative.sum()),
        equatorial_px=int(equatorial.sum()), npc_px=int(npc.sum()),
        droplet_interior_px=int(droplet.sum()), droplet_edge_px=int(wall.sum()),
        z_supervised_px=int(z_supervised.sum()), z_supervised_outside_nucleus_px=z_outside,
        padded_px=int((~inside).sum()), supervised_outside_image_px=int((supervised & ~inside).sum()),
        supervised_sentinel_px=supervised_sentinel,
        problems='; '.join(problems),
    )

    labelled, count = ndimage.label(supervised & inside, structure=EIGHT)
    components = []
    masks = []
    if count:
        boxes = ndimage.find_objects(labelled)
        dilated_wall = ndimage.binary_dilation(wall, EIGHT, iterations=4)
        for index, box in enumerate(boxes, start=1):
            if box is None:
                continue
            local = labelled[box] == index
            area = int(local.sum())
            ys, xs = box
            try:
                from skimage.measure import regionprops
                prop = regionprops(local.astype(np.uint8))[0]
                solidity, euler = float(prop.solidity), int(prop.euler_number)
            except Exception:
                solidity, euler = float('nan'), 0
            components.append(dict(
                stem=stem, source=source, t=int(t), z=int(z),
                y0=origin_y + ys.start, x0=origin_x + xs.start,
                y1=origin_y + ys.stop, x1=origin_x + xs.stop,
                area_px=area, solidity=solidity, euler_number=euler,
                npc_label_px=int((npc[box] & local).sum()),
                equatorial_px=int((equatorial[box] & local).sum()),
                wall_contact_px=int((dilated_wall[box] & local).sum()),
                z_supervised_px=int((z_supervised[box] & local).sum()),
                unsupervised_stored_px=int(((stored & ~supervised)[box]).sum()),
                droplet_interior_px=scalars['droplet_interior_px'],
                out_of_image_px=int((local & ~inside[box]).sum()),
            ))
            masks.append(np.packbits(local.ravel()))
    return scalars, components, masks


def _shard_inputs(frame):
    return {r.stem: [str(r.lab), *hn._identity(r.lab)[1:], str(r.wgt), *hn._identity(r.wgt)[1:]]
            for r in frame.itertuples()}


def census(session, *, workers=None, verbose=True, timepoints=None):
    """Label/weight census of every effective patch, cached and resumable.

    One shard per (source, timepoint): an interrupted census keeps the shards it
    finished. A shard is reused only when every input file still has the same
    path, size and mtime it had when the shard was written.
    """
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm

    cfg = session.rec.cfg
    frame = session.frame
    if timepoints is not None:
        frame = frame[frame.t.isin(list(timepoints))]
    height, width = session.image_shape[-2:]
    root = session.cache_dir / 'census'
    root.mkdir(parents=True, exist_ok=True)

    groups = sorted(frame.groupby(['source', 't']).groups)
    patch_tables, component_tables, blobs = [], [], []
    pending = []
    for source, t in groups:
        shard = frame[frame.source.eq(source) & frame.t.eq(t)]
        inputs = _shard_inputs(shard)
        stamp = root / f'inputs_{source}_t{t:03d}.json'
        patches = root / f'patches_{source}_t{t:03d}.csv'
        comps = root / f'components_{source}_t{t:03d}.csv'
        pack = root / f'masks_{source}_t{t:03d}.npy'
        if stamp.is_file() and patches.is_file() and comps.is_file() and pack.is_file() \
                and json.loads(stamp.read_text()) == inputs:
            patch_tables.append(pd.read_csv(patches, keep_default_na=False))
            component_tables.append(pd.read_csv(comps))
            blobs.append((source, t, np.load(pack)))
            continue
        pending.append((source, t, shard, inputs, (stamp, patches, comps, pack)))

    if verbose:
        print(f'census: {len(groups) - len(pending)} of {len(groups)} shards cached; '
              f'{sum(len(s[2]) for s in pending)} patches to read')
    started = time.monotonic()
    for source, t, shard, inputs, (stamp, patches, comps, pack) in pending:
        tasks = [(r.stem, r.source, r.t, r.z, r.cy, r.cx, r.sample_class, r.split,
                  r.corrected, r.lab, r.wgt, cfg.patch_size, height, width)
                 for r in shard.itertuples()]
        bar = tqdm(total=len(tasks), desc=f'census {source} t{t}', unit='patch', leave=False)
        results = []
        with Parallel(n_jobs=workers or -1, backend='loky', return_as='generator') as pool:
            for item in pool(delayed(_census_one)(task) for task in tasks):
                results.append(item)
                bar.update(1)
        bar.close()
        scalars = pd.DataFrame([r[0] for r in results])
        records, blob = [], []
        cursor = 0
        for _, comps_list, masks in results:
            for record, mask in zip(comps_list, masks):
                record = dict(record, pack_offset=cursor, pack_len=len(mask))
                cursor += len(mask)
                records.append(record)
                blob.append(mask)
        components = pd.DataFrame(records, columns=_COMPONENT_COLUMNS)
        packed = np.concatenate(blob) if blob else np.zeros(0, np.uint8)
        scalars.to_csv(patches, index=False)
        components.to_csv(comps, index=False)
        np.save(pack, packed)
        stamp.write_text(json.dumps(inputs, indent=1))
        patch_tables.append(scalars)
        component_tables.append(components)
        blobs.append((source, t, packed))
        if verbose:
            print(f'  {source} t{t}: {len(tasks)} patches, {len(components)} supervised '
                  f'nuclear components, {time.monotonic() - started:.0f}s elapsed')

    session.patches = pd.concat(patch_tables, ignore_index=True) if patch_tables else pd.DataFrame()
    component_tables = [c for c in component_tables if len(c)]
    session.components = (pd.concat(component_tables, ignore_index=True) if component_tables
                          else pd.DataFrame(columns=_COMPONENT_COLUMNS))
    session.mask_blobs = {(s, t): b for s, t, b in blobs}
    broken = session.patches[session.patches.problems.astype(str).str.len() > 0]
    if verbose:
        print(f'census complete: {len(session.patches)} patches, '
              f'{len(session.components)} supervised nuclear components, '
              f'{len(broken)} patches with integrity problems')
    return session.patches


_COMPONENT_COLUMNS = ['stem', 'source', 't', 'z', 'y0', 'x0', 'y1', 'x1', 'area_px',
                      'solidity', 'euler_number', 'npc_label_px', 'equatorial_px',
                      'wall_contact_px', 'z_supervised_px', 'unsupervised_stored_px',
                      'droplet_interior_px', 'out_of_image_px', 'pack_offset', 'pack_len']


def component_mask(session, row):
    """Unpack one component's boolean mask in its own global bounding box."""
    blob = session.mask_blobs[(row.source, int(row.t))]
    height, width = int(row.y1) - int(row.y0), int(row.x1) - int(row.x0)
    bits = blob[int(row.pack_offset):int(row.pack_offset) + int(row.pack_len)]
    return np.unpackbits(bits)[:height * width].astype(bool).reshape(height, width)


# ------------------------------------------------------- global region index

def _paste(target, box, mask, source_box):
    y0, x0, y1, x1 = box
    sy0, sx0, _, _ = source_box
    target[y0 - sy0:y1 - sy0, x0 - sx0:x1 - sx0] |= mask


def _touching(a_box, a_mask, b_box, b_mask):
    """True when two components overlap or are 8-adjacent in global coordinates."""
    ay0, ax0, ay1, ax1 = a_box
    by0, bx0, by1, bx1 = b_box
    if ay1 + 1 <= by0 or by1 + 1 <= ay0 or ax1 + 1 <= bx0 or bx1 + 1 <= ax0:
        return False
    y0, x0 = min(ay0, by0) - 1, min(ax0, bx0) - 1
    y1, x1 = max(ay1, by1) + 1, max(ax1, bx1) + 1
    canvas_a = np.zeros((y1 - y0, x1 - x0), bool)
    canvas_b = np.zeros_like(canvas_a)
    _paste(canvas_a, a_box, a_mask, (y0, x0, y1, x1))
    _paste(canvas_b, b_box, b_mask, (y0, x0, y1, x1))
    return bool((ndimage.binary_dilation(canvas_a, EIGHT) & canvas_b).any())


def build_regions(session, *, verbose=True):
    """Merge per-patch components into global (t, z, XY) regions.

    Indexing is global on purpose: the same physical object appears in several
    overlapping crops under different stems, and correcting one crop while leaving
    the others contradictory is exactly the failure this audit must avoid.
    """
    from tqdm.auto import tqdm

    cfg = session.rec.cfg
    components = session.components
    if components.empty:
        session.regions = pd.DataFrame()
        session.region_masks = {}
        return session.regions

    rows, masks = [], {}
    groups = list(components.groupby(['t', 'z']))
    for (t, z), group in tqdm(groups, desc='group components into global regions', unit='plane'):
        items = []
        for row in group.itertuples():
            items.append((row, (int(row.y0), int(row.x0), int(row.y1), int(row.x1)),
                          component_mask(session, row)))
        parent = list(range(len(items)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if find(i) == find(j):
                    continue
                if _touching(items[i][1], items[i][2], items[j][1], items[j][2]):
                    parent[find(j)] = find(i)
        clusters = {}
        for i in range(len(items)):
            clusters.setdefault(find(i), []).append(i)

        for members in clusters.values():
            boxes = [items[i][1] for i in members]
            y0 = min(b[0] for b in boxes); x0 = min(b[1] for b in boxes)
            y1 = max(b[2] for b in boxes); x1 = max(b[3] for b in boxes)
            canvas = np.zeros((y1 - y0, x1 - x0), bool)
            for i in members:
                _paste(canvas, items[i][1], items[i][2], (y0, x0, y1, x1))
            digest = hashlib.sha256(np.packbits(canvas.ravel()).tobytes()
                                    + f'{t},{z},{y0},{x0}'.encode()).hexdigest()[:8]
            region_id = f't{int(t):03d}_z{int(z):03d}_y{_signed(y0)}_x{_signed(x0)}_{digest}'
            member_rows = [items[i][0] for i in members]
            stems = sorted({r.stem for r in member_rows})
            rows.append(dict(
                region_id=region_id, t=int(t), z=int(z), y0=y0, x0=x0, y1=y1, x1=x1,
                area_px=int(canvas.sum()), area_um2=float(canvas.sum()) * cfg.pixel_size_um ** 2,
                n_components=len(members), n_patches=len(stems), patch_stems=';'.join(stems),
                solidity=float(np.nanmin([r.solidity for r in member_rows])),
                euler_number=int(min(r.euler_number for r in member_rows)),
                npc_label_px=int(sum(r.npc_label_px for r in member_rows)),
                equatorial_px=int(sum(r.equatorial_px for r in member_rows)),
                wall_contact_px=int(max(r.wall_contact_px for r in member_rows)),
                z_supervised_px=int(max(r.z_supervised_px for r in member_rows)),
                unsupervised_stored_px=int(max(r.unsupervised_stored_px for r in member_rows)),
                droplet_interior_px=int(max(r.droplet_interior_px for r in member_rows)),
                out_of_image_px=int(sum(r.out_of_image_px for r in member_rows)),
            ))
            masks[region_id] = canvas

    regions = pd.DataFrame(rows)
    regions['droplet_fill_fraction'] = regions.area_px / regions.droplet_interior_px.clip(lower=1)
    session.regions = regions
    session.region_masks = masks
    if verbose:
        print(f'{len(regions)} global supervised nuclear-positive regions across '
              f'{regions.groupby(["t", "z"]).ngroups} (t, z) planes')
    return regions


def annotate_regions(session, *, verbose=True):
    """Split membership, holdout verdicts, seam distance, Z support and prior review."""
    cfg = session.rec.cfg
    regions = session.regions
    if regions.empty:
        return regions
    patch_split = dict(zip(session.patches.stem, session.patches.split))
    patch_center = {r.stem: (int(r.cy), int(r.cx), int(r.t)) for r in session.patches.itertuples()}

    correctable, held, splits, reasons = [], [], [], []
    for row in regions.itertuples():
        stems = row.patch_stems.split(';')
        member_splits = sorted({patch_split.get(s, 'unknown') for s in stems})
        ok, blocked, why = [], [], set()
        for stem in stems:
            cy, cx, t = patch_center[stem]
            reason = hn.holdout_reason(t, cy, cx, cfg)
            if reason == 'train' and patch_split.get(stem) in ('train', 'hard_negative_train'):
                ok.append(stem)
            else:
                blocked.append(stem)
                why.add(reason if reason != 'train' else patch_split.get(stem, 'unknown'))
        correctable.append(';'.join(ok))
        held.append(';'.join(blocked))
        splits.append(','.join(member_splits))
        reasons.append(','.join(sorted(why)))
    regions['correctable_stems'] = correctable
    regions['withheld_stems'] = held
    regions['splits'] = splits
    regions['withheld_reasons'] = reasons
    regions['n_correctable_patches'] = [len(s.split(';')) if s else 0 for s in correctable]

    step = cfg.mosaic_tile_px - cfg.mosaic_overlap_px
    centre_y = (regions.y0 + regions.y1) / 2
    centre_x = (regions.x0 + regions.x1) / 2
    regions['seam_distance_px'] = np.minimum(
        np.minimum(centre_y % step, step - centre_y % step),
        np.minimum(centre_x % step, step - centre_x % step))
    regions['below_z_floor'] = regions.z < cfg.z_floor

    # Z support: does a supervised positive overlap this region's XY on z +/- 1?
    by_plane = {}
    for row in regions.itertuples():
        by_plane.setdefault((row.t, row.z), []).append(row)
    support = []
    for row in regions.itertuples():
        count = 0
        for dz in (-1, 1):
            for other in by_plane.get((row.t, row.z + dz), []):
                if row.y1 > other.y0 and other.y1 > row.y0 and row.x1 > other.x0 and other.x1 > row.x0:
                    count += 1
                    break
        support.append(count)
    regions['z_neighbour_support'] = support

    overlap, prior = [], []
    reviewed = session.reviewed_artifacts
    for row in regions.itertuples():
        mask = session.region_masks[row.region_id]
        total, names = 0, []
        for rec in reviewed.get((row.t, row.z), []):
            y0, x0 = max(row.y0, rec['y0']), max(row.x0, rec['x0'])
            y1, x1 = min(row.y1, rec['y1']), min(row.x1, rec['x1'])
            if y0 >= y1 or x0 >= x1:
                continue
            hit = int((mask[y0 - row.y0:y1 - row.y0, x0 - row.x0:x1 - row.x0]
                       & rec['mask'][y0 - rec['y0']:y1 - rec['y0'], x0 - rec['x0']:x1 - rec['x0']]).sum())
            if hit:
                total += hit
                names.append(f"{rec['candidate_id']}:{rec['decision']}")
        overlap.append(total)
        prior.append(';'.join(sorted(names)))
    regions['confirmed_artifact_overlap_px'] = overlap
    regions['prior_review'] = prior
    if verbose:
        print(f'{int((regions.n_correctable_patches > 0).sum())} regions have at least one '
              f'eligible training crop; {int((regions.n_correctable_patches == 0).sum())} are '
              'holdout/seam-only and can only be reported')
        print(f'{int((regions.confirmed_artifact_overlap_px > 0).sum())} regions overlap a '
              'previously reviewed artifact mask')
    return regions


def load_reviewed_artifacts(session, *, verbose=True):
    """Exact masks from the committed review, by decision, for overlap signals."""
    review = Path(session.eff.commit['review'])
    manifest = json.loads((review / 'candidates.json').read_text())
    table = pd.read_csv(review / 'review.csv').fillna('')
    decisions = dict(zip(table.candidate_id, table.decision))
    by_plane = {}
    for rec in manifest['candidates']:
        path = (review / rec['mask_file']).resolve()
        if review.resolve() not in path.parents or hn._digest(path) != rec['mask_sha256']:
            raise ValueError('Committed reviewed mask changed: ' + rec['candidate_id'])
        mask = np.load(path, allow_pickle=False).astype(bool)
        item = dict(candidate_id=rec['candidate_id'], decision=decisions.get(rec['candidate_id'], ''),
                    y0=int(rec['y0']), x0=int(rec['x0']), y1=int(rec['y1']), x1=int(rec['x1']),
                    mask=mask)
        if item['decision'] == 'artifact_negative':
            by_plane.setdefault((int(rec['t']), int(rec['z'])), []).append(item)
    session.reviewed_artifacts = by_plane
    if verbose:
        counts = table.decision.value_counts().to_dict()
        print('committed review decisions:', counts)
        print(f'{sum(len(v) for v in by_plane.values())} confirmed artifact masks loaded '
              '(these are prior decisions, not new candidates)')
    return by_plane


# ------------------------------------------------------- conflict scan (exact)

def scan_conflicts(session, region_ids, *, verbose=True):
    """Exact cross-crop label conflicts for the given regions.

    For every effective training crop that overlaps a region, read only that
    region's window and count pixels the crop supervises as NOT nuclear, and
    stored positives the crop leaves unweighted. A weight-positive conflict and an
    ignored stored positive are reported separately: the negative-pool builder
    guards on STORED positives, so the two have different consequences.
    """
    from tqdm.auto import tqdm

    regions = session.regions.set_index('region_id')
    lookup = {}
    for row in session.frame.itertuples():
        lookup.setdefault((row.t, row.z), []).append(row)
    out = []
    for region_id in tqdm(list(region_ids), desc='exact cross-crop conflict scan', unit='region'):
        row = regions.loc[region_id]
        mask = session.region_masks[region_id]
        negative_px = positive_px = ignored_px = 0
        conflicting, members = [], set(str(row.patch_stems).split(';'))
        for patch in lookup.get((int(row.t), int(row.z)), []):
            oy, ox = int(patch.cy) - session.patch_size // 2, int(patch.cx) - session.patch_size // 2
            y0, x0 = max(row.y0, oy), max(row.x0, ox)
            y1, x1 = min(row.y1, oy + session.patch_size), min(row.x1, ox + session.patch_size)
            if y0 >= y1 or x0 >= x1:
                continue
            window = mask[y0 - row.y0:y1 - row.y0, x0 - row.x0:x1 - row.x0]
            if not window.any():
                continue
            lab = np.load(patch.lab, allow_pickle=False, mmap_mode='r')
            wgt = np.load(patch.wgt, allow_pickle=False, mmap_mode='r')
            sl = (slice(y0 - oy, y1 - oy), slice(x0 - ox, x1 - ox))
            local_lab = np.asarray(lab[sl[0], sl[1], :])
            local_wgt = np.asarray(wgt[sl[0], sl[1], :], dtype=np.float32)
            stored, supervised, negative = nuclear_masks(local_lab, local_wgt)
            hits = int((window & negative).sum())
            if hits and patch.stem not in members:
                conflicting.append(f'{patch.stem}:{hits}')
            negative_px += hits
            positive_px += int((window & supervised).sum())
            ignored_px += int((window & stored & ~supervised).sum())
        out.append(dict(region_id=region_id, negative_supervision_px=negative_px,
                        positive_supervision_px=positive_px,
                        ignored_stored_px=ignored_px,
                        conflicting_crops=';'.join(sorted(conflicting))))
    frame = pd.DataFrame(out)
    if verbose and len(frame):
        print(f'{int((frame.negative_supervision_px > 0).sum())} of {len(frame)} scanned regions '
              'have pixels another effective crop supervises as NOT nuclear')
    return frame


# ------------------------------------------------------------ image signals

def add_image_signals(session, region_ids, *, verbose=True):
    """NLS enrichment and NPC rim support from the STORED normalized inputs.

    The stored img patches are the exact globally-normalized tensors the model was
    trained on (plane-major [NLS, NPC, Mem]; centre plane = channels 6, 7, 8). No
    per-crop contrast stretch is applied: an independently stretched picture cannot
    quantify enrichment.
    """
    from tqdm.auto import tqdm

    cfg = session.cfg
    regions = session.regions.set_index('region_id')
    by_stem = {r.stem: r for r in session.frame.itertuples()}
    out = []
    for region_id in tqdm(list(region_ids), desc='image signals (stored normalized input)', unit='region'):
        row = regions.loc[region_id]
        mask = session.region_masks[region_id]
        stems = [s for s in str(row.patch_stems).split(';') if s in by_stem and by_stem[s].img]
        record = dict(region_id=region_id, nls_region_median=np.nan, nls_surround_median=np.nan,
                      nls_enrichment=np.nan, npc_rim_support=np.nan, image_signal_stem='')
        for stem in stems:
            patch = by_stem[stem]
            oy, ox = int(patch.cy) - session.patch_size // 2, int(patch.cx) - session.patch_size // 2
            y0, x0 = max(row.y0, oy), max(row.x0, ox)
            y1, x1 = min(row.y1, oy + session.patch_size), min(row.x1, ox + session.patch_size)
            if y0 >= y1 or x0 >= x1:
                continue
            window = mask[y0 - row.y0:y1 - row.y0, x0 - row.x0:x1 - row.x0]
            if window.sum() < 0.9 * mask.sum():
                continue                        # need the whole region inside one crop
            img = np.load(patch.img, allow_pickle=False, mmap_mode='r')
            lab = np.load(patch.lab, allow_pickle=False, mmap_mode='r')
            wgt = np.load(patch.wgt, allow_pickle=False, mmap_mode='r')
            nls = np.asarray(img[..., 6], dtype=np.float32)
            npc = np.asarray(img[..., 7], dtype=np.float32)
            full = np.zeros((session.patch_size, session.patch_size), bool)
            full[y0 - oy:y1 - oy, x0 - ox:x1 - ox] = window
            droplet = (np.asarray(lab[..., DROPLET_INTERIOR_HEAD]) == 1) & \
                      (np.asarray(wgt[..., DROPLET_INTERIOR_HEAD], dtype=np.float32) > 0)
            surround = droplet & ~ndimage.binary_dilation(full, EIGHT, iterations=6)
            # The NPC label is a band just INSIDE the traced outline, so the rim is an
            # annulus straddling the envelope, not a ring outside it.
            band = max(2, int(round(cfg.npc_band_width_um / cfg.pixel_size_um)))
            rim = (ndimage.binary_dilation(full, EIGHT, iterations=band)
                   & ~ndimage.binary_erosion(full, EIGHT, iterations=band))
            if not full.any() or not surround.any():
                continue
            region_median = float(np.median(nls[full]))
            surround_median = float(np.median(nls[surround]))
            record.update(
                nls_region_median=region_median, nls_surround_median=surround_median,
                nls_enrichment=region_median / surround_median if surround_median > 1e-6 else np.inf,
                npc_rim_support=(float(np.median(npc[rim])) / float(np.median(npc[surround]))
                                 if rim.any() and float(np.median(npc[surround])) > 1e-6 else np.nan),
                image_signal_stem=stem)
            break
        out.append(record)
    frame = pd.DataFrame(out)
    if verbose and len(frame):
        measured = frame.nls_enrichment.notna().sum()
        print(f'image signals measured for {measured} of {len(frame)} regions '
              '(a region split across crops is left unmeasured rather than stitched)')
    return frame


# --------------------------------------------------------------- nomination

@dataclass
class NominationSettings:
    droplet_fill_max: float = 0.45        # above this a positive fills its droplet
    min_area_um2: float = None            # defaults to cfg.roi_nuc_area_min_um2
    seam_px: int = 64
    min_solidity: float = 0.85
    nls_enrichment_min: float = 1.20      # cfg.nucleus_min_enrichment by default
    npc_rim_min: float = 1.05
    sample_per_timepoint: int = 20
    negative_controls_per_timepoint: int = 6
    max_candidates: int = None
    seed: int = 42


FLAG_RULES = (
    ('confirmed_artifact_overlap', 3.0, 'overlaps a mask already confirmed as artifact'),
    ('negative_supervision_conflict', 3.0, 'another effective training crop supervises these pixels as NOT nuclear'),
    ('below_z_floor', 2.0, 'centre plane is below the nuclear Z floor'),
    ('droplet_filling', 2.0, 'positive covers an implausible fraction of its droplet'),
    ('ignored_stored_positive', 1.0, 'stored positive left unweighted in some crop'),
    ('wall_contact', 1.5, 'touches supervised droplet wall / edge labels'),
    ('z_isolated', 1.5, 'no supervised positive in either neighbouring Z plane'),
    ('seam_proximity', 1.0, 'centre lies near a mosaic stitching seam'),
    ('no_npc_label', 1.0, 'no supervised NPC label anywhere in the region'),
    ('odd_shape', 1.0, 'low solidity or more than one hole'),
    ('tiny', 1.0, 'smaller than the gold ROI minimum nuclear area'),
    ('out_of_image', 1.0, 'includes padded, out-of-image pixels'),
    ('weak_nls_enrichment', 2.0, 'NLS median inside barely exceeds the droplet surround'),
    ('weak_npc_support', 1.0, 'no NPC rim enrichment around the region'),
)


def apply_flags(session, regions, settings):
    """Attach named flags and an additive, fully itemised priority score."""
    cfg = session.rec.cfg
    min_area = settings.min_area_um2 if settings.min_area_um2 is not None else cfg.roi_nuc_area_min_um2
    enrich_min = settings.nls_enrichment_min or cfg.nucleus_min_enrichment
    frame = regions.copy()

    def column(name, default=np.nan):
        """A column as a Series even when the pass that fills it has not run yet."""
        if name in frame.columns:
            return frame[name]
        return pd.Series(default, index=frame.index, dtype='float64')

    tests = {
        'confirmed_artifact_overlap': column('confirmed_artifact_overlap_px', 0) > 0,
        'negative_supervision_conflict': column('negative_supervision_px', 0) > 0,
        'below_z_floor': frame.below_z_floor,
        'droplet_filling': frame.droplet_fill_fraction > settings.droplet_fill_max,
        'ignored_stored_positive': frame.unsupervised_stored_px > 0,
        'wall_contact': frame.wall_contact_px > 0,
        'z_isolated': frame.z_neighbour_support == 0,
        'seam_proximity': frame.seam_distance_px < settings.seam_px,
        'no_npc_label': frame.npc_label_px == 0,
        'odd_shape': (frame.solidity < settings.min_solidity) | (frame.euler_number < 0),
        'tiny': frame.area_um2 < min_area,
        'out_of_image': frame.out_of_image_px > 0,
        'weak_nls_enrichment': column('nls_enrichment') < enrich_min,
        'weak_npc_support': column('npc_rim_support') < settings.npc_rim_min,
    }
    score = pd.Series(0.0, index=frame.index)
    names = [[] for _ in range(len(frame))]
    for flag, weight, _ in FLAG_RULES:
        hit = tests[flag].fillna(False).astype(bool)
        frame['flag_' + flag] = hit
        score = score + weight * hit
        for position, on in enumerate(hit.to_numpy()):
            if on:
                names[position].append(flag)
    frame['flags'] = [','.join(n) for n in names]
    frame['n_flags'] = [len(n) for n in names]
    frame['priority_score'] = score.to_numpy()
    return frame


def nominate(session, *, settings=None, verbose=True):
    """Rank candidates, then add a stratified random sample of unflagged positives.

    A flag is a reason to look. The sample exists so prevalence can be estimated
    at all; without it the review can only confirm what the signals already
    suspected.
    """
    settings = settings or NominationSettings()
    regions = session.regions
    if regions.empty:
        session.candidates = pd.DataFrame()
        return session.candidates

    # Pass 1 flags over every region, to choose who gets an exact conflict scan.
    tier1 = apply_flags(session, regions, settings)
    shortlist = tier1[(tier1.n_flags > 0) & (tier1.n_correctable_patches > 0)]
    shortlist = shortlist.sort_values('priority_score', ascending=False)
    if settings.max_candidates:
        shortlist = shortlist.head(settings.max_candidates)

    rng = np.random.default_rng(settings.seed)
    unflagged = tier1[(tier1.n_flags == 0) & (tier1.n_correctable_patches > 0)]
    sampled = []
    for _, group in unflagged.groupby('t'):
        take = min(settings.sample_per_timepoint, len(group))
        if take:
            sampled.append(group.iloc[rng.choice(len(group), take, replace=False)])
    sample = pd.concat(sampled) if sampled else unflagged.head(0)

    selected = pd.concat([shortlist, sample]).drop_duplicates('region_id')
    conflicts = scan_conflicts(session, selected.region_id, verbose=verbose)
    images = add_image_signals(session, selected.region_id, verbose=verbose)

    enriched = regions.merge(conflicts, on='region_id', how='left').merge(images, on='region_id', how='left')
    for column in ('negative_supervision_px', 'positive_supervision_px', 'ignored_stored_px'):
        enriched[column] = enriched[column].fillna(-1)      # -1 = not scanned
    session.regions = enriched
    final = apply_flags(session, enriched, settings)
    session.scored_regions = final

    chosen = final[final.region_id.isin(set(selected.region_id))].copy()
    chosen['kind'] = np.where(chosen.region_id.isin(set(shortlist.region_id)),
                              'flagged', 'random_sample')
    chosen = chosen.sort_values(['kind', 'priority_score'], ascending=[True, False])
    negatives = sample_negative_controls(session, settings, verbose=verbose)
    session.candidates = pd.concat([chosen, negatives], ignore_index=True)
    if verbose:
        print(f'{len(chosen)} candidates for review: '
              f'{int((chosen.kind == "flagged").sum())} flagged, '
              f'{int((chosen.kind == "random_sample").sum())} stratified random positives, '
              f'plus {len(negatives)} negative controls')
        print('Flags are review priorities, not labels. T0, weak NPC, small size, '
              'non-roundness and multinuclear droplets cannot alone prove an artifact.')
    return session.candidates


def sample_negative_controls(session, settings, *, verbose=True):
    """Stratified sample of supervised NEGATIVE nuclear regions, as review controls.

    Confirms that existing negatives look like negatives. These are controls, not
    corrections: a negative control has no correctable region and is never written.
    """
    from tqdm.auto import tqdm

    count = settings.negative_controls_per_timepoint
    if not count:
        return pd.DataFrame()
    rng = np.random.default_rng(settings.seed + 1)
    eligible = session.patches[(session.patches.split.isin(['train', 'hard_negative_train']))
                               & (session.patches.supervised_negative_px > 0)]
    picks = []
    for _, group in eligible.groupby('t'):
        take = min(count, len(group))
        if take:
            picks += list(group.iloc[rng.choice(len(group), take, replace=False)].stem)
    by_stem = {r.stem: r for r in session.frame.itertuples()}
    rows = []
    for stem in tqdm(picks, desc='negative controls', unit='patch'):
        patch = by_stem[stem]
        lab = np.load(patch.lab, allow_pickle=False)
        wgt = np.load(patch.wgt, allow_pickle=False).astype(np.float32)
        _, _, negative = nuclear_masks(lab, wgt)
        droplet = (lab[..., DROPLET_INTERIOR_HEAD] == 1) & (wgt[..., DROPLET_INTERIOR_HEAD] > 0)
        target = negative & droplet
        if not target.any():
            continue
        labelled, n = ndimage.label(target, structure=EIGHT)
        if not n:
            continue
        sizes = ndimage.sum(target, labelled, range(1, n + 1))
        index = int(np.argmax(sizes)) + 1
        box = ndimage.find_objects(labelled)[index - 1]
        local = labelled[box] == index
        oy, ox = int(patch.cy) - session.patch_size // 2, int(patch.cx) - session.patch_size // 2
        y0, x0 = oy + box[0].start, ox + box[1].start
        y1, x1 = oy + box[0].stop, ox + box[1].stop
        digest = hashlib.sha256(np.packbits(local.ravel()).tobytes()
                                + f'{patch.t},{patch.z},{y0},{x0}'.encode()).hexdigest()[:8]
        region_id = f't{int(patch.t):03d}_z{int(patch.z):03d}_y{_signed(y0)}_x{_signed(x0)}_{digest}'
        session.region_masks[region_id] = local
        rows.append(dict(region_id=region_id, t=int(patch.t), z=int(patch.z), y0=y0, x0=x0,
                         y1=y1, x1=x1, area_px=int(local.sum()),
                         area_um2=float(local.sum()) * session.rec.cfg.pixel_size_um ** 2,
                         n_components=1, n_patches=1, patch_stems=stem,
                         correctable_stems='', withheld_stems='',
                         splits=patch.split, withheld_reasons='negative_control',
                         n_correctable_patches=0, flags='', n_flags=0, priority_score=0.0,
                         kind='negative_control', prior_review=''))
    frame = pd.DataFrame(rows)
    if verbose:
        print(f'{len(frame)} supervised-negative control regions sampled across timepoints')
    return frame


# -------------------------------------------------------- augmentation check

def augmentation_check(session, stems=None, *, seed=7, count=3):
    """Reproduce the training augmentation on real patches with a fixed seed.

    Checks what augmentation can actually break: that no UNANNOTATED sentinel is
    blended into a real value, that pixels mapped from outside the patch lose
    their weight instead of inventing supervision, and that image and packed
    target stay aligned. This measures the transform; it does not claim that
    augmentation caused any particular artifact.
    """
    ns, cfg = session.rec.ns, session.rec.cfg
    layout = ns['p3_layout'](cfg)
    frame = session.frame[session.frame.split.eq('train') & session.frame.img.astype(bool)]
    chosen = list(stems) if stems else list(frame.stem.head(count))
    rows = {r.stem: r for r in frame.itertuples()}
    out = []
    for stem in chosen:
        row = rows[stem]
        x, packed = ns['p3_load_numpy'](str(row.img), str(row.lab), str(row.wgt), layout, cfg)
        ax, apacked = ns['p3_augment_numpy'](x.copy(), packed.copy(), seed, layout, cfg)
        n_y = layout['n_y']
        labels, weights = apacked[..., :n_y], apacked[..., n_y:2 * n_y]
        sentinel_supervised = int(((labels == 255) & (weights > 0)).sum())
        blended = int((~np.isin(labels[..., :layout['n_mask']], (0.0, 1.0, 255.0))).sum())
        out.append(dict(
            stem=stem, seed=seed,
            input_changed=bool(not np.allclose(x, ax)),
            target_changed=bool(not np.allclose(packed, apacked)),
            input_in_unit_range=bool(np.isfinite(ax).all() and ax.min() >= 0 and ax.max() <= 1),
            mask_values_still_discrete=blended == 0,
            blended_mask_values=blended,
            supervised_sentinel_after_aug=sentinel_supervised,
            weight_total_before=float(packed[..., n_y:2 * n_y].sum()),
            weight_total_after=float(weights.sum()),
            before=(x, packed), after=(ax, apacked), layout=layout))
    table = pd.DataFrame([{k: v for k, v in r.items() if k not in ('before', 'after', 'layout')}
                          for r in out])
    return dict(table=table, examples=out, layout=layout)


# ------------------------------------------------------------------ session

@dataclass
class AuditSession:
    rec: object
    eff: object
    frame: pd.DataFrame
    cache_dir: Path
    image_shape: tuple
    patch_size: int
    patches: pd.DataFrame = None
    components: pd.DataFrame = None
    regions: pd.DataFrame = None
    scored_regions: pd.DataFrame = None
    candidates: pd.DataFrame = None
    mask_blobs: dict = field(default_factory=dict)
    region_masks: dict = field(default_factory=dict)
    reviewed_artifacts: dict = field(default_factory=dict)

    @property
    def cfg(self):
        return self.rec.cfg

    @property
    def provenance(self):
        return self.eff.provenance

    def summary(self):
        """Supervised nuclear positives by timepoint, split, source and class."""
        patches = self.patches
        by_split = patches.groupby('split').agg(
            patches=('stem', 'size'),
            with_supervised_positive=('supervised_nuclear_px', lambda s: int((s > 0).sum())),
            supervised_positive_px=('supervised_nuclear_px', 'sum'),
            stored_positive_px=('stored_nuclear_px', 'sum'),
            unsupervised_stored_px=('unsupervised_stored_px', 'sum'),
            supervised_negative_px=('supervised_negative_px', 'sum'),
            z_supervised_outside_nucleus_px=('z_supervised_outside_nucleus_px', 'sum'),
            integrity_problems=('problems', lambda s: int((s.astype(str).str.len() > 0).sum())))
        regions = self.regions if self.regions is not None else pd.DataFrame()
        by_t = pd.DataFrame()
        if len(regions):
            by_t = regions.groupby('t').agg(regions=('region_id', 'size'),
                                            median_area_um2=('area_um2', 'median'),
                                            correctable=('n_correctable_patches', lambda s: int((s > 0).sum())))
        return dict(by_split=by_split, by_timepoint=by_t,
                    by_class=patches.groupby(['source', 'sample_class']).agg(
                        patches=('stem', 'size'),
                        with_supervised_positive=('supervised_nuclear_px', lambda s: int((s > 0).sum()))))


def open_audit(run_dir=vr.DEFAULT_RUN_DIR, commit_dir=vr.DEFAULT_COMMIT_DIR, *,
               notebook=vr.DEFAULT_NOTEBOOK, cache_dir=None, verbose=True):
    """Fresh-kernel entry point: provenance, effective labels and an empty census."""
    import tifffile
    rec = vr.recover_config(run_dir, notebook=notebook, verbose=verbose)
    vr.attach_norm_stats(rec, verbose=verbose)
    eff = vr.effective_rows(rec, commit_dir, verbose=verbose)
    frame = vr.row_frame(eff)
    raw = tifffile.memmap(rec.cfg.image_file, mode='r')
    shape = tuple(int(n) for n in raw.shape)
    del raw
    if cache_dir is None:
        cache_dir = Path(rec.cfg.out_root) / ('training_patch_audit_cache_' + rec.cfg.run_hash)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = AuditSession(rec=rec, eff=eff, frame=frame, cache_dir=cache_dir,
                           image_shape=shape, patch_size=int(rec.cfg.patch_size))
    if verbose:
        print('raw hyperstack     :', shape, '(T,Z,C,Y,X)')
        print('audit cache        :', cache_dir)
    load_reviewed_artifacts(session, verbose=verbose)
    return session


def run_audit(run_dir=vr.DEFAULT_RUN_DIR, commit_dir=vr.DEFAULT_COMMIT_DIR, *,
              notebook=vr.DEFAULT_NOTEBOOK, cache_dir=None, workers=None,
              settings=None, timepoints=None, verbose=True):
    """One contained audit: recover, census, index globally, nominate.

    Read-only throughout; the only writes are the resumable census cache and, if a
    review is later opened, the review state. Nothing trains or rebuilds a pool.
    """
    started = time.monotonic()
    session = open_audit(run_dir, commit_dir, notebook=notebook, cache_dir=cache_dir, verbose=verbose)
    census(session, workers=workers, verbose=verbose, timepoints=timepoints)
    build_regions(session, verbose=verbose)
    annotate_regions(session, verbose=verbose)
    nominate(session, settings=settings, verbose=verbose)
    if verbose:
        print(f'audit finished in {(time.monotonic() - started) / 60:.1f} min. '
              'No training, inference, background jobs, or pool rebuilds.')
    return session
