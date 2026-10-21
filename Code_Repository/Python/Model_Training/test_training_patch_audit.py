"""Synthetic end-to-end tests for the training-patch audit, resolver and commit.

Builds a miniature but structurally real pool: a raw hyperstack, a complete Phase 2
gold pool, a committed review with a gold-correction pool and a reviewed
hard-negative pool, and a training run directory whose config.json hashes to its
own name. The fixture drives the PRODUCTION code (gold_artifact_corrections,
reviewed_hard_negatives, the notebook's own p3_rows/p3_split) rather than
re-implementing it, so these tests exercise the same paths the real pool will.

Run directly for a plain-English PASS/FAIL report:
    python test_training_patch_audit.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gold_artifact_corrections as gc
import reviewed_hard_negatives as hn
import training_patch_audit as tpa
import training_patch_resolver as tpr
import vulcan_audit_recovery as vr

T, Z, C, H, W = 3, 10, 3, 256, 384
PATCH = 64
HALF = PATCH // 2
CONFIRMED = 'Confirmed artifact; conflicting gold annotation requires correction.'

# (name, t, z, cy, cx) -- centres are deliberately mixed: in-tile, holdout, and one
# signed/padded centre, because the real pool contains all three.
PATCHES = [
    ('A', 0, 4, 40, 40), ('B', 0, 4, 40, 60), ('G', 0, 4, 40, -1),
    ('C', 0, 4, 64, 176), ('D', 1, 4, 40, 40), ('I', 0, 6, 40, 40),
    ('E', 2, 5, 160, 40), ('F', 2, 5, 160, 56), ('J', 2, 7, 232, 92),
]
# Supervised nuclear objects in GLOBAL coordinates: name -> (t, z, y0, x0, y1, x1, bright)
OBJECTS = dict(
    real=(0, 4, 30, 30, 50, 50, True),           # stays real; must end byte-identical
    corrected=(0, 6, 30, 30, 46, 46, True),      # already fixed by the existing correction pool
    artifact=(2, 5, 150, 30, 166, 46, False),    # the new finding, spanning two train crops
    holdout=(0, 4, 55, 170, 75, 190, True),      # spatial holdout only
    temporal=(1, 4, 30, 30, 50, 50, True),       # temporal holdout only
    padded=(0, 4, 30, -20, 46, 10, True),        # straddles the image border
)
DROPLETS = {  # (t, z) -> list of (centre_y, centre_x, radius) in global coordinates
    (0, 4): [(40, 40, 46), (65, 180, 40), (40, -5, 40)],
    (0, 6): [(40, 40, 46)],
    (1, 4): [(40, 40, 46)],
    (2, 5): [(158, 44, 46)],
    (2, 7): [(210, 70, 46)],
}


def _disc(shape, centre_y, centre_x, radius):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy - centre_y) ** 2 + (xx - centre_x) ** 2 <= radius * radius


def _build_patch(t, z, cy, cx):
    """Labels, weights and a globally-plausible normalized input for one crop."""
    origin_y, origin_x = cy - HALF, cx - HALF
    lab = np.full((PATCH, PATCH, 10), 255, np.uint8)
    wgt = np.zeros((PATCH, PATCH, 9), np.float32)
    lab[..., 9] = 0

    inside = np.zeros((PATCH, PATCH), bool)
    y_lo, y_hi = max(0, -origin_y), min(PATCH, H - origin_y)
    x_lo, x_hi = max(0, -origin_x), min(PATCH, W - origin_x)
    if y_lo < y_hi and x_lo < x_hi:
        inside[y_lo:y_hi, x_lo:x_hi] = True

    droplet = np.zeros((PATCH, PATCH), bool)
    for centre_y, centre_x, radius in DROPLETS.get((t, z), []):
        droplet |= _disc((PATCH, PATCH), centre_y - origin_y, centre_x - origin_x, radius)
    droplet &= inside
    from scipy import ndimage
    wall = droplet & ~ndimage.binary_erosion(droplet, np.ones((3, 3), bool), iterations=3)
    # Droplet family is supervised wherever the droplet was traced.
    for head, mask in ((0, droplet | inside), (1, droplet), (2, wall)):
        lab[..., head][inside] = 0
        wgt[..., head][inside] = 1.0
        lab[..., head][mask] = 1
    lab[..., 9][droplet] = 1

    nucleus = np.zeros((PATCH, PATCH), bool)
    nls_boost = np.zeros((PATCH, PATCH), np.float32)
    for name, (ot, oz, y0, x0, y1, x1, bright) in OBJECTS.items():
        if (ot, oz) != (t, z):
            continue
        local = np.zeros((PATCH, PATCH), bool)
        ly0, lx0 = max(0, y0 - origin_y), max(0, x0 - origin_x)
        ly1, lx1 = min(PATCH, y1 - origin_y), min(PATCH, x1 - origin_x)
        if ly0 >= ly1 or lx0 >= lx1:
            continue
        local[ly0:ly1, lx0:lx1] = True
        local &= inside
        nucleus |= local
        nls_boost[local] = 0.55 if bright else 0.06
    edge = nucleus & ~ndimage.binary_erosion(nucleus, np.ones((3, 3), bool), iterations=2)

    # Nuclear family: supervised 0 inside the droplet, 1 on the annotated nucleus.
    for head in (4, 5, 6):
        lab[..., head][droplet] = 0
        wgt[..., head][droplet] = 1.0
    lab[..., 4][nucleus] = 1
    lab[..., 5][edge] = 1
    lab[..., 6][nucleus] = 1
    lab[..., 3][droplet] = 0
    wgt[..., 3][droplet] = 1.0
    lab[..., 3][edge] = 1                      # NPC band on the envelope
    # z_offset: a real target only on the nucleus, 10 units per micron.
    lab[..., 8][nucleus] = 20
    wgt[..., 8][nucleus] = 1.0

    rng = np.random.default_rng(abs(hash((t, z, cy, cx))) % (2 ** 31))
    img = rng.uniform(0.05, 0.12, (PATCH, PATCH, 15)).astype(np.float32)
    for plane in range(5):
        img[..., plane * 3 + 0] += nls_boost * (1.0 if plane == 2 else 0.5)
        img[..., plane * 3 + 1] += edge * 0.4
        img[..., plane * 3 + 2] += droplet * 0.1
    return lab, wgt, np.clip(img, 0, 1).astype(np.float32)


def _stem(t, z, index, cy, cx):
    return f't{t:03d}_z{z:03d}_d{index:06d}_y{cy:05d}_x{cx:05d}_roi_p{index:06d}'


def build_fixture(root):
    """Create the whole miniature project and return the recovered audit session."""
    root = Path(root)
    data_root = root / 'data' / 'Nuclear_Scaling'
    out_root = data_root / 'Inputs' / 'Training_Data'
    image_root = data_root / 'Inputs' / 'Raw_Images' / 'Condition' / 'Control'
    roi_root = image_root / 'Regions_of_Interest'
    model_root = root / 'project' / 'Models'
    for folder in (out_root, image_root, roi_root, model_root):
        folder.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    raw = rng.integers(80, 160, (T, Z, C, H, W)).astype(np.uint16)
    for name, (t, z, y0, x0, y1, x1, bright) in OBJECTS.items():
        yy0, xx0, yy1, xx1 = max(0, y0), max(0, x0), min(H, y1), min(W, x1)
        if yy0 < yy1 and xx0 < xx1:
            raw[t, max(0, z - 1):z + 2, 1, yy0:yy1, xx0:xx1] = 2200 if bright else 300
    image_file = image_root / 'control_fixture.tif'
    tifffile.imwrite(image_file, raw)
    for name in ('NucleiRoiSet.zip', 'DropletRoiSet.zip', 'NPCRoiSet.zip'):
        (roi_root / name).write_bytes(name.encode() * 32)

    # --- configuration: the real PipelineConfig, scaled down -----------------
    cells = vr.select_definition_cells(vr.DEFAULT_NOTEBOOK)
    paths = dict(PROJECT_ROOT=root / 'project', CODE_ROOT=root / 'project' / 'Code_Repository',
                 DATA_ROOT=data_root, CONTROL_DATA_DIR=image_root, ROI_ROOT=roi_root,
                 TRAINING_DATA_ROOT=out_root)
    ns = vr._base_namespace(paths)
    for marker, index, source in cells:
        if marker in vr.PRE_CONFIG_MARKERS:
            exec(vr.definition_only(source, f'cell{index}')[0], ns)
    cfg = ns['PipelineConfig']()
    cfg.model_root, cfg.image_root, cfg.out_root = model_root, image_root, out_root
    cfg.data_root, cfg.quarantine_root = data_root, root / 'project' / '_to_delete'
    cfg.image_filename = 'control_fixture.tif'
    cfg.model_name, cfg.gold_pool_name = 'Fixture_1.0', 'Fixture_phase2'
    cfg.pixel_size_um, cfg.z_step_um = 0.1625, 2.0
    cfg.membrane_channel_idx, cfg.nucleus_channel_idx, cfg.npc_channel_idx = 0, 1, 2
    cfg.patch_size, cfg.n_z_context, cfg.z_floor = PATCH, 5, 2
    cfg.mosaic_tile_px, cfg.mosaic_overlap_px, cfg.holdout_margin_px = 128, 16, 8
    cfg.holdout_tile, cfg.holdout_timepoint, cfg.holdout_mode = (0, 1), 1, 'spatial'
    cfg.label_source, cfg.use_abnormal_head = 'roi', False
    cfg.gold_complete_timepoints = tuple(range(T))
    ns['cfg'] = cfg
    for marker, index, source in cells:
        if marker not in vr.PRE_CONFIG_MARKERS:
            exec(vr.definition_only(source, f'cell{index}')[0], ns)

    # --- Phase 2 gold pool ---------------------------------------------------
    pool = cfg.reviewed_root
    for folder in ('images', 'labels', 'weights'):
        (pool / folder).mkdir(parents=True, exist_ok=True)
    records = []
    for index, (name, t, z, cy, cx) in enumerate(PATCHES):
        lab, wgt, img = _build_patch(t, z, cy, cx)
        stem = _stem(t, z, index, cy, cx)
        np.save(pool / 'images' / f'img_{stem}.npy', img, allow_pickle=False)
        np.save(pool / 'labels' / f'lab_{stem}.npy', lab, allow_pickle=False)
        np.save(pool / 'weights' / f'wgt_{stem}.npy', wgt, allow_pickle=False)
        records.append(dict(stem=stem, t=t, z=z, droplet=index, cy=cy, cx=cx,
                            sample_class='gold_nucleus', sample_weight=1.0, fixture=name))
    pd.DataFrame(records).to_csv(pool / 'patch_metadata.csv', index=False)
    roi_paths = {kind: str(roi_root / f'{kind}RoiSet.zip') for kind in ('Nuclei', 'Droplet', 'NPC')}
    roi_paths = dict(nuc=roi_paths['Nuclei'], drop=roi_paths['Droplet'], npc=roi_paths['NPC'])
    (pool / 'manifest.json').write_text(json.dumps(dict(
        status='complete', gen_hash=cfg.gen_hash, gold_hash=cfg.gold_hash,
        n_patches=len(records), roi_paths=roi_paths,
        roi_sha256={k: hn._digest(v) for k, v in roi_paths.items()}), indent=2))
    (pool / 'z_cluster_summary.json').write_text(json.dumps(dict(n_validated=4)))
    cfg._z_label_cache = None

    original_rows = ns['p3_rows'](cfg)
    train, val = ns['p3_split'](cfg, verbose=False)
    by_fixture = {r['fixture']: r['stem'] for r in records}

    # --- committed review (the analogue of the existing 185+124 commit) ------
    review = out_root / 'fixture_review'
    (review / 'masks').mkdir(parents=True, exist_ok=True)
    candidates, rows = [], []
    for candidate_id, (t, z, y0, x0, y1, x1), decision in (
            ('T0_Z6_L1', (0, 6, 30, 30, 46, 46), 'artifact_negative'),      # conflicts with gold
            ('T2_Z7_L2', (2, 7, 200, 60, 216, 76), 'artifact_negative')):   # conflicts with nothing
        mask = np.ones((y1 - y0, x1 - x0), np.uint8)
        path = review / 'masks' / f'{candidate_id}.npy'
        np.save(path, mask, allow_pickle=False)
        candidates.append(dict(candidate_id=candidate_id, review_id=candidate_id, t=t, z=z,
                               label=1, nucleus_3d_id=1, y0=y0, x0=x0, y1=y1, x1=x1,
                               area_pixels=int(mask.sum()),
                               mask_file=f'masks/{candidate_id}.npy', mask_sha256=hn._digest(path)))
        rows.append(dict(candidate_id=candidate_id, t=t, z=z, decision=decision,
                         reviewer='fixture', notes=CONFIRMED, conflict_resolution='negative',
                         failure_type='cytoplasm', mask_file=f'masks/{candidate_id}.npy'))
    hn._json(review / 'candidates.json', dict(
        schema=1, status='candidates_only', raw_image=hn._identity(cfg.image_file),
        source_masks=hn._identity(cfg.image_file), pixel_size_um=float(cfg.pixel_size_um),
        channel_indices=dict(nls=1, npc=2, membrane=0), source_model=str(cfg.model_name),
        candidates=candidates))
    pd.DataFrame(rows).to_csv(review / 'review.csv', index=False)

    correction_pool = gc.build_corrections(review, out_root / 'fixture_corrections', cfg,
                                           original_rows, train)
    corrected = gc.corrected_rows(correction_pool, cfg, original_rows)
    negative_pool = hn.build_reviewed_negative_pool(
        review, out_root / 'fixture_hard_negatives', cfg, gold_rows=corrected,
        extract_input_stack=ns['extract_input_stack'], ensure_norm_stats=ns['ensure_norm_stats'])

    commit_dir = out_root / 'reviewed_training_commit_fixture'
    commit_dir.mkdir()
    hn._json(commit_dir / 'manifest.json', dict(
        status='complete', source_review=str(review), review=str(review),
        correction_pool=str(correction_pool), negative_pool=str(negative_pool),
        decisions=pd.DataFrame(rows).decision.value_counts().to_dict()))

    # --- training run directory ---------------------------------------------
    cfg.norm_stats = None
    cfg.model_name = cfg.model_name + '_hn' + hn._digest(negative_pool / 'manifest.json')[:12]
    cfg.__dict__.pop('_frozen_run_hash', None)
    cfg.stale_guard(write=True)
    hard_rows, _ = hn.load_reviewed_negative_rows(negative_pool, cfg)
    split = dict(train=len(train) + len(hard_rows), spatial=len(ns['SPLITS']['spatial']),
                 temporal=len(ns['SPLITS']['temporal']), val=len(val), independent=0)
    hn._json(cfg.run_dir / 'phase3_run.json', dict(
        pools={'gold': dict(root=str(pool), n_patches=len(records), gen_hash=cfg.gen_hash,
                            gold_hash=cfg.gold_hash)},
        label_source=cfg.label_source, class_weights=[1.0] * 6, mask_heads=['background'],
        has_z=True, seeds=[42], split=split, val_classes={}))
    return dict(root=root, cfg=cfg, run_dir=cfg.run_dir, commit_dir=commit_dir,
                pool=pool, correction_pool=correction_pool, negative_pool=negative_pool,
                by_fixture=by_fixture, review=review, out_root=out_root)


# ----------------------------------------------------------------- the tests

def _digests(paths):
    return {str(p): hn._digest(p) for p in paths}


def run_checks(verbose=True):
    results = []

    def check(name, condition, detail=''):
        results.append((name, bool(condition), detail))
        if verbose:
            print(f'{"PASS" if condition else "FAIL"}  {name}' + (f'  — {detail}' if detail else ''))

    workspace = Path(tempfile.mkdtemp(prefix='vulcan_audit_test_'))
    try:
        fixture = build_fixture(workspace)
        run_dir, commit_dir = fixture['run_dir'], fixture['commit_dir']

        # 1. Fresh-kernel recovery, with no notebook globals and no training.
        session = tpa.run_audit(run_dir, commit_dir, workers=1, verbose=False,
                                settings=tpa.NominationSettings(sample_per_timepoint=3,
                                                                negative_controls_per_timepoint=1))
        check('fresh-kernel recovery reproduces the run configuration',
              session.cfg.run_dir.resolve() == run_dir.resolve(), str(run_dir.name))
        check('reconstructed split matches phase3_run.json',
              session.provenance['split'] == json.loads((run_dir / 'phase3_run.json').read_text())['split'],
              str(session.provenance['split']))
        check('hard negatives appended to training only',
              session.provenance['n_hard_negative_rows'] > 0
              and all(r['source'] == 'gold' for r in session.eff.val))

        # 2. A patch corrected by the existing pool reads as corrected, not active.
        patches = session.patches.set_index('stem')
        corrected_stem = fixture['by_fixture']['I']
        check('existing correction is reflected in the effective labels',
              bool(patches.loc[corrected_stem, 'corrected'])
              and int(patches.loc[corrected_stem, 'supervised_nuclear_px']) == 0,
              f'supervised nuclear px now {int(patches.loc[corrected_stem, "supervised_nuclear_px"])}')
        regions = session.regions
        check('already-corrected object is not re-nominated as an active positive',
              not ((regions.t == 0) & (regions.z == 6)).any())

        # 3. Global indexing merges the same object seen in overlapping crops.
        artifact = regions[(regions.t == 2) & (regions.z == 5)]
        check('overlapping crops of one object form a single global region',
              len(artifact) == 1 and int(artifact.iloc[0].n_patches) == 2,
              f'{len(artifact)} region(s), {int(artifact.iloc[0].n_patches) if len(artifact) else 0} crops')
        check('signed / padded crop centres are parsed and padding excluded',
              bool((session.patches.cx < 0).any())
              and int(session.patches[session.patches.cx < 0].supervised_outside_image_px.iloc[0]) == 0,
              'out-of-image pixels are never counted as supervised')

        # Heldout-only objects are reported but never correctable.
        holdout_regions = regions[regions.n_correctable_patches == 0]
        check('holdout-only objects are reported and never correctable',
              len(holdout_regions) >= 2
              and set(holdout_regions.splits) <= {'spatial_holdout', 'temporal_holdout'},
              f'{len(holdout_regions)} holdout-only regions: {sorted(set(holdout_regions.splits))}')

        # 3b. Nuclear-mask semantics. A nucleus interior is legitimately 0 on the
        # edge head; testing heads independently would call every nucleus negative.
        lab = np.zeros((4, 4, 10), np.uint8)
        wgt = np.zeros((4, 4, 9), np.float32)
        lab[..., 4:6] = 0
        wgt[..., 4:6] = 1.0
        lab[1:3, 1:3, 4] = 1          # interior, so 0 on the edge head
        lab[0, 0, 5] = 1              # edge only
        stored, supervised, negative = tpa.nuclear_masks(lab, wgt)
        check('a nucleus interior is never counted as supervised NOT-nuclear',
              int(negative[1:3, 1:3].sum()) == 0 and int(supervised[1:3, 1:3].sum()) == 4
              and bool(negative[3, 3]) and bool(stored[0, 0]),
              f'{int(negative.sum())} negative pixels of 16')
        lab[1, 1, 4] = 1
        wgt[1, 1, 4:6] = 0.0          # stored positive left unweighted
        stored2, supervised2, negative2 = tpa.nuclear_masks(lab, wgt)
        check('an unweighted stored positive still blocks a negative',
              bool(stored2[1, 1]) and not bool(supervised2[1, 1]) and not bool(negative2[1, 1]))

        # 3c. Image signals separate a bright real nucleus from a dim artifact.
        scored = session.scored_regions.set_index('region_id')
        artifact_signal = float(scored.loc[artifact.iloc[0].region_id, 'nls_enrichment'])
        real_signals = scored[(scored.t == 0) & (scored.z == 4) & scored.nls_enrichment.notna()]
        check('genuine nuclei carry no cross-crop negative conflict',
              int(real_signals.negative_supervision_px.max()) == 0,
              'the conflict scan fires on contradictions, not on every nucleus')
        check('NLS enrichment ranks the dim artifact below the bright nuclei',
              len(real_signals) > 0 and artifact_signal < float(real_signals.nls_enrichment.min()),
              f'artifact {artifact_signal:.2f} vs real >= {float(real_signals.nls_enrichment.min()):.2f}')

        # 3d. The existing correction pool leaves live Z-offset targets behind.
        check('audit detects Z-offset targets left behind by the existing corrections',
              int(session.patches.z_supervised_outside_nucleus_px.sum()) > 0,
              f'{int(session.patches.z_supervised_outside_nucleus_px.sum())} px')

        # 4. Sentinel / weight / head integrity across the whole census.
        check('no supervised UNANNOTATED sentinel anywhere',
              int(session.patches.supervised_sentinel_px.sum()) == 0)
        check('every census patch passed the integrity checks',
              int((session.patches.problems.astype(str).str.len() > 0).sum()) == 0,
              str(sorted(set(session.patches.problems)) ))

        # 5. Review state: autosave, resume, and refusal to reuse a stale review.
        state = tpr.ReviewState(session, verbose=False)
        artifact_region = artifact.iloc[0].region_id
        state.save_decision(artifact_region, 'artifact_negative', 'tester', 'cytoplasm', 'fixture')
        version = state.version
        reopened = tpr.ReviewState(session, root=state.root, verbose=False)
        check('interrupted review resumes with its decisions intact',
              reopened.decisions[artifact_region]['decision'] == 'artifact_negative'
              and reopened.version == version)
        stale = session.candidates.copy()
        session.candidates = session.candidates.iloc[:-1]
        try:
            tpr.ReviewState(session, root=state.root, verbose=False)
            check('a changed candidate set refuses to reuse saved decisions', False)
        except RuntimeError:
            check('a changed candidate set refuses to reuse saved decisions', True)
        session.candidates = stale

        # 6. Non-correcting decisions write nothing.
        for region_id in list(state.decisions):
            if region_id == artifact_region:
                continue
            kind = state.decisions[region_id].get('kind')
            decision = 'mixed_redraw' if kind == 'flagged' else ('real' if kind == 'random_sample' else 'skip')
            state.save_decision(region_id, decision, 'tester')
        check('all candidates decided before commit is allowed', not state.pending())
        plan = tpr.preview_commit(session, state, verbose=False)
        check('mixed / uncertain / skipped produce no correction',
              set(plan['per_patch']) == {fixture['by_fixture']['E'], fixture['by_fixture']['F']},
              f'crops to overlay: {sorted(plan["per_patch"])}')
        check('no contradictory overlapping decisions', len(plan['contradictions']) == 0)

        # 7. Commit: propagation, byte-for-byte preservation, idempotence.
        everything = sorted(set(session.frame.lab) | set(session.frame.wgt) | set(session.frame.img))
        before = _digests([Path(p) for p in everything if p])
        committed = tpr.commit(session, state, verbose=False)
        manifest = json.loads((committed / 'manifest.json').read_text())
        check('commit completes with staging -> complete status', manifest['status'] == 'complete')
        check('confirmed artifact corrected in every eligible training crop',
              sorted(p['stem'] for p in manifest['patches'])
              == sorted([fixture['by_fixture']['E'], fixture['by_fixture']['F']]))
        check('no original pool file changed', _digests([Path(p) for p in everything if p]) == before,
              'validation, temporal holdout and the existing pools are byte-identical')
        check('commit is idempotent', tpr.commit(session, state, verbose=False) == committed)

        # 8. The overlay itself: exact pixels, nuclear heads only, Z de-supervised.
        item = manifest['patches'][0]
        stem = item['stem']
        row = session.frame.set_index('stem').loc[stem]
        original_lab = np.load(row.lab, allow_pickle=False)
        original_wgt = np.load(row.wgt, allow_pickle=False)
        new_lab = np.load(committed / item['lab'], allow_pickle=False)
        new_wgt = np.load(committed / item['wgt'], allow_pickle=False)
        mask = np.load(committed / 'masks' / (artifact_region + '.npy'), allow_pickle=False).astype(bool)
        record = state.record(artifact_region)
        origin_y, origin_x = int(row.cy) - HALF, int(row.cx) - HALF
        local = np.zeros((PATCH, PATCH), bool)
        y0, x0 = max(int(record.y0), origin_y), max(int(record.x0), origin_x)
        y1 = min(int(record.y1), origin_y + PATCH); x1 = min(int(record.x1), origin_x + PATCH)
        local[y0 - origin_y:y1 - origin_y, x0 - origin_x:x1 - origin_x] = mask[
            y0 - int(record.y0):y1 - int(record.y0), x0 - int(record.x0):x1 - int(record.x0)]
        check('nuclear heads 4/5/6 set to target 0 with explicit weight',
              bool((new_lab[..., 4:7][local] == 0).all())
              and bool((new_wgt[..., 4:7][local] == 1.0).all()))
        check('Z-offset de-supervised over the corrected pixels, never invented',
              bool((new_lab[..., 8][local] == 255).all()) and bool((new_wgt[..., 8][local] == 0).all())
              and int((original_wgt[..., 8] > 0).sum()) > 0)
        untouched = ~local
        check('non-nuclear heads and uncorrected pixels are unchanged',
              bool((new_lab[..., :4] == original_lab[..., :4]).all())
              and bool((new_lab[..., 9] == original_lab[..., 9]).all())
              and bool((new_lab[..., 4:7][untouched] == original_lab[..., 4:7][untouched]).all())
              and bool((new_wgt[..., :4] == original_wgt[..., :4]).all()))
        check('a whole patch is never zeroed for a partial artifact',
              int((new_lab[..., 1] == 1).sum()) == int((original_lab[..., 1] == 1).sum()))

        # 9. The committed overlay reads back, and refuses stale inputs.
        rows = tpr.audit_corrected_rows(committed, session.cfg, session.eff.effective_train)
        swapped = [r for r in rows if r['stem'] == stem][0]
        check('committed overlay is readable through audit_corrected_rows',
              Path(swapped['lab']).parent.parent == committed)
        poisoned = np.load(committed / item['lab'], allow_pickle=False)
        poisoned[0, 0, 0] = 7
        np.save(committed / item['lab'], poisoned, allow_pickle=False)
        try:
            tpr.audit_corrected_rows(committed, session.cfg, session.eff.effective_train)
            check('a modified overlay file is rejected by its hash', False)
        except ValueError:
            check('a modified overlay file is rejected by its hash', True)

        # 10. Contradiction abort, on a fresh review of the same candidates.
        second = tpr.ReviewState(session, root=session.cache_dir / 'review_contradiction', verbose=False)
        for region_id in second.decisions:
            second.save_decision(region_id, 'skip', 'tester')
        second.save_decision(artifact_region, 'artifact_negative', 'tester')
        clash_id = 'synthetic_overlap'
        second.decisions[clash_id] = dict(decision='real', reviewer='tester', failure_type='',
                                          notes='', updated='', kind='flagged')
        second.candidates = pd.concat([second.candidates, pd.DataFrame([
            dict(second.candidates[second.candidates.region_id.eq(artifact_region)].iloc[0],
                 region_id=clash_id)])], ignore_index=True)
        np.save(second.masks_dir / (clash_id + '.npy'),
                second.mask(artifact_region), allow_pickle=False)
        clashes = tpr.contradictions(session, second)
        check('contradictory resolutions on the same pixels are detected',
              len(clashes) == 1 and int(clashes.iloc[0].shared_px) > 0,
              f'{len(clashes)} clash(es)')
        try:
            tpr.commit(session, second, verbose=False)
            check('a contradictory review aborts the commit', False)
        except ValueError as exc:
            check('a contradictory review aborts the commit', 'Contradictory' in str(exc))

        # 11. Augmentation reproduction on real patches with a fixed seed.
        check_aug = tpa.augmentation_check(session, seed=11, count=2)
        table = check_aug['table']
        check('augmentation keeps mask targets discrete and sentinels unsupervised',
              bool(table.mask_values_still_discrete.all())
              and int(table.supervised_sentinel_after_aug.sum()) == 0
              and bool(table.input_in_unit_range.all()))

        # 12. Census cache resume.
        shard = next((session.cache_dir / 'census').glob('patches_*.csv'))
        stamp_count = len(list((session.cache_dir / 'census').glob('inputs_*.json')))
        again = tpa.open_audit(run_dir, commit_dir, cache_dir=session.cache_dir, verbose=False)
        tpa.census(again, workers=1, verbose=False)
        check('census shards are reused rather than recomputed',
              len(again.patches) == len(session.patches) and shard.is_file()
              and stamp_count == len(list((session.cache_dir / 'census').glob('inputs_*.json'))))
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    failed = [name for name, ok, _ in results if not ok]
    if verbose:
        print(f'\n{len(results) - len(failed)}/{len(results)} checks passed.')
        if failed:
            print('FAILED:')
            for name in failed:
                print('  -', name)
    return results


def test_training_patch_audit():
    results = run_checks(verbose=False)
    failed = [name for name, ok, _ in results if not ok]
    assert not failed, 'failed checks: ' + ', '.join(failed)


if __name__ == '__main__':
    outcome = run_checks()
    sys.exit(1 if any(not ok for _, ok, _ in outcome) else 0)
