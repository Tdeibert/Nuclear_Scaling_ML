"""Fresh-kernel recovery of a Vulcan training run's configuration and effective labels.

Read-only. Nothing here trains, runs inference, starts a background job, rebuilds a
pool, or writes into an existing pool.

The notebook's authoritative definitions (PipelineConfig, pool readers, split,
packed loader, augmentation) are loaded by AST filter: only definitions and
constant assignments are compiled, so a generation/training/export trigger cannot
fire as a side effect of loading a reader. Nothing is imported from the notebook
as a module and no notebook stage cell runs.

Effective label order, matching the production hooks:
    original pool rows -> gold_corrections overrides by stem -> split
    -> append eligible hard-negative rows to TRAINING ONLY.
Validation and the temporal holdout are never corrected and never extended.
"""
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / 'Nuclear_Scaling_Core'
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import reviewed_hard_negatives as hn
import gold_artifact_corrections as gc

DEFAULT_NOTEBOOK = HERE / 'vulcan_training_2_5_3.ipynb'
DEFAULT_RUN_DIR = Path('/home/tdeibert/Projects/Nuclear_Scaling/Models/runs/'
                       'Vulcan_2.5.4_hardneg_hn2392da6c18e0_iezqw_a0cf3b2a13')
DEFAULT_COMMIT_DIR = Path('/data/user/tdeibert/Nuclear_Scaling/Inputs/Training_Data/'
                          'reviewed_training_commit_20261013_121152_980805')

# Storage layout asserted against the loaded notebook constants; see check_layout().
NUCLEUS_HEADS = (4, 5, 6)      # interior, edge, equatorial
Z_OFFSET_HEAD = 8
DROPLET_SOURCE_CHANNEL = 9     # metadata, not a loss head

# Notebook cells carrying the definitions this audit reuses, in notebook order.
# Each marker must match exactly one cell. Later cells legitimately redefine
# earlier names (cell 43 supersedes the Phase 2 list_phase2_patch_files), so the
# order is the notebook's own top-to-bottom order.
DEFINITION_MARKERS = (
    '# A3. Label storage schema',
    '# A4. PipelineConfig',
    '# B1. Channel access, normalization, compact masks',
    '# B10. Patch extraction, file naming and discovery',
    'Phase 2 notebook cell source. Embedded by build_vulcan_phase2_notebook.py',
    '# F1. Phase 3 layout, split, sampling and class weights (definitions)',
    '# F2. Phase 3 data pipeline and augmentation (definitions)',
    # Anchored (^) because cell 33 defines an earlier list_phase2_patch_files with
    # the same docstring; this later cell supersedes it, as a notebook run would.
    '^def list_phase2_patch_files(cfg=cfg):',
)
# Cells loaded before a cfg exists. Everything after them uses `cfg=cfg` default
# arguments, which bind at definition time and therefore need the audit cfg first.
PRE_CONFIG_MARKERS = DEFINITION_MARKERS[:2]

# Calls permitted on the right-hand side of a kept top-level assignment. Anything
# else makes the assignment a dropped statement rather than a silent side effect.
SAFE_CALLS = frozenset({
    'len', 'enumerate', 'range', 'tuple', 'list', 'dict', 'set', 'frozenset',
    'sorted', 'zip', 'str', 'int', 'float', 'bool', 're.compile', 'dict.fromkeys',
})


def _call_name(node):
    parts = []
    target = node.func
    while isinstance(target, ast.Attribute):
        parts.append(target.attr)
        target = target.value
    if not isinstance(target, ast.Name):
        return None
    parts.append(target.id)
    return '.'.join(reversed(parts))


def _immediate_calls(node):
    """Calls that run when this statement executes.

    Lambda bodies are skipped: they are deferred, so the head table's
    `lambda c: c.z_supervised_labels()` is a definition, not a side effect.
    """
    found = []
    stack = [node]
    while stack:
        current = stack.pop()
        for child in ast.iter_child_nodes(current):
            if isinstance(child, ast.Lambda):
                continue
            if isinstance(child, ast.Call):
                found.append(_call_name(child))
            stack.append(child)
    return found


def definition_only(source, where):
    """Compile only definitions and constant assignments out of a notebook cell."""
    tree = ast.parse(source)
    keep, dropped = [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Import, ast.ImportFrom)):
            keep.append(node)
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            keep.append(node)            # module/cell docstring
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if all(name in SAFE_CALLS for name in _immediate_calls(node)):
                keep.append(node)
                continue
        dropped.append(f'{where}:{getattr(node, "lineno", "?")} {type(node).__name__}')
    module = ast.Module(body=keep, type_ignores=[])
    return compile(ast.fix_missing_locations(module), f'<notebook {where}>', 'exec'), dropped


def select_definition_cells(notebook, markers=DEFINITION_MARKERS):
    """(marker, cell index, source) for each marker, in notebook order."""
    cells = json.loads(Path(notebook).read_text(encoding='utf8'))['cells']
    sources = [''.join(c['source']) if c['cell_type'] == 'code' else '' for c in cells]
    chosen = []
    for marker in markers:
        if marker.startswith('^'):
            hits = [i for i, s in enumerate(sources) if s.startswith(marker[1:])]
        else:
            hits = [i for i, s in enumerate(sources) if marker in s]
        if len(hits) != 1:
            raise RuntimeError(f'Notebook marker {marker!r} matched {len(hits)} code cells; '
                               'the notebook changed — update DEFINITION_MARKERS.')
        chosen.append((marker, hits[0], sources[hits[0]]))
    if [i for _, i, _ in chosen] != sorted(i for _, i, _ in chosen):
        raise RuntimeError('DEFINITION_MARKERS are not in notebook order.')
    return chosen


def _base_namespace(paths):
    """The subset of the notebook's A1 imports the loaded definitions need.

    TensorFlow is deliberately absent: only p3_dataset touches it, the audit never
    builds a tf.data pipeline, and importing it would pull a GPU runtime into a
    read-only audit. Calling p3_dataset from this namespace raises NameError.
    """
    import hashlib
    import os
    import math
    import time
    import gc as _gc
    import itertools
    import copy
    import warnings
    import tifffile as tiff
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from scipy import ndimage
    from scipy.ndimage import distance_transform_edt, affine_transform, gaussian_filter
    from skimage import filters, morphology, measure, segmentation
    from skimage.morphology import h_maxima, disk as _disk
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from sklearn.cluster import DBSCAN
    import roifile
    from dataclasses import dataclass as _dataclass, field as _field
    try:
        import cv2
    except ImportError:
        cv2 = None
    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(x, **kwargs):
            return x
    from nuclear_scaling.acquisition_metadata import Acquisition, imagej_calibration
    from nuclear_scaling.roi_zcluster import assign_z

    ns = dict(
        np=np, pd=pd, Path=Path, json=json, hashlib=hashlib, re=re, os=os, math=math,
        time=time, gc=_gc, sys=sys, itertools=itertools, copy=copy, warnings=warnings,
        tiff=tiff, plt=plt, ListedColormap=ListedColormap, ndimage=ndimage,
        distance_transform_edt=distance_transform_edt, affine_transform=affine_transform,
        gaussian_filter=gaussian_filter, filters=filters, morphology=morphology,
        measure=measure, segmentation=segmentation, h_maxima=h_maxima, _disk=_disk,
        watershed=watershed, peak_local_max=peak_local_max, DBSCAN=DBSCAN,
        roifile=roifile, cv2=cv2, tqdm=tqdm, dataclass=_dataclass, field=_field,
        Acquisition=Acquisition, imagej_calibration=imagej_calibration, assign_z=assign_z,
        SEED=42,
    )
    ns.update(paths)
    return ns


@dataclass
class Recovered:
    run_dir: Path
    notebook: Path
    cfg: object
    ns: dict
    run_record: dict
    cached_config: dict
    gold_root: Path
    paths: dict
    dropped_statements: list = field(default_factory=list)
    norm_stats_source: object = None

    @property
    def signature(self):
        return hn._training_signature(self.cfg)


def recover_config(run_dir=DEFAULT_RUN_DIR, *, notebook=DEFAULT_NOTEBOOK, verbose=True):
    """Rebuild the run's own PipelineConfig from its saved config.json.

    Fresh-kernel safe: needs no notebook globals, no acquisition sidecar, and no
    upstream generation cell. Every reconstructed value is checked against the
    cached config.json and against the run directory's own name, so a silently
    different configuration cannot masquerade as this run.
    """
    run_dir = Path(run_dir).resolve()
    notebook = Path(notebook).resolve()
    for name in ('config.json', 'phase3_run.json'):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(run_dir / name)
    cached = json.loads((run_dir / 'config.json').read_text())
    record = json.loads((run_dir / 'phase3_run.json').read_text())

    gold_root = Path(record['pools']['gold']['root']).resolve()
    out_root = gold_root.parent
    data_root = out_root.parent.parent
    image_root = data_root / 'Inputs' / 'Raw_Images' / 'Condition' / 'Control'
    model_root = run_dir.parent.parent
    if out_root.name != 'Training_Data' or run_dir.parent.name != 'runs' or model_root.name != 'Models':
        raise ValueError('Unexpected run/pool layout; pass an explicit run_dir for this project.')
    project_root = next((d for d in (notebook.parent, *notebook.parents)
                         if (d / 'Repository_Structure.json').is_file()), None)
    if project_root is None:
        raise FileNotFoundError('Repository_Structure.json not found above ' + str(notebook))
    paths = dict(PROJECT_ROOT=project_root, CODE_ROOT=project_root / 'Code_Repository',
                 DATA_ROOT=data_root, CONTROL_DATA_DIR=image_root,
                 ROI_ROOT=image_root / 'Regions_of_Interest', TRAINING_DATA_ROOT=out_root)

    cells = select_definition_cells(notebook)
    ns = _base_namespace(paths)
    dropped = []
    pre = [c for c in cells if c[0] in PRE_CONFIG_MARKERS]
    post = [c for c in cells if c[0] not in PRE_CONFIG_MARKERS]
    for marker, index, source in pre:
        code, lost = definition_only(source, f'cell{index}')
        exec(code, ns)
        dropped += lost

    cfg = ns['PipelineConfig']()
    cfg.model_root, cfg.image_root, cfg.out_root = model_root, image_root, out_root
    cfg.data_root, cfg.quarantine_root = data_root, project_root / '_to_delete'
    unknown = sorted(set(cached) - set(cfg.__dataclass_fields__))
    if unknown:
        raise ValueError('Cached config has fields this PipelineConfig does not define: ' + str(unknown))
    for key, text in cached.items():
        try:
            setattr(cfg, key, ast.literal_eval(text))
        except (ValueError, SyntaxError):
            setattr(cfg, key, text)
    cfg.norm_stats = None
    cfg._z_label_cache = None

    # Round-trip check: the live configuration must serialise back to the bytes the
    # training stage cached, and must still hash to this run directory's own name.
    live = {k: str(v) for k, v in cfg.__dict__.items()
            if k not in ('norm_stats', '_frozen_run_hash', '_z_label_cache',
                         'max_parallel_workers', 'data_root', 'model_root',
                         'image_root', 'out_root', 'quarantine_root')}
    diff = {k: (cached.get(k), live[k]) for k in live if cached.get(k) != live[k]}
    missing = sorted(set(cached) - set(live))
    if diff or missing:
        raise RuntimeError(f'Recovered configuration disagrees with {run_dir/"config.json"}: '
                           f'differing {diff}, unrecovered {missing}')
    if cfg.run_dir.resolve() != run_dir:
        raise RuntimeError(f'Recovered configuration hashes to {cfg.run_dir}, not {run_dir}. '
                           'Its fields do not describe this run.')
    if cfg.reviewed_root.resolve() != gold_root:
        raise RuntimeError(f'Recovered gold pool {cfg.reviewed_root} is not the pool this run '
                           f'recorded ({gold_root}).')
    if not cfg.image_file.is_file():
        raise FileNotFoundError(cfg.image_file)
    cfg._frozen_run_hash = cfg.run_hash

    ns['cfg'] = cfg
    for marker, index, source in post:
        code, lost = definition_only(source, f'cell{index}')
        exec(code, ns)
        dropped += lost

    rec = Recovered(run_dir=run_dir, notebook=notebook, cfg=cfg, ns=ns, run_record=record,
                    cached_config=cached, gold_root=gold_root, paths=paths,
                    dropped_statements=dropped)
    check_layout(rec)
    if verbose:
        print('Recovered run      :', run_dir.name)
        print('  model_name       :', cfg.model_name, '| caps', cfg.caps_tag, '| run_hash', cfg.run_hash)
        print('  gold pool        :', cfg.reviewed_root.name, '|', record['pools']['gold'].get('n_patches'), 'stored patches')
        print('  classical pool   :', cfg.training_root.name,
              '(present)' if cfg.training_root.exists() else '(absent — gold-only census)')
        print('  raw image        :', cfg.image_file)
        print('  label_source     :', cfg.label_source, '| holdout_mode', cfg.holdout_mode)
        print('  active heads     :', cfg.active_heads())
        print('  notebook defs    :', len(cells), 'cells loaded definition-only;',
              len(dropped), 'non-definition statements skipped')
        print('No training, inference, or writes. Definitions only.')
    return rec


def check_layout(rec):
    """Refuse a notebook whose storage layout is not the one this audit assumes."""
    ns = rec.ns
    expected = ['background', 'droplet_interior', 'droplet_edge', 'npc', 'nucleus_interior',
                'nucleus_edge', 'nucleus_equatorial', 'abnormal_nucleus', 'z_offset']
    if ns['HEAD_NAMES'] != expected:
        raise RuntimeError('Head order changed: ' + str(ns['HEAD_NAMES']))
    if [ns['HEAD_INDEX'][n] for n in ('nucleus_interior', 'nucleus_edge', 'nucleus_equatorial')] != list(NUCLEUS_HEADS):
        raise RuntimeError('Nuclear head indices changed.')
    if ns['HEAD_INDEX']['z_offset'] != Z_OFFSET_HEAD or ns['DROPLET_SOURCE_IDX'] != DROPLET_SOURCE_CHANNEL:
        raise RuntimeError('Z-offset / droplet_source channel moved.')
    if ns['N_LABEL_CHANNELS'] != 10 or ns['UNANNOTATED'] != 255 or ns['Z_OFFSET_SCALE'] != 10.0:
        raise RuntimeError('Label storage schema changed.')
    cfg = rec.cfg
    if cfg.n_channels != 3 or cfg.n_z_context != 5 or cfg.patch_size % 2:
        raise RuntimeError('Expected five-plane, three-channel inputs and an even patch size.')
    if cfg.normalization_mode != 'global_t':
        raise RuntimeError('Expected global_t normalization.')
    return True


def attach_norm_stats(rec, verbose=True):
    """Load the cached global_t percentiles. Never computes: that rescans the stack.

    Without them raw context can still be displayed with explicit per-view scaling,
    but it cannot be compared to the stored normalized inputs.
    """
    cfg = rec.cfg
    baseline = re.sub(r'_hn[0-9a-f]{12}$', '', str(cfg.model_name))
    candidates = [cfg.norm_stats_path,
                  rec.gold_root / 'qc' / 'norm_stats.json',
                  cfg.out_root / f'training_patches_{baseline}_{cfg.gen_hash}' / 'qc' / 'norm_stats.json']
    for path in candidates:
        if Path(path).is_file():
            cfg.norm_stats = rec.ns['load_norm_stats'](path)
            rec.norm_stats_source = dict(path=str(path), sha256=hn._digest(path),
                                         pairs=len(cfg.norm_stats))
            if verbose:
                print('norm stats       :', len(cfg.norm_stats), '(t,channel) pairs <-', path)
            return cfg.norm_stats
    rec.norm_stats_source = None
    if verbose:
        print('norm stats       : not cached at', [str(p) for p in candidates],
              '\n                   raw views will be shown with explicit per-view scaling only.')
    return None


@dataclass
class Effective:
    """The label set the audited run actually trained on."""
    original_rows: list
    rows: list                      # original rows with gold_corrections applied
    train: list
    val: list
    spatial: list
    temporal: list
    hard_negative_rows: list
    commit: dict
    correction_manifest: dict
    negative_manifest: dict
    provenance: dict

    @property
    def effective_train(self):
        return list(self.train) + list(self.hard_negative_rows)


def effective_rows(rec, commit_dir=DEFAULT_COMMIT_DIR, *, verbose=True):
    """Reproduce the audited run's effective labels and splits.

    Uses the production readers: gold_artifact_corrections.enable_corrections wraps
    the pool reader exactly as the training hook did, then the notebook's own
    p3_split runs on the corrected rows, then validated hard-negative rows are
    appended to TRAINING only. Validation and the temporal holdout are untouched.
    """
    cfg, ns = rec.cfg, rec.ns
    if ns.get('_AUDIT_CORRECTIONS_APPLIED'):
        raise RuntimeError('This recovered namespace already has the correction reader '
                           'installed. Call recover_config() again for a second pass '
                           'rather than wrapping the reader twice.')
    commit_dir = Path(commit_dir).resolve()
    commit = json.loads((commit_dir / 'manifest.json').read_text())
    if commit.get('status') != 'complete':
        raise ValueError(f'{commit_dir}: commit status is {commit.get("status")!r}, not complete')
    correction_pool = Path(commit['correction_pool']).resolve() if commit.get('correction_pool') else None
    negative_pool = Path(commit['negative_pool']).resolve() if commit.get('negative_pool') else None

    original = ns['p3_rows'](cfg)
    if not original:
        raise ValueError('The gold pool produced no rows; check pool hashes and paths.')

    rows = original
    correction_manifest = {}
    if correction_pool is not None:
        # corrected_rows re-verifies both the baseline digests and the correction
        # digests, so a changed original or a changed override aborts here.
        rows = gc.corrected_rows(correction_pool, cfg, original)
        correction_manifest = json.loads((correction_pool / 'manifest.json').read_text())
        ns['p3_rows'] = lambda active_cfg=cfg, _p=correction_pool, _b=ns['p3_rows']: gc.corrected_rows(_p, active_cfg, _b(active_cfg))
        ns['_AUDIT_CORRECTIONS_APPLIED'] = True

    train, val = ns['p3_split'](cfg, verbose=False)
    splits = ns['SPLITS']
    spatial, temporal = list(splits['spatial']), list(splits['temporal'])

    hard_rows, negative_manifest = [], {}
    if negative_pool is not None:
        hard_rows, negative_manifest = hn.load_reviewed_negative_rows(negative_pool, cfg)
        tag = '_hn' + hn._digest(negative_pool / 'manifest.json')[:12]
        if tag not in str(cfg.model_name):
            raise ValueError(f'Negative pool fingerprint {tag} is absent from model name '
                             f'{cfg.model_name}; this commit did not produce this run.')

    recorded = rec.run_record.get('split', {})
    observed = dict(train=len(train) + len(hard_rows), spatial=len(spatial),
                    temporal=len(temporal), val=len(val),
                    independent=len(splits.get('independent', [])))
    mismatch = {k: (recorded.get(k), v) for k, v in observed.items() if recorded.get(k) != v}
    if mismatch:
        raise RuntimeError('Reconstructed split does not match phase3_run.json '
                           f'(recorded, reconstructed): {mismatch}')

    provenance = dict(
        run_dir=str(rec.run_dir), commit_dir=str(commit_dir),
        training_signature=hn._training_signature(cfg),
        raw_image=hn._identity(cfg.image_file),
        gold_pool=str(cfg.reviewed_root),
        gold_pool_manifest_sha256=hn._digest(cfg.reviewed_root / 'manifest.json'),
        commit_manifest_sha256=hn._digest(commit_dir / 'manifest.json'),
        review_csv_sha256=hn._digest(Path(commit['review']) / 'review.csv'),
        candidates_sha256=hn._digest(Path(commit['review']) / 'candidates.json'),
        correction_pool=str(correction_pool) if correction_pool else None,
        correction_manifest_sha256=hn._digest(correction_pool / 'manifest.json') if correction_pool else None,
        negative_pool=str(negative_pool) if negative_pool else None,
        negative_manifest_sha256=hn._digest(negative_pool / 'manifest.json') if negative_pool else None,
        n_original_rows=len(original), n_corrected_patches=len(correction_manifest.get('patches', [])),
        n_hard_negative_rows=len(hard_rows), split=observed,
    )
    eff = Effective(original_rows=original, rows=rows, train=train, val=val, spatial=spatial,
                    temporal=temporal, hard_negative_rows=hard_rows, commit=commit,
                    correction_manifest=correction_manifest, negative_manifest=negative_manifest,
                    provenance=provenance)
    if verbose:
        print('Effective labels   :', len(original), 'stored pool rows;',
              len(correction_manifest.get('patches', [])), 'gold patches overridden by corrections')
        print('  training         :', len(train), 'corrected pool rows +', len(hard_rows),
              'reviewed hard negatives =', len(train) + len(hard_rows))
        print('  spatial holdout  :', len(spatial), '| temporal holdout:', len(temporal),
              '| validation used by this run:', len(val))
        print('  split matches phase3_run.json exactly.')
    return eff


def row_frame(eff):
    """One row per effective patch with split membership and the holdout verdict."""
    records = []
    membership = {}
    for name, rows in (('train', eff.train), ('spatial_holdout', eff.spatial),
                       ('temporal_holdout', eff.temporal), ('hard_negative_train', eff.hard_negative_rows)):
        for r in rows:
            membership[(r.get('source', 'hard_negative'), r['stem'])] = name
    corrected = {p['stem'] for p in eff.correction_manifest.get('patches', [])}
    originals = {(r['source'], r['stem']): r for r in eff.original_rows}
    for r in list(eff.rows) + list(eff.hard_negative_rows):
        source = r.get('source', 'hard_negative')
        key = (source, r['stem'])
        base = originals.get(key, {})
        records.append(dict(
            source=source, stem=r['stem'], t=int(r['t']), z=int(r['z']),
            cy=int(r['cy']), cx=int(r['cx']),
            sample_class=r.get('sample_class', 'unknown'),
            split=membership.get(key, 'dropped_seam_or_classical'),
            corrected=r['stem'] in corrected and source == 'gold',
            img=str(r.get('img', '')), lab=str(r['lab']), wgt=str(r['wgt']),
            original_lab=str(base.get('lab', r['lab'])), original_wgt=str(base.get('wgt', r['wgt'])),
        ))
    frame = pd.DataFrame(records)
    if frame.stem.duplicated().any():
        raise ValueError('Duplicate patch stems across the effective pools.')
    return frame
