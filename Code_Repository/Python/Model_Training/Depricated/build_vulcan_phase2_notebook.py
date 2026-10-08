"""Reproducibly assemble the self-contained Phase 2 notebook; source untouched."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'vulcan_training_3.2.ipynb'
TARGET = ROOT/'vulcan_training_2_5_3.ipynb'
nb = json.loads(SOURCE.read_text(encoding='utf8'))

def code(text):
    return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=text.splitlines(True))

def md(text):
    return dict(cell_type='markdown',metadata={},source=text.splitlines(True))

nb['cells'][0] = md('''# Vulcan Training 2.5.3 — Phase 2 label generation

This notebook builds gold and classical image/label/weight pools. Training and
inference remain Phase 3. The source `vulcan_training_3.2.ipynb` is unchanged.
Historical diagnostic outputs are preserved in the optional appendix and are
not evidence that this build has passed E1/E2.

Run the definitions and E1/E2, inspect the read-only Phase 2 preview, then call
`p2_accept_checks()` after accepting current geometry and normalization. Enable
the gold write stage only after reviewing the labels. All write flags default to
False; full dataset generation has not been run as part of creating this file.

## Label contract

- Gold NLS ROIs are complete for the explicitly declared `gold_complete_timepoints`.
  Their exact masks override classical segmentation in **both** patch pools.
- Otsu + enrichment 1.20 + solidity 0.80 remains the classical detector. Component
  splitting and the proposed NPC-envelope gate are not production methods.
- The reported 34.5% early object rejection was accepted as a coverage tradeoff.
  It is conditional on the audit's assignable gold objects, not whole-dataset recall.
- Detector rejection/absence does not establish a negative. Rejected raw/cleaned
  foreground remains unknown outside complete gold annotation.
- Empty droplets and empty focal planes have distinct sampling metadata. A one-plane
  spatial guard around gold NLS observations remains unknown. Every preview and
  training center obeys `z >= z_floor` (default 6; z0-5 excluded, z6 included).
  Lower planes remain available as 2.5D input neighbors and as gold evidence for
  droplet occupancy, z-column identity, and boundary guards.
- Droplet RANSAC runs on the complete inventory. Accepted manual droplet circles
  replace matching fitted circles and receive higher weights.
- Paired NPC outer ROIs minus NLS interiors are manual NPC labels. Other NPC labels
  are calculated from raw NPC with lower weight. Early orphan hulls dilated by
  1.5 um are unknown for nucleus/NPC; blur is unknown for NPC; late puncta are negative.
- Ten uint8 label channels are retained; aligned nine-head float32 weight sidecars
  encode source confidence and zero out unknown targets. Phase 3 must use these weights.
- Gold and classical nucleus z targets intentionally use the 2.0 um acquisition
  step. Droplet sphere geometry independently uses the provisional calibrated
  2.18 um step. Failed columns retain masks and lose only axial supervision;
  no sphere-derived nucleus absence or cap negatives are emitted.

Sampling uses deterministic droplet centres on supported planes plus every gold
nucleus centre and manual droplet centre. Empty categories have separate sampling
fractions. This build is sequential to bound geometry/cache memory; legacy jitter,
parallel-worker and inferred-cap options do not control these new writers.

Pool writes refuse existing contents. `manifest.json` stays `building` until all
image/label/weight triplets and metadata are complete. ROI source hashes, configuration,
candidate QC, droplet ROI flags, NPC orphan flags/ZIP and z-column flags are saved.
''')

c = ''.join(nb['cells'][8]['source'])
c = c.replace('algorithm_version: str = "3.2"','algorithm_version: str = "2.5.3-phase2.2"')
c = c.replace('model_name: str = "Vulcan_3.2"','model_name: str = "Vulcan_2.5.3"')
c = c.replace('gold_pool_name: str = "Vulcan_3.2"','gold_pool_name: str = "Vulcan_2.5.3_phase2"')
c = c.replace('(refractive-index mismatch makes them differ)', '(physical cause is not established)')
c = c.replace('# Every field here is read by code in this notebook. A field whose code arrives\n# in a later build phase says so in its comment ("phase 2"); nothing else is\n# allowed to sit here unused.', '# Shared configuration includes Phase 3 and legacy generation options.\n# The Phase 2 sampling contract in the introduction lists the active writer behavior.')
extra = '''    # ---- Phase 2 source precedence, uncertainty, and sampling ----
    gold_complete_timepoints: tuple = tuple(range(10))
    gold_roi_weight: float = 1.0
    empty_plane_guard_dz: int = 1
    empty_droplet_sample_fraction: float = 1.0
    empty_focal_sample_fraction: float = 1.0
    npc_early_max_t: int = 2
    npc_roi_pair_frac: float = 0.50
    npc_roi_near_frac: float = 0.50
    npc_roi_blur_dz: int = 2
    npc_orphan_dilate_um: float = 1.5
    z_target_step_um: float = 2.0
    label_weight_dtype: str = "float16"

'''
c = c.replace('    def __post_init__(self):', extra+'    def __post_init__(self):')
fields = ('gold_complete_timepoints','gold_roi_weight','empty_plane_guard_dz',
          'empty_droplet_sample_fraction','empty_focal_sample_fraction','npc_early_max_t',
          'npc_roi_pair_frac','npc_roi_near_frac','npc_roi_blur_dz','npc_orphan_dilate_um',
          'classical_filler_weight','negative_patch_weight','seed','equatorial_band_planes',
          'nucleus_channel_idx','npc_channel_idx','membrane_channel_idx','n_channels','image_root',
          'z_target_step_um','label_weight_dtype')
c = c.replace('    _GEN_FIELDS = _INPUT_FIELDS + _SHARED_LABEL_FIELDS + _GEOM_FIELDS + _CLASSICAL_FIELDS',
              '    _PHASE2_FIELDS = '+repr(fields)+'\n'
              '    _GEN_FIELDS = tuple(dict.fromkeys(_INPUT_FIELDS + _SHARED_LABEL_FIELDS + _GEOM_FIELDS + _CLASSICAL_FIELDS + _GOLD_FIELDS + _PHASE2_FIELDS))')
c = c.replace('    _GOLD_HASH_FIELDS = _INPUT_FIELDS + _SHARED_LABEL_FIELDS + _GEOM_FIELDS + _GOLD_FIELDS',
              '    _GOLD_HASH_FIELDS = _GEN_FIELDS  # both pools contain gold precedence and calculated NPC')
nb['cells'][8] = code(c)

controls = ''.join(nb['cells'][10]['source'])
controls += '''\n# Phase 2 writes and historical experiments are opt-in.
RUN_LEGACY_DIAGNOSTICS = False
RUN_PHASE2_PREVIEW = False
RUN_GOLD_IMPORT = False
RUN_CLASSICAL_GENERATION = False
PHASE2_TIMEPOINTS = None  # None means every TIFF timepoint for a write stage
'''
nb['cells'][10] = code(controls)
nb['cells'][12]['source']=''.join(nb['cells'][12]['source']).replace(
    'return tiff.memmap(str(path))','return tiff.memmap(str(path), mode="r")').splitlines(True)
# Both calculated and manual NPC labels exclude the authoritative NLS interior.
nb['cells'][22]['source']=''.join(nb['cells'][22]['source']).replace(
    'mask = (np.asarray(npc_crop_raw) > threshold) & shell',
    'mask = (np.asarray(npc_crop_raw) > threshold) & shell & ~nucleus').splitlines(True)

# Stamp actual E1/E2 executions rather than accepting old live-kernel caches.
for index, flag, name, hash_expr in [
    (28,'RUN_GEOMETRY_QC','E1_CONFIG_HASH',"cfg._hash_of(cfg._GEOM_FIELDS + ('pixel_size_um',))"),
    (29,'RUN_NORM_STATS_CHECK','E2_CONFIG_HASH','cfg._hash_of(cfg._INPUT_FIELDS)'),
]:
    text=''.join(nb['cells'][index]['source'])
    text += f'\nif {flag}:\n    {name} = {hash_expr}\n'
    nb['cells'][index]=code(text)

appendix=[]
for index in [25,26,30,31,32,33,34,35,36,37,38]:
    cell=nb['cells'][index]
    if cell['cell_type']=='code':
        text=''.join(cell['source'])
        cell['source']=('if RUN_LEGACY_DIAGNOSTICS:\n'+
                        ''.join('    '+line+'\n' for line in text.splitlines())).splitlines(True)
        cell['execution_count']=None
        cell.setdefault('metadata',{})['historical_output']=True
    appendix.append(cell)
nb['cells']=[cell for i,cell in enumerate(nb['cells']) if i not in [25,26,30,31,32,33,34,35,36,37,38]]

module=(ROOT/'vulcan_phase2.py').read_text(encoding='utf8')
nb['cells'] += [md('## C/D. Phase 2 label assembly, source weights, and pool writers\n\nDefinitions only. The source is embedded so this notebook is self-contained.\n'),code(module)]
nb['cells'] += [md('''## E5. Phase 2 preview and explicit write stages

Run `PHASE2_PREVIEW = p2_preview((2, 9))` for the read-only label review.
After accepting current E1/E2 and the preview, run `p2_accept_checks()`.
Set `RUN_GOLD_IMPORT=True` to build the gold pool. Classical generation is optional;
gold NLS precedence still applies within declared complete timepoints, so the two
pools cannot teach contradictory labels. All patches require their weight sidecar.

The preview and writer compute geometry outside the E1 cache as needed. Do not use
Run All to approve checks automatically; no acceptance call is hidden here.
'''),code('''if RUN_PHASE2_PREVIEW:
    PHASE2_PREVIEW = p2_preview((2, 9))

if RUN_GOLD_IMPORT:
    GOLD_BUILD = p2_build('gold', PHASE2_TIMEPOINTS)

if RUN_CLASSICAL_GENERATION:
    CLASSICAL_BUILD = p2_build('classical', PHASE2_TIMEPOINTS)
'''),md('## Appendix. Historical diagnostics (optional)\n\nCopied outputs are historical. Set `RUN_LEGACY_DIAGNOSTICS=True` only to rerun these exploratory cells.\n')] + appendix

# Generated notebooks carry source checksums so rebuilding cannot silently replace
# a hand-edited notebook. A new version must be chosen if edits are detected.
old = None
if TARGET.exists():
    old=json.loads(TARGET.read_text(encoding='utf8'))
    expected=old.get('metadata',{}).get('phase2_generated_source_sha256')
    actual=hashlib.sha256(json.dumps([c['source'] for c in old['cells']],sort_keys=True).encode()).hexdigest()
    if actual != expected:
        raise RuntimeError('Target notebook has user source edits; refusing rebuild')
for i,cell in enumerate(nb['cells']):
    if cell['cell_type']=='code': ast.parse(''.join(cell['source']),filename=f'cell_{i}')
    cell['id']=hashlib.sha256((str(i)+''.join(cell['source'])).encode()).hexdigest()[:12]
if old is not None:
    old_by_source={''.join(c['source']):c for c in old['cells'] if c['cell_type']=='code'}
    for cell in nb['cells']:
        previous=old_by_source.get(''.join(cell['source']))
        if cell['cell_type']=='code' and previous is not None:
            cell['outputs']=previous.get('outputs',[])
            cell['execution_count']=previous.get('execution_count')
nb.setdefault('metadata',{})['phase2_generated_source_sha256']=hashlib.sha256(
    json.dumps([c['source'] for c in nb['cells']],sort_keys=True).encode()).hexdigest()
TARGET.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf8')
print(TARGET.name, len(nb['cells']), 'cells; all code parsed')
