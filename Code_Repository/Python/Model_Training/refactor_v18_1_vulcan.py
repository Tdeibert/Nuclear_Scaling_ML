"""Reproducibly embed the Vulcan contract in v18.1; preserve a byte-exact backup."""
import ast
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB = ROOT.parent / 'Image_Segmentation/Large_FOV_Nuclear_Pipeline_v18.1.ipynb'
BACKUP = NB.with_name('Large_FOV_Nuclear_Pipeline_v18.1.pre_vulcan_refactor.ipynb')


def main():
    if not BACKUP.exists():
        shutil.copy2(NB, BACKUP)
    n = json.loads(BACKUP.read_text(encoding='utf8'))
    def get(i): return ''.join(n['cells'][i]['source'])
    def put(i, s): n['cells'][i]['source'] = s.splitlines(keepends=True)
    contract = (ROOT/'vulcan_pipeline_contract.py').read_text(encoding='utf8')
    tree = ast.parse(contract)
    functions = {x.name: ast.get_source_segment(contract, x)+'\n' for x in tree.body if isinstance(x, ast.FunctionDef)}
    consumed = set()
    for i,c in enumerate(n['cells']):
        if c['cell_type']!='code': continue
        s=get(i); lines=s.splitlines(keepends=True)
        for node in reversed(ast.parse(s).body):
            if isinstance(node,ast.FunctionDef) and node.name in functions:
                lines[node.lineno-1:node.end_lineno]=[functions[node.name]]
                consumed.add(node.name)
        put(i,''.join(lines))
    extras='\n\n'.join(v for k,v in functions.items() if k not in consumed)
    constant=next(ast.get_source_segment(contract,x) for x in tree.body if isinstance(x,ast.Assign))
    put(20,get(20)+'\n\n'+constant+'\n\n'+extras+'\n')
    config=get(4).replace('z_step_um: float = 1.0','z_step_um: float = 2.0')
    config=config.replace('require_stage2_gate: bool = True','require_stage2_gate: bool = False')
    config=config.replace('use_focus_z_selection: bool = True','use_focus_z_selection: bool = False')
    config=config.replace('focus_edge_z_exclusion: int = 2','focus_edge_z_exclusion: int = 0')
    config=config.replace('patch_stride: int = 256','patch_stride: int = 384')
    config=config.replace('npc_threshold:     float = 0.3','npc_threshold:     float = 0.5')
    config=config.replace('min_nucleus_area_px: int = 1065','min_nucleus_area_px: int = 0')
    added='''    # Vulcan baseline; set all result-affecting options BEFORE run bootstrap.
    pipeline_revision: str = "v18.1-vulcan253-contract1"
    model_sha256: str = ""
    input_fingerprint: str = ""
    tta: bool = True
    best_z_mode: str = "argmin_z_offset"
    watershed_core_thresh: float = 0.7
    watershed_edge_weight: float = 1.0
    watershed_min_marker_dist_px: int = 20
    sharpness_window_px: int = 7
    repair_enabled: bool = False
    repair_flatten_um: float = 8.0
    repair_npc_slack: float = 1.10
    repair_min_solidity: float = 0.93
    repair_area_deficit: float = 0.50
    repair_z_pad: int = 6
    repair_max_frac_droplet: float = 0.60
    repair_min_signed_margin: float = 1.15
    repair_edge_margin_um: float = 2.0
    repair_npc_min_rays: int = 36
    repair_win_um: float = 45.0

'''
    config=config.replace('    # ── Imaging',added+'    # ── Imaging',1)
    config=config.replace('    # Vulcan 2.5.3 output: seven mask LOGITS followed by z_offset regression.',
                          '    # Four-channel compatibility view; all eight heads are also saved separately.')
    put(4,config)
    put(8,'''# Configure all options before bootstrap; fingerprint the actual model bytes.
import hashlib
cfg = PipelineConfig()
paths = ProjectPaths.for_cheaha(project_name="Nuclear_Scaling")
_model_path = paths.models_dir / cfg.model_name
_image_path = paths.raw_images_dir / cfg.image_subdir / cfg.input_image_name
_sha = hashlib.sha256()
with _model_path.open("rb") as _stream:
    for _block in iter(lambda: _stream.read(8 * 1024**2), b""):
        _sha.update(_block)
cfg.model_sha256 = _sha.hexdigest()
_st = _image_path.stat()
cfg.input_fingerprint = f"{_image_path.resolve()}:{_st.st_size}:{_st.st_mtime_ns}"
run = paths.for_run(ProjectPaths.make_run_id(
    label=Path(cfg.input_image_name).stem, config_dict=cfg.to_run_signature()))
cfg.paths = run
run.assert_config_matches(cfg.to_run_signature())
run.snapshot_config(cfg.to_run_signature())
run.write_manifest(input_image=cfg.input_image_name, model=cfg.model_name,
                   model_sha256=cfg.model_sha256, run_id=run.run_id)
print("Run:", run.run_dir)
print("Model SHA256:", cfg.model_sha256)
cfg
''')
    put(7,get(7).replace('        snap = self.load_snapshot()',
        '        config_dict = _json.loads(_json.dumps(config_dict, default=str))\n'
        '        snap = self.load_snapshot()'))
    put(9,'# Calibration and optional repair settings are now dataclass fields above bootstrap.\n'
          'print(f"Pixel size={cfg.pixel_size_um} um; Z step={cfg.z_step_um} um; repair={cfg.repair_enabled}")\n')
    # Refuse shape/dtype changes instead of silently unlinking artifacts.
    s=get(22); start=s.index('    if out_path.exists():',s.index('def open_hyperstack_memmap'))
    end=s.index('    return tiff.memmap',start)
    s=s[:start]+'''    if out_path.exists():
        mm = tiff.memmap(str(out_path), mode="r+")
        if tuple(mm.shape) != shape or np.dtype(mm.dtype) != dtype:
            raise RuntimeError(f"Incompatible existing artifact: {out_path}; choose a new run")
        return mm

'''+s[end:]
    s=s.replace('bigtiff=True, metadata={"axes": axes})',
                'bigtiff=True, photometric="minisblack", metadata={"axes": axes})')
    put(22,s)
    s=get(23)
    s=s.replace('    if save_probability_tiff:\n        t["probability_hyperstack"]',
        '    t["full_heads"] = ("file", full_head_path(config))\n'
        '    t["full_heads_metadata"] = ("file", full_head_metadata_path(config))\n'
        '    if save_probability_tiff:\n        t["probability_hyperstack"]')
    s=s.replace('    checks = [','    checks = [\n        (full_head_path(config), (T, Z, 8, Y, X)),\n'
        '        (config.segmentation_probability_hyperstack_path, (T, Z, 4, Y, X)),')
    s=s.replace('    want = {"T": T, "Z": Z, "Y": Y, "X": X}',
        '    if cp.get("config_signature") != config.to_run_signature():\n'
        '        return SegmentationState("inconsistent", [], [], cp, [], "Checkpoint configuration differs; create a new run")\n'
        '    want = {"T": T, "Z": Z, "Y": Y, "X": X}')
    # JSON turns tuples into lists; normalize both operands.
    s=s.replace('cp.get("config_signature") != config.to_run_signature()',
        'cp.get("config_signature") != json.loads(json.dumps(config.to_run_signature()))')
    s=s.replace('"code_version": "v18.1",','"code_version": config.pipeline_revision,\n'
        '        "config_signature": config.to_run_signature(),\n        "model_sha256": config.model_sha256,')
    s=s.replace('    missing = _verify_seg_artifacts(cp, needs=needs or None)',
        '    if cp.get("config_signature") != json.loads(json.dumps(config.to_run_signature())):\n'
        '        raise RuntimeError("Configuration changed after segmentation; create a new run")\n'
        '    missing = _verify_seg_artifacts(cp, needs=needs or None)')
    s=s.replace('        for p in (config.segmentation_class_hyperstack_path,',
        '        for p in (full_head_path(config), full_head_metadata_path(config), config.segmentation_class_hyperstack_path,')
    put(23,s)
    # Correct debug path: identical inputs/inference to production.
    s=get(65);a=s.index('    plane_yxc =');b=s.index('    fig, axes',a)
    s=s[:a]+'''    prob_map, label_map, _, _ = segment_single_plane_with_overlap(
        img_5d, DEBUG_T_IDX, DEBUG_Z_IDX, model, cfg)

'''+s[b:];put(65,s)
    # No regeneration of indices from legacy files in a fresh baseline.
    put(68,'# Plane indices are written transactionally by segmentation; no legacy reconstruction.\n')
    # Optional repair is explicitly opt-in and has provenance in run signature.
    put(78,get(78).replace('RUN_FRAGMENTATION_REPAIR = True','RUN_FRAGMENTATION_REPAIR = cfg.repair_enabled'))
    put(76,get(76).replace('if not (0 <= z < Z):','if z not in eligible_z_planes(Z, config):'))
    # Old ad-hoc repair cells would mutate best_z_df on Run All; retain as documentation.
    for i in range(149,len(n['cells'])):
        if n['cells'][i]['cell_type']=='code':
            s=get(i);n['cells'][i]={'cell_type':'markdown','metadata':{},'source':
                ('Legacy optional diagnostic, disabled in the Vulcan baseline.\n\n```python\n'+s+'\n```\n').splitlines(keepends=True)}
    put(0,'''# Large-FOV Nuclear Pipeline v18.1 — Vulcan 2.5.3 baseline

Start with a fresh kernel and execute in order. Set configuration before run bootstrap.
The model SHA256 and complete configuration determine a new run directory; prior results remain available.

Inputs: five edge-replicated Z planes, each ordered NLS/NPC/membrane, globally normalized per timepoint.
Inference: Hann overlap blending and eight dihedral TTA views. All eight output heads are retained;
the four-channel probability TIFF is a compatibility view for existing analysis code.
Target planes: Z >= 6, with no focus-window cropping. Lower planes may supply input context.
Nuclei: edge-guided watershed; instance IDs survive measurement and mask recovery.
Z selection: mean predicted Z-offset minimum, with maximum-area comparison saved in QC.
Legacy NPC/membrane rejection and raw-image fragmentation repair are disabled by default.
Exceptions stop the run before committing the affected timepoint. Resume uses completed timepoints only.

The complete model output is `segmentation/vulcan_all_heads.tif` (actual directory follows the path resolver),
with channel names and Z-offset units in its JSON sidecar. Z-offset is in micrometres, not probability.
Old four-channel re-extraction is disabled for this baseline; run fresh inference.
''')
    put(63,'Restart the kernel after updating the notebook, then execute from the configuration downward.\n'
           'The new revision and model fingerprint isolate this run from earlier segmentation outputs.\n')
    put(75,'## 30b. Optional legacy fragmentation repair\n\nDisabled by default. Enabling it changes the run signature and measurements.\n')
    put(5,'Calibration is defined once in PipelineConfig: 0.1625 µm/pixel and 2.0 µm acquisition Z step.\n')
    put(16,'## 6. Optional focus-scoring utilities\n\nQC only for this baseline; target planes are all Z >= 6.\n')
    put(18,'## 7a. Optional legacy NPC/membrane gate\n\nRetained for explicit experiments, disabled by default. '
           'These are Vulcan 1.1-era criteria and are not required by Vulcan 2.5.3.\n')
    put(28,'## 10b. Legacy re-extraction guard\n\nFour-channel re-extraction is disabled because it lacks edge and Z heads. '
           'This baseline requires fresh full-head inference.\n')
    put(30,'## 11. Object extraction from saved instance measurements\n\nPreserves watershed IDs and per-instance head statistics.\n')
    put(34,'## 13. Link adjacent Z planes by mask overlap\n')
    put(36,'## 14. Select best Z by predicted offset; save maximum-area comparison\n')
    # Strip stale outputs/execution counts; never alter the backup.
    for c in n['cells']:
        if c['cell_type']=='code':
            c['outputs']=[];c['execution_count']=None
            compile(''.join(c['source']),'<notebook-cell>','exec')
    NB.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n',encoding='utf8')
    print('Updated',NB,'; backup',BACKUP)


if __name__=='__main__': main()
