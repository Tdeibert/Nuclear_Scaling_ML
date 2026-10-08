"""Append self-contained post-segmentation audits, preserving existing cells."""
import ast
import difflib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB = ROOT.parent / 'Image_Segmentation/Large_FOV_Nuclear_Pipeline_v18.1.ipynb'
MARKER = '## 38. Recovered Vulcan post-segmentation audits'


def main():
    original = NB.read_text(encoding='utf8')
    nb = json.loads(original)
    if any(MARKER in ''.join(c['source']) for c in nb['cells']):
        raise RuntimeError('Recovery section already exists; refusing to duplicate it')
    old_cells = list(nb['cells'])

    def add(kind, source):
        cell = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True),
                    id='vulcan-recovery-%d' % len(nb['cells']))
        if kind == 'code':
            ast.parse(source)
            cell.update(execution_count=None, outputs=[])
        nb['cells'].append(cell)

    add('markdown', MARKER + '''

Run this section against an existing completed run after configuring `cfg` with
that run's paths. No segmentation rerun is required. These diagnostics do not
modify segmentation masks or replace production analysis tables.

Run the cells in order. Audit functions save new timestamped QC outputs.
This population does NOT enforce proximity or multi-nucleus droplet exclusion.
Reviewed T0 examples were cytoplasm artifacts; timepoint alone is not a rejection rule.
Real nuclei in multi-nucleus droplets remain included pending a separate implementation.
''')
    add('code', '''from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

if 'cfg' not in globals():
    raise RuntimeError('Configure cfg for the existing segmentation run first.')
print('Loading saved objects from:', cfg.obj_dir)
# Deliberately reload from this run, rather than reuse possibly stale kernel tables.
objects_df = pd.read_pickle(cfg.obj_dir / 'plane_objects.pkl')
grouped_z_df = pd.read_pickle(cfg.obj_dir / 'grouped_z_objects.pkl')
best_z_df = pd.read_pickle(cfg.obj_dir / 'best_z_nuclei.pkl')
print('Plane objects:', len(objects_df), '| linked rows:', len(grouped_z_df))
''')
    add('markdown', '''### 38a. Enrichment and solidity filter

Training-like thresholds: enrichment >= 1.20, solidity >= 0.80, at least 50
reference-ring pixels. Support here is the segmented droplet, not the training
circle. Unassessable references are rejected by the gate, not confirmed artifacts.
Definitions are embedded below so this notebook does not depend on external scripts.
''')
    gate = (ROOT / 'segmented_artifact_gate.py').read_text(encoding='utf8')
    add('code', gate.split("\nif 'cfg' in globals():")[0])
    add('code', '''# Reads the raw image from cfg to avoid stale img_5d from another run.
ARTIFACT_GATE = audit_segmented_artifacts(
    cfg, image=None, objects=objects_df, best_z=best_z_df)
''')
    add('markdown', '''### 38b. Maximum nuclear cross-sectional area before and after filtering

One value per (T, linked nuclear ID): maximum single-component area across Z.
After filtering, take the maximum among passing components; fully rejected IDs
are missing (NaN), not zero. Branched IDs are flagged because a linked ID is not
necessarily one biological nucleus. No best-Z selection or proximity filter is used.
''')
    add('code', '''keys = ['t', 'z', 'label']
identity = grouped_z_df[keys + ['nucleus_3d_id']].copy()
gate_rows = ARTIFACT_GATE['all_instances']
area_planes = identity.merge(gate_rows, on=keys, how='left',
                             validate='one_to_one', indicator=True)
if not area_planes['_merge'].eq('both').all():
    raise ValueError('Grouped objects and artifact gate do not match.')
if not area_planes.gate_pass.isin([True, False]).all():
    raise ValueError('Invalid gate decisions.')
groups = ['t', 'nucleus_3d_id']
before = area_planes.groupby(groups).area_um2.max().rename('before_um2')
after = area_planes.loc[area_planes.gate_pass].groupby(groups).area_um2.max().rename('after_um2')
branches = (area_planes.groupby(groups + ['z']).size().gt(1)
            .groupby(level=[0, 1]).any().rename('multiple_instances_same_z'))
MAX_AREA_FILTER_COMPARISON = pd.concat([before, after, branches], axis=1).reset_index()
comparison = MAX_AREA_FILTER_COMPARISON
display(comparison.groupby('t').agg(
    before_count=('before_um2', 'count'), after_count=('after_um2', 'count'),
    before_median=('before_um2', 'median'), after_median=('after_um2', 'median'),
    branched_ids=('multiple_instances_same_z', 'sum')))
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
rng = np.random.default_rng(0)
times = sorted(comparison.t.unique())
for i, t in enumerate(times):
    sub = comparison[comparison.t == t]
    jitter = rng.uniform(-0.055, 0.055, len(sub))
    for offset, col, color, label in [(-0.15, 'before_um2', 'gray', 'Before'),
                                    (0.15, 'after_um2', 'teal', 'After')]:
        vals = sub[col].to_numpy()
        axs[0].scatter(i + offset + jitter, vals, s=12, alpha=.35,
                       color=color, label=label if i == 0 else None)
        finite = vals[np.isfinite(vals)]
        if len(finite):
            axs[0].plot([i+offset-.09, i+offset+.09], [np.median(finite)]*2,
                        color=color, linewidth=3)
paired = comparison.dropna(subset=['after_um2'])
axs[1].scatter(paired.before_um2, paired.after_um2, s=14, alpha=.4)
limit = max(1., float(comparison.before_um2.max())) if len(comparison) else 1.
axs[1].plot([0, limit], [0, limit], '--', color='gray')
axs[0].set_xticks(range(len(times))); axs[0].set_xticklabels(times)
axs[0].set(xlabel='Timepoint index', ylabel='Maximum nuclear area (um²)',
           title='All linked IDs; points = nuclei, bars = medians')
axs[0].legend()
axs[1].set(xlabel='Before (um²)', ylabel='After (um²)',
           title='Surviving IDs only; identity line shown')
fig.suptitle('Enrichment/solidity only — multi-nucleus droplets NOT excluded')
fig.tight_layout(); plt.show()
if ARTIFACT_GATE.get('output_dir') is not None:
    out = Path(ARTIFACT_GATE['output_dir'])
    comparison.to_csv(out / 'max_area_before_after.csv', index=False)
    fig.savefig(out / 'max_area_before_after.png', dpi=160, bbox_inches='tight')
''')
    add('markdown', '''### 38c. Survivor audit across Z

Default: six area-ranked survivors each from T0, T1 and T2. Cyan = passing
component; red = rejected component. Review sheets are saved for manual labels.
The profile sums components at each Z for branched IDs and flags those IDs;
38b instead reports maximum single-component area. Neither establishes droplet multiplicity.
''')
    survivor = (ROOT / 'survivor_audit.py').read_text(encoding='utf8')
    add('code', survivor.split("\nif all(k in globals()")[0])
    add('code', '''SURVIVOR_AUDIT = audit_survivors(
    cfg, grouped_z_df, ARTIFACT_GATE, image=None,
    times=(0, 1, 2), per_t=6, save=True)
''')
    assert nb['cells'][:len(old_cells)] == old_cells
    replacement = json.dumps(nb, indent=1, ensure_ascii=False) + '\n'
    json.loads(replacement)
    backup = NB.with_name(NB.stem + '.pre_audit_recovery_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.ipynb')
    shutil.copy2(NB, backup)
    diff = list(difflib.unified_diff(original.splitlines(True), replacement.splitlines(True)))
    hunks = ''.join('@@\n' if line.startswith('@@') else line for line in diff[2:])
    patch = '*** Begin Patch\n*** Update File: ' + str(NB) + '\n' + hunks + '*** End Patch\n'
    subprocess.run([shutil.which('apply_patch')], input=patch, text=True, check=True)
    saved = json.loads(NB.read_text(encoding='utf8'))
    assert saved == nb
    print('Preserved %d existing cells; appended %d cells.' % (len(old_cells), len(nb['cells'])-len(old_cells)))
    print('Backup:', backup)


if __name__ == '__main__':
    main()
