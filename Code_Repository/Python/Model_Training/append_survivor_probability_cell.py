"""Embed probability audit definitions and invocation without changing existing cells."""
import ast
import difflib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parent
path = root.parent / 'Image_Segmentation/Large_FOV_Nuclear_Pipeline_v18.1.ipynb'
original = path.read_text(encoding='utf8')
nb = json.loads(original)
marker = '### 38d. Survivor probability audit'
if any(marker in ''.join(c['source']) for c in nb['cells']):
    raise RuntimeError('Probability audit already exists')
sources = [('markdown', marker + '''

Run after 38c. Reads saved full-head probabilities (no second sigmoid), not the
four-channel compatibility TIFF. No segmentation rerun or new filtering is applied.
Metrics cover survivors in SURVIVOR_AUDIT; maps use its selected examples.
Probability thresholds are hypothetical; labels must be manually supplied by review ID.
Multi-nucleus droplets are a separate review category, not segmentation artifacts.
'''), ('code', (root / 'survivor_probability_audit.py').read_text(encoding='utf8')),
('code', '''# Optional manually confirmed review labels. Never label by timepoint alone.
PROBABILITY_REVIEW_LABELS = {
    # 'T0_N123': 'artifact',
    # 'T1_N456': 'real nucleus',
    # 'T2_N789': 'multi-nucleus droplet',
}
SURVIVOR_PROBABILITY_AUDIT = audit_survivor_probabilities(
    cfg, SURVIVOR_AUDIT, core_px=2,
    review_labels=PROBABILITY_REVIEW_LABELS, save=True)
''')]
old = len(nb['cells'])
for kind, source in sources:
    c = dict(cell_type=kind, metadata={}, id='probability-audit-%d' % len(nb['cells']),
             source=source.splitlines(keepends=True))
    if kind == 'code':
        ast.parse(source); c.update(execution_count=None, outputs=[])
    nb['cells'].append(c)
replacement = json.dumps(nb, indent=1, ensure_ascii=False) + '\n'
backup = path.with_name(path.stem + '.pre_probability_audit_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.ipynb')
shutil.copy2(path, backup)
diff = list(difflib.unified_diff(original.splitlines(True), replacement.splitlines(True)))
hunks = ''.join('@@\n' if x.startswith('@@') else x for x in diff[2:])
subprocess.run([shutil.which('apply_patch')], input='*** Begin Patch\n*** Update File: '+str(path)+'\n'+hunks+'*** End Patch\n', text=True, check=True)
assert json.loads(path.read_text(encoding='utf8'))['cells'][:old] == json.loads(original)['cells']
print('Appended three cells; preserved existing cells. Backup:', backup)
