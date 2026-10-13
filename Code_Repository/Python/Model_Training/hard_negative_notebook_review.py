"""Inline review of existing hard-negative candidates; no labels are inferred."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import shutil
import tempfile
import pandas as pd


class ReviewSession:
    decisions = ('', 'artifact_negative', 'real', 'uncertain', 'skip')

    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.path = self.folder / 'review.csv'
        manifest = json.loads((self.folder / 'candidates.json').read_text(encoding='utf8'))
        self.records = {r['candidate_id']: r for r in manifest['candidates']}
        self.reload()

    def digest(self):
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def reload(self):
        self.table = pd.read_csv(self.path, keep_default_na=False)
        if self.table.empty or self.table.candidate_id.duplicated().any():
            raise ValueError('Empty review or duplicate candidate IDs')
        if set(self.table.candidate_id) != set(self.records):
            raise ValueError('Review CSV does not match candidate manifest')
        if not set(self.table.decision) <= set(self.decisions):
            raise ValueError('Unknown decision in review CSV')
        self.version = self.digest()

    def save(self, candidate_id, decision, reviewer, notes, failure_type=None):
        if decision not in self.decisions: raise ValueError('Invalid decision')
        if decision == 'artifact_negative' and not reviewer.strip():
            raise ValueError('Enter your name before approving a negative.')
        if self.digest() != self.version:
            raise RuntimeError('CSV changed elsewhere. Click Reload CSV before saving.')
        updated = self.table.copy()
        match = updated.candidate_id.eq(candidate_id)
        if match.sum() != 1: raise ValueError('Unknown candidate')
        updated.loc[match, ['decision', 'reviewer', 'notes']] = [decision, reviewer.strip(), notes]
        if failure_type is not None:
            if 'failure_type' not in updated:updated['failure_type']=''
            updated.loc[match,'failure_type']=failure_type
        backup = self.folder / 'review_backups'
        backup.mkdir(exist_ok=True)
        shutil.copy2(self.path, backup / ('review_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.csv'))
        fd, name = tempfile.mkstemp(prefix='.review_', suffix='.csv', dir=self.folder)
        try:
            with os.fdopen(fd, 'w', encoding='utf8', newline='') as stream:
                updated.to_csv(stream, index=False)
            os.replace(name, self.path)
        finally:
            if Path(name).exists(): Path(name).unlink()
        self.table = updated
        self.version = self.digest()


def review_hard_negatives_inline(folder, reviewer=''):
    """Display one existing PNG at a time. Only Save buttons write decisions."""
    import html
    import ipywidgets as w
    from IPython.display import display, Image, clear_output
    session = ReviewSession(folder)
    blank = session.table.index[session.table.decision.eq('')].tolist()
    position = [blank[0] if blank else 0]
    title = w.HTML()
    message = w.HTML()
    name = w.Text(value=reviewer, description='Reviewer:')
    choice = w.Dropdown(options=[('Not reviewed', ''), ('Artifact — exact mask negative', 'artifact_negative'),
        ('Real nucleus', 'real'), ('Uncertain', 'uncertain'), ('Skip / mixed mask', 'skip')], description='Decision:', layout=w.Layout(width='470px'))
    notes = w.Textarea(description='Notes:', layout=w.Layout(width='95%', height='70px'))
    failure = w.Dropdown(options=['','cytoplasm','droplet_edge','empty_focal_plane',
        'empty_droplet','stitching','tiny_fragment','real_nucleus','mixed','uncertain','other'],
        description='Type:',layout=w.Layout(width='400px'))
    picture = w.Output()
    back = w.Button(description='Previous')
    next_button = w.Button(description='Next (no save)')
    save_button = w.Button(description='Save', button_style='success')
    save_next = w.Button(description='Save & next', button_style='success')
    pending = w.Button(description='Next unreviewed')
    reload_button = w.Button(description='Reload CSV')
    jump = w.BoundedIntText(value=position[0]+1, min=1, max=len(session.table), description='Go to:')
    go = w.Button(description='Go')

    def show():
        row = session.table.iloc[position[0]]
        rec = session.records[row.candidate_id]
        choice.value = row.decision
        if row.reviewer: name.value = row.reviewer
        notes.value = row.notes
        failure.value = row.get('failure_type','') if row.get('failure_type','') in failure.options else 'other'
        jump.value = position[0]+1
        counts = session.table.decision.replace('', 'unreviewed').value_counts().to_dict()
        title.value = '<b>%d / %d — %s — %s</b><br>T%d Z%d, %d pixels<br>%s' % (
            position[0]+1, len(session.table), html.escape(row.candidate_id), html.escape(rec['review_id']),
            rec['t'], rec['z'], rec['area_pixels'], html.escape(str(counts)))
        if 'candidate_signals' in row:
            title.value += '<br>Mining signals (NOT labels): ' + html.escape(str(row.candidate_signals))
        with picture:
            clear_output(wait=True)
            preview = session.folder / 'previews' / (row.candidate_id + '.png')
            if preview.exists(): display(Image(filename=str(preview)))
            else: print('Preview missing:', preview)

    def move(delta):
        position[0] = max(0, min(len(session.table)-1, position[0]+delta))
        message.value = 'Navigation does not save edits.'
        show()

    def save(advance):
        try:
            cid = session.table.iloc[position[0]].candidate_id
            if choice.value == 'artifact_negative' and not (session.folder/'previews'/(cid+'.png')).exists():
                raise ValueError('Cannot approve a negative without its preview.')
            session.save(cid, choice.value, name.value, notes.value, failure.value)
            message.value = '<b>Saved %s.</b> CSV updated; previous CSV backed up.' % html.escape(cid)
            if advance: position[0] = min(position[0]+1, len(session.table)-1)
            show()
        except Exception as exc:
            message.value = '<b>Not saved:</b> ' + html.escape(str(exc))

    def next_unreviewed(_):
        candidates = [i for i in range(position[0]+1,len(session.table))] + list(range(position[0]+1))
        match = next((i for i in candidates if session.table.iloc[i].decision == ''), None)
        if match is None: message.value = 'All candidates have a saved decision.'
        else:
            position[0] = match; message.value = 'Navigation does not save edits.'; show()

    def reload_csv(_):
        try: session.reload(); message.value = 'Reloaded saved CSV; unsaved form edits discarded.'; show()
        except Exception as exc: message.value = html.escape(str(exc))

    def goto(_):
        position[0] = jump.value-1; message.value = 'Navigation does not save edits.'; show()

    back.on_click(lambda _: move(-1)); next_button.on_click(lambda _: move(1))
    save_button.on_click(lambda _: save(False)); save_next.on_click(lambda _: save(True))
    pending.on_click(next_unreviewed); reload_button.on_click(reload_csv); go.on_click(goto)
    show()
    display(w.VBox([w.HTML('<b>Judge the CENTER mask only.</b> Adjacent outlines are projections. '
        'Approve only if the entire mask is non-nuclear; skip mixed masks. '
        'Only Save / Save & next write your decision.'), title, picture, name, choice, failure, notes,
        w.HBox([back,next_button,save_button,save_next]), w.HBox([pending,jump,go,reload_button]),message]))
    return session
