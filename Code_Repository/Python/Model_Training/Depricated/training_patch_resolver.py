"""Interactive resolver and explicit commit for the training-patch audit.

Nothing is written to any existing pool. Review state is autosaved into its own
versioned folder; a commit creates a NEW timestamped directory holding sparse
overlays composed on top of the CURRENT effective labels, so the existing
185-patch gold-correction pool and the reviewed hard-negative pool keep their
provenance and their bytes.

Decisions
    real            keep the existing nuclear positive. Changes nothing: restoring
                    or adding a positive needs reviewed boundaries and consistent
                    edge/equatorial/Z targets, which a button cannot supply.
    artifact_negative  the ENTIRE exact region is non-nuclear. Writes target 0 and
                    an explicit weight on nuclear heads 4/5/6 for exactly those
                    pixels, and de-supervises the Z-offset there so a masked-out
                    regression never becomes an invented target.
    mixed_redraw    part artifact, part real. Recorded and left UNRESOLVED: local
                    partial redraw is not implemented, so no correction is written.
    uncertain       reviewed, not decided. No correction.
    skip            deliberately passed over. No correction.

An unreviewed candidate is never treated as real.
"""
import hashlib
import html
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import reviewed_hard_negatives as hn
import training_patch_audit as tpa

DECISIONS = ('real', 'artifact_negative', 'mixed_redraw', 'uncertain', 'skip')
CORRECTING = ('artifact_negative',)
FAILURE_TYPES = ('', 'cytoplasm', 'empty_droplet', 'empty_focal_plane', 'droplet_wall',
                 'stitching_seam', 'annotation_spillover', 'neighbouring_droplet',
                 'debris', 'unknown_mechanism', 'real_nucleus')
NUCLEAR_WRITE_HEADS = (4, 5, 6)
Z_OFFSET_HEAD = 8
UNANNOTATED = 255


def _json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding='utf8')


def _atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + datetime.now().strftime('%H%M%S_%f') + '.tmp')
    _json(tmp, obj)
    os.replace(tmp, path)


class ReviewState:
    """Autosaved, resumable review of one immutable candidate set.

    The candidate set and its exact masks are snapshotted on creation and hashed.
    Reopening with the same candidates resumes; reopening after the candidate set
    changed refuses to reuse the old decisions and starts a new versioned review
    rather than silently erasing or mislabelling them.
    """

    def __init__(self, session, root=None, *, reviewer='', verbose=True):
        self.session = session
        candidates = session.candidates
        if candidates is None or candidates.empty:
            raise ValueError('No candidates to review; run the audit cell first.')
        self.signature = self._candidate_signature(session, candidates)
        base = Path(root) if root else session.cache_dir / ('review_' + self.signature[:16])
        self.root = Path(base)
        self.masks_dir = self.root / 'masks'
        self.state_path = self.root / 'review_state.json'
        self.history_path = self.root / 'history.jsonl'
        self.reviewer = reviewer
        self.committed = None
        if self.state_path.is_file():
            saved = json.loads(self.state_path.read_text())
            if saved.get('candidate_signature') != self.signature:
                raise RuntimeError(
                    f'{self.root} holds a review of a different candidate set. Pass an '
                    'explicit root for a new review; existing decisions are not reused.')
            self._verify_masks()
            self.decisions = saved['decisions']
            self.version = int(saved.get('version', 0))
            self.committed = saved.get('committed')
            self.candidates = pd.read_csv(self.root / 'candidates.csv', keep_default_na=False)
            missing = set(candidates.region_id) - set(self.decisions)
            if missing:
                raise RuntimeError(f'Saved review is missing {len(missing)} candidates.')
            if verbose:
                print(f'Resumed review {self.root.name}: {len(self.decisions)} candidates, '
                      f'{len(self.pending())} pending, version {self.version}'
                      + (f', COMMITTED to {self.committed}' if self.committed else ''))
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(exist_ok=True)
        self.candidates = candidates.reset_index(drop=True)
        self.version = 0
        self.decisions = {r.region_id: dict(decision='', reviewer='', failure_type='', notes='',
                                            updated='', kind=r.kind)
                          for r in self.candidates.itertuples()}
        for region_id in self.decisions:
            path = self.masks_dir / (region_id + '.npy')
            with path.open('xb') as stream:
                np.save(stream, session.region_masks[region_id], allow_pickle=False)
        self.candidates.to_csv(self.root / 'candidates.csv', index=False)
        _json(self.root / 'provenance.json', dict(
            created=datetime.now().isoformat(), candidate_signature=self.signature,
            mask_sha256={rid: hn._digest(self.masks_dir / (rid + '.npy')) for rid in self.decisions},
            **session.provenance))
        self._save('created')
        if verbose:
            print(f'Opened review {self.root}: {len(self.decisions)} candidates, '
                  f'{len(self.pending())} pending')

    @staticmethod
    def _candidate_signature(session, candidates):
        digest = hashlib.sha256()
        digest.update(json.dumps(session.provenance, sort_keys=True, default=str).encode())
        for region_id in sorted(candidates.region_id):
            digest.update(region_id.encode())
            digest.update(np.packbits(session.region_masks[region_id].ravel()).tobytes())
        return digest.hexdigest()

    def _verify_masks(self):
        recorded = json.loads((self.root / 'provenance.json').read_text())['mask_sha256']
        for region_id, expected in recorded.items():
            path = self.masks_dir / (region_id + '.npy')
            if not path.is_file() or hn._digest(path) != expected:
                raise RuntimeError('Reviewed mask changed or missing: ' + region_id)

    def _save(self, action, region_id=None):
        self.version += 1
        _atomic_json(self.state_path, dict(
            candidate_signature=self.signature, version=self.version,
            updated=datetime.now().isoformat(), committed=self.committed,
            decisions=self.decisions))
        with self.history_path.open('a', encoding='utf8') as stream:
            stream.write(json.dumps(dict(version=self.version, action=action, region_id=region_id,
                                         at=datetime.now().isoformat(),
                                         decision=self.decisions.get(region_id))) + '\n')

    def mask(self, region_id):
        return np.load(self.masks_dir / (region_id + '.npy'), allow_pickle=False).astype(bool)

    def record(self, region_id):
        rows = self.candidates[self.candidates.region_id.eq(region_id)]
        if rows.empty:
            raise KeyError(region_id)
        return rows.iloc[0]

    def save_decision(self, region_id, decision, reviewer, failure_type='', notes=''):
        if self.committed:
            raise RuntimeError('This review was already committed; open a new review to revise.')
        if region_id not in self.decisions:
            raise KeyError(region_id)
        if decision not in DECISIONS:
            raise ValueError('Decision must be one of ' + ', '.join(DECISIONS))
        if not str(reviewer).strip():
            raise ValueError('Enter a reviewer name.')
        if decision == 'artifact_negative' and int(self.record(region_id).n_correctable_patches) == 0:
            # Still recorded: the finding is real even though nothing may be written.
            pass
        self.decisions[region_id] = dict(
            decision=decision, reviewer=str(reviewer).strip(), failure_type=failure_type,
            notes=str(notes), updated=datetime.now().isoformat(),
            kind=str(self.record(region_id).kind))
        self.reviewer = str(reviewer).strip()
        self._save('decide', region_id)

    def clear_decision(self, region_id):
        if self.committed:
            raise RuntimeError('This review was already committed.')
        kind = self.decisions[region_id].get('kind', '')
        self.decisions[region_id] = dict(decision='', reviewer='', failure_type='', notes='',
                                         updated='', kind=kind)
        self._save('clear', region_id)

    def pending(self):
        return [r for r, d in self.decisions.items()
                if not d['decision'] or not str(d['reviewer']).strip()]

    def table(self):
        rows = []
        for region_id, d in self.decisions.items():
            record = self.record(region_id)
            rows.append(dict(region_id=region_id, kind=d.get('kind', record.kind),
                             t=int(record.t), z=int(record.z), area_um2=float(record.area_um2),
                             flags=record.flags, priority_score=float(record.priority_score),
                             n_correctable_patches=int(record.n_correctable_patches),
                             decision=d['decision'], reviewer=d['reviewer'],
                             failure_type=d['failure_type'], notes=d['notes']))
        return pd.DataFrame(rows)

    def progress(self):
        table = self.table()
        done = table[table.decision.ne('') & table.reviewer.ne('')]
        return dict(total=len(table), decided=len(done), pending=len(table) - len(done),
                    by_decision=done.decision.value_counts().to_dict(),
                    by_kind={f'{kind}/{decision}': int(n) for (kind, decision), n
                             in done.groupby(['kind', 'decision']).size().items()})


# ------------------------------------------------------------------ display

def _raw_window(session, t, z, channel, box):
    import tifffile
    y0, x0, y1, x1 = box
    height, width = session.image_shape[-2:]
    out = np.zeros((y1 - y0, x1 - x0), np.float32)
    sy0, sx0, sy1, sx1 = max(0, y0), max(0, x0), min(height, y1), min(width, x1)
    if sy0 < sy1 and sx0 < sx1:
        raw = tifffile.memmap(session.cfg.image_file, mode='r')
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = np.asarray(
            raw[int(t), int(z), int(channel), sy0:sy1, sx0:sx1], dtype=np.float32)
        del raw
    return out


def _contour(axis, mask, color, width=0.9):
    if mask.any():
        axis.contour(np.arange(-1, mask.shape[1] + 1), np.arange(-1, mask.shape[0] + 1),
                     np.pad(mask, 1), levels=[.5], colors=[color], linewidths=width)


def show_candidate(session, state, region_id, *, pad=48):
    """Raw, normalized, original and effective targets, weights and context."""
    import matplotlib.pyplot as plt
    cfg = session.cfg
    record = state.record(region_id)
    mask = state.mask(region_id)
    t, z = int(record.t), int(record.z)
    y0, x0 = int(record.y0) - pad, int(record.x0) - pad
    y1, x1 = int(record.y1) + pad, int(record.x1) + pad
    view = np.zeros((y1 - y0, x1 - x0), bool)
    view[int(record.y0) - y0:int(record.y1) - y0, int(record.x0) - x0:int(record.x1) - x0] = mask

    prior = np.zeros_like(view)
    for item in session.reviewed_artifacts.get((t, z), []):
        ay0, ax0 = max(y0, item['y0']), max(x0, item['x0'])
        ay1, ax1 = min(y1, item['y1']), min(x1, item['x1'])
        if ay0 < ay1 and ax0 < ax1:
            prior[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] |= item['mask'][
                ay0 - item['y0']:ay1 - item['y0'], ax0 - item['x0']:ax1 - item['x0']]

    stems = [s for s in str(record.patch_stems).split(';') if s]
    by_stem = {r.stem: r for r in session.frame.itertuples()}
    display_stem = next((s for s in stems if s in by_stem), None)

    figure, axes = plt.subplots(3, 4, figsize=(19, 14))
    channels = [('NLS (raw)', cfg.nucleus_channel_idx), ('NPC (raw)', cfg.npc_channel_idx),
                ('Membrane (raw)', cfg.membrane_channel_idx)]
    for axis, (title, channel) in zip(axes[0, :3], channels):
        image = _raw_window(session, t, z, channel, (y0, x0, y1, x1))
        lo, hi = np.percentile(image, [1, 99.8])
        axis.imshow(image, cmap='gray', vmin=lo, vmax=max(hi, lo + 1e-6))
        _contour(axis, view, 'cyan')
        _contour(axis, prior, 'red')
        axis.set_title(f'{title}\nper-view display scale {lo:.0f}-{hi:.0f} (not for quantitation)',
                       fontsize=8)

    # Globally normalized input, exactly as trained: plane-major [NLS, NPC, Mem],
    # centre plane channels 6, 7, 8. Never re-stretched per crop.
    axis = axes[0, 3]
    if display_stem and by_stem[display_stem].img:
        patch = by_stem[display_stem]
        img = np.load(patch.img, allow_pickle=False, mmap_mode='r')
        oy = int(patch.cy) - session.patch_size // 2
        ox = int(patch.cx) - session.patch_size // 2
        sl = (slice(max(0, y0 - oy), min(session.patch_size, y1 - oy)),
              slice(max(0, x0 - ox), min(session.patch_size, x1 - ox)))
        axis.imshow(np.asarray(img[sl[0], sl[1], 6]), cmap='gray', vmin=0, vmax=1)
        axis.set_title(f'Stored normalized NLS (global_t, vmin=0 vmax=1)\n{display_stem}', fontsize=8)
    else:
        axis.set_title('No stored input patch for this region', fontsize=8)

    for axis, dz in zip(axes[1, :3], (-1, 0, 1)):
        zz = int(np.clip(z + dz, 0, session.image_shape[1] - 1))
        image = _raw_window(session, t, zz, cfg.nucleus_channel_idx, (y0, x0, y1, x1))
        lo, hi = np.percentile(image, [1, 99.8])
        axis.imshow(image, cmap='gray', vmin=lo, vmax=max(hi, lo + 1e-6))
        _contour(axis, view, 'cyan')
        axis.set_title(f'NLS Z{zz}{" CENTRE" if dz == 0 else " (centre mask projected for context)"}',
                       fontsize=8)

    if display_stem:
        patch = by_stem[display_stem]
        oy = int(patch.cy) - session.patch_size // 2
        ox = int(patch.cx) - session.patch_size // 2
        # Target panels are rendered in the DISPLAY box frame, zero-filled where the
        # crop does not reach, so every contour lands on the pixel it describes even
        # when the region sits against a patch border.
        dy0, dx0 = max(y0, oy), max(x0, ox)
        dy1, dx1 = min(y1, oy + session.patch_size), min(x1, ox + session.patch_size)
        source = (slice(dy0 - oy, dy1 - oy), slice(dx0 - ox, dx1 - ox))
        target = (slice(dy0 - y0, dy1 - y0), slice(dx0 - x0, dx1 - x0))

        def panel(array, channels, reducer):
            out = np.zeros(view.shape, np.float32)
            if dy0 < dy1 and dx0 < dx1:
                out[target] = reducer(np.asarray(array[source[0], source[1], channels]))
            return out

        original = np.load(patch.original_lab, allow_pickle=False, mmap_mode='r')
        effective = np.load(patch.lab, allow_pickle=False, mmap_mode='r')
        weights = np.load(patch.wgt, allow_pickle=False, mmap_mode='r')
        nuclear = lambda block: np.any(block == 1, axis=-1).astype(np.float32)
        panels = [
            (panel(effective, 1, lambda b: (b == 1).astype(np.float32)),
             'Droplet interior target (context)', dict(vmin=0, vmax=1)),
            (panel(original, slice(4, 7), nuclear),
             f'ORIGINAL stored nuclear targets (4/5/6)\n{display_stem}', dict(vmin=0, vmax=1)),
            (panel(effective, slice(4, 7), nuclear),
             'EFFECTIVE nuclear targets after corrections', dict(vmin=0, vmax=1)),
            (panel(weights, 4, lambda b: b.astype(np.float32)),
             'nucleus_interior supervision weight (head 4)', dict(vmin=0)),
        ]
        for axis, (image, title, kwargs) in zip(axes[1, 3:4].tolist() + axes[2, :3].tolist(), panels):
            axis.imshow(image, **kwargs)
            axis.set_title(title, fontsize=8)
            _contour(axis, view, 'cyan')

    decision = state.decisions[region_id]
    axis = axes[2, 3]
    axis.axis('off')
    lines = [
        f'{region_id}',
        f'kind: {record.kind}   t={t} z={z}   {record.area_um2:.1f} um^2 ({int(record.area_px)} px)',
        f'global bbox y{int(record.y0)}:{int(record.y1)} x{int(record.x0)}:{int(record.x1)}',
        f'splits: {record.splits}',
        f'correctable training crops: {int(record.n_correctable_patches)} of {int(record.n_patches)}',
        f'withheld: {record.withheld_reasons or "none"}',
        f'why flagged: {record.flags or "none (random sample)"}',
        f'priority score: {float(record.priority_score):.1f}',
        f'prior review: {record.prior_review or "none"}',
        f'already corrected crops: {sum(1 for s in stems if s in by_stem and by_stem[s].corrected)}',
        '',
        f'decision: {decision["decision"] or "UNREVIEWED"}  by {decision["reviewer"] or "-"}',
        f'failure type: {decision["failure_type"] or "-"}',
    ]
    for key in ('nls_enrichment', 'npc_rim_support', 'negative_supervision_px',
                'ignored_stored_px', 'droplet_fill_fraction', 'z_neighbour_support',
                'confirmed_artifact_overlap_px', 'seam_distance_px'):
        if key in record.index:
            lines.append(f'{key}: {record[key]}')
    axis.text(0, 1, '\n'.join(lines), va='top', family='monospace', fontsize=8)

    for axis in axes.flat:
        if axis.images or axis.collections:
            axis.set_xticks([]); axis.set_yticks([])
    figure.suptitle(f'{region_id} — cyan: exact region under review; red: previously confirmed artifact. '
                    'Adjacent planes are context only.', fontsize=10)
    figure.tight_layout()
    plt.show()
    plt.close(figure)


# ------------------------------------------------------------- commit checks

def contradictions(session, state):
    """Approved negative and retained-real regions must not share pixels."""
    planes = {}
    for region_id, decision in state.decisions.items():
        if decision['decision'] not in ('artifact_negative', 'real'):
            continue
        record = state.record(region_id)
        planes.setdefault((int(record.t), int(record.z)), []).append(
            (region_id, decision['decision'], record, state.mask(region_id)))
    clashes = []
    for key, items in planes.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a[1] == b[1]:
                    continue
                ay0, ax0 = int(a[2].y0), int(a[2].x0)
                by0, bx0 = int(b[2].y0), int(b[2].x0)
                y0, x0 = max(ay0, by0), max(ax0, bx0)
                y1 = min(int(a[2].y1), int(b[2].y1)); x1 = min(int(a[2].x1), int(b[2].x1))
                if y0 >= y1 or x0 >= x1:
                    continue
                shared = int((a[3][y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0]
                              & b[3][y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]).sum())
                if shared:
                    clashes.append(dict(t=key[0], z=key[1], region_a=a[0], decision_a=a[1],
                                        region_b=b[0], decision_b=b[1], shared_px=shared))
    return pd.DataFrame(clashes)


def preview_commit(session, state, *, weight=1.0, verbose=True):
    """What a commit would change, before anything is written."""
    approved, withheld = [], []
    for region_id, decision in state.decisions.items():
        if decision['decision'] not in CORRECTING:
            continue
        record = state.record(region_id)
        stems = [s for s in str(record.correctable_stems).split(';') if s]
        if not stems:
            withheld.append(dict(region_id=region_id, reason=record.withheld_reasons or 'no eligible training crop',
                                 area_px=int(record.area_px)))
            continue
        approved.append((region_id, record, stems))

    by_stem = {r.stem: r for r in session.frame.itertuples()}
    per_patch = {}
    for region_id, record, stems in approved:
        mask = state.mask(region_id)
        for stem in stems:
            patch = by_stem[stem]
            oy = int(patch.cy) - session.patch_size // 2
            ox = int(patch.cx) - session.patch_size // 2
            y0, x0 = max(int(record.y0), oy), max(int(record.x0), ox)
            y1 = min(int(record.y1), oy + session.patch_size)
            x1 = min(int(record.x1), ox + session.patch_size)
            if y0 >= y1 or x0 >= x1:
                continue
            entry = per_patch.setdefault(stem, dict(
                stem=stem, source=patch.source, t=int(patch.t), z=int(patch.z),
                lab=patch.lab, wgt=patch.wgt, regions=[],
                mask=np.zeros((session.patch_size, session.patch_size), bool)))
            entry['mask'][y0 - oy:y1 - oy, x0 - ox:x1 - ox] |= mask[
                y0 - int(record.y0):y1 - int(record.y0), x0 - int(record.x0):x1 - int(record.x0)]
            entry['regions'].append(region_id)

    rows = []
    for stem, entry in sorted(per_patch.items()):
        lab = np.load(entry['lab'], allow_pickle=False, mmap_mode='r')
        wgt = np.load(entry['wgt'], allow_pickle=False, mmap_mode='r')
        mask = entry['mask']
        stored = np.any(np.asarray(lab[..., 4:7]) == 1, axis=-1)
        supervised = np.zeros(mask.shape, bool)
        for k in (4, 5):
            supervised |= (np.asarray(lab[..., k]) == 1) & (np.asarray(wgt[..., k], dtype=np.float32) > 0)
        z_live = np.asarray(wgt[..., Z_OFFSET_HEAD], dtype=np.float32) > 0
        rows.append(dict(stem=stem, source=entry['source'], t=entry['t'], z=entry['z'],
                         regions=';'.join(sorted(set(entry['regions']))),
                         corrected_px=int(mask.sum()),
                         removed_supervised_positive_px=int((mask & supervised).sum()),
                         removed_stored_positive_px=int((mask & stored).sum()),
                         desupervised_z_offset_px=int((mask & z_live).sum())))
    preview = pd.DataFrame(rows)
    clash = contradictions(session, state)
    if verbose:
        print(f'{len(approved)} approved artifact regions -> {len(preview)} training crops to overlay; '
              f'{len(withheld)} approved regions have no eligible training crop and will only be reported')
        if len(preview):
            print(f'  {int(preview.corrected_px.sum())} pixels set to nuclear target 0 with weight {weight}')
            print(f'  {int(preview.removed_supervised_positive_px.sum())} of them are currently '
                  'supervised positives; '
                  f'{int(preview.desupervised_z_offset_px.sum())} Z-offset pixels will be de-supervised')
        counts = state.table().decision.value_counts().to_dict()
        print('  decisions:', counts)
        print('  mixed_redraw / uncertain / skip produce NO correction.')
        if len(clash):
            print(f'  {len(clash)} CONTRADICTORY overlaps — a commit will abort.')
    return dict(patches=preview, withheld=pd.DataFrame(withheld), contradictions=clash,
                per_patch=per_patch, approved=[a[0] for a in approved])


def commit(session, state, *, weight=1.0, verbose=True):
    """Explicit commit. Writes a NEW versioned directory; existing pools untouched.

    Idempotent: a completed commit recorded in the review state is returned rather
    than written twice. Separate from preparing a new model and from any training.
    """
    if state.committed:
        if verbose:
            print('Already committed:', state.committed)
        return Path(state.committed)
    pending = state.pending()
    if pending:
        raise ValueError(f'{len(pending)} candidates still need a decision and a reviewer '
                         '(uncertain and skip are valid completed reviews).')
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError('Weight must be positive and finite.')
    if session.scored_regions is None:
        raise RuntimeError('Run the audit cell first: the commit records the full region '
                           'census alongside the decisions.')
    state._verify_masks()
    _check_sources_unchanged(session, verbose=verbose)

    plan = preview_commit(session, state, weight=weight, verbose=False)
    if len(plan['contradictions']):
        raise ValueError('Contradictory overlapping decisions; resolve them before committing:\n'
                         + plan['contradictions'].to_string(index=False))
    if not plan['per_patch'] and not plan['withheld']:
        raise ValueError('No approved artifact regions; nothing to commit.')

    root = Path(session.cfg.out_root) / ('training_patch_audit_commit_'
                                         + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema=1, status='staging', created=datetime.now().isoformat(),
                    review_dir=str(state.root), review_version=state.version,
                    candidate_signature=state.signature, weight=float(weight),
                    policy=('explicitly approved exact regions only; nuclear heads 4/5/6 set to '
                            'target 0 with weight; Z-offset de-supervised over the same pixels; '
                            'all other heads, channels and pixels unchanged; sparse overlays '
                            'composed on top of the CURRENT effective labels'),
                    source_provenance=session.provenance, patches=[], withheld=plan['withheld'].to_dict('records'))
    _json(root / 'manifest.json', manifest)
    try:
        for folder in ('labels', 'weights', 'masks', 'review'):
            (root / folder).mkdir()
        for region_id in plan['approved']:
            shutil.copy2(state.masks_dir / (region_id + '.npy'), root / 'masks' / (region_id + '.npy'))
        for name in ('candidates.csv', 'review_state.json', 'provenance.json', 'history.jsonl'):
            if (state.root / name).is_file():
                shutil.copy2(state.root / name, root / 'review' / name)
        state.table().to_csv(root / 'review' / 'decisions.csv', index=False)
        session.scored_regions.to_csv(root / 'review' / 'region_census.csv', index=False)

        for stem, entry in sorted(plan['per_patch'].items()):
            original_lab, original_wgt = Path(entry['lab']), Path(entry['wgt'])
            before = (hn._digest(original_lab), hn._digest(original_wgt))
            lab = np.load(original_lab, allow_pickle=False)
            wgt = np.load(original_wgt, allow_pickle=False)
            if lab.shape[-1] != 10 or wgt.shape != lab.shape[:2] + (9,):
                raise ValueError('Unexpected label/weight layout for ' + stem)
            mask = entry['mask']
            stored = np.any(lab[..., 4:7] == 1, axis=-1)
            z_live = wgt[..., Z_OFFSET_HEAD] > 0
            item = dict(stem=stem, source=entry['source'], t=entry['t'], z=entry['z'],
                        regions=sorted(set(entry['regions'])), corrected_px=int(mask.sum()),
                        removed_stored_positive_px=int((mask & stored).sum()),
                        desupervised_z_offset_px=int((mask & z_live).sum()),
                        effective_lab=str(original_lab), effective_lab_sha256=before[0],
                        effective_wgt=str(original_wgt), effective_wgt_sha256=before[1])
            for k in NUCLEAR_WRITE_HEADS:
                lab[..., k][mask] = 0
                wgt[..., k][mask] = weight
            # A nucleus that is not there has no depth: never leave a live regression
            # target behind a removed mask.
            lab[..., Z_OFFSET_HEAD][mask] = UNANNOTATED
            wgt[..., Z_OFFSET_HEAD][mask] = 0
            for key, folder, array in (('lab', 'labels', lab), ('wgt', 'weights', wgt)):
                path = root / folder / (stem + '.npy')
                with path.open('xb') as stream:
                    np.save(stream, array, allow_pickle=False)
                item[key] = str(path.relative_to(root))
                item[key + '_sha256'] = hn._digest(path)
            after = (hn._digest(original_lab), hn._digest(original_wgt))
            if after != before:
                raise RuntimeError('Source patch changed while committing: ' + stem)
            manifest['patches'].append(item)

        plan['withheld'].to_csv(root / 'withheld_regions.csv', index=False)
        plan['patches'].to_csv(root / 'affected_patches.csv', index=False)
        held = session.scored_regions[session.scored_regions.n_correctable_patches.eq(0)]
        held.to_csv(root / 'holdout_only_regions.csv', index=False)
        manifest.update(status='complete', progress=state.progress(),
                        n_patches=len(manifest['patches']),
                        n_regions=len(plan['approved']),
                        holdout_only_regions=int(len(held)))
        _json(root / 'manifest.json', manifest)
        state.committed = str(root)
        state._save('commit')
        if verbose:
            print('Committed:', root)
            print(f'  {len(manifest["patches"])} overlay patches, {len(plan["approved"])} approved regions, '
                  f'{len(plan["withheld"])} approved-but-withheld regions')
            print(f'  {int(len(held))} holdout-only regions reported, not corrected')
            print('  Original pools, the 185-patch correction pool, the hard-negative pool, '
                  'validation and the temporal holdout are byte-identical.')
            print('  No model prepared and no training started.')
        return root
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        _json(root / 'manifest.json', manifest)
        raise


def _check_sources_unchanged(session, *, verbose=True):
    """Re-verify every provenance hash the audit was built on."""
    provenance = session.provenance
    checks = [
        ('raw image', hn._identity(session.cfg.image_file), provenance['raw_image']),
        ('gold pool manifest', hn._digest(session.cfg.reviewed_root / 'manifest.json'),
         provenance['gold_pool_manifest_sha256']),
        ('commit manifest', hn._digest(Path(provenance['commit_dir']) / 'manifest.json'),
         provenance['commit_manifest_sha256']),
    ]
    for key, name in (('correction_pool', 'gold correction manifest'),
                      ('negative_pool', 'hard-negative manifest')):
        if provenance.get(key):
            checks.append((name, hn._digest(Path(provenance[key]) / 'manifest.json'),
                           provenance[key.replace('_pool', '') + '_manifest_sha256']))
    if hn._training_signature(session.cfg) != provenance['training_signature']:
        raise RuntimeError('Training signature changed since the audit; rerun the audit cell.')
    for name, observed, expected in checks:
        if observed != expected:
            raise RuntimeError(f'{name} changed since the audit; rerun the audit cell.')
    if verbose:
        print('Provenance re-verified:', ', '.join(name for name, _, _ in checks),
              'and the training signature.')
    return True


def audit_corrected_rows(commit_dir, cfg, rows):
    """Reader that applies a committed audit overlay. Enabling it is a separate step.

    Deliberately NOT wired into training here: this notebook commits decisions and
    stops. A later run must opt in explicitly, with a new model identity.
    """
    root = Path(commit_dir).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Incomplete audit commit')
    if manifest['source_provenance']['training_signature'] != hn._training_signature(cfg):
        raise ValueError('Audit commit was built for a different input/split contract')
    changes = {p['stem']: p for p in manifest['patches']}
    result, seen = [], set()
    for row in rows:
        item = dict(row)
        patch = changes.get(item['stem'])
        if patch is not None:
            seen.add(item['stem'])
            if hn.holdout_reason(item['t'], item['cy'], item['cx'], cfg) != 'train':
                raise ValueError('Holdout correction refused: ' + item['stem'])
            for key in ('lab', 'wgt'):
                if str(Path(item[key]).resolve()) != patch['effective_' + key] \
                        or hn._digest(item[key]) != patch['effective_' + key + '_sha256']:
                    raise ValueError('Effective label changed since the audit commit: ' + item['stem'])
                path = (root / patch[key]).resolve()
                if root not in path.parents or hn._digest(path) != patch[key + '_sha256']:
                    raise ValueError('Audit overlay file changed: ' + item['stem'])
                item[key] = path
            item['audit_commit_manifest'] = str(root / 'manifest.json')
        result.append(item)
    if seen != set(changes):
        raise ValueError('Audit commit does not match the supplied rows')
    return result


# -------------------------------------------------------------- notebook UI

def audit_cell(run_dir=None, commit_dir=None, **kwargs):
    """One contained audit/setup cell: provenance, census, nomination, review."""
    from IPython.display import display
    import vulcan_audit_recovery as vr
    session = tpa.run_audit(run_dir or vr.DEFAULT_RUN_DIR, commit_dir or vr.DEFAULT_COMMIT_DIR, **kwargs)
    summary = session.summary()
    print('\n--- effective supervision census ---')
    display(summary['by_split'])
    display(summary['by_timepoint'])
    print('\n--- candidates ---')
    columns = ['region_id', 'kind', 't', 'z', 'area_um2', 'n_correctable_patches',
               'priority_score', 'flags']
    display(session.candidates[columns].head(30))
    state = ReviewState(session)
    print('\nReview progress:', state.progress())
    return session, state


def decision_cell(session, state):
    """Interactive resolver: inspect, classify, autosave, then commit explicitly."""
    import ipywidgets as w
    from IPython.display import display, clear_output

    ids = list(state.decisions)
    order = [0]
    heading = w.HTML()
    picture = w.Output()
    log = w.Output()
    reviewer = w.Text(value=state.reviewer or 'tdeibert', description='Reviewer:')
    choice = w.Dropdown(description='Decision:', layout=w.Layout(width='620px'), options=[
        ('Choose…', ''),
        ('Keep real nuclear positive (no change written)', 'real'),
        ('Confirmed artifact negative — the ENTIRE cyan region', 'artifact_negative'),
        ('Mixed / needs local redraw — left unresolved', 'mixed_redraw'),
        ('Uncertain — reviewed, no correction', 'uncertain'),
        ('Skip — reviewed, no correction', 'skip')])
    failure = w.Dropdown(description='Failure:', options=list(FAILURE_TYPES),
                         layout=w.Layout(width='420px'))
    notes = w.Textarea(description='Notes:', layout=w.Layout(width='95%'))
    jump = w.Dropdown(options=ids, description='Candidate:', layout=w.Layout(width='620px'))
    kind_filter = w.Dropdown(options=['all', 'flagged', 'random_sample', 'negative_control', 'pending'],
                             value='pending', description='Filter:')
    previous = w.Button(description='< Previous')
    following = w.Button(description='Next >')
    save = w.Button(description='Save & next', button_style='info')
    undo = w.Button(description='Clear decision')
    progress = w.HTML()
    preview = w.Button(description='Preview commit', button_style='warning')
    commit_button = w.Button(description='COMMIT decisions', button_style='success',
                             layout=w.Layout(width='220px'))

    def visible():
        if kind_filter.value == 'all':
            return ids
        if kind_filter.value == 'pending':
            return state.pending() or ids
        return [r for r in ids if state.decisions[r].get('kind') == kind_filter.value] or ids

    def refresh():
        current = visible()
        order[0] %= max(len(current), 1)
        region_id = current[order[0]]
        decision = state.decisions[region_id]
        record = state.record(region_id)
        heading.value = (f'<b>{order[0] + 1}/{len(current)} — {html.escape(region_id)}</b> '
                         f'| {record.kind} | t{int(record.t)} z{int(record.z)} '
                         f'| {float(record.area_um2):.1f} um^2 '
                         f'| flags: {html.escape(str(record.flags) or "none")}')
        choice.value = decision['decision']
        reviewer.value = decision['reviewer'] or reviewer.value
        failure.value = decision['failure_type'] if decision['failure_type'] in FAILURE_TYPES else ''
        notes.value = decision['notes']
        if jump.value != region_id:
            jump.unobserve(jumped, names='value')
            jump.value = region_id
            jump.observe(jumped, names='value')
        stats = state.progress()
        progress.value = (f'decided {stats["decided"]}/{stats["total"]}, pending {stats["pending"]} '
                          f'| {html.escape(str(stats["by_decision"]))}'
                          + (f' | COMMITTED: {html.escape(str(state.committed))}' if state.committed else ''))
        commit_button.disabled = bool(stats['pending']) or bool(state.committed)
        with picture:
            clear_output(wait=True)
            show_candidate(session, state, region_id)

    def guarded(action):
        with log:
            clear_output(wait=True)
            try:
                action()
            except Exception as exc:
                print(f'{type(exc).__name__}: {exc}')

    def move(step):
        order[0] = (order[0] + step) % max(len(visible()), 1)
        refresh()

    def do_save():
        current = visible()
        state.save_decision(current[order[0]], choice.value, reviewer.value, failure.value, notes.value)
        move(1)

    def jumped(change):
        if change['name'] == 'value' and change['new'] in visible():
            order[0] = visible().index(change['new'])
            guarded(refresh)

    previous.on_click(lambda _: guarded(lambda: move(-1)))
    following.on_click(lambda _: guarded(lambda: move(1)))
    save.on_click(lambda _: guarded(do_save))
    undo.on_click(lambda _: guarded(lambda: (state.clear_decision(visible()[order[0]]), refresh())))
    kind_filter.observe(lambda c: guarded(lambda: (order.__setitem__(0, 0), refresh()))
                        if c['name'] == 'value' else None, names='value')
    jump.observe(jumped, names='value')

    def do_preview():
        from IPython.display import display as show
        plan = preview_commit(session, state)
        show(plan['patches'].head(40))
        if len(plan['withheld']):
            print('Withheld (approved but no eligible training crop):')
            show(plan['withheld'])

    def do_commit():
        commit(session, state)
        save.disabled = choice.disabled = notes.disabled = failure.disabled = True
        refresh()

    preview.on_click(lambda _: guarded(do_preview))
    commit_button.on_click(lambda _: guarded(do_commit))

    display(w.VBox([
        w.HTML('<b>Exact-region review.</b> Cyan is the exact region under review; red is a '
               'previously confirmed artifact. Only <i>confirmed artifact negative</i> writes '
               'anything, and only over those pixels. Mixed/uncertain/skip are recorded and '
               'produce no correction. Decisions autosave; reopening resumes.'),
        heading, w.HBox([jump, kind_filter]), reviewer, choice, failure, notes,
        w.HBox([previous, following, save, undo]), progress, picture,
        w.HBox([preview, commit_button]), log]))
    guarded(refresh)
    return state
