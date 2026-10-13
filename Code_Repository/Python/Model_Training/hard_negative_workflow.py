"""Two-cell audit/review workflow with staged decisions and an explicit commit."""
import copy
import html
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import reviewed_hard_negatives as hn
import gold_artifact_corrections as gc
from hard_negative_conflict_audit import audit_conflicts, plot_conflicts

CONFIRMED = 'Confirmed artifact; conflicting gold annotation requires correction.'


class Workflow:
    def __init__(self, namespace, review_dir):
        if namespace.get('_HN_SPLIT_ENABLED') or namespace.get('_GOLD_CORRECTION_ENABLED'):
            raise RuntimeError('Existing training hooks are active. Restore baseline F1 definitions and baseline cfg before auditing.')
        for key in ('cfg', 'p3_rows', 'p3_split', 'extract_input_stack', 'ensure_norm_stats'):
            if key not in namespace: raise RuntimeError('Load baseline configuration and F1 helpers first: missing '+key)
        self.ns = namespace; self.cfg = namespace['cfg']; self.root = Path(review_dir).resolve()
        self.manifest = json.loads((self.root/'candidates.json').read_text())
        self.review_hash = hn._digest(self.root/'review.csv')
        self.manifest_hash = hn._digest(self.root/'candidates.json')
        self.signature = hn._training_signature(self.cfg)
        self.rows = namespace['p3_rows'](self.cfg)
        self.train, self.val = namespace['p3_split'](self.cfg, verbose=False)
        self.audit = audit_conflicts(self.root, self.cfg, self.rows)
        self.table = pd.read_csv(self.root/'review.csv').fillna('')
        self.records = {r['candidate_id']: r for r in self.manifest['candidates']}
        self.decisions = {}
        for row in self.table.itertuples():
            confirmed = (str(row.notes).startswith(CONFIRMED)
                         or getattr(row, 'conflict_resolution', '') == 'negative')
            pending = row.candidate_id in self.audit['objects'] and not confirmed
            self.decisions[row.candidate_id] = dict(
                decision='' if pending else row.decision, reviewer=row.reviewer, notes=row.notes)
        self.draft = Path(self.cfg.out_root)/('hard_negative_draft_'+self.review_hash[:16]+'.json')
        if self.draft.exists():
            saved = json.loads(self.draft.read_text())
            if saved['manifest_hash'] == self.manifest_hash and set(saved['decisions']) == set(self.decisions):
                self.decisions = saved['decisions']
        self.committed = None
        self.activated = False

    def check_current(self):
        if hn._digest(self.root/'review.csv') != self.review_hash or hn._digest(self.root/'candidates.json') != self.manifest_hash:
            raise RuntimeError('Source review changed. Rerun the audit cell to load it.')
        if hn._training_signature(self.cfg) != self.signature:
            raise RuntimeError('Configuration changed. Rerun the audit.')

    def save_draft(self, cid, decision, reviewer, notes):
        if self.committed: raise RuntimeError('This review has already been committed.')
        self.check_current()
        if cid not in self.decisions or decision not in ('real','artifact_negative','skip','uncertain'):
            raise ValueError('Choose gold, negative, mixed/skip, or uncertain.')
        if not reviewer.strip(): raise ValueError('Enter a reviewer name.')
        updated = copy.deepcopy(self.decisions)
        updated[cid] = dict(decision=decision,reviewer=reviewer.strip(),notes=notes)
        self.draft.parent.mkdir(parents=True,exist_ok=True)
        tmp = self.draft.with_name(self.draft.name+'.'+datetime.now().strftime('%H%M%S_%f')+'.tmp')
        hn._json(tmp,dict(manifest_hash=self.manifest_hash,decisions=updated))
        os.replace(tmp,self.draft);self.decisions=updated

    def pending(self):
        return [cid for cid,d in self.decisions.items() if not d['decision'] or not d['reviewer'].strip()]

    def plot(self,cid):
        if cid in self.audit['objects']:
            plot_conflicts(self.audit,self.cfg,candidate_ids=[cid],count=1);return
        rec=self.records[cid];mp=(self.root/rec['mask_file']).resolve()
        if self.root not in mp.parents or hn._digest(mp)!=rec['mask_sha256']:raise ValueError('Reviewed mask changed')
        mask=np.load(mp,allow_pickle=False).astype(bool)
        obj=dict(record=rec,mask=mask,positive_union=np.zeros_like(mask),overlap_pixels=0,pairs=[])
        plot_conflicts(dict(objects={cid:obj}),self.cfg,candidate_ids=[cid],count=1)

    def commit(self):
        if self.committed:return self.committed
        self.check_current()
        if self.pending(): raise ValueError(f'{len(self.pending())} decisions still need saving.')
        if self.ns.get('_HN_SPLIT_ENABLED') or self.ns.get('_GOLD_CORRECTION_ENABLED'):
            raise RuntimeError('Restore baseline readers before committing.')
        # Revalidate baseline pool readers before writing a new version.
        fresh=self.ns['p3_rows'](self.cfg)
        if [(r['source'],r['stem'],str(r['lab']),str(r['wgt'])) for r in fresh] != [(r['source'],r['stem'],str(r['lab']),str(r['wgt'])) for r in self.rows]:
            raise RuntimeError('Baseline rows changed; rerun audit.')
        root=Path(self.cfg.out_root)/('reviewed_training_commit_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        root.mkdir(parents=True,exist_ok=False)
        status=dict(status='building',source_review=str(self.root),source_review_sha256=self.review_hash)
        hn._json(root/'manifest.json',status)
        try:
            snapshot=root/'review';snapshot.mkdir();(snapshot/'masks').mkdir()
            table=self.table.copy();manifest=copy.deepcopy(self.manifest)
            for rec in manifest['candidates']:
                src=(self.root/rec['mask_file']).resolve()
                if self.root not in src.parents or hn._digest(src)!=rec['mask_sha256']:raise ValueError('Mask changed: '+rec['candidate_id'])
                relative='masks/'+rec['candidate_id']+'.npy';shutil.copy2(src,snapshot/relative);rec['mask_file']=relative
                cid=rec['candidate_id'];d=self.decisions[cid];sel=table.candidate_id.eq(cid)
                table.loc[sel,['decision','reviewer','notes']]=[d['decision'],d['reviewer'],d['notes']]
                table.loc[sel,'mask_file']=relative
                table.loc[sel,'conflict_resolution']='negative' if d['decision']=='artifact_negative' else d['decision']
                if d['decision']=='real':table.loc[sel,'failure_type']='real_nucleus'
                if d['decision']=='skip':table.loc[sel,'failure_type']='mixed'
            table.to_csv(snapshot/'review.csv',index=False);hn._json(snapshot/'candidates.json',manifest)
            final_audit=audit_conflicts(snapshot,self.cfg,self.rows)
            safe_keys={(r['source'],r['stem']) for r in self.train if hn.holdout_reason(r['t'],r['cy'],r['cx'],self.cfg)=='train'}
            correction_needed=any(self.decisions[cid]['decision']=='artifact_negative' and
                any(('gold',p['gold_stem']) in safe_keys for p in obj['pairs']) for cid,obj in final_audit['objects'].items())
            correction=None;rows=self.rows
            if correction_needed:
                correction=gc.build_corrections(snapshot,root/'gold_corrections',self.cfg,rows,self.train)
                rows=gc.corrected_rows(correction,self.cfg,rows)
            negative=None
            # A batch containing only gold/skip decisions can still be committed.
            if table.decision.eq('artifact_negative').any():
                try:
                    negative=hn.build_reviewed_negative_pool(snapshot,root/'hard_negatives',self.cfg,
                        gold_rows=rows,extract_input_stack=self.ns['extract_input_stack'],ensure_norm_stats=self.ns['ensure_norm_stats'])
                except ValueError as exc:
                    if str(exc)!='No training-safe patches; review output holdout policy before proceeding':raise
                    print('All negative tiles were withheld by the holdout policy; no negative pool will be enabled.')
            status.update(status='complete',review=str(snapshot),correction_pool=str(correction) if correction else None,
                          negative_pool=str(negative) if negative else None,decisions=table.decision.value_counts().to_dict())
            hn._json(root/'manifest.json',status);self.committed=root
            print('Committed review and training pools:',root)
            print(status['decisions']);return root
        except Exception as exc:
            status.update(status='failed',error=str(exc));hn._json(root/'manifest.json',status)
            raise

    def prepare_training(self,model_name):
        if not self.committed:raise RuntimeError('Commit decisions first.')
        if self.activated:raise RuntimeError('New model is already prepared.')
        m=json.loads((self.committed/'manifest.json').read_text())
        if not model_name.strip() or model_name==self.cfg.model_name:raise ValueError('Enter a new model name.')
        if not m['negative_pool']:raise RuntimeError('This commit has no training-safe negative pool to enable.')
        new=copy.copy(self.cfg);new.__dict__.pop('_frozen_run_hash',None);new.model_name=model_name.strip()
        saved={key:self.ns.get(key) for key in ('p3_rows','p3_split','_GOLD_CORRECTION_ENABLED','_HN_SPLIT_ENABLED')}
        try:
            if m['correction_pool']:gc.enable_corrections(self.ns,self.cfg,m['correction_pool'])
            hn.enable_hard_negative_training(self.ns,new,m['negative_pool'],baseline_cfg=self.cfg)
        except Exception:
            for key,value in saved.items():
                if value is None:self.ns.pop(key,None)
                else:self.ns[key]=value
            raise
        self.prepared_cfg=new
        bindings=dict(BASELINE_TRAINING_CFG=self.cfg,cfg=new,HN_PREPARED_CFG=new,
                      HN_POOL_DIR=Path(m['negative_pool']),RUN_TRAIN=False)
        self.ns.update(bindings)
        # Widget callbacks run outside ordinary cell execution. Publish to the
        # interactive namespace as well as the captured function-global mapping.
        from IPython import get_ipython
        shell=get_ipython()
        if shell is not None:shell.push(bindings)
        self.activated=True
        print('Prepared',new.model_name,'— RUN_TRAIN remains False.')


def audit_cell(namespace,review_dir):
    from IPython.display import display
    workflow=Workflow(namespace,review_dir)
    display(workflow.table.groupby(['t','decision']).size().unstack(fill_value=0))
    display(workflow.audit['summary'])
    print(len(workflow.table),'candidates;',len(workflow.pending()),'decisions pending. Existing saved resolutions were loaded.')
    if workflow.audit['objects']:
        workflow.plot(next(iter(workflow.audit['objects'])))
    return workflow


def decision_cell(workflow):
    import ipywidgets as w
    from IPython.display import display,clear_output
    ids=list(workflow.decisions);pending=workflow.pending();position=[ids.index(pending[0]) if pending else 0]
    heading=w.HTML();picture=w.Output();log=w.Output()
    reviewer=w.Text(value='tdeibert',description='Reviewer:')
    choice=w.Dropdown(options=[('Choose…',''),('Gold — retain existing nuclear label','real'),
        ('Negative — entire cyan mask is artifact','artifact_negative'),('Mixed — skip','skip'),('Uncertain — exclude','uncertain')],
        description='Decision:',layout=w.Layout(width='570px'))
    notes=w.Textarea(description='Notes:',layout=w.Layout(width='95%'))
    jump=w.Dropdown(options=ids,description='Candidate:')
    previous=w.Button(description='Previous (no save)');next_button=w.Button(description='Next (no save)')
    save=w.Button(description='Save draft & next',button_style='info');next_pending=w.Button(description='Next pending')
    commit=w.Button(description='Commit all decisions',button_style='success',layout=w.Layout(width='200px'))
    model=w.Text(value='Vulcan_2.5.4_hardneg',description='New model:')
    prepare=w.Button(description='Prepare new model',disabled=not bool(workflow.committed))
    def show():
        cid=ids[position[0]];d=workflow.decisions[cid]
        heading.value=f'<b>{position[0]+1}/{len(ids)} — {html.escape(cid)}</b> | {len(workflow.pending())} pending'
        choice.value=d['decision'];reviewer.value=d['reviewer'] or reviewer.value;notes.value=d['notes']
        commit.disabled=bool(workflow.pending()) or bool(workflow.committed)
        with picture:
            clear_output(wait=True);workflow.plot(cid)
    def edited(change):
        d=workflow.decisions[ids[position[0]]]
        dirty=(choice.value,reviewer.value,notes.value)!=(d['decision'],d['reviewer'],d['notes'])
        commit.disabled=dirty or bool(workflow.pending()) or bool(workflow.committed)
    for field in (choice,reviewer,notes):field.observe(edited,names='value')
    def execute(fn):
        with log:
            clear_output(wait=True)
            try:fn()
            except Exception as exc:print(type(exc).__name__+': '+str(exc))
    def move(step):position[0]=(position[0]+step)%len(ids);show()
    def save_next():
        workflow.save_draft(ids[position[0]],choice.value,reviewer.value,notes.value);move(1)
    def go_pending():
        pending=workflow.pending()
        if pending:position[0]=ids.index(pending[0]);show()
    def do_commit():
        commit.disabled=True
        try:
            workflow.commit();prepare.disabled=False;save.disabled=True
            choice.disabled=True;notes.disabled=True;reviewer.disabled=True
            print('Decisions committed. Training has not started.')
        finally:commit.disabled=bool(workflow.committed) or bool(workflow.pending())
    previous.on_click(lambda _:execute(lambda:move(-1)));next_button.on_click(lambda _:execute(lambda:move(1)))
    save.on_click(lambda _:execute(save_next));next_pending.on_click(lambda _:execute(go_pending))
    commit.on_click(lambda _:execute(do_commit));prepare.on_click(lambda _:execute(lambda:workflow.prepare_training(model.value)))
    def jumped(change):
        if change['name']=='value':position[0]=ids.index(change['new']);execute(show)
    jump.observe(jumped,names='value')
    display(w.VBox([w.HTML('Gold retains existing annotations. Negative corrects overlapping training gold and builds negative patches. Save drafts, then commit the complete batch.'),
        heading,jump,reviewer,choice,notes,w.HBox([previous,next_button,save,next_pending]),picture,commit,w.HBox([model,prepare]),log]))
    show()
    return workflow
