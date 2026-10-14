"""Resume the October 13 hard-negative checkpoint on an existing GPU allocation."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from contextlib import contextmanager
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
RUN = Path('/home/tdeibert/Projects/Nuclear_Scaling/Models/runs/Vulcan_2.5.4_hardneg_hn2392da6c18e0_iezqw_a0cf3b2a13')
COMMIT = Path('/data/user/tdeibert/Nuclear_Scaling/Inputs/Training_Data/reviewed_training_commit_20261013_121152_980805')
CELL_IDS = ['cbaa77e08eb9','58d1f5526de3','66ee6d9017b3','6e5e547eb474','e7f6e35ef96d',
            '5aa0324330f6','8bc68c549bac','b838b3114857','ba0491fe338c','c52032b66821',
            'd8c5dfd59f31','19c38bfa45f3','3e7454f20e82','35d2cac9','895c4f61',
            'a91732d0','8763f7e6','5aa245d6','c252f9a7']


@contextmanager
def stage(title):
    print('\n'+title,flush=True)
    stop=threading.Event()
    bar=tqdm(total=None,desc=title,bar_format='{desc} | elapsed {elapsed}',mininterval=10)
    def heartbeat():
        while not stop.wait(30):bar.refresh()
    worker=threading.Thread(target=heartbeat,daemon=True);worker.start()
    try:yield
    finally:stop.set();worker.join();bar.close()


def remaining(job):
    value=subprocess.check_output(['squeue','-h','-j',job,'-o','%L'],text=True).strip()
    days=0
    if '-' in value:day,value=value.split('-',1);days=int(day)
    parts=list(map(int,value.split(':')))
    if len(parts)==2:parts=[0]+parts
    if len(parts)!=3:raise ValueError('Cannot determine allocation time remaining')
    return days*86400+parts[0]*3600+parts[1]*60+parts[2]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job-id',required=True);ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args();os.environ.setdefault('MPLBACKEND','Agg')
    os.chdir(HERE)
    lock=(RUN/'resume_hardneg.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cells={c.get('id'):''.join(c.get('source',[])) for c in json.loads((HERE/'vulcan_training_2_5_3.ipynb').read_text())['cells']}
    ns={'__name__':'__main__','__file__':str(HERE/'vulcan_training_2_5_3.ipynb')}
    with stage('Load baseline definitions'):
        for cid in tqdm(CELL_IDS,desc='Notebook definition cells',unit='cell'):
            source=cells[cid]
            if cid=='8763f7e6':
                anchor='hist = model.fit(train_ds, validation_data=val_ds'
                assert anchor in source
                source=source.replace(anchor,'cbs.append(ResumeProgress())\n    '+anchor,1)
            exec(compile(source,'notebook:'+cid,'exec'),ns)
            for key in list(ns):
                if key.startswith('RUN_'):ns[key]=False
    tf=ns['tf']
    if not tf.config.list_physical_devices('GPU'):raise RuntimeError('No GPU available')
    class ResumeProgress(tf.keras.callbacks.Callback):
        def on_epoch_begin(self,epoch,logs=None):
            self.bar=tqdm(total=self.params.get('steps'),desc=f'Epoch {epoch+1}',unit='batch',mininterval=30)
        def on_train_batch_end(self,batch,logs=None):self.bar.update(1)
        def on_test_begin(self,logs=None):
            self.validation_start=time.monotonic();print('\nValidation started',flush=True)
        def on_test_end(self,logs=None):print(f'Validation finished in {time.monotonic()-self.validation_start:.0f}s',flush=True)
        def on_epoch_end(self,epoch,logs=None):self.bar.close()
        def on_train_end(self,logs=None):
            if hasattr(self,'bar'):self.bar.close()
    ns['ResumeProgress']=ResumeProgress
    import gold_artifact_corrections as gc
    import reviewed_hard_negatives as hn
    commit=json.loads((COMMIT/'manifest.json').read_text());assert commit['status']=='complete'
    baseline=ns['cfg'];assert baseline.model_name=='Vulcan_2.5.3'
    with stage('Validate and restore corrected gold pool'):
        gc.enable_corrections(ns,baseline,commit['correction_pool'])
    cfg=copy.copy(baseline);cfg.__dict__.pop('_frozen_run_hash',None);cfg.model_name='Vulcan_2.5.4_hardneg'
    with stage('Validate and restore hard-negative pool'):
        hn.enable_hard_negative_training(ns,cfg,commit['negative_pool'],baseline_cfg=baseline)
    ns['cfg']=cfg
    assert cfg is ns['p3_split'].prepared_cfg
    assert cfg.run_dir.resolve()==RUN.resolve(), 'Run configuration changed; refusing a different run'
    state=json.loads((RUN/'resume/s42/state.json').read_text())
    assert (RUN/'resume/s42/checkpoint').exists()
    print('Verified resume state:',state,flush=True)
    if args.check_only:return
    budget=remaining(args.job_id)-45*60
    if budget<30*60:raise RuntimeError('Insufficient remaining time for another epoch plus checkpoint margin')
    print(f'Training budget {budget/3600:.2f} hours; 45-minute allocation margin. Resuming after epoch {state["epoch"]}.',flush=True)
    with stage('Training and checkpointing'):
        result=ns['p3_train'](cfg,time_budget_s=budget)
    print('Training returned:',result['status'],flush=True)


if __name__=='__main__':main()
