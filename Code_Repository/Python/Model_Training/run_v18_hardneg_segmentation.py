"""Execute v18.1 through segmentation, using its configured hard-negative model."""
import fcntl
import json
import os
from pathlib import Path
import time
from tqdm import tqdm


def main():
    os.environ.setdefault('MPLBACKEND','Agg')
    notebook=Path(__file__).resolve().parent.parent/'Image_Segmentation/Large_FOV_Nuclear_Pipeline_v18.1.ipynb'
    os.chdir(notebook.parent)
    nb=json.loads(notebook.read_text())
    ns={'__name__':'__main__','__file__':str(notebook)}
    lock=None;started=time.monotonic()
    cells=[]
    for cell in nb['cells']:
        if cell['cell_type']=='code':cells.append(cell)
        if cell.get('id')=='8133a6bd':break
    assert cells[-1]['id']=='8133a6bd'
    for cell in tqdm(cells,desc='v18.1 setup and segmentation',unit='cell'):
        print(f'\nCell {cell["id"]}; elapsed {(time.monotonic()-started)/60:.1f} min',flush=True)
        exec(compile(''.join(cell['source']),str(notebook)+':'+cell['id'],'exec'),ns)
        if cell['id']=='953ca2cd':
            cfg=ns['cfg']
            if 'Vulcan_2.5.4_hardneg_hn2392da6c18e0' not in cfg.model_name:raise ValueError('Wrong model selected')
            lock=(ns['run'].run_dir/'segmentation_runner.lock').open('a')
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            print('COMPARISON RUN:',ns['run'].run_dir,flush=True)
        if cell['id']=='07e8d944':ns['RUN_SINGLE_PLANE_DEBUG']=False
    print('Segmentation complete:',ns['run'].run_dir,flush=True)
    print('Continue object extraction and population analysis in v18.1 to compare results.',flush=True)


if __name__=='__main__':main()
