"""In-memory label semantics and isolated temporary-pool integration checks."""
import ast
import copy
from dataclasses import dataclass, field
import hashlib
import itertools
import json
from pathlib import Path
import re
import tempfile
import unittest
import zipfile
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import filters, morphology, measure
import roifile
import tifffile as tiff
try:
    import cv2
except ImportError:
    cv2=None
from nuclear_scaling.roi_zcluster import assign_z

ROOT=Path(__file__).resolve().parent
nb=json.loads((ROOT/'vulcan_training_2_5_3.ipynb').read_text(encoding='utf8'))
ns=dict(globals(),PROJECT_ROOT=Path('/tmp'),DATA_ROOT=Path('/tmp'),
        CONTROL_DATA_DIR=Path('/tmp'),TRAINING_DATA_ROOT=Path('/tmp'))
schema=next(''.join(c['source']) for c in nb['cells'] if 'UNANNOTATED = 255' in ''.join(c['source']))
exec(schema,ns)
config=next(''.join(c['source']) for c in nb['cells'] if 'class PipelineConfig:' in ''.join(c['source']))
exec(config,ns)
cfg=ns['PipelineConfig'](pixel_size_um=.5,z_step_um=2.,patch_size=64,n_z_context=3,
                        nucleus_channel_idx=0,npc_channel_idx=1,membrane_channel_idx=2,
                        normalization_mode='per_patch')
ns['cfg']=cfg
for cell in nb['cells']:
    if cell['cell_type']!='code': continue
    source=''.join(cell['source'])
    if any(anchor in source for anchor in ('def load_memmap_tiff(', 'def _circle_from_3(',
        'def _sphere_lsq(', 'def nucleus_fits_droplet_chord(',
        'def nucleus_envelope_shell(', 'def build_input_patch(')):
        tree=ast.parse(source)
        tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Import,ast.ImportFrom))]
        exec(compile(tree,'notebook_definitions','exec'),ns)
exec((ROOT/'vulcan_phase2.py').read_text(encoding='utf8'),ns)

def roi(i,t,z,cx,cy,r):
    theta=np.linspace(0,2*np.pi,120,endpoint=False)
    poly=np.c_[cx+r*np.cos(theta),cy+r*np.sin(theta)]
    local=ns['p2_roi_local'](poly,(128,128))
    obj=roifile.ImagejRoi.frompoints(poly,name=f'roi{i}',t=t,z=z)
    return dict(i=i,t=t,z=z,poly=poly,loc=local,area=int(local[1].sum()),
                name=f'roi{i}',original=obj,cls='nucleus_interior')

class Phase2Tests(unittest.TestCase):
    def test_embedded_source_matches(self):
        module=(ROOT/'vulcan_phase2.py').read_text(encoding='utf8')
        self.assertTrue(any(''.join(c['source'])==module for c in nb['cells']))

    def setUp(self):
        self.cfg=copy.copy(cfg)
        self.hs=np.full((10,20,3,128,128),100,np.uint16)
        self.n=[roi(i,2,z,64,64,r) for i,(z,r) in enumerate([(7,5),(8,8),(9,5)])]
        self.r=dict(nuc=self.n,drop=[],npc=[roi(0,2,8,64,64,12)])
        ns['p2_classify_npc'](self.r,self.cfg)
        self.ctx=dict(hs=self.hs,rois=self.r,zcols=ns['p2_z_columns'](self.n,self.cfg))
        circle=(64.,64.,40.)
        plane=ns['p2_detect_plane'](self.hs,2,8,circle,self.cfg)
        self.state=dict(t=2,planes={(0,8):plane},gold_overrides={},occupied={0},
                        ambiguous=set(),gold_z={0:{7,8,9}})
        self.box=(0,0,128,128)

    def compose(self,z=8,mode='gold'):
        return ns['p2_compose'](self.ctx,self.state,z,self.box,mode,self.cfg)

    def test_gold_overrides_rejection_and_npc_subtraction(self):
        lab,w=self.compose()
        idx=ns['HEAD_INDEX']
        self.assertEqual(lab[64,64,idx['nucleus_interior']],1)
        self.assertEqual(lab[64,64,idx['npc']],0)
        self.assertEqual(lab[64,75,idx['npc']],1)
        self.assertEqual(w[64,75,idx['npc']],1)
        self.assertEqual(lab[64,64,idx['z_offset']],0)
        self.assertEqual(lab[0,0,idx['nucleus_interior']],0)
        self.assertEqual(lab[0,0,idx['droplet_interior']],255)
        self.assertEqual(lab[64,109,idx['background']],1)

    def test_classical_cannot_contradict_gold(self):
        result=self.state['planes'][(0,8)]['result']
        result['mask'][:]=True
        lab,w=self.compose(mode='classical')
        self.assertEqual(lab[64,90,ns['HEAD_INDEX']['nucleus_interior']],0)
        self.assertEqual(lab[64,64,ns['HEAD_INDEX']['nucleus_interior']],1)

    def test_guard_and_empty_focal_planes(self):
        guard,_=self.compose(z=6)
        empty,_=self.compose(z=5)
        idx=ns['HEAD_INDEX']['nucleus_interior']
        self.assertEqual(guard[64,64,idx],255)
        self.assertEqual(empty[64,64,idx],0)

    def test_early_orphan_and_blur_and_late_punctum(self):
        early=roi(1,2,8,23,23,4)
        blur=roi(2,2,10,64,64,5)
        linked=roi(3,2,11,64,64,5)
        late=roi(4,9,8,23,23,4)
        self.r['npc'] += [early,blur,linked,late]
        flags=ns['p2_classify_npc'](self.r,self.cfg)
        self.assertEqual(early['policy'],'early_unknown')
        self.assertEqual(blur['policy'],'blur')
        self.assertEqual(linked['policy'],'blur')
        self.assertEqual(late['policy'],'late_negative')
        lab,w=self.compose(); idx=ns['HEAD_INDEX']
        self.assertEqual(lab[23,23,idx['nucleus_interior']],255)
        self.assertEqual(lab[23,23,idx['npc']],255)
        self.assertEqual(w[23,23,idx['npc']],0)
        self.state['t']=9
        lab,w=self.compose()
        self.assertEqual(lab[23,23,idx['npc']],0)
        self.assertEqual(w[23,23,idx['npc']],1)

    def test_padded_pixels_never_supervised(self):
        sample=dict(t=2,z=8,cy=0,cx=0,sample_weight=1.)
        x,lab,w=ns['p2_patch'](self.ctx,self.state,sample,cfg=self.cfg)
        self.assertTrue((lab[:32,:,:ns['N_HEADS']]==255).all())
        self.assertTrue((w[:32]==0).all())

    def test_below_floor_gold_still_establishes_occupancy(self):
        self.r['nuc']=[roi(50,2,1,64,64,4)]
        self.r['npc']=[]
        original=ns['p2_geometry']
        ns['p2_geometry']=lambda hs,t,cfg: {0:dict(prof={8:(64.,64.,40.)},
            sphere_outliers=[],z_eq=8.,R_um=20.)}
        try:
            state=ns['p2_timepoint'](self.ctx,2,self.cfg)
        finally:
            ns['p2_geometry']=original
        self.assertIn(0,state['occupied'])
        self.assertEqual(state['gold_z'][0],{1})
        samples=list(ns['p2_samples'](self.ctx,state,cfg=self.cfg))
        self.assertTrue(samples)
        self.assertTrue(all(s['z']>=6 for s in samples))
        self.assertTrue(all(s['sample_class']=='empty_focal_plane' for s in samples))

    def test_all_sample_sources_obey_floor_including_six(self):
        self.assertEqual(self.cfg.z_floor,6)
        plane=copy.deepcopy(self.state['planes'][(0,8)])
        plane['result']['mask'][30:40,30:40]=True
        self.state['planes']={(0,z):plane for z in (0,5,6)}
        self.r['nuc']=[roi(50+z,2,z,90,90,4) for z in (0,5,6)]
        self.state['gold_overrides']={(100,z):(20.,20.,15.) for z in (0,5,6)}
        for mode in ('gold','classical'):
            samples=list(ns['p2_samples'](self.ctx,self.state,mode,self.cfg))
            self.assertTrue(samples)
            self.assertEqual({s['z'] for s in samples},{6})
            if mode=='gold':
                self.assertIn('gold_nucleus',{s['sample_class'] for s in samples})
                self.assertIn('gold_droplet',{s['sample_class'] for s in samples})

    def test_direct_patch_rejects_below_floor(self):
        for mode in ('gold','classical'):
            for z in (0,5):
                with self.assertRaisesRegex(ValueError,'outside the training range'):
                    ns['p2_patch'](self.ctx,self.state,
                        dict(t=2,z=z,cy=64,cx=64,sample_weight=1.),mode,self.cfg)

    def test_low_planes_retained_as_input_neighbors(self):
        self.cfg.n_z_context=5
        self.cfg.normalization_mode='global_t'
        self.cfg.norm_stats={(2,c):(0.,100.) for c in range(3)}
        for z in range(20): self.hs[2,z]=z*10
        self.assertEqual(ns['context_planes'](6,20,5),[4,5,6,7,8])
        x,lab,w=ns['p2_patch'](self.ctx,self.state,
            dict(t=2,z=6,cy=64,cx=64,sample_weight=1.),cfg=self.cfg)
        np.testing.assert_allclose(x[32,32],np.repeat([.4,.5,.6,.7,.8],3),atol=1e-6)

    def test_calculated_npc_excludes_nls_interior(self):
        yy,xx=np.ogrid[:64,:64]
        nucleus=(xx-32)**2+(yy-32)**2<=10**2
        support=np.ones((64,64),bool)
        npc=np.zeros((64,64),np.float32)
        npc[ns['nucleus_envelope_shell'](nucleus,self.cfg)]=100
        result=ns['detect_npc_on_envelope'](npc,
            dict(mask=nucleus,unannotated=np.zeros_like(nucleus)),support,self.cfg)
        self.assertTrue(result['mask'].any())
        self.assertFalse((result['mask'] & nucleus).any())

    def test_intentional_z_target_and_weight_storage(self):
        self.assertEqual(getattr(self.cfg,'z_target_step_um',self.cfg.z_step_um),2.0)
        self.assertEqual(self.cfg.geometry_z_step_um,2.18)
        self.assertEqual(self.ctx['zcols'].cfg.z_step_um,2.0)
        lab,w=self.compose()
        self.assertEqual(w.dtype,np.dtype('float16'))

    def test_classical_rejection_stays_unknown(self):
        self.cfg.gold_complete_timepoints=()
        plane=self.state['planes'][(0,8)]
        r=plane['result']; r['raw_foreground'][35:45,35:45]=True
        r['unannotated'][35:45,35:45]=True
        lab,w=self.compose(mode='classical')
        idx=ns['HEAD_INDEX']['nucleus_interior']
        self.assertEqual(lab[64,64,idx],255)
        self.assertEqual(w[64,64,idx],0)

    def test_new_content_parameters_in_hashes(self):
        for name in self.cfg._PHASE2_FIELDS:
            c=copy.copy(self.cfg); original=getattr(c,name)
            value=(original/'different' if isinstance(original,Path) else
                   original+(99,) if isinstance(original,tuple) else original+.1)
            setattr(c,name,value)
            self.assertNotEqual(c.gen_hash,self.cfg.gen_hash,name)
            self.assertNotEqual(c.gold_hash,self.cfg.gold_hash,name)

    def test_temporary_pool_roundtrip_and_refuse_rerun(self):
        with tempfile.TemporaryDirectory(prefix='vulcan_phase2_test_') as td:
            root=Path(td); c=copy.copy(self.cfg)
            c.image_root=root; c.image_filename='test.tif'; c.out_root=root
            c.gold_pool_name='test_gold'; c.model_name='test_model'
            tiff.imwrite(root/'test.tif',self.hs,photometric='minisblack')
            for kind,filename in [('nuc','NucleiRoiSet.zip'),('drop','DropletRoiSet.zip'),('npc','NPCRoiSet.zip')]:
                with zipfile.ZipFile(root/filename,'w') as z:
                    for i,item in enumerate(self.r[kind]):
                        z.writestr(f'{i}.roi',item['original'].tobytes())
            original_root=ns.get('ROI_ROOT'); original_geom=ns['p2_geometry']
            original_sig=ns.get('P2_ACCEPTED_SIGNATURE')
            ns['ROI_ROOT']=root
            ns['p2_geometry']=lambda hs,t,cfg: {0:dict(prof={8:(64.,64.,40.)},
                    sphere_outliers=[],z_eq=8.,R_um=20.)}
            ns['P2_ACCEPTED_SIGNATURE']=ns['p2_signature'](c)
            try:
                result=ns['p2_build']('gold',(2,),c)
                self.assertEqual(result['status'],'complete')
                triples=ns['p2_pool_files'](c.reviewed_root)
                self.assertTrue(triples)
                metadata=pd.read_csv(c.reviewed_root/'patch_metadata.csv')
                self.assertTrue((metadata.z>=6).all())
                self.assertIn(6,metadata.z.values)
                for a,b,w in triples:
                    ns['p2_validate_arrays'](np.load(a),np.load(b),np.load(w),c)
                with self.assertRaises(FileExistsError):
                    ns['p2_build']('gold',(2,),c)
                manifest=c.reviewed_root/'manifest.json'
                obj=json.loads(manifest.read_text()); obj['status']='building'
                manifest.write_text(json.dumps(obj))
                with self.assertRaises(RuntimeError): ns['p2_pool_files'](c.reviewed_root)
            finally:
                ns['p2_geometry']=original_geom; ns['ROI_ROOT']=original_root
                ns['P2_ACCEPTED_SIGNATURE']=original_sig


if __name__=='__main__':
    unittest.main(verbosity=2)
