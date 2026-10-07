"""CPU contract tests; no user images/models/runs are modified."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

ROOT=Path(__file__).resolve().parent
NB=ROOT.parent/'Image_Segmentation/Large_FOV_Nuclear_Pipeline_v18.1.ipynb'


def namespace():
    ns=dict(np=np,pd=pd,tiff=tiff,ndi=ndi,filters=filters,measure=measure,morphology=morphology)
    exec('import os, json, math, time, gc, shutil\nfrom pathlib import Path\n'
         'from datetime import datetime\nfrom dataclasses import dataclass, asdict\n'
         'from typing import *\n',ns)
    n=json.loads(NB.read_text(encoding='utf8'))
    for i in (4,15,20,22,23,25,27,29,31,35,37,45):
        exec(compile(''.join(n['cells'][i]['source']), 'cell%d'%i,'exec'),ns)
    return ns


class FakeModel:
    input_shape=(None,32,32,15)
    output_shape=(None,32,32,8)
    def get_layer(self,name):
        return SimpleNamespace(filters=7 if name=='masks_logits' else 1,
            activation=SimpleNamespace(__name__='linear' if name=='masks_logits' else 'softplus'))
    def __call__(self,x,training=False):
        p=np.zeros(x.shape[:3]+(8,),np.float32)
        p[...,:7]=x[...,:7]*8-4
        p[...,7]=x[...,7]*5
        return SimpleNamespace(numpy=lambda:p)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.ns=namespace()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        ns=self.ns
        self.cfg=ns['PipelineConfig'](patch_size=32,patch_stride=16,batch_size=2,tta=True,
            watershed_min_marker_dist_px=3)
        paths=SimpleNamespace(run_id='test',run_dir=root,
            project=SimpleNamespace(models_dir=root,raw_images_dir=root),
            assert_config_matches=lambda x:None)
        for name in ('seg_dir','obj_dir','track_dir','analysis_dir','qc_dir','mask_tif_dir','exports_dir'):
            p=root/name;p.mkdir();setattr(paths,name,p)
        self.cfg.paths=paths
        self.cfg.model_path.write_bytes(b'synthetic model')
        self.cfg.model_sha256=ns['file_sha256'](self.cfg.model_path)

    def test_input_order_context_normalization_and_tta(self):
        ns,c=self.ns,self.cfg
        rng=np.random.default_rng(1)
        image=rng.normal(size=(1,9,3,40,48)).astype(np.float32)
        raw=ns['get_full_context_yxc'](image,0,6,c)
        self.assertTrue(np.array_equal(raw[...,0],image[0,4,1]))
        self.assertEqual(ns['context_plane_indices'](0,9,5),[0,0,0,1,2])
        self.assertEqual(ns['eligible_z_planes'](9,c),[6,7,8])
        stats=ns['normalization_stats_for_timepoint'](image,0,c)
        norm=ns['_preprocess_patch'](raw,stats)
        pred=ns['predict_vulcan_plane'](image,0,6,FakeModel(),c,stats)
        expected=FakeModel()(norm[None]).numpy()[0]
        expected[...,:7]=1/(1+np.exp(-expected[...,:7]))
        np.testing.assert_allclose(pred,expected,rtol=2e-6,atol=2e-6)
        with self.assertRaises(ValueError): ns['predict_vulcan_plane'](image,0,5,FakeModel(),c,stats)

    def test_watershed_parity(self):
        ns,c=self.ns,self.cfg
        n=json.loads((ROOT/'vulcan_training_2_5_3.ipynb').read_text(encoding='utf8'))
        found={}
        for cell in n['cells']:
            if cell['cell_type']!='code':continue
            s=''.join(cell['source'])
            for node in ast.parse(s).body:
                if isinstance(node,ast.FunctionDef) and node.name in ('normalized_sharpness','watershed_nuclei'):
                    found[node.name]=ast.get_source_segment(s,node)
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
        ref=dict(ns,cfg=c,ndimage=ndi,peak_local_max=peak_local_max,watershed=watershed,distance_transform_edt=ndi.distance_transform_edt)
        c.mask_threshold=c.nucleus_threshold
        for name in ('normalized_sharpness','watershed_nuclei'):exec(found[name],ref)
        yy,xx=np.mgrid[:48,:48]
        h=np.zeros((48,48,8),np.float32)
        h[...,4]=np.maximum(np.exp(-((yy-20)**2+(xx-16)**2)/60),np.exp(-((yy-20)**2+(xx-32)**2)/60))
        h[...,5]=.2;h[...,1]=.9
        nls=h[...,4]*100
        actual,_=ns['vulcan_instances'](h,nls,c)
        expected=ref['watershed_nuclei'](nls,h[...,4],h[...,5],c)
        np.testing.assert_array_equal(actual,expected)

    def test_failure_resume_storage_and_measurements(self):
        ns,c=self.ns,self.cfg
        image=np.zeros((2,8,3,32,32),np.float32)
        real=ns['predict_vulcan_plane']
        counter=[]
        def synthetic(img,t,z,model,config,stats=None):
            counter.append((t,z))
            if (t,z)==(1,6):raise ValueError('injected failure')
            h=np.zeros((32,32,8),np.float32);h[...,1]=.9
            h[10:22,10:22,4]=.99;h[...,7]=8-z;h[...,6]=.8
            return h
        ns['predict_vulcan_plane']=synthetic
        try:
            with self.assertRaisesRegex(RuntimeError,'t=1 z=6'):
                ns['run_segmentation_for_all_planes'](image,FakeModel(),c)
            cp=json.loads((c.seg_dir/'_checkpoint.json').read_text())
            self.assertEqual(cp['completed_t'],[0]);self.assertEqual(cp['status'],'in_progress')
            state=ns['assess_segmentation_state'](c,(2,8,32,32))
            self.assertEqual(state.mode,'resume');self.assertEqual(state.completed_t,[0])
            ns['predict_vulcan_plane']=lambda img,t,z,m,config,stats=None:synthetic(img,0,z,m,config,stats)
            idx=ns['run_segmentation_for_all_planes'](image,FakeModel(),c,state)
            self.assertEqual(len(idx),16);self.assertEqual(idx[idx.included].z.min(),6)
            self.assertEqual(ns['assess_segmentation_state'](c,(2,8,32,32)).mode,'complete')
            h=tiff.memmap(str(ns['full_head_path'](c)),mode='r')
            self.assertEqual(h.shape,(2,8,8,32,32));self.assertFalse(h[:,:6].any())
            self.assertEqual(float(h[0,6,7,15,15]),2)
            obj=ns['extract_objects_from_saved_masks'](idx,c)
            self.assertTrue(obj.mean_z_offset.notna().all())
            groups=ns['group_nuclei_across_z'](obj,c)
            best=ns['select_best_z_per_nucleus'](groups,c)
            self.assertTrue((best.z==7).all())
            for row in best.itertuples():
                mask=ns['recover_nucleus_mask_for_row'](row,c)
                self.assertEqual(int(mask.sum()),row.area_px)
            c.focus_min_z=7
            with self.assertRaises(RuntimeError):ns['require_segmentation'](c)
        finally: ns['predict_vulcan_plane']=real

    def test_adjacent_instances_not_merged(self):
        ns,c=self.ns,self.cfg
        mm=ns['open_hyperstack_memmap'](c.nucleus_instance_hyperstack_path,(1,8,32,32),np.uint16,'TZYX')
        mm[0,6,10:20,10:16]=1;mm[0,6,10:20,16:22]=2;mm.flush()
        row=SimpleNamespace(t=0,z=6,label=1,nucleus_3d_id=1)
        mask=ns['recover_nucleus_mask_for_row'](row,c)
        self.assertEqual(int(mask.sum()),60)
        with self.assertRaises(RuntimeError): ns['open_hyperstack_memmap'](c.nucleus_instance_hyperstack_path,(1,9,32,32),np.uint16,'TZYX')


if __name__=='__main__': unittest.main()
