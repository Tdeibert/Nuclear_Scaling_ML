import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import tifffile
import reviewed_hard_negatives as hn


class HardNegativeTests(unittest.TestCase):
    def cfg(self,image):
        return SimpleNamespace(image_file=image,pixel_size_um=1.,z_step_um=2.,z_floor=6,
            n_channels=3,n_z_context=5,patch_size=8,nucleus_channel_idx=1,npc_channel_idx=2,
            membrane_channel_idx=0,normalization_mode='global_t',norm_stats_stride=4,
            norm_stats_z_stride=4,holdout_tile=(1,1),holdout_timepoint=5,mosaic_tile_px=32,
            mosaic_overlap_px=0,holdout_margin_px=2,model_name='old')

    def test_weights_and_holdout(self):
        mask=np.eye(8,dtype=bool);lab,w=hn.make_negative_targets(mask)
        for k in [4,5,6]:
            self.assertTrue((lab[...,k][mask]==0).all())
            self.assertTrue((w[...,k][mask]==1).all())
        self.assertTrue((w[~mask]==0).all())
        self.assertTrue((w[...,[0,1,2,3,7,8]]==0).all())
        cfg=self.cfg(Path('/unused'))
        self.assertEqual(hn.holdout_reason(5,4,4,cfg),'temporal_holdout')
        self.assertEqual(hn.holdout_reason(0,30,30,cfg),'spatial_holdout_or_margin')
        self.assertEqual(hn.holdout_reason(0,4,4,cfg),'train')

    def test_review_build_load_and_new_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);image=root/'image.tif';masks=root/'masks.tif'
            tifffile.imwrite(image,np.ones((1,8,3,64,64),np.uint16),metadata={'axes':'TZCYX'})
            inst=np.zeros((1,8,64,64),np.uint16);inst[0,6,4:8,4:8]=1
            tifffile.imwrite(masks,inst,metadata={'axes':'TZYX'})
            cfg=self.cfg(image)
            infer=SimpleNamespace(input_image_path=image,nucleus_instance_hyperstack_path=masks,
                pixel_size_um=1.,nuclear_channel_index=1,npc_channel_index=2,membrane_channel_index=0,model_name='old')
            grouped=pd.DataFrame([dict(t=0,z=6,label=1,nucleus_3d_id=1)])
            review=hn.export_artifact_candidates(infer,grouped,['T0_N1'],root/'review')
            csv=pd.read_csv(review/'review.csv').fillna('')
            self.assertEqual(csv.decision.iloc[0],'')
            lab=root/'gold.npy';np.save(lab,np.zeros((8,8,10),np.uint8))
            gold=[dict(t=0,z=6,cy=8,cx=8,lab=lab,source='gold',stem='gold')]
            kw=dict(gold_rows=gold,extract_input_stack=lambda *args,**kwargs:np.zeros((8,8,15),np.float32),ensure_norm_stats=lambda cfg:None)
            with self.assertRaisesRegex(ValueError,'No exact'):hn.build_reviewed_negative_pool(review,root/'pool',cfg,**kw)
            csv.loc[0,['decision','reviewer']]=['artifact_negative','tester'];csv.to_csv(review/'review.csv',index=False)
            pos=np.zeros((8,8,10),np.uint8);pos[0,0,4]=1;np.save(lab,pos)
            with self.assertRaisesRegex(ValueError,'conflicts'):hn.build_reviewed_negative_pool(review,root/'pool',cfg,**kw)
            np.save(lab,np.zeros((8,8,10),np.uint8))
            pool=hn.build_reviewed_negative_pool(review,root/'pool',cfg,**kw)
            rows,m=hn.load_reviewed_negative_rows(pool,cfg);self.assertEqual(len(rows),1)
            target=np.load(rows[0]['lab']);weights=np.load(rows[0]['wgt'])
            self.assertEqual(int((weights[...,4]>0).sum()),16)
            self.assertTrue((target[...,4][weights[...,4]==0]==255).all())
            calls=[]
            def split(c,verbose=True):calls.append(c);return [dict(source='gold')],[dict(source='gold')]
            ns={'p3_split':split,'SPLITS':{}}
            new=copy.copy(cfg);new.model_name='new'
            hn.enable_hard_negative_training(ns,new,pool,baseline_cfg=cfg)
            train,val=ns['p3_split'](new)
            self.assertIs(calls[0],cfg);self.assertEqual(len(train),2);self.assertEqual(len(val),1)
            self.assertTrue(new.model_name.startswith('new_hn'))
            with self.assertRaises(FileExistsError):hn.build_reviewed_negative_pool(review,pool,cfg,**kw)
            with rows[0]['wgt'].open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):hn.load_reviewed_negative_rows(pool,cfg)


if __name__=='__main__':unittest.main()
