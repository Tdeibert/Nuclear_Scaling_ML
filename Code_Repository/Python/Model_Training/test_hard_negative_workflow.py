import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import reviewed_hard_negatives as hn
import hard_negative_workflow as flow
import test_reviewed_hard_negatives as fixtures


class WorkflowTests(unittest.TestCase):
    def setup_case(self,root):
        image=root/'image.tif';tifffile.imwrite(image,np.ones((1,8,3,64,64),np.uint16),photometric='minisblack')
        cfg=fixtures.HardNegativeTests().cfg(image);cfg.out_root=root/'output'
        review=root/'review';review.mkdir();(review/'masks').mkdir()
        mask=np.ones((2,2),np.uint8);mp=review/'masks/a.npy';np.save(mp,mask)
        rec=dict(candidate_id='a',review_id='T0_N1',t=0,z=6,label=1,nucleus_3d_id=1,y0=4,x0=4,y1=6,x1=6,
                 area_pixels=4,mask_file='masks/a.npy',mask_sha256=hn._digest(mp))
        manifest=dict(raw_image=hn._identity(image),pixel_size_um=1.,channel_indices=dict(nls=1,npc=2,membrane=0),candidates=[rec])
        hn._json(review/'candidates.json',manifest)
        pd.DataFrame([dict(rec,decision='artifact_negative',reviewer='tester',notes='')]).to_csv(review/'review.csv',index=False)
        lab=np.ones((8,8,10),np.uint8);weight=np.full((8,8,9),.5,np.float32)
        lp=root/'lab.npy';wp=root/'wgt.npy';ip=root/'img.npy'
        np.save(lp,lab);np.save(wp,weight);np.save(ip,np.zeros((8,8,15),np.float32))
        row=dict(source='gold',stem='t000_z006_d1_y00008_x00008',t=0,z=6,cy=8,cx=8,img=ip,lab=lp,wgt=wp)
        ns=dict(cfg=cfg,p3_rows=lambda cfg:[dict(row)],p3_split=lambda cfg,verbose=False:([dict(row)],[]),
                extract_input_stack=lambda *a,**kw:np.zeros((8,8,15),np.float32),ensure_norm_stats=lambda cfg:None)
        return review,ns,row

    def test_commit_corrections_and_negative_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);review,ns,row=self.setup_case(root);original=hn._digest(review/'review.csv');labhash=hn._digest(row['lab'])
            work=flow.Workflow(ns,review)
            self.assertEqual(work.pending(),['a'])
            with self.assertRaisesRegex(ValueError,'decisions'):work.commit()
            work.save_draft('a','artifact_negative','tester','Confirmed cytoplasm')
            self.assertEqual(hn._digest(review/'review.csv'),original)
            resumed=flow.Workflow(ns,review);self.assertFalse(resumed.pending())
            output=resumed.commit();m=json.loads((output/'manifest.json').read_text())
            self.assertEqual(m['status'],'complete');self.assertIsNotNone(m['correction_pool']);self.assertIsNotNone(m['negative_pool'])
            self.assertEqual(hn._digest(row['lab']),labhash);self.assertEqual(hn._digest(review/'review.csv'),original)
            self.assertEqual(resumed.commit(),output)
            patch=json.loads((Path(m['correction_pool'])/'manifest.json').read_text())['patches'][0]
            lab=np.load(Path(m['correction_pool'])/patch['lab']);self.assertTrue((lab[:2,:2,4:7]==0).all())
            self.assertTrue((lab[...,0:4]==1).all())
            hn.load_reviewed_negative_rows(m['negative_pool'],ns['cfg'])

    def test_keep_gold_and_stale_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);review,ns,row=self.setup_case(root)
            work=flow.Workflow(ns,review);work.save_draft('a','real','tester','Keep gold')
            output=work.commit();m=json.loads((output/'manifest.json').read_text())
            self.assertIsNone(m['negative_pool']);self.assertIsNone(m['correction_pool'])
            fresh=flow.Workflow(ns,review)
            table=pd.read_csv(review/'review.csv').fillna('');table.loc[0,'notes']='changed externally';table.to_csv(review/'review.csv',index=False)
            with self.assertRaisesRegex(RuntimeError,'Source review changed'):fresh.commit()


if __name__=='__main__':unittest.main()
