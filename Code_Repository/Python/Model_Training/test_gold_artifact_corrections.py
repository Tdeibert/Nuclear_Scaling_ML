import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import reviewed_hard_negatives as hn
from test_reviewed_hard_negatives import HardNegativeTests
import gold_artifact_corrections as gc


class CorrectionTests(unittest.TestCase):
    def test_versioned_exact_mask_and_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'raw.tif'
            tifffile.imwrite(image,np.ones((6,8,3,64,64),np.uint16),photometric='minisblack')
            cfg=HardNegativeTests().cfg(image);review=root/'review';review.mkdir();(review/'masks').mkdir()
            mask=np.ones((2,2),np.uint8);mp=review/'masks/a.npy';np.save(mp,mask)
            rec=dict(candidate_id='a',review_id='T0_N1',t=0,z=6,y0=4,x0=4,y1=6,x1=6,area_pixels=4,mask_file='masks/a.npy',mask_sha256=hn._digest(mp))
            (review/'candidates.json').write_text(json.dumps(dict(raw_image=hn._identity(image),pixel_size_um=1.,candidates=[rec])))
            csv=pd.DataFrame([dict(candidate_id='a',decision='artifact_negative',reviewer='test',notes='')]);csv.to_csv(review/'review.csv',index=False)
            lab=np.ones((8,8,10),np.uint8);w=np.full((8,8,9),.5,np.float32)
            lp=root/'lab.npy';wp=root/'wgt.npy';np.save(lp,lab);np.save(wp,w)
            row=dict(source='gold',stem='s',t=0,z=6,cy=8,cx=8,lab=lp,wgt=wp)
            with self.assertRaisesRegex(ValueError,'explicitly resolve'):
                gc.build_corrections(review,root/'pool',cfg,[row],[row])
            csv.loc[0,'notes']='Confirmed artifact; conflicting gold annotation requires correction. test';csv.to_csv(review/'review.csv',index=False)
            with self.assertRaisesRegex(ValueError,'No explicitly'):
                gc.build_corrections(review,root/'held',cfg,[row],[])
            pool=gc.build_corrections(review,root/'pool',cfg,[row],[row])
            out=gc.corrected_rows(pool,cfg,[row])[0];new=np.load(out['lab']);weights=np.load(out['wgt'])
            self.assertTrue((new[:2,:2,4:7]==0).all());self.assertTrue((weights[:2,:2,4:7]==1).all())
            self.assertTrue((new[...,0:4]==lab[...,0:4]).all());self.assertTrue((new[...,7:]==lab[...,7:]).all())
            self.assertTrue((new[2:]==lab[2:]).all());self.assertTrue((np.load(lp)==lab).all());self.assertTrue((np.load(wp)==w).all())
            self.assertTrue(gc.audit_conflicts(review,cfg,[out])['pairs'].empty)
            with self.assertRaises(FileExistsError):gc.build_corrections(review,pool,cfg,[row],[row])
            csv.loc[0,'decision']='skip';csv.to_csv(review/'review.csv',index=False)
            with self.assertRaisesRegex(ValueError,'Review changed'):gc.corrected_rows(pool,cfg,[row])


if __name__=='__main__':unittest.main()
