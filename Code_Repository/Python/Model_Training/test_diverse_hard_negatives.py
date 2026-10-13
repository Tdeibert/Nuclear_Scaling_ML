import unittest
import numpy as np
import pandas as pd
from mine_diverse_hard_negatives import select_diverse_candidates


class MiningTests(unittest.TestCase):
    def fixture(self):
        rows=[];comps=[];circles=[]
        for t in range(4):
            for n in range(1,13):
                for z in [6,7]:
                    rows.append(dict(t=t,z=z,label=n,nucleus_3d_id=n,centroid_x_px=10.,centroid_y_px=10.))
                    comps.append(dict(t=t,z=z,label=n,nucleus_3d_id=n,parent=n,area_pixels=1 if n%3==0 else 100,
                        enrichment=1. if n%2 else 2.,geometry_status='assigned',status='accepted'))
                    circles.append(dict(t=t,z=z,parent=n,cx=10.,cy=10.,radius_px=20.))
        return pd.DataFrame(rows),dict(components=pd.DataFrame(comps),circles=pd.DataFrame(circles),
            flags=pd.DataFrame([dict(t=0,parent=2)]))

    def test_diversity_and_exclusion(self):
        grouped,pop=self.fixture()
        a=select_diverse_candidates(grouped,pop,1.,excluded_ids=['T0_N1'],per_timepoint=5)
        b=select_diverse_candidates(grouped,pop,1.,excluded_ids=['T0_N1'],per_timepoint=5)
        self.assertEqual(len(a),20)
        self.assertFalse(a.review_id.eq('T0_N1').any())
        self.assertFalse(a.duplicated(['t','nucleus_3d_id']).any())
        self.assertFalse(a.duplicated(['t','parent']).any())
        self.assertEqual(a.review_id.tolist(),b.review_id.tolist())
        self.assertTrue((a.groupby('t').size()==5).all())

    def test_partial_population_refused(self):
        grouped,pop=self.fixture();pop['components']=pop['components'][pop['components'].t<3]
        with self.assertRaisesRegex(ValueError,'every timepoint'):
            select_diverse_candidates(grouped,pop,1.)


if __name__=='__main__':unittest.main()
