"""Synthetic checks; no user data or checkpoints modified."""
import unittest
import numpy as np
from gold_selection_validation import fit_growth, saturation, raster, inside

class GoldSelectionTests(unittest.TestCase):
    def test_saturation_recovery(self):
        t=np.linspace(0,100,30)
        result=fit_growth(t,saturation(t,50,300,.06))
        self.assertAlmostEqual(result['plateau_um2'],350,places=3)
        self.assertAlmostEqual(result['k_per_min'],.06,places=5)
        self.assertEqual(result['status'],'plateau locally constrained')

    def test_unconstrained_linear_trend(self):
        t=np.linspace(0,60,30)
        result=fit_growth(t,50+3*t)
        self.assertEqual(result['status'],'plateau poorly constrained')

    def test_short_series_not_fitted(self):
        self.assertIsNone(fit_growth([0,1,2],[10,20,30])['curve'])

    def test_raster_and_boundary_exclusion(self):
        box,mask=raster(np.array([[2,2],[5,2],[5,5],[2,5]]),(10,10))
        self.assertEqual(box,(2,2,6,6))
        self.assertEqual(int(mask.sum()),16)
        self.assertTrue(inside(box,(0,0,10,10)))
        self.assertFalse(inside(box,(2,2,6,6)))
        with self.assertRaises(ValueError):
            raster(np.array([[-1,2],[5,2],[5,5]]),(10,10))

if __name__=='__main__':unittest.main()
