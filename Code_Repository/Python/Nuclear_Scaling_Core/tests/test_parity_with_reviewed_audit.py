"""Parity with the reviewed section-38g/38h diagnostics, on the real geometry code.

The 5 % overlap tolerance was approved by looking at the output of
`ransac_droplet_audit.py` and `droplet_overlap_audit.py`. This test runs those
two scripts and the new package modules on the same synthetic images, using the
actual training notebook's geometry functions, and requires identical circles,
parent assignments, measurements, flags and per-timepoint impact.

Skipped when Model_Training (the scripts and vulcan_training_2_5_3.ipynb) is not
next to Nuclear_Scaling_Core. Set VULCAN_MODEL_TRAINING_DIR to point elsewhere.
Takes roughly a minute; nothing outside a temporary directory is written.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from nuclear_scaling import droplet_geometry as dg          # noqa: E402
from nuclear_scaling import droplet_population as dp        # noqa: E402

MODEL_TRAINING = Path(os.environ.get("VULCAN_MODEL_TRAINING_DIR",
                                     HERE.parents[1] / "Model_Training"))
TRAINING_NOTEBOOK = MODEL_TRAINING / "vulcan_training_2_5_3.ipynb"
SCRIPTS = ("segmented_artifact_gate.py", "ransac_droplet_audit.py", "droplet_overlap_audit.py")
AVAILABLE = TRAINING_NOTEBOOK.exists() and all((MODEL_TRAINING / s).exists() for s in SCRIPTS)
KEYS = ["t", "z", "label"]


@unittest.skipUnless(AVAILABLE, "reviewed audit scripts or training notebook not found")
class ParityWithReviewedAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import synthetic_spheres

        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        root = Path(cls.tmp.name)
        cls.data = synthetic_spheres.build_run(root / "run")
        cfg, grouped = cls.data.cfg, cls.data.grouped

        # ── The reviewed diagnostics, exactly as the notebook called them ──
        sys.path.insert(0, str(MODEL_TRAINING))
        import IPython.display
        IPython.display.display = lambda *a, **k: None      # keep test output quiet
        import segmented_artifact_gate as old_gate
        import ransac_droplet_audit as old_ransac
        import droplet_overlap_audit as old_overlap
        for module in (old_gate, old_ransac, old_overlap):
            module.display = lambda *a, **k: None
            module.plt.show = lambda *a, **k: None
        cls.old_gate_module = old_gate
        with open(os.devnull, "w") as quiet:
            stdout, sys.stdout = sys.stdout, quiet
            try:
                gate = old_gate.audit_segmented_artifacts(
                    cfg, image=None, objects=grouped, best_z=None, save=False)
                cls.old_ransac = old_ransac.audit_ransac_droplets(
                    cfg, gate, training_notebook=TRAINING_NOTEBOOK, timepoints=(0, 1, 2),
                    cache_dir=root / "old_cache", save=False)
                cls.old_overlap = old_overlap.audit_droplet_overlaps(
                    cfg, cls.old_ransac, grouped, thresholds=(0.05, 0.10, 0.20),
                    examples_per_threshold=1, save=False)
            finally:
                sys.stdout = stdout
        cls.gate = gate

        # ── The new stage, geometry computed independently of the old cache ──
        cls.geometry = dg.compute_geometry(
            cfg.input_image_path, TRAINING_NOTEBOOK, pixel_size_um=cfg.pixel_size_um,
            npc_channel_index=cfg.npc_channel_index, cache_dir=root / "new_cache", progress=False)
        cls.assigned = dg.assign_components(
            cfg.input_image_path, cfg.nucleus_instance_hyperstack_path, grouped, cls.geometry,
            nuclear_channel_index=cfg.nuclear_channel_index, pixel_size_um=cfg.pixel_size_um,
            valid_planes=dg.included_planes(pd.read_pickle(cfg.segmentation_index_path),
                                            cfg.focus_min_z),
            progress=False)
        cls.population = dp.build_population(cls.assigned["components"], cls.assigned["circles"],
                                             dp.PopulationSettings(), cls.assigned["upstream"])
        cls.root = root

    def test_run_is_not_trivial(self):
        """The comparison only means something if every state actually occurs."""
        statuses = set(self.population["components"].geometry_status)
        self.assertGreaterEqual(statuses, {"assigned", "partial_parent", "missing_parent",
                                           "ambiguous_parent"})
        self.assertGreater(len(self.population["flags"]), 0)
        self.assertEqual(set(self.population["circles"].source), {dp.OBSERVED, dp.PREDICTED})
        self.assertGreater(int(self.population["impact"].linked_ids_lost.sum()), 0)
        reasons = set(self.population["reasons"].reason)
        self.assertGreaterEqual(reasons, {"enrichment", "overlapping_droplet_geometry"})
        self.assertGreater(len(self.population["accepted"]), 20)

    def test_geometry_signature_and_cache_are_interchangeable(self):
        import json
        old = json.loads((self.root / "old_cache" / dg.GEOMETRY_MANIFEST).read_text())
        self.assertTrue(dg._same_signature(old, self.geometry["signature"]))
        # A fresh cache directory fills itself entirely from the reviewed audit's cache.
        reused = dg.compute_geometry(
            self.data.cfg.input_image_path, TRAINING_NOTEBOOK,
            pixel_size_um=self.data.cfg.pixel_size_um,
            npc_channel_index=self.data.cfg.npc_channel_index,
            cache_dir=self.root / "adopted", reuse_cache_dirs=[self.root / "old_cache"],
            progress=False)
        self.assertTrue(reused["coverage"].origin.str.startswith("reused").all())

    def test_circles_identical(self):
        cols = ["t", "z", "parent", "cx", "cy", "radius_px", "support_pixels", "source"]
        old = self.old_ransac["circles"][cols].sort_values(["t", "z", "parent"]).reset_index(drop=True)
        new = self.assigned["circles"][cols].sort_values(["t", "z", "parent"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(old, new, check_dtype=False)

    def test_component_assignment_and_measurements_identical(self):
        cols = ["area_pixels", "parent_candidates", "parent", "parent_source",
                "parent_overlap_fraction", "geometry_status", "geometry_gate_pass",
                "contained_gate_pass", "enrichment", "solidity", "gate_reason"]
        old = self.old_ransac["comparison"].sort_values(KEYS).reset_index(drop=True)
        new = self.assigned["components"].sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(old[KEYS], new[KEYS], check_dtype=False)
        for col in cols:
            with self.subTest(column=col):
                a, b = old[col], new[col]
                if col in ("parent_overlap_fraction", "enrichment", "solidity"):
                    np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float),
                                               rtol=0, atol=1e-12, equal_nan=True)
                else:
                    self.assertTrue((a.isna() == b.isna()).all())
                    self.assertTrue((a[a.notna()] == b[b.notna()]).all())

    def test_overlap_pairs_identical(self):
        def canonical(pairs):
            p = pairs.copy()
            lo = p[["parent_a", "parent_b"]].min(axis=1)
            hi = p[["parent_a", "parent_b"]].max(axis=1)
            p["parent_a"], p["parent_b"] = lo, hi
            return (p[["t", "z", "parent_a", "parent_b", "overlap_fraction", "observed_pair"]]
                    .sort_values(["t", "z", "parent_a", "parent_b"]).reset_index(drop=True))
        old, new = canonical(self.old_overlap["pairs"]), canonical(self.population["pairs"])
        pd.testing.assert_frame_equal(old.drop(columns="overlap_fraction"),
                                      new.drop(columns="overlap_fraction"), check_dtype=False)
        np.testing.assert_allclose(old.overlap_fraction, new.overlap_fraction, rtol=0, atol=1e-12)

    def test_flags_decisions_and_impact_identical_at_every_reviewed_tolerance(self):
        trials = dp.compare_tolerances(self.assigned["components"], self.assigned["circles"],
                                       (0.05, 0.10, 0.20))
        rename = dict(baseline_nuclear_ids="baseline_linked_ids",
                      remaining_nuclear_ids="remaining_linked_ids",
                      nuclear_ids_lost="linked_ids_lost")
        shared = ["droplets_flagged", "overlapping_observed_pairs", "baseline_components",
                  "remaining_components", "components_removed", "baseline_linked_ids",
                  "remaining_linked_ids", "linked_ids_lost", "unresolved_assignments",
                  "unknown_gate_decisions"]
        old = self.old_overlap["summary"].rename(columns=rename)
        for tolerance in (0.05, 0.10, 0.20):
            with self.subTest(tolerance=tolerance):
                a = old[np.isclose(old.threshold, tolerance)].sort_values("t").reset_index(drop=True)
                b = trials[np.isclose(trials.tolerance, tolerance)].sort_values("t").reset_index(drop=True)
                pd.testing.assert_frame_equal(a[["t"] + shared], b[["t"] + shared], check_dtype=False)
                flags_old = self.old_overlap["flags"]
                flags_old = flags_old[np.isclose(flags_old.threshold, tolerance)]
                flags_new = dp.flag_overlapping_droplets(self.population["pairs"], tolerance)
                self.assertEqual(set(zip(flags_old.t, flags_old.parent)),
                                 set(zip(flags_new.t, flags_new.parent)))
        # Component by component at the approved tolerance.
        decisions = self.old_overlap["decisions"]
        decisions = decisions[np.isclose(decisions.threshold, 0.05)]
        merged = decisions[KEYS + ["hypothetical_pass"]].merge(
            self.population["components"][KEYS + ["status"]], on=KEYS, validate="one_to_one")
        self.assertEqual(len(merged), len(self.population["components"]))
        old_pass = merged.hypothetical_pass.fillna(False).astype(bool)
        self.assertTrue((old_pass == (merged.status == dp.STATUS_ACCEPTED)).all())

    def test_gate_formula_identical_to_script(self):
        rng = np.random.default_rng(0)
        yy, xx = np.ogrid[:80, :80]
        for trial in range(25):
            image = rng.normal(1000, 80, (80, 80))
            candidate = (xx - 40) ** 2 + (yy - 40) ** 2 <= rng.uniform(1, 12) ** 2
            image[candidate] += rng.uniform(-200, 900)
            support = (xx - 40) ** 2 + (yy - 40) ** 2 <= rng.uniform(8, 45) ** 2
            foreground = (image > rng.uniform(1000, 1500)) & support
            old, old_ring = self.old_gate_module.measure_segment_gate(
                image, candidate, support, foreground, 0.1625)
            new, new_ring = dg.measure_segment_gate(image, candidate, support, foreground, 0.1625)
            self.assertEqual(set(old), set(new))
            for key in old:
                if isinstance(old[key], float):
                    np.testing.assert_equal(old[key], new[key])
                else:
                    self.assertEqual(old[key], new[key])
            np.testing.assert_array_equal(old_ring, new_ring)
        self.assertEqual(self.old_gate_module.ArtifactGateSettings().__dict__,
                         dg.ArtifactGateSettings().__dict__)

    def test_inputs_unchanged_by_both_implementations(self):
        # Masks and image are opened read-only; a write would have raised.
        for path in (self.data.cfg.input_image_path, self.data.cfg.nucleus_instance_hyperstack_path):
            self.assertTrue(Path(path).exists())
        pd.testing.assert_frame_equal(self.data.grouped,
                                      pd.read_pickle(self.data.cfg.obj_dir / "grouped_z_objects.pkl"))


if __name__ == "__main__":
    unittest.main()
