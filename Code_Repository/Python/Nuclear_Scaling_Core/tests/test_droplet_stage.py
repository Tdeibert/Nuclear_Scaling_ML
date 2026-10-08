"""The geometry -> assignment -> population stage on a synthetic run (temporary files only).

Run from Nuclear_Scaling_Core:
    python -m unittest discover -s tests -v
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import synthetic                                              # noqa: E402
from nuclear_scaling import droplet_geometry as dg            # noqa: E402
from nuclear_scaling import droplet_population as dp          # noqa: E402

KEYS = ["t", "z", "label"]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(root) -> dict:
    return {str(p.relative_to(root)): sha256(p) for p in sorted(Path(root).rglob("*")) if p.is_file()}


class StageCase(unittest.TestCase):
    """One synthetic run and one stage execution shared by the read-only tests."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        cls.data = synthetic.build_run(cls.root / "run")
        cls.inputs_before = tree(cls.data.root)
        cls.frames_before = {name: getattr(cls.data, name).copy(deep=True)
                             for name in ("grouped", "best_z", "tracked", "radial")}
        cls.geometry = cls.compute(cls.root / "geometry")
        cls.assigned = cls.assign(cls.geometry, cache_dir=cls.root / "components")
        cls.population = dp.build_population(cls.assigned["components"], cls.assigned["circles"],
                                             dp.PopulationSettings(), cls.assigned["upstream"])
        cls.saved_to = dp.save_population(cls.population, cls.root / "populations")

    @classmethod
    def compute(cls, cache_dir, data=None, **kwargs):
        data = data or cls.data
        kwargs.setdefault("progress", False)
        return dg.compute_geometry(data.image_path, data.training_notebook,
                                   pixel_size_um=data.pixel_size_um,
                                   npc_channel_index=data.npc_channel_index,
                                   cache_dir=cache_dir, **kwargs)

    @classmethod
    def assign(cls, geometry, data=None, components=None, **kwargs):
        data = data or cls.data
        kwargs.setdefault("progress", False)
        return dg.assign_components(
            data.image_path, data.labels_path,
            data.grouped if components is None else components, geometry,
            nuclear_channel_index=data.nuclear_channel_index, pixel_size_um=data.pixel_size_um,
            **kwargs)


class TrainingGeometryLoader(StageCase):
    def test_only_declared_geometry_settings_are_read(self):
        training = self.geometry["training"]
        self.assertEqual(set(training.settings),
                         {"inventory_ref_z", "geometry_z_step_um", "erosion_px",
                          "pixel_size_um", "npc_channel_idx"})
        # Calibration and channel come from inference, not from the training notebook.
        self.assertEqual(training.settings["pixel_size_um"], synthetic.PIXEL_UM)
        self.assertEqual(training.settings["npc_channel_idx"], synthetic.NPC)
        self.assertEqual(training.settings["geometry_z_step_um"], synthetic.GEOMETRY_Z_STEP_UM)

    def test_training_cells_are_never_executed(self):
        # The miniature notebook has a cell that raises if it runs; loading succeeded.
        self.assertIn("p2_circle", self.geometry["training"].namespace)
        self.assertNotIn("PipelineConfig", self.geometry["training"].namespace)

    def test_missing_function_is_reported(self):
        broken = synthetic._FAKE_GEOMETRY_CELL.replace("def p2_circle", "def p2_circle_renamed")
        path = synthetic.write_training_notebook(self.root / "broken.ipynb", broken)
        with self.assertRaisesRegex(ValueError, "p2_circle"):
            dg.load_training_geometry(path, 0.5, 2)

    def test_code_hash_tracks_geometry_functions_only(self):
        base = self.geometry["training"].code_sha256
        edited = synthetic._FAKE_GEOMETRY_CELL.replace('"fitted", tuple', '"fitted",  tuple')
        changed = dg.load_training_geometry(
            synthetic.write_training_notebook(self.root / "edited.ipynb", edited), 0.5, 2)
        self.assertNotEqual(base, changed.code_sha256)
        unrelated = synthetic._FAKE_GEOMETRY_CELL + "\ndef some_training_helper():\n    return 1\n"
        same = dg.load_training_geometry(
            synthetic.write_training_notebook(self.root / "unrelated.ipynb", unrelated), 0.5, 2)
        self.assertEqual(base, same.code_sha256)


class GeometryCache(StageCase):
    def test_all_timepoints_by_default(self):
        coverage = self.geometry["coverage"]
        self.assertEqual(list(coverage.t), [0, 1, 2])
        self.assertEqual(list(coverage.inventory), [7, 2, 1])
        self.assertTrue((coverage.status == "ok").all())
        self.assertEqual(self.geometry["timepoints"], [0, 1, 2])
        for t in range(3):
            self.assertTrue((self.root / "geometry" / f"geometry_t{t:03d}.pkl").exists())

    def test_second_run_loads_everything_from_cache(self):
        files = sorted((self.root / "geometry").glob("geometry_t*.pkl"))
        stamps = [f.stat().st_mtime_ns for f in files]
        again = self.compute(self.root / "geometry")
        self.assertEqual(list(again["coverage"].origin), ["cache"] * 3)
        self.assertEqual(stamps, [f.stat().st_mtime_ns for f in files])
        self.assertEqual(again["signature_hash"], self.geometry["signature_hash"])

    def test_interrupted_run_resumes_only_missing_timepoints(self):
        cache = self.root / "resume"
        self.compute(cache, timepoints=[0, 2])
        self.assertFalse((cache / "geometry_t001.pkl").exists())
        resumed = self.compute(cache)
        self.assertEqual(list(resumed["coverage"].origin), ["cache", "computed", "cache"])
        self.assertFalse(list(cache.glob("*.tmp")))

    def test_mismatched_provenance_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = synthetic.build_run(Path(tmp) / "run")
            cache = Path(tmp) / "geometry"
            self.compute(cache, data=data)
            # Different training geometry settings -> different signature.
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                dg.compute_geometry(data.image_path, data.training_notebook, pixel_size_um=0.25,
                                    npc_channel_index=data.npc_channel_index, cache_dir=cache,
                                    progress=False)
            # A changed input image -> different signature.
            stat = data.image_path.stat()
            os.utime(data.image_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                self.compute(cache, data=data)

    def test_directory_without_provenance_is_refused(self):
        stray = self.root / "stray"
        stray.mkdir()
        (stray / "geometry_t000.pkl").write_bytes(b"not ours")
        with self.assertRaisesRegex(ValueError, "fresh directory"):
            self.compute(stray)

    def test_matching_older_cache_is_reused_and_mismatched_one_ignored(self):
        donor = self.root / "donor"
        self.compute(donor, timepoints=[0, 1])
        other = self.root / "other_donor"
        other.mkdir()
        (other / dg.GEOMETRY_MANIFEST).write_text(json.dumps(dict(settings={}, code_sha256="x", image=[])))
        shutil.copy(donor / "geometry_t000.pkl", other / "geometry_t002.pkl")
        fresh = self.compute(self.root / "fresh", reuse_cache_dirs=[other, donor])
        origin = list(fresh["coverage"].origin)
        self.assertTrue(origin[0].startswith("reused from donor"))
        self.assertTrue(origin[1].startswith("reused from donor"))
        self.assertEqual(origin[2], "computed")                    # never taken from `other`
        found = dg.find_geometry_caches(self.root, self.geometry["signature"]).set_index(
            dg.find_geometry_caches(self.root, self.geometry["signature"]).cache_dir.map(lambda p: p.name))
        self.assertTrue(found.loc["donor", "provenance_matches"])
        self.assertFalse(found.loc["other_donor", "provenance_matches"])
        self.assertEqual(found.loc["donor", "timepoints"], [0, 1])

    def test_invalid_timepoints(self):
        for bad in ([], [-1], [3], [0, 99]):
            with self.assertRaises(ValueError):
                self.compute(self.root / "bad_t", timepoints=bad)

    def test_parallel_workers_give_the_same_geometry(self):
        parallel = self.compute(self.root / "parallel", n_workers=2)
        self.assertEqual(list(parallel["coverage"].origin), ["computed"] * 3)
        for t in range(3):
            self.assertEqual(json.dumps(parallel["geometry"][t], sort_keys=True, default=str),
                             json.dumps(self.geometry["geometry"][t], sort_keys=True, default=str))

    def test_failed_timepoint_is_recorded_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            truth = dict(synthetic.TRUTH)
            truth[2] = []                                           # no droplet inventory at T2
            data = synthetic.build_run(Path(tmp) / "run", truth=truth)
            with self.assertRaisesRegex(RuntimeError, "no droplet inventory"):
                self.compute(Path(tmp) / "strict", data=data)
            # The timepoints before the failure stay cached for the next attempt.
            self.assertTrue((Path(tmp) / "strict" / "geometry_t001.pkl").exists())
            geometry = self.compute(Path(tmp) / "strict", data=data, on_error="record")
            coverage = geometry["coverage"].set_index("t")
            self.assertEqual(list(coverage.status), ["ok", "ok", "failed"])
            self.assertIn("no droplet inventory", coverage.loc[2, "error"])
            assigned = self.assign(geometry, data=data)
            t2 = assigned["components"][assigned["components"].t == 2]
            self.assertTrue((t2.geometry_status == "no_timepoint_geometry").all())
            population = dp.build_population(assigned["components"], assigned["circles"])
            t2 = population["components"][population["components"].t == 2]
            # A timepoint without geometry is unjudged, never rejected wholesale.
            self.assertTrue((t2.status == "unresolved").all())
            self.assertTrue((t2.reason == "no_timepoint_geometry").all())


class ComponentAssignment(StageCase):
    def test_every_synthetic_component_gets_the_expected_parent_state(self):
        merged = self.assigned["components"].merge(self.data.expected, on=KEYS, validate="one_to_one")
        self.assertEqual(len(merged), len(self.data.expected))
        wrong = merged[merged.geometry_status != merged.expect_geometry]
        self.assertTrue(wrong.empty, wrong[KEYS + ["geometry_status", "expect_geometry"]].to_string())

    def test_observed_and_predicted_support_are_distinguished(self):
        circles = self.assigned["circles"]
        d5 = circles[(circles.t == 0) & (circles.parent == 5)]
        self.assertTrue((d5.source == dp.PREDICTED).all())
        self.assertTrue((circles[circles.parent != 5].source == dp.OBSERVED).all())
        c = self.assigned["components"].set_index("nucleus_3d_id")
        self.assertEqual(c.loc[12, "parent_source"], dp.PREDICTED)     # support, but labelled as predicted
        self.assertEqual(c.loc[11, "parent_source"], dp.OBSERVED)

    def test_circles_use_uneroded_radius_and_only_nuclear_planes(self):
        circles = self.assigned["circles"]
        z3 = circles[(circles.t == 0) & (circles.z == 3) & (circles.parent == 1)].iloc[0]
        self.assertAlmostEqual(z3.radius_px, 40.0, places=9)
        eroded_area = np.pi * (40.0 - synthetic.EROSION_PX) ** 2
        self.assertLess(abs(z3.support_pixels - eroded_area) / eroded_area, 0.01)
        self.assertEqual(sorted(circles[circles.t == 0].z.unique()), [1, 2, 3, 4, 5])
        self.assertEqual(sorted(circles[circles.t == 1].z.unique()), [2, 3, 4])

    def test_ambiguous_and_missing_components_keep_no_parent(self):
        c = self.assigned["components"].set_index("nucleus_3d_id")
        self.assertEqual(c.loc[8, "parent"], 0)
        self.assertEqual(c.loc[8, "parent_candidates"], 2)
        self.assertEqual(set(x.split(":")[0] for x in c.loc[8, "candidate_parents"].split(";")), {"4", "5"})
        self.assertTrue(np.isnan(c.loc[8, "enrichment"]))
        self.assertEqual(c.loc[9, "parent_candidates"], 0)
        self.assertIsNone(c.loc[9, "geometry_gate_pass"])
        self.assertLess(c.loc[10, "parent_overlap_fraction"], 0.85)
        self.assertEqual(c.loc[10, "parent"], 6)

    def test_measurements(self):
        c = self.assigned["components"].set_index(["nucleus_3d_id", "z"])
        self.assertAlmostEqual(c.loc[(3, 3), "enrichment"], 3.0, places=9)
        self.assertAlmostEqual(c.loc[(4, 3), "enrichment"], 1.1, places=9)
        self.assertLess(c.loc[(5, 3), "solidity"], 0.8)
        self.assertFalse(c.loc[(13, 3), "reference_ok"])
        self.assertLess(c.loc[(13, 3), "ring_pixels"], 50)
        self.assertEqual(c.loc[(6, 3), "area_pixels"], 1)

    def test_component_cache_is_reused_and_invalidated(self):
        cache = self.root / "components"
        files = sorted(cache.glob("components_t*.pkl"))
        self.assertEqual(len(files), 3)
        stamps = [f.stat().st_mtime_ns for f in files]
        again = self.assign(self.geometry, cache_dir=cache)
        pd.testing.assert_frame_equal(again["components"], self.assigned["components"])
        self.assertEqual(stamps, [f.stat().st_mtime_ns for f in files])
        self.assertEqual(again["upstream"], self.assigned["upstream"])
        # Different measurement settings must not be served from this cache.
        own = self.root / "components_copy"
        shutil.copytree(cache, own)
        changed = self.assign(self.geometry, cache_dir=own,
                              gate_settings=dg.ArtifactGateSettings(min_ring_pixels=500))
        self.assertFalse(changed["components"].reference_ok.any())
        self.assertNotEqual(changed["upstream"]["components"], self.assigned["upstream"]["components"])
        # Different object rows for one timepoint invalidate only that timepoint.
        fewer = self.data.grouped[~((self.data.grouped.t == 1) & (self.data.grouped.z == 4))]
        before = {f.name: f.stat().st_mtime_ns for f in sorted(cache.glob("components_t*.pkl"))}
        self.assign(self.geometry, components=fewer, cache_dir=cache)
        after = {f.name: f.stat().st_mtime_ns for f in sorted(cache.glob("components_t*.pkl"))}
        self.assertEqual(before["components_t000.pkl"], after["components_t000.pkl"])
        self.assertNotEqual(before["components_t001.pkl"], after["components_t001.pkl"])
        self.assign(self.geometry, cache_dir=cache)                # restore for other tests

    def test_object_table_from_another_run_is_refused(self):
        wrong = self.data.grouped.copy()
        wrong.loc[wrong.index[0], "area_px"] += 1
        with self.assertRaisesRegex(ValueError, "same segmentation run"):
            self.assign(self.geometry, components=wrong)
        absent = self.data.grouped.copy()
        absent.loc[absent.index[0], "label"] = 999
        with self.assertRaisesRegex(ValueError, "same segmentation run"):
            self.assign(self.geometry, components=absent)

    def test_components_without_requested_geometry_are_refused(self):
        partial = self.compute(self.root / "geometry", timepoints=[0, 1])
        with self.assertRaisesRegex(ValueError, r"timepoint\(s\) \[2\]"):
            self.assign(partial)
        ok = self.assign(partial, components=self.data.grouped[self.data.grouped.t < 2])
        self.assertEqual(sorted(ok["components"].t.unique()), [0, 1])

    def test_planes_outside_segmentation_are_refused(self):
        valid = dg.included_planes(self.data.segmentation_index, min_z=3)
        with self.assertRaisesRegex(ValueError, "did not include"):
            self.assign(self.geometry, valid_planes=valid)
        self.assign(self.geometry, valid_planes=dg.included_planes(self.data.segmentation_index,
                                                                   self.data.min_z))


class PopulationOnSyntheticRun(StageCase):
    def test_every_component_gets_the_expected_status_and_reason(self):
        merged = self.population["components"].merge(self.data.expected, on=KEYS, validate="one_to_one")
        wrong = merged[(merged.status != merged.expect_status) | (merged.reason != merged.expect_reason)]
        self.assertTrue(wrong.empty, wrong[KEYS + ["status", "reason", "expect_status", "expect_reason"]].to_string())

    def test_overlap_rule_end_to_end(self):
        """Evidence on Z3 only (6.8 %); Z2/Z4 are 4.3 % and do not trigger by themselves."""
        pairs = self.population["pairs"]
        pair = pairs[(pairs.parent_a == 1) & (pairs.parent_b == 2)].set_index("z").overlap_fraction
        self.assertEqual(sorted(pair.index), [2, 3, 4])
        self.assertGreater(pair[3], 0.05)
        self.assertLess(pair[2], 0.05)
        self.assertLess(pair[4], 0.05)
        flags = self.population["flags"]
        self.assertEqual(list(zip(flags.t, flags.parent)), [(0, 1), (0, 2)])
        self.assertEqual(list(flags.evidence_z), ["3", "3"])
        c = self.population["components"]
        in_flagged = c[(c.t == 0) & c.parent.isin([1, 2])]
        self.assertEqual(sorted(in_flagged.z.unique()), [1, 2, 3, 4, 5])      # propagated across Z
        self.assertTrue((in_flagged.status == "excluded").all())
        # Same droplet IDs at T1 are untouched.
        self.assertTrue((c[(c.t == 1)].status == "accepted").all())

    def test_predicted_only_pair_does_not_remove_anything(self):
        pairs = self.population["pairs"]
        pair = pairs[(pairs.parent_a == 4) & (pairs.parent_b == 5)]
        self.assertGreater(pair.overlap_fraction.max(), 0.2)
        self.assertFalse(pair.observed_pair.any())
        ids = self.population["linked_ids"].set_index("nucleus_3d_id")
        self.assertEqual(ids.loc[11, "status"], "accepted")
        self.assertEqual(ids.loc[12, "status"], "accepted")

    def test_linked_id_outcomes(self):
        ids = self.population["linked_ids"].set_index("nucleus_3d_id")
        self.assertTrue(ids.loc[1, "lost_to_overlap"])
        self.assertTrue(ids.loc[2, "lost_to_overlap"])
        self.assertEqual(ids.loc[7, "status"], "accepted")
        self.assertTrue(ids.loc[7, "partially_removed"])
        self.assertFalse(ids.loc[7, "lost_to_overlap"])
        impact = self.population["impact"].set_index("t")
        self.assertEqual(impact.loc[0, "linked_ids_lost"], 2)
        self.assertEqual(impact.loc[0, "components_removed"], 9)
        self.assertEqual(impact.loc[1, "droplets_flagged"], 0)

    def test_exploratory_options_change_only_what_they_name(self):
        base = self.population["components"].set_index(KEYS).status
        one_pixel = dp.build_population(self.assigned["components"], self.assigned["circles"],
                                        dp.PopulationSettings(exclude_one_pixel_components=True))
        changed = one_pixel["components"].set_index(KEYS).status
        differs = base[base != changed]
        self.assertEqual(len(differs), 1)
        self.assertEqual(one_pixel["linked_ids"].set_index("nucleus_3d_id").loc[6, "reasons"],
                         "one_pixel_component")
        evaluate = dp.build_population(self.assigned["components"], self.assigned["circles"],
                                       dp.PopulationSettings(partial_parent_policy="evaluate"))
        changed = evaluate["components"].set_index(KEYS).status
        self.assertEqual(len(base[base != changed]), 1)
        self.assertEqual(evaluate["linked_ids"].set_index("nucleus_3d_id").loc[10, "status"], "accepted")
        self.assertEqual(len({self.population["tag"], one_pixel["tag"], evaluate["tag"]}), 3)


class SelectionDownstream(StageCase):
    def test_accepted_is_the_default_and_excludes_everything_unjudged(self):
        selected = dp.select_population(self.data.best_z, self.population)
        # ID 7 is absent on purpose: see test_best_z_row_is_judged_on_its_own_component.
        self.assertEqual(sorted(selected.nucleus_3d_id), [3, 6, 11, 12, 20, 21, 30])
        self.assertTrue((selected.population_status == "accepted").all())

    def test_other_populations_are_explicit(self):
        best = self.data.best_z
        self.assertEqual(len(dp.select_population(best, self.population, "baseline")), len(best))
        self.assertNotIn("population_status",
                         dp.select_population(best, self.population, "baseline").columns)
        unresolved = dp.select_population(best, self.population, "unresolved")
        self.assertEqual(sorted(unresolved.nucleus_3d_id), [8, 9, 10, 13])
        excluded = dp.select_population(best, self.population, "excluded")
        self.assertEqual(sorted(excluded.nucleus_3d_id), [1, 2, 4, 5, 7])
        both = dp.select_population(best, self.population, "accepted_plus_unresolved")
        self.assertEqual(len(both), 7 + 4)
        parts = [dp.select_population(best, self.population, w) for w in ("accepted", "excluded", "unresolved")]
        self.assertEqual(sum(map(len, parts)), len(best))
        with self.assertRaises(ValueError):
            dp.select_population(best, self.population, "clean")

    def test_best_z_row_is_judged_on_its_own_component(self):
        # ID 7 is accepted as a linked ID (its Z3 component passes), but its
        # best-Z row is the Z2 component, which sits in an overlap-flagged
        # droplet. The best-Z row is dropped; no other Z plane is substituted.
        self.assertEqual(self.population["linked_ids"].set_index("nucleus_3d_id").loc[7, "status"],
                         "accepted")
        best = self.data.best_z[self.data.best_z.nucleus_3d_id == 7]
        self.assertEqual(list(best.z), [2])
        self.assertEqual(len(dp.select_population(best, self.population)), 0)
        grouped = self.data.grouped
        row = grouped[(grouped.nucleus_3d_id == 7) & (grouped.z == 3)]
        self.assertEqual(len(dp.select_population(row, self.population)), 1)

    def test_table_without_labels_is_matched_through_tracking(self):
        radial = dp.select_population(self.data.radial, self.population,
                                      key_lookup=self.data.tracked)
        accepted_tracks = set(dp.select_population(self.data.tracked, self.population).track_id)
        self.assertEqual(set(radial.track_id), accepted_tracks)
        self.assertEqual(len(radial), 4 * len(accepted_tracks))
        with self.assertRaisesRegex(ValueError, "key_lookup"):
            dp.select_population(self.data.radial, self.population)
        with self.assertRaises(KeyError):
            dp.select_population(pd.DataFrame(dict(x=[1])), self.population)

    def test_rows_without_a_decision_are_never_silently_kept(self):
        early = dp.build_population(
            self.assigned["components"][self.assigned["components"].t < 2],
            self.assigned["circles"][self.assigned["circles"].t < 2])
        with self.assertRaisesRegex(ValueError, r"timepoints \[2\]"):
            dp.select_population(self.data.best_z, early)
        dropped = dp.select_population(self.data.best_z, early, on_missing="drop")
        self.assertNotIn(2, set(dropped.t))

    def test_non_unique_index_is_handled(self):
        table = self.data.best_z.copy()
        table.index = [0] * len(table)
        selected = dp.select_population(table, self.population)
        self.assertEqual(sorted(selected.nucleus_3d_id), [3, 6, 11, 12, 20, 21, 30])

    def test_gate_shaped_view_for_the_section_38_audits(self):
        gate = dp.population_as_gate(self.population, self.data.pixel_size_um)
        table = gate["all_instances"]
        self.assertEqual(int(table.gate_pass.sum()), len(self.population["accepted"]))
        self.assertTrue(table.gate_pass.isin([True, False]).all())
        self.assertEqual(table.gate_pass.dtype, bool)
        # Same shape as the original gate table: merges with the linked object
        # table on (t, z, label) without a column clash.
        self.assertNotIn("nucleus_3d_id", table.columns)
        merged = self.data.grouped[KEYS + ["nucleus_3d_id"]].merge(table, on=KEYS, validate="one_to_one")
        self.assertEqual(len(merged), len(self.data.grouped))
        self.assertIn("nucleus_3d_id", merged.columns)
        for column in ("area_um2", "enrichment", "solidity", "parent_overlap_fraction",
                       "gate_reason", "parent_droplet"):
            self.assertIn(column, table.columns)
        np.testing.assert_allclose(table.area_um2, table.area_pixels * synthetic.PIXEL_UM ** 2)
        self.assertEqual(gate["settings"]["min_enrichment"], 1.20)
        legacy = dict(all_instances=table[KEYS].assign(parent_droplet=77))
        carried = dp.population_as_gate(self.population, self.data.pixel_size_um, legacy_gate=legacy)
        self.assertTrue((carried["all_instances"].parent_droplet == 77).all())
        self.assertTrue((carried["all_instances"].fitted_parent == table.parent).all())


class NothingExistingIsModified(StageCase):
    """Handoff case 8: masks, image and baseline tables are byte-identical afterwards."""

    def test_input_files_are_byte_identical(self):
        self.assertEqual(tree(self.data.root), self.inputs_before)

    def test_input_tables_are_not_mutated(self):
        for name, before in self.frames_before.items():
            pd.testing.assert_frame_equal(getattr(self.data, name), before)
        for which in dp.POPULATIONS:
            dp.select_population(self.data.best_z, self.population, which)
            dp.select_population(self.data.radial, self.population, which, key_lookup=self.data.tracked)
        for name, before in self.frames_before.items():
            pd.testing.assert_frame_equal(getattr(self.data, name), before)

    def test_population_tables_do_not_alias_the_measurements(self):
        self.population["accepted"].loc[:, "enrichment"] = -1.0
        self.assertTrue((self.assigned["components"].enrichment.dropna() > 0).all())
        self.assertTrue((self.population["components"].enrichment.dropna() > 0).all())

    def test_images_are_opened_read_only(self):
        for path in (self.data.image_path, self.data.labels_path):
            mapped = tifffile.memmap(str(path), mode="r")
            with self.assertRaises(ValueError):
                mapped[(0,) * mapped.ndim] = 1

    def test_saving_writes_only_inside_its_own_tagged_directory(self):
        self.assertEqual(self.saved_to, self.root / "populations" / self.population["tag"])
        written = sorted(p.name for p in self.saved_to.iterdir())
        self.assertIn("accepted_components.pkl", written)
        self.assertIn("excluded_components.csv", written)
        self.assertIn("unresolved_components.csv", written)
        self.assertIn("linked_id_decisions.csv", written)
        self.assertIn("population_manifest.json", written)
        self.assertFalse([name for name in written if name.endswith(".tmp")])
        self.assertEqual([p.name for p in (self.root / "populations").iterdir()], [self.population["tag"]])
        self.assertEqual(tree(self.data.root), self.inputs_before)

    def test_saved_population_round_trips(self):
        loaded = dp.load_population(self.saved_to)
        self.assertEqual(loaded["tag"], self.population["tag"])
        self.assertEqual(loaded["settings"], self.population["settings"])
        pd.testing.assert_frame_equal(loaded["linked_ids"], self.population["linked_ids"])
        pd.testing.assert_frame_equal(
            dp.select_population(self.data.best_z, loaded),
            dp.select_population(self.data.best_z, self.population))
        manifest = json.loads((self.saved_to / "population_manifest.json").read_text())
        self.assertEqual(manifest["rows"]["components"], len(self.population["components"]))
        self.assertIn("multi-nucleus droplets not excluded", manifest["definition"]["not_enforced"])

    def test_different_rules_are_saved_side_by_side(self):
        other = dp.build_population(self.assigned["components"], self.assigned["circles"],
                                    dp.PopulationSettings(overlap_tolerance=0.10),
                                    self.assigned["upstream"])
        with tempfile.TemporaryDirectory() as tmp:
            first = dp.save_population(self.population, tmp)
            second = dp.save_population(other, tmp)
            self.assertNotEqual(first, second)
            before = tree(first)
            dp.save_population(other, tmp)                         # re-saving one leaves the other alone
            self.assertEqual(tree(first), before)


if __name__ == "__main__":
    unittest.main()
