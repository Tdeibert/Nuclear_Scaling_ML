"""The v18.1 notebook itself: valid, ordered, and its population stage runnable.

Handoff case 9. The stage cells (ids starting `pop-`) and the figure-binding
cell are executed in notebook order against the synthetic run, so a cell that
uses a name before it is defined, or reads a baseline table where it should
read the selected one, fails here rather than on Cheaha.

Skipped when the notebook is not next to Nuclear_Scaling_Core.
"""
import ast
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import synthetic                                              # noqa: E402

NOTEBOOK = Path(os.environ.get(
    "VULCAN_PIPELINE_NOTEBOOK",
    HERE.parents[1] / "Image_Segmentation" / "Large_FOV_Nuclear_Pipeline_v18.1.ipynb"))
BIND_CELL, FIGURE_HELPERS, FIGURES_36, FIGURES_37 = "acb5eced", "4876d8d9", "c6d7e90a", "d333cde8"
TRAINING_RELATIVE = "Projects/Nuclear_Scaling/Code_Repository/Python/Model_Training/vulcan_training_2_5_3.ipynb"


@unittest.skipUnless(NOTEBOOK.exists(), f"notebook not found: {NOTEBOOK}")
class PipelineNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = json.loads(NOTEBOOK.read_text(encoding="utf8"))
        cls.cells = cls.nb["cells"]
        cls.ids = [c["id"] for c in cls.cells]
        cls.source = {c["id"]: "".join(c["source"]) for c in cls.cells}

    def position(self, cell_id):
        return self.ids.index(cell_id)

    def stage_cells(self):
        """Section 35c-35g cells, i.e. the pop-NN cells that precede section 36."""
        return [i for i in self.ids[:self.position(BIND_CELL)] if re.match(r"pop-\d\d-", i)]

    def test_notebook_is_valid_nbformat(self):
        try:
            import nbformat
        except ImportError:
            self.skipTest("nbformat not installed")
        nbformat.validate(nbformat.read(str(NOTEBOOK), as_version=4))

    def test_cell_ids_are_unique_and_every_code_cell_compiles(self):
        self.assertEqual(len(self.ids), len(set(self.ids)))
        for index, cell in enumerate(self.cells):
            if cell["cell_type"] == "code":
                with self.subTest(cell=index, id=cell["id"]):
                    ast.parse("".join(cell["source"]))

    def test_stage_order(self):
        stage = self.stage_cells()
        self.assertEqual(stage, sorted(stage))
        self.assertEqual(len([i for i in stage if self.cells[self.position(i)]["cell_type"] == "code"]), 10)
        order = [self.position(i) for i in
                 ("pop-01-imports", "pop-02-config", "pop-04-geometry", "pop-06-assign",
                  "pop-08-decide", "pop-10-qc-tables", BIND_CELL, FIGURE_HELPERS,
                  FIGURES_36, FIGURES_37, "vulcan-recovery-150", "vulcan-recovery-153",
                  "pop-38-bind", "vulcan-recovery-155")]
        self.assertEqual(order, sorted(order))

    def test_cells_the_inference_contract_test_indexes_have_not_moved(self):
        expected = {4: "class PipelineConfig", 35: "def group_nuclei_across_z",
                    37: "def select_best_z_per_nucleus", 45: "def build_cumulative_halo_masks"}
        for index, marker in expected.items():
            self.assertIn(marker, "".join(self.cells[index]["source"]))

    def test_population_rules_are_not_fields_of_the_run_hashed_config(self):
        config_cell = "".join(self.cells[4]["source"])
        for name in ("overlap_tolerance", "partial_parent_policy", "ANALYSIS_POPULATION"):
            self.assertNotIn(name, config_cell)

    def test_one_canonical_copy_of_the_tested_logic(self):
        for name in ("measure_segment_gate", "circle_overlap_fraction", "droplet_overlap_pairs",
                     "audit_ransac_droplets", "audit_droplet_overlaps", "load_training_geometry",
                     "fitted_plane_support"):
            defined = [i for i, s in self.source.items() if re.search(rf"^def {name}\(", s, re.M)]
            self.assertEqual(defined, [], f"{name} is defined in the notebook again")
        self.assertNotIn("sys.path.insert", "\n".join(self.source[i] for i in self.ids[100:]))

    def test_figure_cells_read_only_the_selected_tables(self):
        for cell_id in (FIGURES_36, FIGURES_37):
            code = self.source[cell_id]
            for baseline in ("halo_df", "timed_df", "radial_df"):
                self.assertNotRegex(code, rf"\b{baseline}\b", f"{cell_id} reads {baseline}")
            self.assertNotRegex(code, r"\btracked_df\b(?!=)", f"{cell_id} reads tracked_df")
            self.assertNotIn("cfg.exports_dir", code)
        self.assertIn("POP_EXPORT_DIR", self.source[FIGURE_HELPERS])

    def test_restore_script_guard_string_is_still_present(self):
        # restore_v18_1_audits.py refuses to append its section when it finds this heading.
        self.assertTrue(any("## 38. Recovered Vulcan post-segmentation audits" in s
                            for s in self.source.values()))

    def test_debug_and_qc_cells_say_what_they_are(self):
        for cell_id in self.ids:
            if "-qc-" in cell_id and self.cells[self.position(cell_id)]["cell_type"] == "code":
                self.assertRegex(self.source[cell_id].splitlines()[0], r"^# QC, informational")

    def test_population_stage_runs_in_notebook_order(self):
        import matplotlib.pyplot as plt
        import IPython.display

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = synthetic.build_run(root / "inputs")
            run = root / "run"
            for name in ("obj", "seg", "qc", "analysis", "exports", "track"):
                (run / name).mkdir(parents=True)
            data.grouped.to_pickle(run / "obj/grouped_z_objects.pkl")
            data.best_z.to_pickle(run / "obj/best_z_nuclei.pkl")
            data.segmentation_index.to_pickle(run / "seg/segmentation_index.pkl")
            training = root / "home" / TRAINING_RELATIVE
            training.parent.mkdir(parents=True)
            training.write_bytes(data.training_notebook.read_bytes())
            cfg = SimpleNamespace(
                input_image_path=data.image_path, nucleus_instance_hyperstack_path=data.labels_path,
                segmentation_index_path=run / "seg/segmentation_index.pkl",
                obj_dir=run / "obj", qc_dir=run / "qc", analysis_dir=run / "analysis",
                exports_dir=run / "exports", track_dir=run / "track",
                pixel_size_um=data.pixel_size_um,
                npc_channel_index=data.npc_channel_index,
                nuclear_channel_index=data.nuclear_channel_index, focus_min_z=data.min_z,
                paths=SimpleNamespace(run_id="synthetic-run"))

            def digest():
                return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(root.rglob("*"))
                        if p.is_file() and "selected_population" not in p.parts and "exports" not in p.parts}
            before = digest()
            baseline = dict(halo_df=data.tracked.copy(), timed_df=data.tracked.copy(),
                            tracked_df=data.tracked.copy(), radial_df=data.radial.copy())
            namespace = dict(cfg=cfg, **{k: v.copy() for k, v in baseline.items()})
            exec("from pathlib import Path\nfrom typing import *\nimport numpy as np\nimport pandas as pd\n"
                 "import matplotlib.pyplot as plt", namespace)

            run_ids = [i for i in self.stage_cells()
                       if self.cells[self.position(i)]["cell_type"] == "code"]
            run_ids += [BIND_CELL, FIGURE_HELPERS]
            old_home, old_show, old_display = os.environ.get("HOME"), plt.show, IPython.display.display
            os.environ["HOME"] = str(root / "home")
            plt.show = lambda *a, **k: None
            IPython.display.display = lambda *a, **k: None
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    for cell_id in sorted(run_ids, key=self.position):
                        try:
                            exec(compile(self.source[cell_id], f"<cell {cell_id}>", "exec"), namespace)
                        except Exception as exc:
                            raise AssertionError(f"cell {cell_id} failed: {type(exc).__name__}: {exc}") from exc
                        plt.close("all")
            finally:
                plt.show, IPython.display.display = old_show, old_display
                if old_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = old_home

            population = namespace["POPULATION"]
            merged = population["components"].merge(data.expected, on=["t", "z", "label"],
                                                    validate="one_to_one")
            self.assertEqual(len(merged), len(data.expected))                 # every timepoint, by default
            self.assertTrue((merged.status == merged.expect_status).all())
            self.assertTrue((merged.reason == merged.expect_reason).all())
            self.assertEqual(namespace["POP_SETTINGS"].overlap_tolerance, 0.05)
            self.assertEqual(namespace["ANALYSIS_POPULATION"], "accepted")
            self.assertEqual(sorted(namespace["halo_sel"].nucleus_3d_id), [3, 6, 11, 12, 20, 21, 30])
            self.assertEqual(set(namespace["radial_sel"].track_id), set(namespace["tracked_sel"].track_id))
            export = namespace["POP_EXPORT_DIR"]
            self.assertEqual(export, run / "exports" / "populations" / f"accepted__{population['tag']}")
            self.assertEqual(namespace["figures_dir"](cfg), export / "figures")
            self.assertTrue((population["output_dir"] / "accepted_components.csv").exists())
            self.assertTrue(list(population["output_dir"].glob("qc_geometry_T*.png")))
            self.assertTrue(list(population["output_dir"].glob("qc_pair_T0_*.png")))
            # Nothing that existed before the stage was touched, on disk or in the kernel.
            self.assertEqual(digest(), before)
            for name, frame in baseline.items():
                pd.testing.assert_frame_equal(namespace[name], frame)


if __name__ == "__main__":
    unittest.main()
