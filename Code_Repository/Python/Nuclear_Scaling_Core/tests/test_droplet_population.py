"""Rules of the selected population, tested on hand-built tables (no files).

Run from Nuclear_Scaling_Core:
    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nuclear_scaling import droplet_population as dp        # noqa: E402

OBS, PRED = dp.OBSERVED, dp.PREDICTED


def circle(t, z, parent, cx, cy, r, source=OBS):
    return dict(t=t, z=z, parent=parent, cx=float(cx), cy=float(cy), radius_px=float(r),
                support_pixels=1, source=source)


def circles(*rows):
    return pd.DataFrame(list(rows), columns=["t", "z", "parent", "cx", "cy", "radius_px",
                                             "support_pixels", "source"])


def component(t, z, label, nid, parent, status="assigned", enrichment=2.0, solidity=0.95,
              reference_ok=True, area=200, overlap=1.0):
    return dict(t=t, z=z, label=label, nucleus_3d_id=nid, area_pixels=area, parent=parent,
                parent_overlap_fraction=overlap, geometry_status=status,
                enrichment=enrichment, solidity=solidity, reference_ok=reference_ok)


def spacing_for_overlap(fraction, r=100.0):
    """Centre distance of two equal circles whose overlap fraction is `fraction`."""
    lo, hi = 0.0, 2 * r
    for _ in range(200):
        mid = (lo + hi) / 2
        if dp.circle_overlap_fraction(0, 0, r, mid, 0, r) > fraction:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class CircleOverlap(unittest.TestCase):
    """Handoff case 1: tangent, disjoint, contained and partially overlapping circles."""

    def test_disjoint_is_zero(self):
        self.assertEqual(dp.circle_overlap_fraction(0, 0, 10, 30, 0, 10), 0.0)

    def test_externally_tangent_is_zero(self):
        self.assertEqual(dp.circle_overlap_fraction(0, 0, 10, 20, 0, 10), 0.0)
        self.assertEqual(dp.circle_overlap_fraction(0, 0, 3, 0, 8, 5), 0.0)

    def test_contained_is_one(self):
        self.assertEqual(dp.circle_overlap_fraction(0, 0, 10, 2, 1, 4), 1.0)
        self.assertEqual(dp.circle_overlap_fraction(5, 5, 7, 5, 5, 7), 1.0)     # identical

    def test_internally_tangent_is_one(self):
        self.assertEqual(dp.circle_overlap_fraction(0, 0, 10, 6, 0, 4), 1.0)

    def test_partial_overlap_matches_closed_form(self):
        # Two unit circles one radius apart: lens = 2*pi/3 - sqrt(3)/2.
        expected = (2 * np.pi / 3 - np.sqrt(3) / 2) / np.pi
        self.assertAlmostEqual(dp.circle_overlap_fraction(0, 0, 1, 1, 0, 1), expected, places=12)

    def test_normalised_by_the_smaller_full_circle(self):
        # Small circle centred on the big one's edge: half of it lies inside (to first order).
        fraction = dp.circle_overlap_fraction(0, 0, 1000, 1000, 0, 1)
        self.assertAlmostEqual(fraction, 0.5, places=3)

    def test_symmetric_and_translation_invariant(self):
        a = dp.circle_overlap_fraction(3, 4, 20, 25, 9, 12)
        self.assertAlmostEqual(a, dp.circle_overlap_fraction(25, 9, 12, 3, 4, 20), places=12)
        self.assertAlmostEqual(a, dp.circle_overlap_fraction(1003, -96, 20, 1025, -91, 12), places=10)

    def test_matches_pixel_rasterisation(self):
        yy, xx = np.mgrid[:600, :600]
        for (x1, y1, r1, x2, y2, r2) in [(200, 300, 120, 380, 300, 100), (250, 250, 90, 300, 330, 150)]:
            a = (xx - x1) ** 2 + (yy - y1) ** 2 <= r1 ** 2
            b = (xx - x2) ** 2 + (yy - y2) ** 2 <= r2 ** 2
            raster = (a & b).sum() / min(a.sum(), b.sum())
            self.assertAlmostEqual(dp.circle_overlap_fraction(x1, y1, r1, x2, y2, r2), raster, places=2)

    def test_not_clipped_to_the_image(self):
        # Centres outside any plausible image: the full circles are still used.
        self.assertGreater(dp.circle_overlap_fraction(-50, -50, 100, 20, -50, 100), 0.5)

    def test_invalid_circles_raise(self):
        for bad in [(0, 0, 0, 1, 1, 1), (0, 0, -1, 1, 1, 1), (np.nan, 0, 1, 1, 1, 1), (0, 0, 1, 1, 1, np.inf)]:
            with self.assertRaises(ValueError):
                dp.circle_overlap_fraction(*bad)

    def test_bulk_evaluation_matches_reference(self):
        rng = np.random.default_rng(1)
        n = 60
        cx, cy, r = rng.uniform(0, 400, n), rng.uniform(0, 400, n), rng.uniform(5, 80, n)
        i, j, fraction = dp._pairwise_overlap(cx, cy, r)
        self.assertEqual(len(fraction), n * (n - 1) // 2)
        reference = [dp.circle_overlap_fraction(cx[a], cy[a], r[a], cx[b], cy[b], r[b])
                     for a, b in zip(i, j)]
        np.testing.assert_allclose(fraction, reference, rtol=0, atol=1e-12)
        self.assertTrue({0.0, 1.0} <= set(np.round(fraction, 12)))   # both extremes occur


class OverlapPairsAndFlags(unittest.TestCase):
    def test_empty_circle_table(self):
        """Handoff case 6: nothing in, nothing out, columns intact."""
        pairs = dp.droplet_overlap_pairs(circles())
        self.assertTrue(pairs.empty)
        self.assertEqual(list(pairs.columns), dp.PAIR_COLUMNS)
        flags = dp.flag_overlapping_droplets(pairs, 0.05)
        self.assertTrue(flags.empty)
        self.assertEqual(list(flags.columns), dp.FLAG_COLUMNS)

    def test_no_overlap_gives_no_pairs_and_no_flags(self):
        table = circles(circle(0, 6, 1, 0, 0, 10), circle(0, 6, 2, 100, 0, 10),
                        circle(0, 6, 3, 20, 0, 10))               # 1 and 3 exactly tangent
        pairs = dp.droplet_overlap_pairs(table)
        self.assertTrue(pairs.empty)
        self.assertTrue(dp.flag_overlapping_droplets(pairs, 0.05).empty)

    def test_single_circle_plane(self):
        self.assertTrue(dp.droplet_overlap_pairs(circles(circle(0, 6, 1, 0, 0, 10))).empty)

    def test_pairs_only_within_one_plane_and_timepoint(self):
        table = circles(circle(0, 6, 1, 0, 0, 10), circle(0, 7, 2, 5, 0, 10),    # other Z
                        circle(1, 6, 3, 5, 0, 10))                                 # other T
        self.assertTrue(dp.droplet_overlap_pairs(table).empty)

    def test_duplicate_circle_keys_raise(self):
        with self.assertRaises(ValueError):
            dp.droplet_overlap_pairs(circles(circle(0, 6, 1, 0, 0, 10), circle(0, 6, 1, 5, 0, 10)))

    def test_strictly_greater_than_tolerance(self):
        """Handoff case 2: exactly 5 % does not trigger; just above does."""
        def pairs_with(value):
            return pd.DataFrame([dict(t=0, z=6, parent_a=1, parent_b=2, overlap_fraction=value,
                                      observed_pair=True, source_a=OBS, source_b=OBS)])
        self.assertTrue(dp.flag_overlapping_droplets(pairs_with(0.05), 0.05).empty)
        self.assertTrue(dp.flag_overlapping_droplets(pairs_with(np.nextafter(0.05, 0)), 0.05).empty)
        flagged = dp.flag_overlapping_droplets(pairs_with(np.nextafter(0.05, 1)), 0.05)
        self.assertEqual(list(flagged.parent), [1, 2])

    def test_threshold_on_real_circles(self):
        below, above = spacing_for_overlap(0.049), spacing_for_overlap(0.051)
        for distance, expect_flag in [(below, False), (above, True)]:
            table = circles(circle(0, 6, 1, 0, 0, 100), circle(0, 6, 2, distance, 0, 100))
            flags = dp.flag_overlapping_droplets(dp.droplet_overlap_pairs(table), 0.05)
            self.assertEqual(not flags.empty, expect_flag)

    def test_tolerance_must_be_a_fraction(self):
        pairs = dp.droplet_overlap_pairs(circles())
        for bad in (0, 1, -0.1, 5):
            with self.assertRaises(ValueError):
                dp.flag_overlapping_droplets(pairs, bad)

    def test_both_droplets_flagged_with_evidence(self):
        """Handoff case 3 (first half): a hit flags BOTH members of the pair."""
        d = spacing_for_overlap(0.20)
        table = circles(circle(0, 8, 4, 0, 0, 100), circle(0, 8, 9, d, 0, 100),
                        circle(0, 9, 4, 0, 0, 100), circle(0, 9, 9, d, 0, 100),
                        circle(0, 8, 5, 5000, 0, 100))
        flags = dp.flag_overlapping_droplets(dp.droplet_overlap_pairs(table), 0.05)
        self.assertEqual(list(zip(flags.t, flags.parent)), [(0, 4), (0, 9)])
        self.assertEqual(list(flags.evidence_planes), [2, 2])
        self.assertEqual(list(flags.evidence_z), ["8;9", "8;9"])
        self.assertEqual(list(flags.partners), ["9", "4"])
        self.assertTrue((flags.reason == "overlapping_droplet_geometry").all())
        np.testing.assert_allclose(flags.max_observed_overlap, 0.20, atol=1e-9)

    def test_predicted_only_overlap_never_flags(self):
        """Handoff case 4: a predicted circle cannot trigger exclusion, however large the overlap."""
        for source_a, source_b in [(OBS, PRED), (PRED, OBS), (PRED, PRED)]:
            table = circles(circle(0, 6, 1, 0, 0, 100, source_a), circle(0, 6, 2, 20, 0, 100, source_b))
            pairs = dp.droplet_overlap_pairs(table)
            self.assertEqual(len(pairs), 1)                       # still reported
            self.assertGreater(pairs.overlap_fraction.iloc[0], 0.8)
            self.assertFalse(bool(pairs.observed_pair.iloc[0]))
            self.assertTrue(dp.flag_overlapping_droplets(pairs, 0.05).empty)

    def test_observed_evidence_on_one_plane_is_enough(self):
        # Predicted on Z6, observed on Z7: the Z7 pair alone triggers.
        table = circles(circle(0, 6, 1, 0, 0, 100, PRED), circle(0, 6, 2, 20, 0, 100),
                        circle(0, 7, 1, 0, 0, 100), circle(0, 7, 2, 20, 0, 100))
        flags = dp.flag_overlapping_droplets(dp.droplet_overlap_pairs(table), 0.05)
        self.assertEqual(list(flags.evidence_z), ["7", "7"])


class ComponentDecisions(unittest.TestCase):
    def setUp(self):
        # Droplets 1 and 2 overlap at T0 on Z8 only. Droplet 3 is clean.
        # The same IDs 1 and 2 exist at T1 and do not overlap there.
        d = spacing_for_overlap(0.30)
        self.circles = circles(
            circle(0, 8, 1, 0, 0, 100), circle(0, 8, 2, d, 0, 100), circle(0, 8, 3, 5000, 0, 100),
            circle(0, 6, 1, 0, 0, 60), circle(0, 6, 2, d, 0, 60),            # disjoint at Z6
            circle(1, 8, 1, 0, 0, 100), circle(1, 8, 2, 4000, 0, 100))
        self.flags = dp.flag_overlapping_droplets(dp.droplet_overlap_pairs(self.circles), 0.05)

    def decide(self, rows, **settings):
        table = pd.DataFrame(rows)
        before = table.copy(deep=True)
        out = dp.decide_components(table, self.flags, dp.PopulationSettings(**settings))
        pd.testing.assert_frame_equal(table, before)            # input untouched
        return out.set_index(["t", "z", "label"])

    def test_flag_applies_across_z_within_the_timepoint(self):
        """Handoff case 3 (second half): evidence on Z8, exclusion on every Z of that T."""
        out = self.decide([component(0, z, 1, 1, parent=1) for z in (6, 7, 8, 9, 12)] +
                          [component(0, z, 2, 2, parent=2) for z in (6, 12)] +
                          [component(0, z, 3, 3, parent=3) for z in (6, 8)])
        self.assertTrue((out.loc[(0, slice(None), 1), "status"] == "excluded").all())
        self.assertTrue((out.loc[(0, slice(None), 2), "status"] == "excluded").all())
        self.assertTrue((out.loc[(0, slice(None), 1), "reason"] == "overlapping_droplet_geometry").all())
        self.assertTrue((out.loc[(0, slice(None), 3), "status"] == "accepted").all())
        self.assertTrue(out.loc[(0, slice(None), 1), "removed_by_overlap"].all())

    def test_flag_does_not_cross_timepoints(self):
        out = self.decide([component(0, 8, 1, 1, parent=1), component(1, 8, 1, 10, parent=1),
                           component(1, 8, 2, 11, parent=2)])
        self.assertEqual(out.loc[(0, 8, 1), "status"], "excluded")
        self.assertEqual(out.loc[(1, 8, 1), "status"], "accepted")
        self.assertEqual(out.loc[(1, 8, 2), "status"], "accepted")
        self.assertFalse(out.loc[(1, 8, 1), "parent_flagged"])

    def test_unknown_assignments_stay_explicit(self):
        """Handoff case 5: missing / ambiguous / partial parents are never accepted,
        never called artifacts, and never moved to another droplet."""
        out = self.decide([
            component(0, 8, 1, 1, parent=0, status="missing_parent", enrichment=np.nan, reference_ok=False),
            component(0, 8, 2, 2, parent=0, status="ambiguous_parent", enrichment=np.nan, reference_ok=False),
            component(0, 8, 3, 3, parent=3, status="partial_parent", overlap=0.6),
            component(0, 8, 4, 4, parent=1, status="partial_parent", overlap=0.6),   # best parent flagged
            component(0, 8, 5, 5, parent=0, status="no_timepoint_geometry", enrichment=np.nan, reference_ok=False),
            component(0, 8, 6, 6, parent=3, reference_ok=False, enrichment=np.nan)])
        self.assertTrue((out.status == "unresolved").all())
        self.assertEqual(list(out.reason), ["missing_parent", "ambiguous_parent", "partial_parent",
                                            "partial_parent", "no_timepoint_geometry",
                                            "unassessable_cytoplasm"])
        self.assertTrue((out.exclusion_reasons == "").all())
        # Nothing was reassigned: parents are exactly as measured.
        self.assertEqual(list(out.parent), [0, 0, 3, 1, 0, 3])
        # A flagged best-candidate does not turn an unresolved component into an exclusion.
        self.assertFalse(out.loc[(0, 8, 4), "removed_by_overlap"])

    def test_partial_parent_policy_evaluate(self):
        rows = [component(0, 8, 3, 3, parent=3, status="partial_parent", overlap=0.6),
                component(0, 8, 4, 4, parent=1, status="partial_parent", overlap=0.6),
                component(0, 8, 5, 5, parent=3, status="partial_parent", overlap=0.6, enrichment=1.0)]
        out = self.decide(rows, partial_parent_policy="evaluate")
        self.assertEqual(list(out.status), ["accepted", "excluded", "excluded"])
        self.assertEqual(list(out.reason), ["accepted", "overlapping_droplet_geometry", "enrichment"])

    def test_gate_thresholds(self):
        out = self.decide([
            component(0, 8, 1, 1, parent=3, enrichment=1.20, solidity=0.80),       # boundaries pass
            component(0, 8, 2, 2, parent=3, enrichment=1.19),
            component(0, 8, 3, 3, parent=3, solidity=0.79),
            component(0, 8, 4, 4, parent=3, enrichment=1.0, solidity=0.5),
            component(0, 8, 5, 5, parent=3, solidity=np.nan),
            component(0, 8, 6, 6, parent=1, enrichment=1.0)])                       # two reasons
        self.assertEqual(list(out.status), ["accepted"] + ["excluded"] * 5)
        self.assertEqual(list(out.reason), ["accepted", "enrichment", "solidity",
                                            "enrichment + solidity", "solidity",
                                            "enrichment + overlapping_droplet_geometry"])
        # Already failing another rule: not counted as removed BY the overlap rule.
        self.assertFalse(out.loc[(0, 8, 6), "removed_by_overlap"])

    def test_solidity_is_applied_without_a_parent_but_enrichment_is_not(self):
        out = self.decide([
            component(0, 8, 1, 1, parent=0, status="missing_parent", solidity=0.4, enrichment=np.nan, reference_ok=False),
            component(0, 8, 2, 2, parent=3, status="partial_parent", enrichment=1.0, overlap=0.5)])
        self.assertEqual(out.loc[(0, 8, 1), "status"], "excluded")
        self.assertEqual(out.loc[(0, 8, 1), "reason"], "solidity")
        self.assertEqual(out.loc[(0, 8, 1), "unresolved_reasons"], "missing_parent")   # still visible
        self.assertEqual(out.loc[(0, 8, 2), "status"], "unresolved")

    def test_one_pixel_option_is_off_by_default(self):
        rows = [component(0, 8, 1, 1, parent=3, area=1), component(0, 8, 2, 2, parent=3, area=2)]
        self.assertEqual(list(self.decide(rows).status), ["accepted", "accepted"])
        on = self.decide(rows, exclude_one_pixel_components=True)
        self.assertEqual(list(on.status), ["excluded", "accepted"])
        self.assertEqual(on.loc[(0, 8, 1), "reason"], "one_pixel_component")

    def test_no_probability_or_npc_cutoff_exists(self):
        fields = set(dp.PopulationSettings().to_dict())
        self.assertEqual(fields, {"overlap_tolerance", "min_enrichment", "min_solidity",
                                  "partial_parent_policy", "exclude_one_pixel_components"})
        standing = dp.PopulationSettings().describe().set_index("setting").standing
        self.assertEqual(standing["overlap_tolerance"], "APPROVED")
        self.assertEqual(standing["partial_parent_policy"], "EXPLORATORY")
        self.assertEqual(dp.PopulationSettings().overlap_tolerance, 0.05)

    def test_invalid_settings_and_inputs_raise(self):
        with self.assertRaises(ValueError):
            dp.PopulationSettings(overlap_tolerance=0)
        with self.assertRaises(ValueError):
            dp.PopulationSettings(partial_parent_policy="accept")
        with self.assertRaises(ValueError):
            self.decide([component(0, 8, 1, 1, parent=3, status="something_new")])
        with self.assertRaises(ValueError):
            self.decide([component(0, 8, 1, 1, parent=3), component(0, 8, 1, 2, parent=3)])
        with self.assertRaises(KeyError):
            dp.decide_components(pd.DataFrame([dict(t=0, z=1, label=1)]), self.flags)

    def test_empty_components(self):
        """Handoff case 6: an empty table goes through every step."""
        empty = pd.DataFrame(columns=list(component(0, 0, 0, 0, 0)))
        population = dp.build_population(empty, circles())
        for key in ("components", "accepted", "excluded", "unresolved", "linked_ids", "pairs", "flags"):
            self.assertTrue(population[key].empty, key)
        self.assertEqual(len(dp.select_population(empty, population)), 0)


class LinkedIdOutcomes(unittest.TestCase):
    """Handoff case 7: losing some components is not losing the linked ID."""

    def setUp(self):
        d = spacing_for_overlap(0.30)
        table = circles(circle(0, 8, 1, 0, 0, 100), circle(0, 8, 2, d, 0, 100),
                        circle(0, 8, 3, 5000, 0, 100))
        rows = (
            [component(0, z, 1, 100, parent=1) for z in (7, 8, 9)] +                # all flagged
            [component(0, 7, 2, 200, parent=1), component(0, 8, 2, 200, parent=3),  # mixed
             component(0, 9, 2, 200, parent=3)] +
            [component(0, z, 3, 300, parent=3) for z in (7, 8)] +                   # clean
            [component(0, 8, 4, 400, parent=1, enrichment=1.0)] +                   # artifact in flagged
            [component(0, 8, 5, 500, parent=0, status="missing_parent", enrichment=np.nan, reference_ok=False),
             component(0, 9, 5, 500, parent=3, enrichment=1.0)] +                   # unknown + failed
            [component(0, 8, 6, 600, parent=3, area=50), component(0, 8, 7, 600, parent=3, area=80)])  # branch
        self.population = dp.build_population(pd.DataFrame(rows), table)
        self.ids = self.population["linked_ids"].set_index("nucleus_3d_id")

    def test_complete_loss(self):
        r = self.ids.loc[100]
        self.assertEqual((r.status, r.n_accepted, r.components_removed_by_overlap), ("excluded", 0, 3))
        self.assertTrue(r.lost_to_overlap)
        self.assertFalse(r.partially_removed)
        self.assertTrue(np.isnan(r.max_area_pixels_accepted))

    def test_component_removal_without_losing_the_id(self):
        r = self.ids.loc[200]
        self.assertEqual((r.status, r.n_accepted, r.n_excluded), ("accepted", 2, 1))
        self.assertEqual(r.components_removed_by_overlap, 1)
        self.assertFalse(r.lost_to_overlap)
        self.assertTrue(r.partially_removed)
        self.assertTrue(r.has_overlap_flagged_component)
        self.assertEqual(r.reasons, "overlapping_droplet_geometry")

    def test_untouched_id(self):
        r = self.ids.loc[300]
        self.assertEqual((r.status, r.n_accepted, r.reasons), ("accepted", 2, ""))
        self.assertFalse(r.partially_removed)

    def test_id_already_failing_is_not_attributed_to_overlap(self):
        r = self.ids.loc[400]
        self.assertEqual(r.status, "excluded")
        self.assertFalse(r.lost_to_overlap)
        self.assertEqual(r.reasons, "enrichment + overlapping_droplet_geometry")

    def test_id_with_an_unjudged_component_is_unresolved_not_excluded(self):
        r = self.ids.loc[500]
        self.assertEqual((r.status, r.n_excluded, r.n_unresolved), ("unresolved", 1, 1))
        self.assertEqual(r.reasons, "enrichment + missing_parent")

    def test_branch_flag_is_kept_visible(self):
        self.assertTrue(self.ids.loc[600].multiple_instances_same_z)
        self.assertFalse(self.ids.loc[300].multiple_instances_same_z)
        self.assertEqual(self.ids.loc[600].max_area_pixels_accepted, 80)

    def test_impact_table_counts_components_and_ids_separately(self):
        impact = self.population["impact"].iloc[0]
        self.assertEqual(impact.droplets_flagged, 2)
        self.assertEqual(impact.components_removed, 4)          # 3 of ID 100 + 1 of ID 200
        self.assertEqual(impact.linked_ids_lost, 1)             # only ID 100
        self.assertEqual(impact.baseline_linked_ids - impact.remaining_linked_ids, 1)
        self.assertEqual(impact.unknown_gate_decisions, 1)

    def test_tables_partition_the_components(self):
        p = self.population
        self.assertEqual(len(p["accepted"]) + len(p["excluded"]) + len(p["unresolved"]),
                         len(p["components"]))
        keys = ["t", "z", "label"]
        seen = pd.concat([p["accepted"][keys], p["excluded"][keys], p["unresolved"][keys]])
        self.assertFalse(seen.duplicated().any())
        self.assertTrue((p["excluded"].exclusion_reasons != "").all())
        self.assertTrue((p["unresolved"].unresolved_reasons != "").all())
        self.assertTrue((p["accepted"].reason == "accepted").all())

    def test_summary_and_reason_counts(self):
        summary = self.population["summary"].iloc[0]
        self.assertEqual(summary.linked_ids, 6)
        self.assertEqual((summary.linked_ids_accepted, summary.linked_ids_excluded,
                          summary.linked_ids_unresolved), (3, 2, 1))
        self.assertEqual(summary.linked_ids_branched, 1)
        reasons = self.population["reasons"].set_index("reason").components
        self.assertEqual(reasons["overlapping_droplet_geometry"], 5)
        self.assertEqual(reasons["enrichment"], 2)
        self.assertEqual(reasons["missing_parent"], 1)

    def test_reviewed_examples_are_reported_not_applied(self):
        labels = {"T0_N100": "real nucleus", "T0_N400": "artifact", "T0_N999": "artifact"}
        before = self.population["components"].copy(deep=True)
        report = dp.reviewed_outcomes(self.population, labels).set_index("review_id")
        self.assertEqual(report.loc["T0_N100", "status"], "excluded")
        self.assertTrue(report.loc["T0_N100", "lost_to_overlap"])
        self.assertEqual(report.loc["T0_N400", "reasons"], "enrichment + overlapping_droplet_geometry")
        self.assertEqual(report.loc["T0_N999", "status"], "not_in_population")
        pd.testing.assert_frame_equal(self.population["components"], before)
        with self.assertRaises(ValueError):
            dp.reviewed_outcomes(self.population, {"T2_N11185x": "artifact"})

    def test_tag_depends_on_rules_and_upstream(self):
        base = dp.population_tag(dp.PopulationSettings(), dict(geometry="a"))
        self.assertEqual(base, dp.population_tag(dp.PopulationSettings(), dict(geometry="a")))
        self.assertNotEqual(base, dp.population_tag(dp.PopulationSettings(overlap_tolerance=0.10), dict(geometry="a")))
        self.assertNotEqual(base, dp.population_tag(dp.PopulationSettings(), dict(geometry="b")))

    def test_compare_tolerances_is_monotonic_and_saves_nothing(self):
        trials = dp.compare_tolerances(self.population["components"].drop(
            columns=[c for c in self.population["components"].columns
                     if c in ("status", "reason", "exclusion_reasons", "unresolved_reasons",
                              "assignment_resolved", "parent_flagged", "status_without_overlap",
                              "removed_by_overlap")]),
            self.population["circles"], (0.05, 0.20, 0.50))
        self.assertEqual(list(trials.tolerance), [0.05, 0.20, 0.50])
        self.assertEqual(list(trials.droplets_flagged), [2, 2, 0])
        self.assertEqual(list(trials.linked_ids_lost), [1, 1, 0])


if __name__ == "__main__":
    unittest.main()
