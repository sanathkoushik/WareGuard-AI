"""
WareGuard AI - Behavior Engine Tests (Phase 2)

Runs without ultralytics, torch, cv2, numpy or pandas installed - the whole
point of keeping `behavior` on the standard library.

    python -m unittest tests.test_behavior -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from behavior import BehaviorEngine, SimConfig, build_demo_scene, build_scenario
from behavior.detectors import (
    EVENT_DRAG,
    EVENT_DROP,
    EVENT_IMPROPER_STACK,
    EVENT_ROUGH_HANDLING,
    EVENT_THROW,
    find_runs,
    saturate,
)
from behavior.features import moving_average, percentile
from behavior.schema import SceneContext, tracks_from_rows
from behavior.simulation import SceneBuilder


def types_in(report):
    return sorted({e.event_type for e in report.events})


class TestNumericHelpers(unittest.TestCase):
    def test_percentile(self):
        values = [1, 2, 3, 4, 5]
        self.assertEqual(percentile(values, 0.0), 1)
        self.assertEqual(percentile(values, 1.0), 5)
        self.assertEqual(percentile(values, 0.5), 3)
        self.assertEqual(percentile([], 0.5), 0.0)

    def test_moving_average_preserves_length_and_edges(self):
        out = moving_average([0.0, 0.0, 9.0, 0.0, 0.0], 3)
        self.assertEqual(len(out), 5)
        # The spike spreads but must not shift: the result stays symmetric
        # about the original peak. (Checking argmax would be wrong here -
        # indices 1..3 tie, so it proves nothing about centring.)
        self.assertAlmostEqual(out[0], out[4])
        self.assertAlmostEqual(out[1], out[3])
        self.assertGreater(out[2], out[0])

    def test_saturate(self):
        self.assertEqual(saturate(0.5, 1.0, 2.0), 0.0)
        self.assertEqual(saturate(2.0, 1.0, 2.0), 1.0)
        self.assertAlmostEqual(saturate(1.5, 1.0, 2.0), 0.5)

    def test_find_runs_tolerates_a_single_gap(self):
        class K:
            def __init__(self, v):
                self.v = v

        seq = [K(1), K(1), K(0), K(1), K(1)]
        runs = find_runs(seq, lambda k: k.v == 1, min_length=2, max_break=1)
        self.assertEqual(runs, [(0, 4)], "one bad frame must not split a run")

        runs = find_runs(seq, lambda k: k.v == 1, min_length=2, max_break=0)
        self.assertEqual(runs, [(0, 1), (3, 4)])


class TestScenarioDetection(unittest.TestCase):
    """Each simulated behavior must produce its own event type and no other."""

    def setUp(self):
        self.engine = BehaviorEngine()

    def _run(self, scenario):
        tracks, ctx = build_scenario(scenario)
        return self.engine.analyze(tracks, ctx)

    def test_normal_carry_produces_no_events(self):
        """The control case. A detector that fires on careful handling is
        worse than no detector, because it trains supervisors to ignore it."""
        report = self._run("normal_carry")
        self.assertEqual(
            report.events, [],
            f"careful handling flagged as {types_in(report)}",
        )

    def test_drop(self):
        self.assertEqual(types_in(self._run("drop")), [EVENT_DROP])

    def test_drag(self):
        self.assertEqual(types_in(self._run("drag")), [EVENT_DRAG])

    def test_throw(self):
        self.assertEqual(types_in(self._run("throw")), [EVENT_THROW])

    def test_improper_stack(self):
        self.assertEqual(types_in(self._run("improper_stack")), [EVENT_IMPROPER_STACK])

    def test_rough_handling(self):
        self.assertEqual(types_in(self._run("rough_handling")), [EVENT_ROUGH_HANDLING])

    def test_drop_reports_physically_sensible_impact_speed(self):
        report = self._run("drop")
        drop = report.events[0]
        speed = drop.metrics["impact_speed_mps"]
        # A 300px fall at 175 px/m is ~1.7m; v = sqrt(2gh) ~ 5.8 m/s. Smoothing
        # shaves the peak, so accept a band rather than a point value.
        self.assertGreater(speed, 3.5)
        self.assertLess(speed, 7.0)
        self.assertGreater(drop.metrics["gravity_ratio"], 0.7)


class TestDiscrimination(unittest.TestCase):
    """The separations between detectors, tested directly."""

    def setUp(self):
        self.engine = BehaviorEngine()

    def test_carried_box_is_not_a_throw(self):
        """A carried box moves fast and high but is *supported*: its vertical
        acceleration is ~0, not g. Without that test every carry across the
        frame reads as a throw."""
        tracks, ctx = build_scenario("normal_carry")
        report = self.engine.analyze(tracks, ctx)
        self.assertNotIn(EVENT_THROW, types_in(report))

    def test_shoved_box_is_not_a_throw(self):
        tracks, ctx = build_scenario("rough_handling")
        report = self.engine.analyze(tracks, ctx)
        self.assertNotIn(EVENT_THROW, types_in(report))

    def test_throw_outranks_drop_when_both_match(self):
        """A thrown box also falls. Only the more specific claim survives, and
        the suppressed one is recorded rather than lost."""
        tracks, ctx = build_scenario("throw")
        report = self.engine.analyze(tracks, ctx)
        self.assertEqual(types_in(report), [EVENT_THROW])
        self.assertGreaterEqual(report.suppressed_events, 1)
        self.assertIn("also matched", report.events[0].description)

    def test_dragged_box_is_not_a_drop(self):
        tracks, ctx = build_scenario("drag")
        self.assertNotIn(EVENT_DROP, types_in(self.engine.analyze(tracks, ctx)))


class TestScaleInvariance(unittest.TestCase):
    """The core design claim: thresholds in object-heights per second hold
    across resolution and frame rate without recalibration."""

    def _drop_scene(self, scale=1.0, fps=30.0):
        cfg = SimConfig(
            fps=fps,
            width=int(1280 * scale),
            height=int(720 * scale),
            floor_y=620.0 * scale,
            box_w=90.0 * scale,
            box_h=70.0 * scale,
            person_w=70.0 * scale,
            person_h=200.0 * scale,
        )
        b = SceneBuilder(cfg)
        b.add_drop(1, 0, cx=500.0 * scale, release_height_px=300.0 * scale)
        return b.build()

    def test_same_events_at_2x_resolution(self):
        engine = BehaviorEngine()
        small = engine.analyze(*self._drop_scene(scale=1.0))
        large = engine.analyze(*self._drop_scene(scale=2.0))

        self.assertEqual(types_in(small), [EVENT_DROP])
        self.assertEqual(types_in(large), [EVENT_DROP])

        # Normalised metrics must agree despite 4x the pixels.
        self.assertAlmostEqual(
            small.events[0].metrics["fall_distance_heights"],
            large.events[0].metrics["fall_distance_heights"],
            places=1,
        )
        self.assertAlmostEqual(
            small.events[0].metrics["gravity_ratio"],
            large.events[0].metrics["gravity_ratio"],
            places=1,
        )

    def test_same_events_at_half_frame_rate(self):
        engine = BehaviorEngine()
        fast = engine.analyze(*self._drop_scene(fps=30.0))
        slow = engine.analyze(*self._drop_scene(fps=15.0))

        self.assertEqual(types_in(fast), [EVENT_DROP])
        self.assertEqual(types_in(slow), [EVENT_DROP])
        self.assertAlmostEqual(
            fast.events[0].metrics["fall_distance_heights"],
            slow.events[0].metrics["fall_distance_heights"],
            delta=0.6,
        )


class TestRobustness(unittest.TestCase):
    def test_full_demo_scene_finds_every_behavior_once(self):
        tracks, ctx = build_demo_scene()
        report = BehaviorEngine().analyze(tracks, ctx)
        self.assertEqual(
            types_in(report),
            sorted([EVENT_DROP, EVENT_THROW, EVENT_DRAG,
                    EVENT_IMPROPER_STACK, EVENT_ROUGH_HANDLING]),
        )
        self.assertEqual(len(report.events), 5)

    def test_survives_bbox_jitter_and_dropped_frames(self):
        """Real detectors flicker. The engine must still find the major events
        at 15% dropout with 3px of corner jitter."""
        tracks, ctx = build_demo_scene(SimConfig(noise_px=3.0, dropout=0.15))
        report = BehaviorEngine().analyze(tracks, ctx)
        found = types_in(report)
        for expected in (EVENT_DROP, EVENT_DRAG, EVENT_THROW):
            self.assertIn(expected, found)

    def test_event_ids_are_assigned_and_ordered(self):
        tracks, ctx = build_demo_scene()
        report = BehaviorEngine().analyze(tracks, ctx)
        ids = [e.event_id for e in report.events]
        self.assertEqual(ids, sorted(ids))
        self.assertTrue(all(i.startswith("EVT-") for i in ids))

    def test_threshold_profiles_change_sensitivity(self):
        tracks, ctx = build_demo_scene(SimConfig(noise_px=4.0, dropout=0.25))
        strict = len(BehaviorEngine("strict").analyze(tracks, ctx).events)
        sensitive = len(BehaviorEngine("sensitive").analyze(tracks, ctx).events)
        self.assertGreaterEqual(sensitive, strict)

    def test_unknown_profile_is_rejected_loudly(self):
        with self.assertRaises(ValueError):
            BehaviorEngine("aggressive")


class TestDataQualityHonesty(unittest.TestCase):
    """Zero events must never be reported as 'no risk' when the input could
    not support analysis. This is the failure mode that would make the tool
    dangerous in a real warehouse."""

    def test_empty_input(self):
        report = BehaviorEngine().analyze([], SceneContext())
        self.assertEqual(report.events, [])
        self.assertIsNotNone(report.data_quality_warning())

    def test_flickering_tracks_are_reported_as_unusable(self):
        """Mirrors the real sample_warehouse log: many one- and two-frame
        tracks with no continuity."""
        rows = []
        for tid in range(20):
            for f in range(2):
                rows.append({
                    "frame": tid * 10 + f * 7,
                    "timestamp": (tid * 10 + f * 7) / 30.0,
                    "track_id": 9000 + tid,
                    "class_name": "box",
                    "confidence": 0.31,
                    "bbox": (100.0, 100.0, 150.0, 150.0),
                })
        report = BehaviorEngine().analyze(tracks_from_rows(rows), SceneContext())
        self.assertEqual(report.events, [])
        warning = report.data_quality_warning()
        self.assertIsNotNone(warning)
        self.assertIn("usable", warning)

    def test_no_cargo_class_is_called_out(self):
        rows = [{
            "frame": f, "timestamp": f / 30.0, "track_id": 1,
            "class_name": "traffic light", "confidence": 0.8,
            "bbox": (10.0, 10.0, 40.0, 60.0),
        } for f in range(20)]
        report = BehaviorEngine().analyze(tracks_from_rows(rows), SceneContext())
        self.assertIn("cargo", report.data_quality_warning())

    def test_confidence_is_reduced_on_gappy_tracks(self):
        clean, ctx = build_scenario("drop")
        gappy, ctx2 = build_scenario("drop", SimConfig(dropout=0.35))
        engine = BehaviorEngine()
        a = engine.analyze(clean, ctx)
        b = engine.analyze(gappy, ctx2)
        if a.events and b.events:
            self.assertLessEqual(b.events[0].confidence, a.events[0].confidence)


if __name__ == "__main__":
    unittest.main(verbosity=2)
