"""
WareGuard AI - Risk Engine Tests (Phase 3)

    python -m unittest tests.test_risk -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from behavior import BehaviorEngine, SceneContext, build_demo_scene, build_scenario
from behavior.detectors import EVENT_DROP
from behavior.engine import BehaviorReport
from behavior.schema import BehaviorEvent
from risk import (
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_ORDER,
    RiskEngine,
    RiskScorer,
    severity_for,
)
from risk.export import (
    assessment_to_assistant_context,
    export_events_csv,
    export_events_json,
)


def make_event(**overrides) -> BehaviorEvent:
    defaults = dict(
        event_type=EVENT_DROP,
        track_id=1,
        class_name="box",
        start_frame=0,
        end_frame=10,
        start_time=0.0,
        end_time=0.33,
        confidence=0.9,
        metrics={"impact_speed_mps": 3.0, "fall_distance_heights": 2.0},
        description="test drop",
        event_id="EVT-0001",
    )
    defaults.update(overrides)
    return BehaviorEvent(**defaults)


def assess_events(events, duration=120.0):
    report = BehaviorReport(events=list(events), context=SceneContext())
    report.total_tracks = 4
    report.cargo_tracks = 2
    report.usable_tracks = 2
    return RiskEngine().assess(report, duration_seconds=duration)


class TestSeverityBands(unittest.TestCase):
    def test_band_boundaries(self):
        self.assertEqual(severity_for(0), SEVERITY_LOW)
        self.assertEqual(severity_for(24.9), SEVERITY_LOW)
        self.assertEqual(severity_for(25), SEVERITY_MEDIUM)
        self.assertEqual(severity_for(49.9), SEVERITY_MEDIUM)
        self.assertEqual(severity_for(50), SEVERITY_HIGH)
        self.assertEqual(severity_for(74.9), SEVERITY_HIGH)
        self.assertEqual(severity_for(75), SEVERITY_CRITICAL)
        self.assertEqual(severity_for(100), SEVERITY_CRITICAL)


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.scorer = RiskScorer()

    def test_score_is_the_sum_of_its_factors(self):
        """Explainability is a hard requirement: the number must equal the
        reasons given for it, or the breakdown is decoration."""
        score, factors = self.scorer.score(make_event())
        self.assertAlmostEqual(score, sum(f.points for f in factors), places=5)

    def test_faster_impact_scores_higher(self):
        slow, _ = self.scorer.score(make_event(metrics={"impact_speed_mps": 2.0}))
        fast, _ = self.scorer.score(make_event(metrics={"impact_speed_mps": 6.0}))
        self.assertGreater(fast, slow)

    def test_worker_proximity_escalates(self):
        alone, _ = self.scorer.score(make_event())
        near, factors = self.scorer.score(
            make_event(metrics={"impact_speed_mps": 3.0, "nearest_person_heights": 0.1})
        )
        self.assertGreater(near, alone)
        self.assertIn("worker_proximity", [f.code for f in factors])

    def test_distant_worker_does_not_escalate(self):
        _, factors = self.scorer.score(
            make_event(metrics={"impact_speed_mps": 3.0, "nearest_person_heights": 5.0})
        )
        self.assertNotIn("worker_proximity", [f.code for f in factors])

    def test_missing_proximity_is_unknown_not_safe(self):
        """No worker detected is not evidence that no worker was there. The
        model must neither invent risk nor silently assume safety."""
        _, factors = self.scorer.score(make_event(metrics={"impact_speed_mps": 3.0}))
        self.assertNotIn("worker_proximity", [f.code for f in factors])

    def test_score_is_bounded(self):
        score, _ = self.scorer.score(
            make_event(metrics={
                "impact_speed_mps": 99.0,
                "fall_distance_heights": 99.0,
                "nearest_person_heights": 0.0,
            })
        )
        self.assertLessEqual(score, 100.0)
        self.assertGreaterEqual(score, 0.0)


class TestPriorityVsSeverity(unittest.TestCase):
    """Severity answers 'how bad if real'; priority answers 'look at this
    first'. Collapsing them hides dangerous-but-uncertain events."""

    def test_priority_is_severity_scaled_by_confidence(self):
        a = assess_events([make_event(confidence=1.0)])
        b = assess_events([make_event(confidence=0.5)])
        self.assertEqual(a.events[0].risk_score, b.events[0].risk_score)
        self.assertGreater(
            a.events[0].metrics["priority_score"],
            b.events[0].metrics["priority_score"],
        )

    def test_ranking_uses_priority(self):
        certain_medium = make_event(
            event_id="EVT-0001", track_id=1, confidence=0.95,
            metrics={"impact_speed_mps": 2.2},
        )
        unsure_severe = make_event(
            event_id="EVT-0002", track_id=2, confidence=0.36,
            start_time=5.0, end_time=5.5,
            metrics={"impact_speed_mps": 7.0, "fall_distance_heights": 4.0,
                     "nearest_person_heights": 0.1},
        )
        a = assess_events([certain_medium, unsure_severe])
        self.assertGreater(
            unsure_severe.risk_score, certain_medium.risk_score,
            "the uncertain event is genuinely more severe",
        )
        self.assertEqual(a.ranked()[0].event_id, "EVT-0001",
                         "but the confident one is what to review first")


class TestRepetition(unittest.TestCase):
    def test_repeat_on_same_load_escalates(self):
        first = make_event(event_id="EVT-0001", track_id=7)
        second = make_event(event_id="EVT-0002", track_id=7,
                            start_time=10.0, end_time=10.4)
        a = assess_events([first, second])
        self.assertGreater(a.events[1].risk_score, a.events[0].risk_score)
        self.assertIn(7, a.summary.repeat_offender_tracks)

    def test_repeated_pattern_escalates_after_the_third(self):
        events = [
            make_event(event_id=f"EVT-000{i}", track_id=i,
                       start_time=i * 5.0, end_time=i * 5.0 + 0.4)
            for i in range(1, 5)
        ]
        a = assess_events(events)
        self.assertGreater(a.events[3].risk_score, a.events[0].risk_score)


class TestShiftSummary(unittest.TestCase):
    def test_shift_severity_is_never_below_the_worst_event(self):
        """One critical incident makes the shift critical. A blended index
        must not be able to launder that into 'High'."""
        severe = make_event(metrics={
            "impact_speed_mps": 9.0,
            "fall_distance_heights": 5.0,
            "nearest_person_heights": 0.0,
        })
        a = assess_events([severe], duration=3600.0)
        self.assertEqual(a.events[0].severity, SEVERITY_CRITICAL)
        self.assertEqual(a.summary.shift_severity, SEVERITY_CRITICAL)

    def test_short_clip_rate_is_flagged_unreliable(self):
        a = assess_events([make_event()], duration=12.0)
        self.assertFalse(a.summary.rate_is_reliable)

    def test_long_clip_rate_is_trusted(self):
        a = assess_events([make_event()], duration=600.0)
        self.assertTrue(a.summary.rate_is_reliable)

    def test_no_events_is_not_reported_as_risk(self):
        a = assess_events([], duration=300.0)
        self.assertEqual(a.summary.total_events, 0)
        self.assertEqual(a.summary.shift_risk_index, 0.0)
        self.assertEqual(a.summary.shift_severity, SEVERITY_LOW)

    def test_zero_events_on_bad_data_never_says_no_risk(self):
        """The most dangerous sentence this tool could produce."""
        report = BehaviorReport(events=[], context=SceneContext())
        report.total_tracks = 40
        report.cargo_tracks = 30
        report.usable_tracks = 0
        a = RiskEngine().assess(report, duration_seconds=60.0)
        headline = a.summary.headline().lower()
        self.assertIn("could not support analysis", headline)
        self.assertNotIn("no unsafe handling", headline)

    def test_timeline_buckets_cover_the_clip(self):
        a = assess_events(
            [make_event(start_time=5.0, end_time=5.4),
             make_event(event_id="EVT-0002", track_id=2,
                        start_time=95.0, end_time=95.4)],
            duration=120.0,
        )
        with_events = [b for b in a.summary.timeline if b["events"]]
        self.assertEqual(len(with_events), 2)
        self.assertEqual(sum(b["events"] for b in a.summary.timeline), 2)


class TestEndToEnd(unittest.TestCase):
    def test_demo_scene_produces_a_scored_assessment(self):
        tracks, ctx = build_demo_scene()
        report = BehaviorEngine().analyze(tracks, ctx)
        a = RiskEngine().assess(report)

        self.assertEqual(len(a.events), 5)
        for e in a.events:
            self.assertIsNotNone(e.risk_score)
            self.assertIn(e.severity, SEVERITY_ORDER)
            self.assertTrue(e.risk_factors, "every event must explain its score")
            self.assertIn("priority_score", e.metrics)

    def test_throw_near_a_worker_outranks_an_isolated_drop(self):
        tracks, ctx = build_demo_scene()
        a = RiskEngine().assess(BehaviorEngine().analyze(tracks, ctx))
        by_type = {e.event_type: e for e in a.events}
        self.assertGreater(by_type["throw"].risk_score, by_type["improper_stack"].risk_score)


class TestExport(unittest.TestCase):
    def setUp(self):
        tracks, ctx = build_demo_scene()
        self.assessment = RiskEngine().assess(BehaviorEngine().analyze(tracks, ctx))

    def test_json_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = export_events_json(self.assessment, Path(tmp) / "events.json")
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(len(data["events"]), 5)
            self.assertIn("summary", data)
            self.assertIn("risk_factors", data)

    def test_csv_has_a_row_per_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = export_events_csv(self.assessment, Path(tmp) / "events.csv")
            lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 6)  # header + 5
            self.assertIn("severity", lines[0])

    def test_assistant_context_is_compact_and_explains_itself(self):
        ctx = assessment_to_assistant_context(self.assessment)
        self.assertIn("shift", ctx)
        self.assertEqual(len(ctx["events"]), 5)
        for e in ctx["events"]:
            self.assertIn("why_this_score", e)
            self.assertIn("what_happened", e)
        # No raw kinematics leak into the LLM payload.
        self.assertNotIn("thresholds", ctx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
