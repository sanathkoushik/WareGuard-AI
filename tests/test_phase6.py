"""
WareGuard AI - Phase 5 fixes + Phase 6 polish tests

Covers the Phase 5 issues fixed during review and the Phase 6 additions.
Kept in its own module so `tests/test_assistant.py` (the original Phase 5
suite) stays untouched.

    python -m unittest tests.test_phase6 -v

Runs with no third-party packages and no API key.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_demo
from assistant import (
    InvalidEventsLog,
    LLMClient,
    WarehouseAssistant,
    build_context,
    is_grounded,
    load_context_from_events_log,
)
from assistant import heuristic
from behavior import BehaviorEngine, build_demo_scene
from risk import RiskEngine
from risk.export import assessment_to_assistant_context

DEMO_LOG = Path(__file__).resolve().parent.parent / "data" / "logs" / "events_sim_demo.json"

OFFLINE = lambda: LLMClient(api_key=None, base_url=None)  # noqa: E731


def demo_context():
    return load_context_from_events_log(DEMO_LOG)


def blocked_context():
    """A real blocked analysis: the bundled detection log yields 1/32 usable."""
    log = Path(__file__).resolve().parent.parent / "data" / "logs" / "detections_sample_warehouse.json"
    payload = RiskEngine().assess(BehaviorEngine().analyze_json(log)).to_dict()
    return build_context(payload)


# --------------------------------------------------------------------------
# Phase 5 fix 1 - severity-qualified counts
# --------------------------------------------------------------------------

class TestSeverityCounts(unittest.TestCase):
    """Regression: "how many critical incidents?" reported the shift total.

    The count branch matched only against by_type, so a severity-qualified
    question fell through and answered "5 event(s)" for a shift with 2
    Critical - a confidently wrong figure on a headline demo question.
    """

    def setUp(self):
        self.context = demo_context()
        self.bot = WarehouseAssistant(self.context, client=OFFLINE())

    def test_critical_count_is_the_critical_count(self):
        expected = self.context["shift"]["by_severity"].get("Critical", 0)
        answer = self.bot.ask("How many critical incidents?")
        self.assertTrue(answer.startswith(f"{expected} Critical"), answer)
        self.assertNotIn("5 event(s) this shift", answer)

    def test_each_present_severity_reports_its_own_count(self):
        for severity, count in self.context["shift"]["by_severity"].items():
            with self.subTest(severity=severity):
                answer = self.bot.ask(f"how many {severity.lower()} events?")
                self.assertTrue(answer.startswith(f"{count} {severity}"), answer)

    def test_absent_severity_reports_zero_not_the_total(self):
        """by_severity omits zero entries; the answer must still be 0."""
        self.assertNotIn("Low", self.context["shift"]["by_severity"])
        answer = self.bot.ask("how many low severity events?")
        self.assertTrue(answer.startswith("0 Low"), answer)

    def test_type_counts_still_work(self):
        answer = self.bot.ask("how many drop events?")
        self.assertTrue(answer.startswith("1 drop"), answer)

    def test_absent_type_reports_zero(self):
        one_type = {
            "shift": dict(self.context["shift"], by_type={"drop": 1}),
            "events": self.context["events"],
        }
        bot = WarehouseAssistant(one_type, client=OFFLINE())
        self.assertTrue(bot.ask("how many throws?").startswith("0 throw"))

    def test_unqualified_count_still_reports_the_total(self):
        answer = self.bot.ask("how many events?")
        self.assertIn(str(self.context["shift"]["total_events"]), answer)


# --------------------------------------------------------------------------
# Phase 5 fix 2 - event-type question
# --------------------------------------------------------------------------

class TestEventTypeQuestion(unittest.TestCase):
    def test_types_question_lists_types(self):
        bot = WarehouseAssistant(demo_context(), client=OFFLINE())
        answer = bot.ask("What types of unsafe behavior occurred?")
        self.assertIn("Unsafe handling types", answer)
        for event_type in demo_context()["shift"]["by_type"]:
            self.assertIn(event_type.replace("_", " "), answer)


# --------------------------------------------------------------------------
# Phase 5 fix 3 - partial-analysis caveat
# --------------------------------------------------------------------------

class TestQualityCaveat(unittest.TestCase):
    def setUp(self):
        base = demo_context()
        self.partial = {
            "shift": dict(base["shift"], data_quality_warning="Only 4 of 30 usable."),
            "events": base["events"],
        }

    def test_findings_on_a_partial_analysis_carry_a_caveat(self):
        bot = WarehouseAssistant(self.partial, client=OFFLINE())
        for question in ["how many critical incidents?", "what was the worst event?",
                         "give me a summary", "what types occurred?"]:
            with self.subTest(question=question):
                self.assertIn("partial analysis", bot.ask(question).lower())

    def test_clean_analysis_has_no_caveat(self):
        bot = WarehouseAssistant(demo_context(), client=OFFLINE())
        self.assertNotIn("partial analysis", bot.ask("how many critical incidents?").lower())

    def test_quality_note_is_empty_without_a_warning(self):
        self.assertEqual(heuristic._quality_note(demo_context()), "")


# --------------------------------------------------------------------------
# Phase 5 fix 4 - LLM output is verified before it is trusted
# --------------------------------------------------------------------------

class TestLLMGrounding(unittest.TestCase):
    def test_grounded_numbers_pass(self):
        self.assertTrue(is_grounded("Risk 74 over 5 events.", "index 74, 5 events"))

    def test_ungrounded_numbers_fail(self):
        self.assertFalse(is_grounded("Risk 999 over 42 events.", "index 74, 5 events"))

    def test_trailing_zeros_are_not_treated_as_fabrication(self):
        self.assertTrue(is_grounded("scored 100", "risk_score: 100.0"))

    def test_text_without_numbers_is_grounded(self):
        self.assertTrue(is_grounded("The shift went badly.", "anything"))

    def test_fabricating_model_is_discarded(self):
        """The core LLM-safety guarantee: an invented score never reaches the UI."""
        class Fabricating(LLMClient):
            def complete(self, system_prompt, user_prompt):
                return "The shift scored 999 with 42 critical incidents."

        bot = WarehouseAssistant(demo_context(), client=Fabricating(api_key="sk-test"))
        answer = bot.ask("What was the worst event?")
        self.assertNotIn("999", answer)
        self.assertNotIn("42 critical", answer)
        self.assertIn("EVT-", answer)

    def test_faithful_model_output_is_kept(self):
        class Faithful(LLMClient):
            def complete(self, system_prompt, user_prompt):
                return "EVT-0003 was the worst event, at risk 100.0."

        bot = WarehouseAssistant(demo_context(), client=Faithful(api_key="sk-test"))
        self.assertEqual(
            bot.ask("worst?"), "EVT-0003 was the worst event, at risk 100.0."
        )

    def test_empty_model_output_falls_back(self):
        class Empty(LLMClient):
            def complete(self, system_prompt, user_prompt):
                return "   "

        bot = WarehouseAssistant(demo_context(), client=Empty(api_key="sk-test"))
        self.assertIn("EVT-", bot.ask("worst event?"))


# --------------------------------------------------------------------------
# Phase 5 fix 5 - malformed payloads produce readable errors
# --------------------------------------------------------------------------

class TestMalformedLogs(unittest.TestCase):
    def test_missing_summary_is_a_readable_error(self):
        with self.assertRaises(InvalidEventsLog) as ctx:
            build_context({"events": []})
        self.assertIn("summary", str(ctx.exception))

    def test_non_dict_payload(self):
        for bad in ([], "text", 42):
            with self.subTest(payload=bad):
                with self.assertRaises(InvalidEventsLog):
                    build_context(bad)

    def test_events_must_be_a_list(self):
        with self.assertRaises(InvalidEventsLog):
            build_context({"summary": {"total_events": 0}, "events": "nope"})

    def test_unknown_summary_keys_are_tolerated(self):
        """Forward compatibility: a new exported field must not break old logs."""
        context = build_context(
            {"summary": {"total_events": 0, "a_future_field": 123}, "events": []}
        )
        self.assertEqual(context["shift"]["total_events"], 0)

    def test_missing_file(self):
        with self.assertRaises(InvalidEventsLog):
            load_context_from_events_log("data/logs/definitely_not_here.json")

    def test_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(InvalidEventsLog) as ctx:
                load_context_from_events_log(path)
            self.assertIn("not valid JSON", str(ctx.exception))


# --------------------------------------------------------------------------
# Data quality - blocked analysis must never read as a safe shift
# --------------------------------------------------------------------------

class TestBlockedAnalysisHonesty(unittest.TestCase):
    def setUp(self):
        self.context = blocked_context()
        self.bot = WarehouseAssistant(self.context, client=OFFLINE())

    def test_the_bundled_sample_is_genuinely_blocked(self):
        self.assertEqual(self.context["shift"]["total_events"], 0)
        self.assertTrue(self.context["shift"]["data_quality_warning"])

    def test_no_question_yields_a_reassuring_answer(self):
        for question in ["What was the worst event?", "Was the shift safe?",
                         "How many critical incidents?", "Give me a summary",
                         "Any repeat offenders?"]:
            with self.subTest(question=question):
                answer = self.bot.ask(question)
                self.assertIn("could not support analysis", answer)


# --------------------------------------------------------------------------
# Phase 6 - shift report
# --------------------------------------------------------------------------

class TestShiftReport(unittest.TestCase):
    def setUp(self):
        tracks, ctx = build_demo_scene()
        self.assessment = RiskEngine().assess(BehaviorEngine().analyze(tracks, ctx))

    def test_report_contains_only_real_figures(self):
        report = run_demo.build_report(self.assessment, "test source")
        s = self.assessment.summary
        self.assertIn(f"{s.shift_risk_index:.0f}/100", report)
        self.assertIn(s.shift_severity, report)
        for event in self.assessment.events:
            self.assertIn(event.event_id, report)
            self.assertIn(event.description, report)

    def test_report_surfaces_a_data_quality_warning(self):
        blocked = RiskEngine().assess(
            BehaviorEngine().analyze_json(
                Path(__file__).resolve().parent.parent
                / "data" / "logs" / "detections_sample_warehouse.json"
            )
        )
        report = run_demo.build_report(blocked, "real detection log")
        self.assertIn("Data quality warning", report)
        self.assertIn("not evidence that the shift was safe", report)

    def test_report_is_markdown_and_names_its_source(self):
        report = run_demo.build_report(self.assessment, "simulated scenario `demo`")
        self.assertTrue(report.startswith("# WareGuard AI"))
        self.assertIn("simulated scenario `demo`", report)


# --------------------------------------------------------------------------
# Phase 6 - end-to-end demo runner
# --------------------------------------------------------------------------

class TestDemoRunner(unittest.TestCase):
    def test_simulated_demo_runs_end_to_end(self):
        self.assertEqual(run_demo.main(["--no-save", "--quick"]), 0)

    def test_demo_with_assistant_stage(self):
        self.assertEqual(run_demo.main(["--no-save"]), 0)

    def test_isolated_scenario(self):
        self.assertEqual(run_demo.main(["--scenario", "drop", "--no-save", "--quick"]), 0)

    def test_real_detection_log_path(self):
        self.assertEqual(
            run_demo.main([
                "--logs", "data/logs/detections_sample_warehouse.json",
                "--no-save", "--quick",
            ]),
            0,
        )

    def test_missing_log_exits_with_an_error_code(self):
        self.assertEqual(
            run_demo.main(["--logs", "data/logs/nope.json", "--no-save"]), 2
        )

    def test_unknown_profile_exits_with_an_error_code(self):
        self.assertEqual(
            run_demo.main(["--profile", "aggressive", "--no-save", "--quick"]), 2
        )

    def test_report_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            self.assertEqual(
                run_demo.main(["--no-save", "--quick", "--report", str(path)]), 0
            )
            self.assertTrue(path.exists())
            self.assertIn("WareGuard AI", path.read_text(encoding="utf-8"))

    def test_degraded_tracking_still_completes(self):
        self.assertEqual(
            run_demo.main(["--noise", "4", "--dropout", "0.2", "--no-save", "--quick"]),
            0,
        )


# --------------------------------------------------------------------------
# Phase 6 - dashboard assistant panel (data layer, no Streamlit required)
# --------------------------------------------------------------------------

class TestAssistantPanel(unittest.TestCase):
    def setUp(self):
        from dashboard import assistant_panel
        self.panel = assistant_panel

    def test_context_from_live_assessment(self):
        tracks, ctx = build_demo_scene()
        assessment = RiskEngine().assess(BehaviorEngine().analyze(tracks, ctx))
        context = self.panel.build_context_from_assessment(assessment)
        self.assertIsNotNone(context)
        self.assertEqual(context["shift"]["total_events"], 5)

    def test_context_from_none_assessment(self):
        self.assertIsNone(self.panel.build_context_from_assessment(None))

    def test_context_from_saved_log(self):
        context = self.panel.build_context_from_log(DEMO_LOG)
        self.assertIsNotNone(context)
        self.assertEqual(context["shift"]["total_events"], 5)

    def test_missing_log_returns_none_rather_than_raising(self):
        self.assertIsNone(self.panel.build_context_from_log("data/logs/nope.json"))

    def test_malformed_log_returns_none_rather_than_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertIsNone(self.panel.build_context_from_log(path))

    def test_panel_matches_the_live_context_shape(self):
        """The saved-log path and the live path must agree, or the assistant
        would answer differently depending on how the dashboard reached it."""
        tracks, ctx = build_demo_scene()
        assessment = RiskEngine().assess(BehaviorEngine().analyze(tracks, ctx))
        live = assessment_to_assistant_context(assessment)
        saved = self.panel.build_context_from_log(DEMO_LOG)
        self.assertEqual(set(live["shift"]), set(saved["shift"]))
        self.assertEqual(set(live["events"][0]), set(saved["events"][0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
