"""
WareGuard AI - Assistant Tests (Phase 5)

Runs with no third-party packages and no API key.

    python -m unittest tests.test_assistant -v

The most important tests here are the ones that assert what the assistant must
NOT say: no invented incidents, and no implication of safety when the data
cannot support one.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant import (
    STATE_BLOCKED,
    STATE_CLEAN,
    STATE_OK,
    STATE_UNAVAILABLE,
    InvalidEventsPayload,
    SafetyAssistant,
    ShiftContext,
    answer_question,
    parse_question,
)
from assistant.intents import (
    INTENT_COUNT,
    INTENT_DATA_QUALITY,
    INTENT_EVENT_TYPES,
    INTENT_FOCUS,
    INTENT_HIGHEST_RISK,
    INTENT_LIST_EVENTS,
    INTENT_MOST_DANGEROUS,
    INTENT_REPEAT_OFFENDER,
    INTENT_SUMMARY,
    INTENT_TIME_QUERY,
    INTENT_UNKNOWN,
    INTENT_WHY_SEVERITY,
)
from assistant.llm import LLMClient, LLMConfig, numbers_are_grounded

DEMO_EVENTS = Path(__file__).resolve().parent.parent / "data" / "logs" / "events_sim_demo.json"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def make_event(**overrides):
    event = {
        "event_id": "EVT-0001",
        "event_type": "drop",
        "severity": "High",
        "risk_score": 60.0,
        "confidence": 0.9,
        "track_id": 2,
        "class_name": "box",
        "start_frame": 100,
        "end_frame": 120,
        "start_time": 3.5,
        "end_time": 4.1,
        "duration_seconds": 0.6,
        "metrics": {"priority_score": 54.0, "impact_speed_mps": 5.0},
        "related_track_ids": [],
        "risk_factors": ["drop baseline (+40)", "impact at 5.0 m/s (+20)"],
        "description": "box #2 fell and stopped abruptly",
    }
    event.update(overrides)
    return event


def payload(events=None, **summary_overrides):
    summary = {
        "total_events": len(events or []),
        "duration_seconds": 120.0,
        "events_by_type": {},
        "events_by_severity": {},
        "max_risk_score": 0.0,
        "mean_risk_score": 0.0,
        "shift_risk_index": 0.0,
        "shift_severity": "Low",
        "events_per_minute": 0.0,
        "rate_is_reliable": True,
        "repeat_offender_tracks": [],
        "top_events": [],
        "timeline": [],
        "data_quality_warning": None,
    }
    summary.update(summary_overrides)
    return {"summary": summary, "events": events or [], "risk_factors": {}}


CRITICAL_PAYLOAD = payload(
    events=[
        make_event(
            event_id="EVT-0001", event_type="throw", severity="Critical",
            risk_score=100.0, confidence=0.97, track_id=4, start_time=7.7,
            end_time=8.4, description="box #4 was thrown",
            metrics={"priority_score": 97.0, "nearest_person_heights": 0.0},
        ),
        make_event(
            event_id="EVT-0002", event_type="drop", severity="Critical",
            risk_score=80.0, confidence=0.95, track_id=4, start_time=12.0,
            end_time=12.6, description="box #4 was dropped",
            metrics={"priority_score": 76.0},
        ),
        make_event(
            event_id="EVT-0003", event_type="drag", severity="Medium",
            risk_score=30.0, confidence=0.8, track_id=9, start_time=20.0,
            end_time=22.0, description="box #9 was dragged",
            metrics={"priority_score": 24.0},
        ),
    ],
    total_events=3,
    events_by_type={"throw": 1, "drop": 1, "drag": 1},
    events_by_severity={"Critical": 2, "Medium": 1, "High": 0, "Low": 0},
    max_risk_score=100.0, mean_risk_score=70.0,
    shift_risk_index=78.0, shift_severity="Critical",
    repeat_offender_tracks=[4],
)

CLEAN_PAYLOAD = payload(events=[], shift_severity="Low", shift_risk_index=0.0)

BLOCKED_PAYLOAD = {
    "summary": dict(
        payload()["summary"],
        data_quality_warning="Only 1 of 32 cargo tracks were usable; results are partial.",
    ),
    "events": [],
    "behavior_summary": {"usable_tracks": 1, "cargo_tracks": 32, "total_tracks": 46},
}


class TestIntentParsing(unittest.TestCase):
    def test_each_supported_question_is_recognised(self):
        expected = {
            "What were the most dangerous incidents?": INTENT_MOST_DANGEROUS,
            "Why was the shift rated Critical?": INTENT_WHY_SEVERITY,
            "Which track had the most incidents?": INTENT_REPEAT_OFFENDER,
            "What types of unsafe behavior occurred?": INTENT_EVENT_TYPES,
            "What should the supervisor focus on?": INTENT_FOCUS,
            "How many critical incidents occurred?": INTENT_COUNT,
            "What happened around 8 seconds?": INTENT_TIME_QUERY,
            "Which event had the highest risk score?": INTENT_HIGHEST_RISK,
            "Can I trust this analysis?": INTENT_DATA_QUALITY,
            "List all events.": INTENT_LIST_EVENTS,
            "Give me a summary": INTENT_SUMMARY,
        }
        for question, intent in expected.items():
            with self.subTest(question=question):
                self.assertEqual(parse_question(question).intent, intent)

    def test_entities_are_extracted(self):
        q = parse_question("how many critical drop events?")
        self.assertEqual(q.severity, "Critical")
        self.assertEqual(q.event_type, "drop")

        self.assertAlmostEqual(parse_question("what happened at 8 seconds?").timestamp, 8.0)
        self.assertAlmostEqual(parse_question("anything around 2 minutes?").timestamp, 120.0)
        self.assertEqual(parse_question("show me track #7").track_id, 7)

    def test_gibberish_is_unknown(self):
        for text in ["", "   ", "asdfghjkl", "what is the meaning of life?"]:
            with self.subTest(text=text):
                self.assertEqual(parse_question(text).intent, INTENT_UNKNOWN)


class TestDemoPayload(unittest.TestCase):
    """The committed simulated shift - the demo path."""

    @classmethod
    def setUpClass(cls):
        if not DEMO_EVENTS.exists():
            raise unittest.SkipTest(f"{DEMO_EVENTS} not present")
        cls.bot = SafetyAssistant.from_file(DEMO_EVENTS, use_llm=False)

    def test_loads_and_is_reliable(self):
        self.assertEqual(self.bot.context.state, STATE_OK)
        self.assertTrue(self.bot.context.is_reliable)
        self.assertEqual(self.bot.context.total_events, 5)

    def test_highest_risk_names_the_real_event(self):
        answer = self.bot.ask("Which event had the highest risk score?")
        worst = self.bot.context.highest_risk_event()
        self.assertIn(str(worst["event_id"]), answer.text)
        self.assertIn(str(int(worst["risk_score"])), answer.text)

    def test_event_types_match_the_summary(self):
        answer = self.bot.ask("What types of unsafe behavior occurred?")
        for event_type in self.bot.context.events_by_type:
            label = event_type.replace("_", " ").split()[0].title()
            self.assertIn(label, answer.text)

    def test_count_matches_the_data(self):
        answer = self.bot.ask("How many critical incidents occurred?")
        expected = len(self.bot.context.events_of_severity("Critical"))
        self.assertTrue(answer.text.startswith(str(expected)))

    def test_time_query_returns_only_nearby_events(self):
        answer = self.bot.ask("What happened around 8 seconds?")
        self.assertTrue(answer.supporting_event_ids)
        for event_id in answer.supporting_event_ids:
            event = next(
                e for e in self.bot.context.events if e["event_id"] == event_id
            )
            self.assertLessEqual(abs(float(event["start_time"]) - 8.0), 6.0)

    def test_every_answer_cites_real_event_ids(self):
        """No answer may reference an event id that is not in the payload."""
        real_ids = {e["event_id"] for e in self.bot.context.events}
        for question in [
            "What were the most dangerous incidents?",
            "Why was the shift rated Critical?",
            "What should the supervisor focus on?",
            "List all events.",
            "Which event had the highest risk score?",
        ]:
            with self.subTest(question=question):
                answer = self.bot.ask(question)
                for event_id in answer.supporting_event_ids:
                    self.assertIn(event_id, real_ids)


class TestCriticalPayload(unittest.TestCase):
    def setUp(self):
        self.bot = SafetyAssistant.from_payload(CRITICAL_PAYLOAD, use_llm=False)

    def test_why_severity_reports_real_index_and_contributors(self):
        answer = self.bot.ask("Why was the shift rated Critical?")
        self.assertIn("78", answer.text)
        self.assertIn("Critical", answer.text)
        self.assertIn("Throw", answer.text)
        self.assertIn("EVT-0001", answer.text)

    def test_repeat_offender_identifies_the_real_track(self):
        answer = self.bot.ask("Which track had the most incidents?")
        self.assertIn("#4", answer.text)
        self.assertIn("2", answer.text)

    def test_repeat_offender_does_not_claim_people(self):
        """Track ids are objects, not workers. Saying otherwise would be a
        privacy and accuracy failure."""
        answer = self.bot.ask("Which worker had the most incidents?")
        self.assertIn("not worker identities", answer.text)

    def test_focus_derives_from_the_top_event(self):
        answer = self.bot.ask("What should the supervisor focus on?")
        self.assertIn("EVT-0001", answer.text)
        self.assertEqual(answer.supporting_event_ids, ["EVT-0001"])

    def test_focus_flags_worker_proximity_only_when_recorded(self):
        answer = self.bot.ask("What should the supervisor focus on?")
        self.assertIn("within reach", answer.text)

    def test_count_by_type(self):
        answer = self.bot.ask("How many drop events were there?")
        self.assertTrue(answer.text.startswith("1 drop"))

    def test_no_repeat_claim_when_every_track_has_one_event(self):
        single = payload(
            events=[
                make_event(event_id="EVT-0001", track_id=1),
                make_event(event_id="EVT-0002", track_id=2),
            ],
            total_events=2,
        )
        bot = SafetyAssistant.from_payload(single, use_llm=False)
        answer = bot.ask("Which track had the most incidents?")
        self.assertIn("No load was involved in more than one incident", answer.text)


class TestCleanPayload(unittest.TestCase):
    """Zero events WITH good data - a genuine all-clear."""

    def setUp(self):
        self.bot = SafetyAssistant.from_payload(CLEAN_PAYLOAD, use_llm=False)

    def test_state_is_clean_and_reliable(self):
        self.assertEqual(self.bot.context.state, STATE_CLEAN)
        self.assertTrue(self.bot.context.is_reliable)

    def test_says_genuine_zero_incident(self):
        answer = self.bot.ask("What were the most dangerous incidents?")
        self.assertIn("No unsafe handling was detected", answer.text)
        self.assertIn("genuine zero-incident", answer.text)

    def test_count_is_zero_not_a_warning(self):
        answer = self.bot.ask("How many critical incidents occurred?")
        self.assertTrue(answer.text.startswith("0 Critical"))
        self.assertNotIn("WARNING", answer.text)

    def test_is_safe_question_answers_from_data(self):
        answer = self.bot.ask("Was the shift safe?")
        self.assertTrue(answer.reliable)
        self.assertIn("No unsafe handling was detected", answer.text)


class TestBlockedPayload(unittest.TestCase):
    """The safety-critical path: zero events because nothing was measurable."""

    def setUp(self):
        self.bot = SafetyAssistant.from_payload(BLOCKED_PAYLOAD, use_llm=False)

    def test_state_is_blocked_and_unreliable(self):
        self.assertEqual(self.bot.context.state, STATE_BLOCKED)
        self.assertFalse(self.bot.context.is_reliable)

    def test_never_claims_the_shift_was_safe(self):
        """The core guarantee. No question may yield a reassuring answer."""
        for question in [
            "What were the most dangerous incidents?",
            "How many critical incidents occurred?",
            "Was the shift safe?",
            "What types of unsafe behavior occurred?",
            "What should the supervisor focus on?",
            "Give me a summary",
            "List all events.",
            "What is the meaning of life?",
        ]:
            with self.subTest(question=question):
                answer = self.bot.ask(question)
                self.assertFalse(answer.reliable)
                self.assertIn("NOT evidence that the shift was safe", answer.text)
                self.assertNotIn("genuine zero-incident", answer.text)
                self.assertNotIn("No unsafe handling was detected", answer.text)

    def test_explains_why_and_quotes_real_track_counts(self):
        answer = self.bot.ask("Can I trust this analysis?")
        self.assertIn("1 of 32", answer.text)
        self.assertIn("46", answer.text)
        self.assertIn("UNKNOWN", answer.text)

    def test_blocked_is_distinguishable_from_clean(self):
        blocked = self.bot.ask("How many critical incidents occurred?").text
        clean = SafetyAssistant.from_payload(
            CLEAN_PAYLOAD, use_llm=False
        ).ask("How many critical incidents occurred?").text
        self.assertNotEqual(blocked, clean)


class TestMalformedPayloads(unittest.TestCase):
    def test_non_dict_is_rejected(self):
        for bad in [[], "text", 42, None]:
            with self.subTest(payload=bad):
                with self.assertRaises(InvalidEventsPayload):
                    ShiftContext.from_payload(bad)

    def test_missing_summary_is_rejected(self):
        """A truncated file must not be readable as a clean shift."""
        with self.assertRaises(InvalidEventsPayload):
            ShiftContext.from_payload({"events": []})

    def test_events_must_be_a_list(self):
        with self.assertRaises(InvalidEventsPayload):
            ShiftContext.from_payload({"summary": {}, "events": "nope"})

    def test_missing_file(self):
        with self.assertRaises(InvalidEventsPayload):
            ShiftContext.from_file("data/logs/does_not_exist.json")

    def test_invalid_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(InvalidEventsPayload):
                ShiftContext.from_file(path)

    def test_answer_question_helper_degrades_gracefully(self):
        answer = answer_question("data/logs/nope.json", "how many events?")
        self.assertEqual(answer.state, STATE_UNAVAILABLE)
        self.assertFalse(answer.reliable)
        self.assertIn("No conclusions can be drawn", answer.text)

    def test_empty_context_is_unavailable(self):
        bot = SafetyAssistant(ShiftContext.empty(), use_llm=False)
        answer = bot.ask("How many events?")
        self.assertEqual(bot.context.state, STATE_UNAVAILABLE)
        self.assertIn("run_analysis.py", answer.text)

    def test_events_with_missing_fields_do_not_crash(self):
        sparse = payload(events=[{"event_id": "EVT-0001"}, {}], total_events=2)
        bot = SafetyAssistant.from_payload(sparse, use_llm=False)
        for question in [
            "What were the most dangerous incidents?",
            "Which event had the highest risk score?",
            "What should the supervisor focus on?",
            "List all events.",
            "Which track had the most incidents?",
        ]:
            with self.subTest(question=question):
                self.assertTrue(bot.ask(question).text)


class TestNoHallucination(unittest.TestCase):
    """Guarantees that unsupported facts never appear in an answer."""

    def test_unknown_question_offers_help_instead_of_guessing(self):
        bot = SafetyAssistant.from_payload(CRITICAL_PAYLOAD, use_llm=False)
        answer = bot.ask("Which forklift driver was responsible?")
        self.assertEqual(answer.intent, INTENT_UNKNOWN)
        self.assertIn("can't answer that", answer.text)

    def test_absent_event_type_is_reported_as_absent(self):
        bot = SafetyAssistant.from_payload(CRITICAL_PAYLOAD, use_llm=False)
        answer = bot.ask("How many rough handling events were there?")
        self.assertTrue(answer.text.startswith("0 rough handling"))

    def test_unknown_track_is_not_invented(self):
        bot = SafetyAssistant.from_payload(CRITICAL_PAYLOAD, use_llm=False)
        answer = bot.ask("Show me track #999")
        self.assertIn("no recorded incidents", answer.text)

    def test_time_outside_the_clip_is_reported_honestly(self):
        bot = SafetyAssistant.from_payload(CRITICAL_PAYLOAD, use_llm=False)
        answer = bot.ask("What happened at 500 seconds?")
        self.assertIn("No events were recorded", answer.text)

    def test_every_event_id_mentioned_exists(self):
        """Scan answer text for EVT- ids and confirm each one is real."""
        import re
        bot = SafetyAssistant.from_payload(CRITICAL_PAYLOAD, use_llm=False)
        real = {e["event_id"] for e in CRITICAL_PAYLOAD["events"]}
        for question in [
            "What were the most dangerous incidents?",
            "Why was the shift rated Critical?",
            "What should the supervisor focus on?",
            "List all events.",
            "How many critical incidents occurred?",
            "Which track had the most incidents?",
        ]:
            with self.subTest(question=question):
                text = bot.ask(question).text
                for event_id in re.findall(r"EVT-\d+", text):
                    self.assertIn(event_id, real)

    def test_no_severity_rating_is_reported_when_absent(self):
        bare = {"summary": {"total_events": 1}, "events": [make_event()]}
        bot = SafetyAssistant.from_payload(bare, use_llm=False)
        answer = bot.ask("Why was the shift rated Critical?")
        self.assertIn("no recorded severity rating", answer.text)


class TestLLMLayerIsOptional(unittest.TestCase):
    def test_disabled_without_an_api_key(self):
        config = LLMConfig.from_env({})
        self.assertFalse(config.enabled)
        self.assertIn("disabled", config.describe())

    def test_never_calls_out_without_a_key(self):
        client = LLMClient(LLMConfig.from_env({}))
        self.assertIsNone(client.rephrase("q", {"a": 1}, "draft"))
        self.assertEqual(client.last_error, "disabled")

    def test_answers_are_identical_with_llm_requested_but_unconfigured(self):
        with_llm = SafetyAssistant(
            ShiftContext.from_payload(CRITICAL_PAYLOAD),
            llm=LLMClient(LLMConfig.from_env({})), use_llm=True,
        )
        without = SafetyAssistant.from_payload(CRITICAL_PAYLOAD, use_llm=False)
        question = "Why was the shift rated Critical?"
        self.assertEqual(with_llm.ask(question).text, without.ask(question).text)
        self.assertFalse(with_llm.ask(question).used_llm)

    def test_config_reads_environment_without_hardcoding(self):
        config = LLMConfig.from_env({
            "WAREGUARD_LLM_API_KEY": "test-key",
            "WAREGUARD_LLM_MODEL": "some-model",
            "WAREGUARD_LLM_PROVIDER": "openai",
        })
        self.assertTrue(config.enabled)
        self.assertEqual(config.model, "some-model")
        self.assertIn("openai", config.url)

    def test_ungrounded_numbers_are_rejected(self):
        """The anti-hallucination check on any LLM rephrasing."""
        grounding = "Shift risk index: 78/100. 3 events."
        self.assertTrue(numbers_are_grounded("Risk was 78 across 3 events.", grounding))
        self.assertFalse(
            numbers_are_grounded("Risk was 92 across 7 events.", grounding),
            "a fabricated score must be rejected",
        )

    def test_rephrasing_with_invented_numbers_is_discarded(self):
        class FabricatingClient(LLMClient):
            def _post(self, user_prompt):
                return "The shift scored 999 with 42 critical incidents."

        client = FabricatingClient(LLMConfig(api_key="test-key"))
        result = client.rephrase("q", {"shift_risk_index": 78}, "Shift risk index: 78/100.")
        self.assertIsNone(result)
        self.assertIn("ungrounded", client.last_error or "")

    def test_faithful_rephrasing_is_accepted(self):
        class FaithfulClient(LLMClient):
            def _post(self, user_prompt):
                return "The shift scored 78 out of 100."

        client = FaithfulClient(LLMConfig(api_key="test-key"))
        result = client.rephrase("q", {"shift_risk_index": 78}, "Shift risk index: 78/100.")
        self.assertEqual(result, "The shift scored 78 out of 100.")

    def test_network_failure_falls_back_silently(self):
        class BrokenClient(LLMClient):
            def _post(self, user_prompt):
                raise OSError("network down")

        bot = SafetyAssistant(
            ShiftContext.from_payload(CRITICAL_PAYLOAD),
            llm=BrokenClient(LLMConfig(api_key="test-key")), use_llm=True,
        )
        answer = bot.ask("Why was the shift rated Critical?")
        self.assertFalse(answer.used_llm)
        self.assertIn("78", answer.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
