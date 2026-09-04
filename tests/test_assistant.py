"""
WareGuard AI - Assistant Tests (Phase 5)

    python -m unittest tests.test_assistant -v
"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant import LLMClient, WarehouseAssistant, load_context_from_events_log
from assistant.context import build_context

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "logs" / "events_sim_demo.json"


def _load_fixture_context():
    return load_context_from_events_log(FIXTURE)


class TestContextBuilding(unittest.TestCase):
    def test_loads_real_events_log(self):
        context = _load_fixture_context()
        self.assertEqual(context["shift"]["total_events"], 5)
        self.assertEqual(len(context["events"]), 5)

    def test_events_are_ranked_by_priority_not_time(self):
        context = _load_fixture_context()
        priorities = [e["risk_score"] * e["confidence"] for e in context["events"]]
        self.assertEqual(priorities, sorted(priorities, reverse=True))

    def test_event_shape_matches_assistant_context_contract(self):
        context = _load_fixture_context()
        event = context["events"][0]
        for key in ("id", "type", "severity", "risk_score", "confidence", "at",
                    "track_id", "what_happened", "why_this_score"):
            self.assertIn(key, event)

    def test_build_context_is_pure(self):
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        context = build_context(raw)
        self.assertEqual(context["shift"]["severity"], raw["summary"]["shift_severity"])


class TestLLMClientAvailability(unittest.TestCase):
    def test_unavailable_with_no_key_and_default_endpoint(self):
        client = LLMClient(api_key=None, base_url=None)
        self.assertFalse(client.available)

    def test_available_with_key(self):
        client = LLMClient(api_key="sk-test")
        self.assertTrue(client.available)

    def test_available_with_local_endpoint_and_no_key(self):
        client = LLMClient(api_key=None, base_url="http://localhost:11434/v1")
        self.assertTrue(client.available)


class TestWarehouseAssistantHeuristicFallback(unittest.TestCase):
    """No API key configured anywhere in these tests - every answer must come
    from the offline heuristic responder, never a real network call."""

    def setUp(self):
        self.context = _load_fixture_context()
        self.assistant = WarehouseAssistant(
            self.context, client=LLMClient(api_key=None, base_url=None)
        )

    def test_worst_event_question_cites_the_top_priority_event(self):
        answer = self.assistant.ask("What was the worst event this shift?")
        self.assertIn(self.context["events"][0]["id"], answer)

    def test_event_id_lookup(self):
        event_id = self.context["events"][0]["id"]
        answer = self.assistant.ask(f"why did {event_id} happen?")
        self.assertIn(event_id, answer)

    def test_repeat_offender_question(self):
        answer = self.assistant.ask("were there any repeat offenders?")
        self.assertIn("repeat", answer.lower())

    def test_unmatched_event_id_says_so(self):
        answer = self.assistant.ask("what happened in EVT-9999?")
        self.assertIn("No event", answer)

    def test_data_quality_question(self):
        answer = self.assistant.ask("is this data reliable?")
        self.assertTrue("quality" in answer.lower())


class TestWarehouseAssistantLLMFailureDegradesGracefully(unittest.TestCase):
    def test_missing_requests_dependency_falls_back_to_heuristic(self):
        context = _load_fixture_context()
        client = LLMClient(api_key="sk-test")
        assistant = WarehouseAssistant(context, client=client)
        # requests is not installed in this environment, so complete() raises
        # ImportError - the assistant must still answer rather than crash.
        answer = assistant.ask("What was the worst event?")
        self.assertIn(context["events"][0]["id"], answer)

    def test_request_failure_falls_back_to_heuristic(self):
        context = _load_fixture_context()
        client = LLMClient(api_key="sk-test")
        assistant = WarehouseAssistant(context, client=client)

        def _boom(system_prompt, user_prompt):
            raise RuntimeError("network unreachable")

        with patch.object(client, "complete", side_effect=_boom):
            answer = assistant.ask("What was the worst event?")
        self.assertIn(context["events"][0]["id"], answer)


class TestLLMClientRequestShape(unittest.TestCase):
    def test_complete_posts_expected_payload(self):
        client = LLMClient(api_key="sk-test", base_url="https://example.test/v1", model="test-model")

        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "  an answer  "}}]}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse()

        fake_requests = type("FakeRequestsModule", (), {"post": staticmethod(fake_post)})

        with patch.dict(sys.modules, {"requests": fake_requests}):
            result = client.complete("system", "user question")

        self.assertEqual(result, "an answer")
        self.assertEqual(captured["url"], "https://example.test/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(captured["json"]["model"], "test-model")


if __name__ == "__main__":
    unittest.main()
