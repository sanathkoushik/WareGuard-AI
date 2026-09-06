"""
WareGuard AI - AI Safety Assistant (Phase 5)

Answers a supervisor's questions about an analysed shift using the Phase 2/3
event and risk data.

The assistant is deterministic by default: it reads `events_<stem>.json`,
classifies the question, and composes an answer from fields that are actually
present. No API key, no network and no model are required for any supported
question. An optional LLM layer can rephrase those answers more naturally, but
it is never a source of facts - see `assistant/llm.py`.

Standard library only; adds nothing to requirements.txt.

Typical use:

    from assistant import SafetyAssistant

    bot = SafetyAssistant.from_file("data/logs/events_sim_demo.json")
    print(bot.ask("Why was the shift rated Critical?"))
    print(bot.ask("What should the supervisor focus on?"))

Or from the command line:

    python -m assistant.cli --events data/logs/events_sim_demo.json
    python -m assistant.cli --events data/logs/events_sim_demo.json --ask "how many critical incidents?"
"""
from .context import (
    STATE_BLOCKED,
    STATE_CLEAN,
    STATE_OK,
    STATE_PARTIAL,
    STATE_UNAVAILABLE,
    InvalidEventsPayload,
    ShiftContext,
    load_context,
)
from .engine import SUPPORTED_QUESTIONS, Answer, SafetyAssistant, answer_question
from .intents import ALL_INTENTS, Question, parse_question
from .llm import LLMClient, LLMConfig

__all__ = [
    "SafetyAssistant",
    "Answer",
    "answer_question",
    "SUPPORTED_QUESTIONS",
    "ShiftContext",
    "load_context",
    "InvalidEventsPayload",
    "parse_question",
    "Question",
    "ALL_INTENTS",
    "LLMClient",
    "LLMConfig",
    "STATE_OK",
    "STATE_CLEAN",
    "STATE_PARTIAL",
    "STATE_BLOCKED",
    "STATE_UNAVAILABLE",
]
