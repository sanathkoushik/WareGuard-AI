"""
WareGuard AI - Supervisor Q&A Interface (Phase 5)

Answers natural-language questions about one shift's events using the compact
context from risk.export.assessment_to_assistant_context (or
assistant.context.load_context_from_events_log). Prefers an LLM when one is
configured; always falls back to the heuristic responder so the assistant
works with nothing installed and nothing configured.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from . import heuristic
from .client import LLMClient

SYSTEM_PROMPT = (
    "You are WareGuard AI's warehouse safety assistant. You answer a "
    "supervisor's questions about ONE shift of loading/unloading footage "
    "using ONLY the JSON context provided below - never invent an event, "
    "track ID, or score that is not in it. Cite event IDs (e.g. EVT-0003) "
    "when referring to a specific incident. Keep answers to a few sentences; "
    "a supervisor is reading this on a shift-floor tablet, not a report."
)


class WarehouseAssistant:
    """Q&A over one shift's scored events."""

    def __init__(self, context: Dict[str, Any], client: Optional[LLMClient] = None):
        self.context = context
        self.client = client if client is not None else LLMClient()

    def ask(self, question: str) -> str:
        if not self.client.available:
            return heuristic.answer(self.context, question)

        user_prompt = (
            f"Shift context:\n{json.dumps(self.context, indent=2)}\n\n"
            f"Question: {question}"
        )
        try:
            return self.client.complete(SYSTEM_PROMPT, user_prompt)
        except Exception:
            # Covers a missing `requests` install, network/API failures, and
            # malformed responses alike - none of them may take a
            # supervisor-facing assistant fully offline. Degrade to the
            # heuristic responder instead of raising into the caller.
            return heuristic.answer(self.context, question)
