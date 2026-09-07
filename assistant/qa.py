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
import re
from typing import Any, Dict, Optional

from . import heuristic
from .client import LLMClient


def _number_tokens(text: str) -> set:
    """Every numeric literal in a string, normalised.

    Trailing zeros are stripped so "74" and "74.0" compare equal - a natural
    rephrasing may write either, and that is not a fabrication.
    """
    tokens = set()
    for raw in re.findall(r"\d+(?:\.\d+)?", text):
        try:
            value = float(raw)
        except ValueError:
            continue
        tokens.add(f"{value:.4f}".rstrip("0").rstrip("."))
    return tokens


def is_grounded(candidate: str, context_json: str) -> bool:
    """True when every number in `candidate` also appears in the context.

    The system prompt tells the model not to invent scores, track IDs or
    timestamps, but a prompt is a request rather than a guarantee. This is the
    check that makes it one for the class of error that matters most here: a
    fabricated risk score or incident count read off a safety dashboard is
    worse than a plainly-worded answer.

    Deliberately narrow. It cannot catch an invented *sentence*, only invented
    *figures* - which is where the damage is concentrated, and which is cheap
    and reliable to verify.
    """
    return _number_tokens(candidate).issubset(_number_tokens(context_json))

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

        context_json = json.dumps(self.context, indent=2)
        user_prompt = (
            f"Shift context:\n{context_json}\n\n"
            f"Question: {question}"
        )
        try:
            reply = self.client.complete(SYSTEM_PROMPT, user_prompt)
        except Exception:
            # Covers a missing `requests` install, network/API failures, and
            # malformed responses alike - none of them may take a
            # supervisor-facing assistant fully offline. Degrade to the
            # heuristic responder instead of raising into the caller.
            return heuristic.answer(self.context, question)

        # Verify before trusting. The model is given the shift context and told
        # not to invent, but nothing about an LLM call enforces that. If the
        # reply contains a figure that is not in the context, it was not read
        # from this shift's data - discard it and answer from the context
        # directly rather than showing a supervisor a fabricated score.
        if not reply or not reply.strip():
            return heuristic.answer(self.context, question)
        if not is_grounded(reply, context_json):
            return heuristic.answer(self.context, question)

        return reply
