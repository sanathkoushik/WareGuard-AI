"""
WareGuard AI - Optional LLM Phrasing Layer (Phase 5)

Entirely optional. The assistant answers every supported question without it,
offline and deterministically. This module only makes those answers read more
naturally when an API key happens to be configured.

The safety design is the important part:

**The model is never a source of facts.** It receives the fact sheet and the
already-derived deterministic answer, and is asked to rephrase. It cannot look
anything up, cannot reach the events file, and cannot add an incident that the
data does not contain.

**Its output is verified before use.** Every number in the rephrased text must
already appear in the grounding material. If the model introduces a figure of
its own, the response is discarded and the deterministic answer is returned
instead. A hallucinated risk score in a safety tool is worse than plain prose.

**It fails closed.** Missing key, network error, timeout, bad status, unexpected
response shape, ungrounded numbers - every path returns None and the caller
falls back to the deterministic answer.

No third-party packages: the HTTP call uses `urllib.request` from the standard
library, so Phase 5 adds nothing to requirements.txt.

Configuration (all via environment, never hard-coded):
    WAREGUARD_LLM_API_KEY    enables the layer when set
    WAREGUARD_LLM_PROVIDER   "anthropic" (default) or "openai"
    WAREGUARD_LLM_MODEL      defaults to claude-sonnet-5
    WAREGUARD_LLM_URL        override the endpoint
    WAREGUARD_LLM_TIMEOUT    seconds, default 12
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

ENV_API_KEY = "WAREGUARD_LLM_API_KEY"
ENV_PROVIDER = "WAREGUARD_LLM_PROVIDER"
ENV_MODEL = "WAREGUARD_LLM_MODEL"
ENV_URL = "WAREGUARD_LLM_URL"
ENV_TIMEOUT = "WAREGUARD_LLM_TIMEOUT"

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"

_DEFAULT_MODEL = "claude-sonnet-5"
_DEFAULT_URLS = {
    PROVIDER_ANTHROPIC: "https://api.anthropic.com/v1/messages",
    PROVIDER_OPENAI: "https://api.openai.com/v1/chat/completions",
}
_ANTHROPIC_VERSION = "2023-06-01"
_MAX_TOKENS = 400

SYSTEM_PROMPT = (
    "You rephrase warehouse safety findings for a shift supervisor.\n"
    "\n"
    "You will be given a JSON fact sheet and a draft answer that was computed "
    "directly from analysed video data. Rewrite the draft so it reads clearly "
    "and concisely.\n"
    "\n"
    "Absolute rules:\n"
    "- Use ONLY facts present in the fact sheet or the draft answer.\n"
    "- Never invent incidents, scores, times, track IDs, workers or counts.\n"
    "- Never change a number. Copy every figure exactly as given.\n"
    "- If the draft says data is unreliable or unavailable, preserve that "
    "warning prominently. Never imply a shift was safe when the draft does not.\n"
    "- Do not add recommendations that the draft does not support.\n"
    "- Keep it under 120 words. Plain text, no markdown headers."
)


@dataclass
class LLMConfig:
    """Resolved configuration. `enabled` is false when no key is present."""

    api_key: Optional[str] = None
    provider: str = PROVIDER_ANTHROPIC
    model: str = _DEFAULT_MODEL
    url: str = _DEFAULT_URLS[PROVIDER_ANTHROPIC]
    timeout: float = 12.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "LLMConfig":
        source = env if env is not None else os.environ

        provider = (source.get(ENV_PROVIDER) or PROVIDER_ANTHROPIC).strip().lower()
        if provider not in _DEFAULT_URLS:
            provider = PROVIDER_ANTHROPIC

        try:
            timeout = float(source.get(ENV_TIMEOUT) or 12.0)
        except (TypeError, ValueError):
            timeout = 12.0

        api_key = (source.get(ENV_API_KEY) or "").strip() or None

        return cls(
            api_key=api_key,
            provider=provider,
            model=(source.get(ENV_MODEL) or _DEFAULT_MODEL).strip(),
            url=(source.get(ENV_URL) or _DEFAULT_URLS[provider]).strip(),
            timeout=timeout,
        )

    def describe(self) -> str:
        if not self.enabled:
            return (
                "LLM phrasing disabled (no "
                f"{ENV_API_KEY} set). Answers are deterministic."
            )
        return f"LLM phrasing enabled via {self.provider} ({self.model})."


def _number_tokens(text: str) -> set:
    """Every numeric literal in a string, normalised.

    Trailing zeros are stripped so "74" and "74.0" compare equal - the draft may
    say one and a natural rephrasing the other, and that is not a fabrication.
    """
    tokens = set()
    for raw in re.findall(r"\d+(?:\.\d+)?", text):
        try:
            value = float(raw)
        except ValueError:
            continue
        tokens.add(f"{value:.4f}".rstrip("0").rstrip("."))
    return tokens


def numbers_are_grounded(candidate: str, grounding: str) -> bool:
    """True when every number in `candidate` also appears in `grounding`.

    The check that stops a rephrasing from quietly inventing a risk score or an
    incident count. Cheap, and it catches the failure mode that matters.
    """
    return _number_tokens(candidate).issubset(_number_tokens(grounding))


class LLMClient:
    """Thin, dependency-free client. Any failure yields None."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self.last_error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def rephrase(
        self, question: str, facts: Dict[str, Any], draft_answer: str
    ) -> Optional[str]:
        """Rewrite `draft_answer`, or return None to keep the deterministic text."""
        self.last_error = None
        if not self.config.enabled:
            self.last_error = "disabled"
            return None
        if not draft_answer.strip():
            self.last_error = "empty draft"
            return None

        try:
            facts_json = json.dumps(facts, indent=2, default=str)
        except (TypeError, ValueError):
            self.last_error = "facts not serialisable"
            return None

        user_prompt = (
            f"Question from the supervisor:\n{question}\n\n"
            f"Fact sheet (the only permitted source of facts):\n{facts_json}\n\n"
            f"Draft answer to rephrase:\n{draft_answer}"
        )

        try:
            text = self._post(user_prompt)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
            self.last_error = f"request failed: {exc}"
            return None
        except (ValueError, KeyError, TypeError) as exc:
            self.last_error = f"unexpected response: {exc}"
            return None

        if not text or not text.strip():
            self.last_error = "empty response"
            return None

        text = text.strip()
        grounding = f"{facts_json}\n{draft_answer}"
        if not numbers_are_grounded(text, grounding):
            # The model produced a figure that is not in the data. Discard it.
            self.last_error = "rejected: response contained ungrounded numbers"
            return None

        return text

    # ------------------------------------------------------------- transport

    def _post(self, user_prompt: str) -> Optional[str]:
        cfg = self.config
        if cfg.provider == PROVIDER_OPENAI:
            payload = {
                "model": cfg.model,
                "max_tokens": _MAX_TOKENS,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.api_key}",
            }
        else:
            payload = {
                "model": cfg.model,
                "max_tokens": _MAX_TOKENS,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            headers = {
                "Content-Type": "application/json",
                "x-api-key": cfg.api_key or "",
                "anthropic-version": _ANTHROPIC_VERSION,
            }

        request = urllib.request.Request(
            cfg.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=cfg.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))

        return self._extract_text(body)

    @staticmethod
    def _extract_text(body: Dict[str, Any]) -> Optional[str]:
        # Anthropic: {"content": [{"type": "text", "text": "..."}]}
        content = body.get("content")
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            joined = "".join(parts).strip()
            if joined:
                return joined

        # OpenAI: {"choices": [{"message": {"content": "..."}}]}
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict):
                text = message.get("content")
                if isinstance(text, str) and text.strip():
                    return text.strip()

        return None
