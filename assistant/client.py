"""
WareGuard AI - LLM Client (Phase 5)

Talks to an OpenAI-compatible chat completions endpoint (OpenAI, Groq,
together.ai, a local Ollama/vLLM server, ...) over plain HTTP so the assistant
never needs a vendor SDK - just `requests`.

`requests` is imported lazily, inside complete(), not at module load time.
behavior/ and risk/ both guarantee they run with nothing but the standard
library installed; this package extends that guarantee to everything except
the one call that actually needs the network, so `import assistant` and the
heuristic responder keep working on a machine that never ran
`pip install -r requirements.txt`.
"""
from __future__ import annotations

import os
from typing import Optional

try:  # optional - keeps `import assistant` working with nothing installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT_S = 30.0


class LLMClient:
    """Minimal OpenAI-compatible chat client.

    Configuration is read from the environment so the same code works against
    a hosted API key or a local, keyless server:
        WAREGUARD_LLM_API_KEY / OPENAI_API_KEY  - bearer token (optional for
                                                   local servers)
        WAREGUARD_LLM_BASE_URL                  - default api.openai.com/v1
        WAREGUARD_LLM_MODEL                     - default gpt-4o-mini
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ):
        self.api_key = (
            api_key
            or os.environ.get("WAREGUARD_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self.base_url = (
            base_url or os.environ.get("WAREGUARD_LLM_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = model or os.environ.get("WAREGUARD_LLM_MODEL") or DEFAULT_MODEL
        self.timeout = timeout

    @property
    def available(self) -> bool:
        """False only when there's no key AND the endpoint is still the hosted
        default - a local server (Ollama, vLLM) is "available" with no key."""
        return bool(self.api_key) or self.base_url != DEFAULT_BASE_URL

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        import requests  # deferred - see module docstring

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
