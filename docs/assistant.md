# Phase 5 — AI Safety Assistant

Answers a supervisor's questions about an analysed shift using the Phase 2/3
event and risk data.

## What it is

A **deterministic** question-answering layer over `events_<stem>.json`. It reads
the payload, classifies the question, and composes an answer from fields that
are actually present.

**No API key, no network and no model are required** for any supported question.
An optional LLM layer can rephrase answers more naturally, but it is never a
source of facts.

**Zero new dependencies.** Standard library only — the optional HTTP call uses
`urllib.request`. Nothing was added to `requirements.txt`. (`requests` is listed
there for Phase 5, but is not used.)

## Quick use

```bash
# interactive
python -m assistant.cli --events data/logs/events_sim_demo.json

# one-shot
python -m assistant.cli --events data/logs/events_sim_demo.json --ask "why was the shift rated critical?"

# the full supervisor question set, for a live walkthrough
python -m assistant.cli --events data/logs/events_sim_demo.json --demo
```

```python
from assistant import SafetyAssistant

bot = SafetyAssistant.from_file("data/logs/events_sim_demo.json")
answer = bot.ask("What should the supervisor focus on?")

answer.text                   # the response
answer.intent                 # which question type was matched
answer.reliable               # False when data quality blocks conclusions
answer.state                  # ok | clean | partial | blocked | unavailable
answer.supporting_event_ids   # event ids the answer is grounded in
answer.used_llm               # whether the optional rephrasing applied
```

## Questions it answers

| Question | Intent |
|---|---|
| "What were the most dangerous incidents?" | `most_dangerous` |
| "Which event had the highest risk score?" | `highest_risk` |
| "Why was the shift rated Critical?" | `why_severity` |
| "Which track had the most incidents?" | `repeat_offender` |
| "What types of unsafe behavior occurred?" | `event_types` |
| "How many critical incidents occurred?" | `count` |
| "What happened around 8 seconds?" | `time_query` |
| "What should the supervisor focus on?" | `focus` |
| "Can I trust this analysis?" | `data_quality` |
| "List all events." | `list_events` |
| "Was the shift safe?" / "Give me a summary" | `summary` |

Anything else returns `unknown` and says so. That is a designed outcome, not a
failure — an honest "I can't answer that from this data" beats a confident
guess.

## The three guarantees

**1. Nothing is invented.** Every figure is read from the payload. Accessors
return `None` when a field is absent rather than substituting a plausible value,
so "the data says zero" and "the data does not say" stay distinguishable. Tests
scan answer text for `EVT-` ids and assert every one exists in the payload.

**2. Unreliable data is never dressed up as safety.** When the analysis is
`blocked` or `unavailable`, *every* answer — including unrecognised questions —
leads with:

> WARNING: Reliable conclusions cannot be drawn from this footage — the tracking
> data was insufficient for behavior analysis. This is NOT evidence that the
> shift was safe.

`test_never_claims_the_shift_was_safe` asserts this across eight different
questions, including "Was the shift safe?".

**3. Recommendations are derived, not imagined.** The `focus` answer names the
actual highest-priority event and its actual scoring factors. It never offers
generic warehouse advice the data does not support.

## Data-quality states

Reuses the Phase 4 classification directly — `assistant/context.py` imports
`data_quality_state` from `dashboard/events_panel.py` rather than
reimplementing it. If the assistant and the dashboard ever disagreed about
whether a shift was analysable, that would be a safety bug, so there is exactly
one implementation.

| State | Meaning | Assistant behaviour |
|---|---|---|
| `ok` | Events found, data sound | Answers normally |
| `clean` | Zero events, data sound | "Genuine zero-incident result" |
| `partial` | Events found, coverage incomplete | Answers + coverage caveat |
| `blocked` | Zero events because nothing was measurable | Refuses; explains why |
| `unavailable` | No or malformed payload | Refuses; explains how to generate one |

The `clean` / `blocked` distinction is the whole point. Both produce zero
events; only one of them means the shift was safe.

## Optional LLM layer

Disabled unless `WAREGUARD_LLM_API_KEY` is set. When enabled it **rephrases an
answer that was already derived from the data** — it cannot look anything up,
cannot reach the events file, and cannot add an incident.

| Variable | Default | Purpose |
|---|---|---|
| `WAREGUARD_LLM_API_KEY` | *(unset)* | Enables the layer. Never hard-coded. |
| `WAREGUARD_LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `WAREGUARD_LLM_MODEL` | `claude-sonnet-5` | Model id |
| `WAREGUARD_LLM_URL` | provider default | Endpoint override |
| `WAREGUARD_LLM_TIMEOUT` | `12` | Seconds |

Two safeguards:

- **Output is verified before use.** Every number in the rephrased text must
  already appear in the fact sheet or the draft. If the model introduces a
  figure of its own, the response is discarded and the deterministic answer is
  returned. A hallucinated risk score in a safety tool is worse than plain
  prose.
- **It fails closed.** Missing key, network error, timeout, bad status,
  unexpected shape, ungrounded numbers — every path returns `None` and the
  deterministic answer stands.

Disable it explicitly with `--no-llm` or `SafetyAssistant(ctx, use_llm=False)`.

## Architecture

```
assistant/
├── context.py    ShiftContext - loads, validates, queries the payload
├── intents.py    keyword-scoring question classifier + entity extraction
├── engine.py     SafetyAssistant - one handler per intent, all data-grounded
├── llm.py        optional rephrasing, verified and fail-closed
└── cli.py        interactive / one-shot / demo terminal interface
```

It **derives no risk**. Every score, severity and threshold is read from the
payload exactly as Phase 3 wrote it. If a number looks wrong, the bug is in
`risk/`, not here.

## Tests

```bash
python -m unittest tests.test_assistant -v
```

46 tests, no dependencies, no API key. Covering the demo payload, a critical
payload, a zero-event clean payload, a blocked payload, malformed payloads,
each question type, unknown questions, and the no-hallucination guarantees
(including a fabricating fake LLM whose output must be rejected).

## Dashboard integration

**Not yet wired in.** The assistant is complete and usable from the CLI and as a
library. Exposing it in the Streamlit dashboard needs a small addition to
`dashboard/` which is deliberately left for review — see the Phase 5 handover
notes.
