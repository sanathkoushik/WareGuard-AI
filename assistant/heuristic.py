"""
WareGuard AI - Heuristic Responder (Phase 5 offline fallback)

Answers the assistant's most common questions directly from the compact
context - no LLM, no network, no API key. This is what WarehouseAssistant
falls back to when no LLMClient is configured or a request fails, so the
assistant degrades to "still answers, just less fluently" rather than going
silent - the same honesty standard behavior/risk hold for a shift with
unusable tracks.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

_EVENT_ID_RE = re.compile(r"\bevt-?\d+\b", re.IGNORECASE)


def _find_event(context: Dict[str, Any], event_id: str) -> Optional[Dict[str, Any]]:
    event_id = event_id.upper()
    for e in context["events"]:
        if e["id"] and e["id"].upper() == event_id:
            return e
    return None


def _format_event(e: Dict[str, Any]) -> str:
    reasons = "; ".join(e["why_this_score"]) if e["why_this_score"] else "no listed factors"
    return (
        f"{e['id']} [{e['severity']}] {e['type']} at {e['at']} "
        f"(risk {e['risk_score']}, confidence {e['confidence']}): "
        f"{e['what_happened']} Why: {reasons}"
    )


def answer(context: Dict[str, Any], question: str) -> str:
    """Best-effort answer using pattern matching over the context dict only."""
    q = question.lower()
    shift = context["shift"]
    events = context["events"]

    match = _EVENT_ID_RE.search(q)
    if match:
        event = _find_event(context, match.group(0))
        if event:
            return _format_event(event)
        return f"No event matching '{match.group(0).upper()}' in this shift."

    if not events:
        return shift["headline"]

    if any(w in q for w in ("how many", "count", "number of")):
        for event_type, count in shift["by_type"].items():
            if event_type.replace("_", " ") in q:
                return f"{count} {event_type.replace('_', ' ')} event(s) this shift."
        return f"{shift['total_events']} event(s) this shift ({shift['headline']})"

    if any(w in q for w in ("worst", "critical", "most severe", "highest risk", "top")):
        return f"Worst event: {_format_event(events[0])}"

    if "repeat" in q or "offend" in q:
        tracks = shift["repeat_offender_tracks"]
        if not tracks:
            return "No repeat offenders this shift - every flagged load appears once."
        return f"Repeat-offender track ID(s): {', '.join(str(t) for t in tracks)}."

    if "quality" in q or "reliable" in q or "trust" in q:
        if shift["data_quality_warning"]:
            return f"Data quality warning: {shift['data_quality_warning']}"
        return "No data quality issues reported for this shift."

    if any(w in q for w in ("summary", "overview", "headline", "how was", "how did")):
        return shift["headline"]

    # Default: headline plus the top 3 events, same priority order the
    # dashboard would rank them in.
    lines = [shift["headline"]]
    for e in events[:3]:
        lines.append(f"  - {_format_event(e)}")
    return "\n".join(lines)
