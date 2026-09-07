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

# Canonical vocabularies for counting questions. `by_severity` and `by_type` in
# the context only carry non-zero entries, so a question about a category with
# no events finds nothing there. Without these lists the answer silently falls
# through to the shift total - reporting "5 events" for "how many critical?".
_SEVERITIES = ("Critical", "High", "Medium", "Low")

_TYPE_ALIASES = {
    "drop": "drop",
    "dropped": "drop",
    "fall": "drop",
    "throw": "throw",
    "thrown": "throw",
    "drag": "drag",
    "dragged": "drag",
    "improper stack": "improper_stack",
    "stacking": "improper_stack",
    "stack": "improper_stack",
    "rough handling": "rough_handling",
    "rough": "rough_handling",
}


def _quality_note(context: Dict[str, Any]) -> str:
    """A coverage caveat to append when the shift was only partly analysable.

    Events reported on a partial analysis are real, but their absence in any
    period is not evidence that nothing happened. Stating findings without this
    invites a supervisor to read incomplete coverage as an all-clear.
    """
    warning = context.get("shift", {}).get("data_quality_warning")
    if not warning:
        return ""
    return (
        f"\n\nNote - partial analysis: {warning} "
        "Absence of an event is not proof that nothing happened."
    )


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
        # Severity first: "how many critical incidents?" must answer with the
        # Critical count, not the shift total. `.get(..., 0)` matters because
        # by_severity omits zero entries, and "0 Low events" is the correct
        # answer rather than a fall-through to the total.
        for severity in _SEVERITIES:
            if severity.lower() in q:
                count = shift.get("by_severity", {}).get(severity, 0)
                return (
                    f"{count} {severity} event(s) this shift."
                    + _quality_note(context)
                )

        for alias, event_type in _TYPE_ALIASES.items():
            if alias in q:
                count = shift.get("by_type", {}).get(event_type, 0)
                return (
                    f"{count} {event_type.replace('_', ' ')} event(s) this shift."
                    + _quality_note(context)
                )

        return (
            f"{shift['total_events']} event(s) this shift ({shift['headline']})"
            + _quality_note(context)
        )

    if any(w in q for w in ("what type", "what kind", "types of", "kinds of",
                            "what behaviour", "what behavior", "breakdown")):
        by_type = shift.get("by_type", {})
        if not by_type:
            return "No unsafe handling types were recorded this shift."
        parts = ", ".join(
            f"{t.replace('_', ' ')} x{n}" for t, n in by_type.items()
        )
        return (
            f"Unsafe handling types this shift: {parts}."
            + _quality_note(context)
        )

    if any(w in q for w in ("worst", "critical", "most severe", "highest risk", "top")):
        return f"Worst event: {_format_event(events[0])}" + _quality_note(context)

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
        return shift["headline"] + _quality_note(context)

    # Default: headline plus the top 3 events, same priority order the
    # dashboard would rank them in.
    lines = [shift["headline"]]
    for e in events[:3]:
        lines.append(f"  - {_format_event(e)}")
    return "\n".join(lines) + _quality_note(context)
