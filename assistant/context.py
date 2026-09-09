"""
WareGuard AI - Assistant Context Builder (Phase 5)

Turns a saved events log (written by risk.export.export_events_json) into the
same compact, token-cheap shape risk.export.assessment_to_assistant_context
produces from a live RiskAssessment - so the assistant can answer questions
about a shift without re-running detection or the behavior/risk engines.

This mirrors run_analysis.py's decoupling: the events log is the stable
interface, and re-reading it is cheap and re-runnable at zero cost.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from behavior.detectors import RECOMMENDED_ACTIONS
from risk.engine import ShiftSummary


def _event_to_context(e: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": e.get("event_id"),
        "type": e.get("event_type"),
        "severity": e.get("severity"),
        "risk_score": e.get("risk_score"),
        "confidence": round(e.get("confidence", 0.0), 2),
        "at": f"{e.get('start_time', 0.0):.1f}s-{e.get('end_time', 0.0):.1f}s",
        "track_id": e.get("track_id"),
        "what_happened": e.get("description", ""),
        "why_this_score": e.get("risk_factors", []),
        "recommended_action": RECOMMENDED_ACTIONS.get(e.get("event_type")),
    }


def build_context(data: Dict[str, Any]) -> Dict[str, Any]:
    """Reshape a raw exported-events payload (the dict risk.export.export_events_json
    writes to disk) into the assistant's compact context - same shape as
    risk.export.assessment_to_assistant_context, without needing a live
    RiskAssessment object in memory.
    """
    summary = ShiftSummary(**data["summary"])

    # Ranked by priority (risk * confidence), same order
    # assessment.ranked() gives the live path - highest-priority event first.
    events: List[Dict[str, Any]] = sorted(
        data.get("events", []),
        key=lambda e: (
            -(e.get("metrics", {}).get("priority_score", 0.0)),
            e.get("start_time", 0.0),
        ),
    )

    return {
        "shift": {
            "headline": summary.headline(),
            "risk_index": round(summary.shift_risk_index, 1),
            "severity": summary.shift_severity,
            "duration_minutes": round(summary.duration_seconds / 60.0, 2),
            "total_events": summary.total_events,
            "by_type": {k: v for k, v in summary.events_by_type.items() if v},
            "by_severity": {k: v for k, v in summary.events_by_severity.items() if v},
            "events_per_minute": round(summary.events_per_minute, 2),
            "rate_is_reliable": summary.rate_is_reliable,
            "repeat_offender_tracks": summary.repeat_offender_tracks,
            "data_quality_warning": summary.data_quality_warning,
        },
        "events": [_event_to_context(e) for e in events],
    }


def load_context_from_events_log(path: Path | str) -> Dict[str, Any]:
    """Load an events_*.json log written by risk.export.export_events_json and
    build the assistant's compact context from it."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return build_context(data)
