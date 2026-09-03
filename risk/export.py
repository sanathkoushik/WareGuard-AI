"""
WareGuard AI - Event Export (Phase 3)

Writes scored events to JSON and CSV for the dashboard, the LLM assistant, and
any downstream WMS integration.

Kept separate from `utils/log_exporter.py` (which serialises raw detections) so
Phase 2/3 output has its own stable schema and the two do not have to change
together.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from .engine import RiskAssessment

EVENT_CSV_FIELDS = [
    "event_id",
    "event_type",
    "severity",
    "risk_score",
    "priority_score",
    "confidence",
    "track_id",
    "class_name",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "duration_seconds",
    "related_track_ids",
    "risk_factors",
    "description",
]


def export_events_json(assessment: RiskAssessment, output_path: Path | str) -> str:
    """Full fidelity: summary, events, metrics and per-event risk factors."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(assessment.to_dict(), f, indent=2)
    return str(output_path)


def export_events_csv(assessment: RiskAssessment, output_path: Path | str) -> str:
    """Flat view for pandas, Excel and quick eyeballing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_CSV_FIELDS)
        writer.writeheader()
        for e in assessment.ranked():
            writer.writerow({
                "event_id": e.event_id,
                "event_type": e.event_type,
                "severity": e.severity or "",
                "risk_score": e.risk_score if e.risk_score is not None else "",
                "priority_score": e.metrics.get("priority_score", ""),
                "confidence": round(e.confidence, 3),
                "track_id": e.track_id,
                "class_name": e.class_name,
                "start_frame": e.start_frame,
                "end_frame": e.end_frame,
                "start_time": round(e.start_time, 3),
                "end_time": round(e.end_time, 3),
                "duration_seconds": round(e.duration_seconds, 3),
                "related_track_ids": "|".join(str(t) for t in e.related_track_ids),
                "risk_factors": "; ".join(e.risk_factors),
                "description": e.description,
            })
    return str(output_path)


def assessment_to_assistant_context(assessment: RiskAssessment) -> Dict[str, Any]:
    """A compact, token-cheap view for the Phase 5 LLM assistant.

    Per-frame kinematics and threshold dumps are stripped: an assistant
    answering "how many drops were there and were any near a worker?" needs the
    events and their reasons, not the raw signal that produced them.
    """
    s = assessment.summary
    return {
        "shift": {
            "headline": s.headline(),
            "risk_index": round(s.shift_risk_index, 1),
            "severity": s.shift_severity,
            "duration_minutes": round(s.duration_seconds / 60.0, 2),
            "total_events": s.total_events,
            "by_type": {k: v for k, v in s.events_by_type.items() if v},
            "by_severity": {k: v for k, v in s.events_by_severity.items() if v},
            "events_per_minute": round(s.events_per_minute, 2),
            "rate_is_reliable": s.rate_is_reliable,
            "repeat_offender_tracks": s.repeat_offender_tracks,
            "data_quality_warning": s.data_quality_warning,
        },
        "events": [
            {
                "id": e.event_id,
                "type": e.event_type,
                "severity": e.severity,
                "risk_score": e.risk_score,
                "confidence": round(e.confidence, 2),
                "at": f"{e.start_time:.1f}s-{e.end_time:.1f}s",
                "track_id": e.track_id,
                "what_happened": e.description,
                "why_this_score": e.risk_factors,
            }
            for e in assessment.ranked()
        ],
    }
