"""
WareGuard AI - Assistant Context Layer (Phase 5)

Loads and validates an `events_<stem>.json` produced by Phase 2/3 and exposes
it as a queryable `ShiftContext`.

Three rules this layer exists to enforce:

**It derives no risk.** Every score, severity and threshold is read from the
payload exactly as Phase 3 wrote it. Nothing here recomputes a risk score,
re-bands a severity, or decides what is dangerous. If a number looks wrong, the
bug is in `risk/`, not here.

**It never fabricates.** Every accessor returns `None` or an empty list when the
underlying field is absent, so a caller can distinguish "the data says zero"
from "the data does not say". The assistant depends on that distinction to
avoid inventing incidents.

**It agrees with the dashboard.** The four-way data-quality classification is
imported from `dashboard.events_panel` rather than reimplemented. The assistant
and the dashboard disagreeing about whether a shift was analysable would be a
safety bug, so there is exactly one implementation of that rule.

Standard library only.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the Phase 4 payload contract. `dashboard.events_panel` guards its own
# Streamlit import, so this works with no Streamlit installed.
from dashboard.events_panel import (  # noqa: E402
    data_quality_state,
    get_events,
    get_factors,
    get_summary,
    sort_events,
)

STATE_OK = "ok"
STATE_CLEAN = "clean"
STATE_PARTIAL = "partial"
STATE_BLOCKED = "blocked"
STATE_UNAVAILABLE = "unavailable"

# States in which no claim about safety can be supported by the data.
UNRELIABLE_STATES = {STATE_BLOCKED, STATE_UNAVAILABLE}

SEVERITY_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}

EVENT_TYPE_LABEL = {
    "drop": "Drop",
    "throw": "Throw",
    "drag": "Drag",
    "improper_stack": "Improper Stack",
    "rough_handling": "Rough Handling",
}


def label_for_type(event_type: Optional[str]) -> str:
    if not event_type:
        return "Unknown"
    return EVENT_TYPE_LABEL.get(event_type, event_type.replace("_", " ").title())


class InvalidEventsPayload(ValueError):
    """Raised when a payload cannot be interpreted as a Phase 2/3 assessment."""


@dataclass
class ShiftContext:
    """A validated, queryable view of one analysed shift."""

    payload: Dict[str, Any]
    source_path: Optional[str] = None

    # ------------------------------------------------------------ construction

    @classmethod
    def from_payload(
        cls, payload: Any, source_path: Optional[str] = None
    ) -> "ShiftContext":
        """Validate a decoded payload.

        Deliberately strict about the `summary` block: a payload without one is
        not an assessment, and treating it as an empty-but-valid shift would let
        a truncated file be reported as a clean one.
        """
        if not isinstance(payload, dict):
            raise InvalidEventsPayload(
                f"Expected a JSON object at the top level, got {type(payload).__name__}."
            )
        if not isinstance(payload.get("summary"), dict):
            raise InvalidEventsPayload(
                "Payload has no 'summary' object - this is not a WareGuard events "
                "file. Generate one with: python run_analysis.py --logs <detections.json>"
            )
        events = payload.get("events")
        if events is not None and not isinstance(events, list):
            raise InvalidEventsPayload(
                f"'events' must be a list, got {type(events).__name__}."
            )
        return cls(payload=payload, source_path=source_path)

    @classmethod
    def from_file(cls, path: Path | str) -> "ShiftContext":
        p = Path(path)
        if not p.exists():
            raise InvalidEventsPayload(f"Events file not found: {p}")
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise InvalidEventsPayload(f"{p.name} is not valid JSON: {exc}") from exc
        except OSError as exc:
            raise InvalidEventsPayload(f"Could not read {p}: {exc}") from exc
        return cls.from_payload(payload, source_path=str(p))

    @classmethod
    def empty(cls) -> "ShiftContext":
        """A context representing 'no analysis available'.

        The payload is genuinely empty, with no `summary` key. An earlier
        version supplied `{"summary": {}, "events": []}`, which classified as
        `clean` - i.e. a confirmed zero-incident shift - because an empty
        summary still looks like an assessment. That is the precise conflation
        this system exists to prevent, so "nothing loaded" must classify as
        `unavailable`.
        """
        return cls(payload={}, source_path=None)

    # ----------------------------------------------------------- basic access

    @property
    def summary(self) -> Dict[str, Any]:
        return get_summary(self.payload)

    @property
    def events(self) -> List[Dict[str, Any]]:
        """All events, ranked by review priority (severity x confidence)."""
        return sort_events(get_events(self.payload))

    @property
    def scene(self) -> Dict[str, Any]:
        scene = self.payload.get("scene")
        return scene if isinstance(scene, dict) else {}

    @property
    def behavior_summary(self) -> Dict[str, Any]:
        bs = self.payload.get("behavior_summary")
        return bs if isinstance(bs, dict) else {}

    @property
    def video_name(self) -> Optional[str]:
        return self.scene.get("video_name")

    # --------------------------------------------------------- data quality

    @property
    def state(self) -> str:
        """One of: ok, clean, partial, blocked, unavailable."""
        return data_quality_state(self.payload)

    @property
    def is_reliable(self) -> bool:
        """False when no conclusion about safety can be drawn from this data."""
        return self.state not in UNRELIABLE_STATES

    @property
    def data_quality_warning(self) -> Optional[str]:
        warning = self.summary.get("data_quality_warning")
        return warning if warning else None

    @property
    def track_quality(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """(usable, cargo, total) track counts, any of which may be None."""
        bs = self.behavior_summary
        return (bs.get("usable_tracks"), bs.get("cargo_tracks"), bs.get("total_tracks"))

    # -------------------------------------------------------- shift metrics
    # All return None when absent, never a substituted zero. "We do not know"
    # and "it is zero" are different answers and the assistant says so.

    @property
    def total_events(self) -> int:
        return len(get_events(self.payload))

    @property
    def shift_risk_index(self) -> Optional[float]:
        return self._opt_float("shift_risk_index")

    @property
    def shift_severity(self) -> Optional[str]:
        value = self.summary.get("shift_severity")
        return str(value) if value else None

    @property
    def max_risk_score(self) -> Optional[float]:
        return self._opt_float("max_risk_score")

    @property
    def mean_risk_score(self) -> Optional[float]:
        return self._opt_float("mean_risk_score")

    @property
    def duration_seconds(self) -> Optional[float]:
        return self._opt_float("duration_seconds")

    @property
    def events_per_minute(self) -> Optional[float]:
        return self._opt_float("events_per_minute")

    @property
    def rate_is_reliable(self) -> bool:
        return bool(self.summary.get("rate_is_reliable", True))

    def _opt_float(self, key: str) -> Optional[float]:
        value = self.summary.get(key)
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @property
    def events_by_type(self) -> Dict[str, int]:
        """Non-zero event-type counts, highest first."""
        raw = self.summary.get("events_by_type") or {}
        if not isinstance(raw, dict):
            return {}
        counts = {k: int(v) for k, v in raw.items() if isinstance(v, (int, float)) and v}
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    @property
    def events_by_severity(self) -> Dict[str, int]:
        """Non-zero severity counts, most severe first."""
        raw = self.summary.get("events_by_severity") or {}
        if not isinstance(raw, dict):
            return {}
        counts = {k: int(v) for k, v in raw.items() if isinstance(v, (int, float)) and v}
        return dict(
            sorted(counts.items(), key=lambda kv: -SEVERITY_RANK.get(kv[0], 0))
        )

    @property
    def repeat_offender_tracks(self) -> List[int]:
        raw = self.summary.get("repeat_offender_tracks") or []
        if not isinstance(raw, list):
            return []
        out = []
        for t in raw:
            try:
                out.append(int(t))
            except (TypeError, ValueError):
                continue
        return out

    @property
    def timeline(self) -> List[Dict[str, Any]]:
        raw = self.summary.get("timeline") or []
        return [b for b in raw if isinstance(b, dict)] if isinstance(raw, list) else []

    # --------------------------------------------------------- event queries

    def highest_risk_event(self) -> Optional[Dict[str, Any]]:
        """The single most severe event by risk score (not by priority).

        Distinct from `events[0]`, which is the one to *review* first. The
        highest-scoring event may carry lower confidence.
        """
        events = get_events(self.payload)
        if not events:
            return None
        return max(events, key=lambda e: float(e.get("risk_score") or 0.0))

    def top_priority_event(self) -> Optional[Dict[str, Any]]:
        events = self.events
        return events[0] if events else None

    def events_of_type(self, event_type: str) -> List[Dict[str, Any]]:
        return [e for e in self.events if e.get("event_type") == event_type]

    def events_of_severity(self, severity: str) -> List[Dict[str, Any]]:
        target = severity.lower()
        return [e for e in self.events if str(e.get("severity", "")).lower() == target]

    def events_for_track(self, track_id: int) -> List[Dict[str, Any]]:
        return [e for e in self.events if e.get("track_id") == track_id]

    def events_near(
        self, timestamp: float, window_seconds: float = 3.0
    ) -> List[Dict[str, Any]]:
        """Events overlapping [t - window, t + window], nearest first."""
        lo, hi = timestamp - window_seconds, timestamp + window_seconds
        hits = []
        for e in self.events:
            start = e.get("start_time")
            end = e.get("end_time", start)
            try:
                start_f = float(start)
                end_f = float(end if end is not None else start)
            except (TypeError, ValueError):
                continue
            if end_f >= lo and start_f <= hi:
                hits.append((abs(start_f - timestamp), e))
        return [e for _, e in sorted(hits, key=lambda pair: pair[0])]

    def track_incident_counts(self) -> Dict[int, int]:
        """Events per track id, descending by count."""
        counts: Dict[int, int] = {}
        for e in get_events(self.payload):
            track = e.get("track_id")
            if track is None:
                continue
            try:
                key = int(track)
            except (TypeError, ValueError):
                continue
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def busiest_track(self) -> Optional[Tuple[int, int]]:
        """(track_id, incident_count) for the most-involved load, or None."""
        counts = self.track_incident_counts()
        if not counts:
            return None
        track_id, count = next(iter(counts.items()))
        return (track_id, count)

    def factors_for(self, event: Dict[str, Any]) -> List[str]:
        """Human-readable scoring factors for one event.

        Prefers the structured `risk_factors` map, falling back to the
        pre-rendered strings stored on the event itself.
        """
        event_id = str(event.get("event_id") or "")
        structured = get_factors(self.payload, event_id)
        if structured:
            out = []
            for f in structured:
                detail = f.get("detail") or f.get("code") or ""
                points = f.get("points")
                try:
                    pts = float(points)
                    sign = "+" if pts >= 0 else ""
                    out.append(f"{detail} ({sign}{pts:.0f})")
                except (TypeError, ValueError):
                    if detail:
                        out.append(str(detail))
            if out:
                return out
        fallback = event.get("risk_factors") or []
        return [str(x) for x in fallback] if isinstance(fallback, list) else []

    # ------------------------------------------------------------- rendering

    @staticmethod
    def describe_event(event: Dict[str, Any], include_id: bool = True) -> str:
        """One-line description built only from fields present on the event."""
        parts: List[str] = []
        if include_id and event.get("event_id"):
            parts.append(str(event["event_id"]))

        label = label_for_type(event.get("event_type"))
        severity = event.get("severity")
        parts.append(f"{label} ({severity})" if severity else label)

        risk = event.get("risk_score")
        if risk is not None:
            try:
                parts.append(f"risk {float(risk):.0f}/100")
            except (TypeError, ValueError):
                pass

        start = event.get("start_time")
        if start is not None:
            try:
                parts.append(f"at {float(start):.1f}s")
            except (TypeError, ValueError):
                pass

        track = event.get("track_id")
        if track is not None:
            parts.append(f"track #{track}")

        return " | ".join(parts)

    def facts(self) -> Dict[str, Any]:
        """A compact, JSON-safe fact sheet.

        Used as grounding for the optional LLM layer, which is permitted to
        rephrase these facts and nothing else.
        """
        top = self.top_priority_event()
        worst = self.highest_risk_event()
        busiest = self.busiest_track()
        return {
            "state": self.state,
            "is_reliable": self.is_reliable,
            "data_quality_warning": self.data_quality_warning,
            "video_name": self.video_name,
            "total_events": self.total_events,
            "shift_risk_index": self.shift_risk_index,
            "shift_severity": self.shift_severity,
            "max_risk_score": self.max_risk_score,
            "mean_risk_score": self.mean_risk_score,
            "duration_seconds": self.duration_seconds,
            "events_per_minute": self.events_per_minute,
            "rate_is_reliable": self.rate_is_reliable,
            "events_by_type": self.events_by_type,
            "events_by_severity": self.events_by_severity,
            "repeat_offender_tracks": self.repeat_offender_tracks,
            "highest_risk_event": self.describe_event(worst) if worst else None,
            "top_priority_event": self.describe_event(top) if top else None,
            "busiest_track": (
                {"track_id": busiest[0], "incidents": busiest[1]} if busiest else None
            ),
        }


def load_context(path: Path | str) -> ShiftContext:
    """Convenience loader. Raises InvalidEventsPayload on any problem."""
    return ShiftContext.from_file(path)
