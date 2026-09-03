"""
WareGuard AI - Risk Assessment Engine (Phase 3)

Scores every behavior event, escalates repeat offenders, and rolls the result
up into a shift-level picture a supervisor can act on.

The shift roll-up is not a sum. Ten minor drags and one carton thrown past
someone's head are not equivalent, and any metric that averages them is
actively misleading. The shift index therefore weights the *worst* event
heavily and the *rate* of events lightly, so a single critical incident cannot
be diluted by a quiet hour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from behavior.detectors import ALL_EVENT_TYPES
from behavior.engine import BehaviorReport
from behavior.schema import BehaviorEvent

from .scoring import (
    DEFAULT_WEIGHTS,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_ORDER,
    RiskFactor,
    RiskScorer,
    RiskWeights,
    severity_for,
)


@dataclass
class ShiftSummary:
    """Aggregate view of one processed clip or shift."""

    total_events: int = 0
    duration_seconds: float = 0.0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_severity: Dict[str, int] = field(default_factory=dict)

    max_risk_score: float = 0.0
    mean_risk_score: float = 0.0
    shift_risk_index: float = 0.0
    shift_severity: str = SEVERITY_LOW

    events_per_minute: float = 0.0
    rate_is_reliable: bool = True
    """False when the clip was too short for events-per-minute to mean much."""
    repeat_offender_tracks: List[int] = field(default_factory=list)
    top_events: List[str] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    data_quality_warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_events": self.total_events,
            "duration_seconds": round(self.duration_seconds, 2),
            "events_by_type": self.events_by_type,
            "events_by_severity": self.events_by_severity,
            "max_risk_score": round(self.max_risk_score, 1),
            "mean_risk_score": round(self.mean_risk_score, 1),
            "shift_risk_index": round(self.shift_risk_index, 1),
            "shift_severity": self.shift_severity,
            "events_per_minute": round(self.events_per_minute, 2),
            "rate_is_reliable": self.rate_is_reliable,
            "repeat_offender_tracks": self.repeat_offender_tracks,
            "top_events": self.top_events,
            "timeline": self.timeline,
            "data_quality_warning": self.data_quality_warning,
        }

    def headline(self) -> str:
        """One line a supervisor can read without opening the dashboard."""
        if self.data_quality_warning and self.total_events == 0:
            # Deliberately never phrased as "no risk found". Zero events on
            # unusable tracks means the footage was not analysable, and saying
            # otherwise would be the most dangerous sentence this tool could
            # produce.
            return (
                "No events detected, but the tracks could not support analysis - "
                f"{self.data_quality_warning}"
            )
        if self.total_events == 0:
            return (
                f"No unsafe handling detected across "
                f"{self.duration_seconds / 60.0:.1f} min of footage."
            )
        crit = self.events_by_severity.get(SEVERITY_CRITICAL, 0)
        high = self.events_by_severity.get(SEVERITY_HIGH, 0)
        parts = [f"{self.total_events} events"]
        if crit:
            parts.append(f"{crit} critical")
        if high:
            parts.append(f"{high} high")
        return (
            f"{self.shift_severity} risk shift ({self.shift_risk_index:.0f}/100): "
            + ", ".join(parts)
            + f" over {self.duration_seconds / 60.0:.1f} min."
        )


@dataclass
class RiskAssessment:
    """Scored events plus the shift roll-up."""

    events: List[BehaviorEvent] = field(default_factory=list)
    summary: ShiftSummary = field(default_factory=ShiftSummary)
    factors_by_event: Dict[str, List[RiskFactor]] = field(default_factory=dict)
    behavior_report: Optional[BehaviorReport] = None

    def ranked(self, limit: Optional[int] = None) -> List[BehaviorEvent]:
        """Events ordered by what a supervisor should look at first."""
        ordered = sorted(
            self.events,
            key=lambda e: (-(e.metrics.get("priority_score", 0.0)), e.start_time),
        )
        return ordered[:limit] if limit else ordered

    def by_severity(self, severity: str) -> List[BehaviorEvent]:
        return [e for e in self.events if e.severity == severity]

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "summary": self.summary.to_dict(),
            "events": [e.to_dict() for e in self.events],
            "risk_factors": {
                eid: [f.to_dict() for f in factors]
                for eid, factors in self.factors_by_event.items()
            },
        }
        if self.behavior_report is not None:
            payload["scene"] = self.behavior_report.to_dict()["scene"]
            payload["behavior_summary"] = self.behavior_report.to_dict()["summary"]
        return payload


class RiskEngine:
    """Phase 3 entry point."""

    # Shift index weighting. Severity dominates on purpose: one critical
    # incident should colour the whole shift, and a high event rate should
    # raise the floor without ever being able to hide a critical event.
    SEVERITY_WEIGHT = 0.65
    FREQUENCY_WEIGHT = 0.35
    FREQUENCY_FULL_EPM = 4.0
    """Events per minute at which the frequency component saturates."""

    RATE_CONFIDENCE_WINDOW_S = 60.0
    """Footage length needed before an events-per-minute figure is trusted at
    full weight. Five events in a 16-second clip extrapolates to 18.7/min,
    which is arithmetic rather than evidence; below this window the frequency
    component is damped in proportion to how much footage actually backs it."""

    TIMELINE_BUCKET_SECONDS = 30.0

    def __init__(self, weights: RiskWeights = DEFAULT_WEIGHTS):
        self.weights = weights
        self.scorer = RiskScorer(weights)

    # ------------------------------------------------------------- scoring

    def assess(
        self,
        report: BehaviorReport,
        duration_seconds: Optional[float] = None,
    ) -> RiskAssessment:
        events = list(report.events)
        events.sort(key=lambda e: e.start_time)

        factors_by_event: Dict[str, List[RiskFactor]] = {}
        seen_per_track: Dict[int, int] = {}
        seen_per_type: Dict[str, int] = {}

        for event in events:
            score, factors = self.scorer.score(event)

            # Repetition is only visible with the whole shift in view, which is
            # why it is applied here rather than in the scorer. Repeated
            # mishandling of the same load, or the same failure recurring
            # across the shift, is a process problem rather than an accident.
            repeat_points = 0.0
            prior_track = seen_per_track.get(event.track_id, 0)
            if prior_track:
                pts = min(
                    self.weights.repeat_max_points,
                    self.weights.repeat_track_points * prior_track,
                )
                factors.append(
                    RiskFactor(
                        "repeat_track", pts,
                        f"{prior_track + 1}th event on this load",
                    )
                )
                repeat_points += pts

            prior_type = seen_per_type.get(event.event_type, 0)
            if prior_type >= 2:
                pts = min(
                    self.weights.repeat_max_points,
                    self.weights.repeat_type_points * (prior_type - 1),
                )
                factors.append(
                    RiskFactor(
                        "repeat_pattern", pts,
                        f"{prior_type + 1}th {event.event_type.replace('_', ' ')} this shift",
                    )
                )
                repeat_points += pts

            score = max(0.0, min(100.0, score + repeat_points))

            event.risk_score = round(score, 1)
            event.severity = severity_for(score)
            event.risk_factors = [str(f) for f in factors]
            event.metrics["priority_score"] = round(score * event.confidence, 1)

            factors_by_event[event.event_id or f"trk{event.track_id}"] = factors
            seen_per_track[event.track_id] = prior_track + 1
            seen_per_type[event.event_type] = prior_type + 1

        summary = self._summarise(events, report, duration_seconds)
        return RiskAssessment(
            events=events,
            summary=summary,
            factors_by_event=factors_by_event,
            behavior_report=report,
        )

    # ------------------------------------------------------------ roll-up

    def _summarise(
        self,
        events: Sequence[BehaviorEvent],
        report: BehaviorReport,
        duration_seconds: Optional[float],
    ) -> ShiftSummary:
        duration = duration_seconds
        if duration is None:
            duration = getattr(report.context, "duration_seconds", None)
        if not duration or duration <= 0:
            duration = max((e.end_time for e in events), default=0.0)
        duration = max(duration, 1e-6)

        by_type = {t: 0 for t in ALL_EVENT_TYPES}
        by_severity = {s: 0 for s in SEVERITY_ORDER}
        for e in events:
            by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
            if e.severity:
                by_severity[e.severity] = by_severity.get(e.severity, 0) + 1

        scores = [e.risk_score or 0.0 for e in events]
        max_score = max(scores) if scores else 0.0
        mean_score = sum(scores) / len(scores) if scores else 0.0
        epm = len(events) / (duration / 60.0)

        frequency_component = min(100.0, epm / self.FREQUENCY_FULL_EPM * 100.0)
        rate_confidence = min(1.0, duration / self.RATE_CONFIDENCE_WINDOW_S)
        frequency_component *= rate_confidence

        index = (
            self.SEVERITY_WEIGHT * max_score
            + self.FREQUENCY_WEIGHT * frequency_component
        ) if events else 0.0

        # A shift is never safer than its worst moment. Damping the frequency
        # term can pull a blended index below the band of a genuinely critical
        # event, and reporting "High" for a shift that contained a Critical
        # incident inverts the meaning of the word. The blend sets the number;
        # the worst event sets the floor on the label.
        index_severity = severity_for(index)
        worst_severity = (
            max((e.severity for e in events if e.severity),
                key=lambda s: SEVERITY_ORDER.index(s), default=SEVERITY_LOW)
            if events else SEVERITY_LOW
        )
        shift_severity = max(
            [index_severity, worst_severity], key=lambda s: SEVERITY_ORDER.index(s)
        )

        track_counts: Dict[int, int] = {}
        for e in events:
            track_counts[e.track_id] = track_counts.get(e.track_id, 0) + 1
        repeat_tracks = sorted(t for t, c in track_counts.items() if c >= 2)

        top = sorted(
            events, key=lambda e: -(e.metrics.get("priority_score", 0.0))
        )[:5]

        return ShiftSummary(
            total_events=len(events),
            duration_seconds=duration,
            events_by_type=by_type,
            events_by_severity=by_severity,
            max_risk_score=max_score,
            mean_risk_score=mean_score,
            shift_risk_index=index,
            shift_severity=shift_severity,
            events_per_minute=epm,
            rate_is_reliable=rate_confidence >= 1.0,
            repeat_offender_tracks=repeat_tracks,
            top_events=[
                f"{e.event_id} {e.severity} {e.risk_score:.0f} - {e.description}"
                for e in top
            ],
            timeline=self._timeline(events, duration),
            data_quality_warning=report.data_quality_warning(),
        )

    def _timeline(
        self, events: Sequence[BehaviorEvent], duration: float
    ) -> List[Dict[str, Any]]:
        """Bucketed event counts and peak risk, for charting the shift."""
        bucket_s = self.TIMELINE_BUCKET_SECONDS
        n_buckets = max(1, int(duration / bucket_s) + 1)
        buckets: List[Dict[str, Any]] = [
            {
                "start_s": round(i * bucket_s, 1),
                "end_s": round((i + 1) * bucket_s, 1),
                "events": 0,
                "max_risk": 0.0,
                "types": [],
            }
            for i in range(n_buckets)
        ]

        for e in events:
            idx = min(n_buckets - 1, int(e.start_time / bucket_s))
            b = buckets[idx]
            b["events"] += 1
            b["max_risk"] = max(b["max_risk"], e.risk_score or 0.0)
            if e.event_type not in b["types"]:
                b["types"].append(e.event_type)

        return buckets


def assess_report(
    report: BehaviorReport, duration_seconds: Optional[float] = None
) -> RiskAssessment:
    """One-line convenience wrapper."""
    return RiskEngine().assess(report, duration_seconds)
