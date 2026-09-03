"""
WareGuard AI - Risk Scoring Model (Phase 3)

Turns a `BehaviorEvent` into a 0-100 risk score plus the reasons for it.

Two design decisions shape this module:

**Severity and certainty are scored separately.** `risk_score` answers "how bad
is this *if it really happened*"; `priority_score` multiplies that by the
detector's confidence to answer "what should a supervisor look at first". A
90%-certain drag and a 40%-certain drop are ranked correctly only when those
two questions are kept apart, and collapsing them into one number is how safety
dashboards end up burying the dangerous-but-uncertain events.

**Every point is attributable.** The score is a base value plus named,
individually-explained contributions, never an opaque weighted sum. A
supervisor who cannot see why a box scored 78 will not act on it, and the
Phase 5 assistant needs these factors to answer "why was this critical?".
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from behavior.detectors import (
    EVENT_DRAG,
    EVENT_DROP,
    EVENT_IMPROPER_STACK,
    EVENT_ROUGH_HANDLING,
    EVENT_THROW,
)
from behavior.schema import BehaviorEvent

# Severity bands. Boundaries are inclusive of the lower value.
SEVERITY_LOW = "Low"
SEVERITY_MEDIUM = "Medium"
SEVERITY_HIGH = "High"
SEVERITY_CRITICAL = "Critical"

SEVERITY_ORDER = [SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH, SEVERITY_CRITICAL]
SEVERITY_BANDS = [(75.0, SEVERITY_CRITICAL), (50.0, SEVERITY_HIGH), (25.0, SEVERITY_MEDIUM)]


def severity_for(score: float) -> str:
    for threshold, label in SEVERITY_BANDS:
        if score >= threshold:
            return label
    return SEVERITY_LOW


@dataclass
class RiskFactor:
    """One named contribution to a score, with its justification."""

    code: str
    points: float
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "points": round(self.points, 1), "detail": self.detail}

    def __str__(self) -> str:
        sign = "+" if self.points >= 0 else ""
        return f"{self.detail} ({sign}{self.points:.0f})"


@dataclass
class RiskWeights:
    """Tunable weights for the scoring model."""

    # Base severity per behavior, before any context is applied.
    # A throw outranks a drop because it is intentional and less predictable;
    # a drag outranks nothing because it damages goods but rarely people.
    base_scores: Dict[str, float] = field(default_factory=lambda: {
        EVENT_THROW: 52.0,
        EVENT_DROP: 40.0,
        EVENT_ROUGH_HANDLING: 34.0,
        EVENT_IMPROPER_STACK: 30.0,
        EVENT_DRAG: 26.0,
    })

    # --- impact energy -----------------------------------------------------
    impact_free_speed_mps: float = 2.0
    """Impact speed below which a landing is unremarkable. ~2 m/s is roughly a
    20cm drop - the height goods are routinely set down from."""
    impact_points_per_mps: float = 8.0
    impact_max_points: float = 25.0

    # --- fall height -------------------------------------------------------
    fall_free_heights: float = 1.0
    fall_points_per_height: float = 4.0
    fall_max_points: float = 12.0

    # --- worker proximity --------------------------------------------------
    proximity_threshold_heights: float = 1.20
    proximity_max_points: float = 25.0
    """A dropped carton beside a worker is an injury risk, not just a damaged
    goods risk. This is the single largest contextual escalation in the model,
    and deliberately so."""

    # --- drag extent -------------------------------------------------------
    drag_free_heights: float = 1.0
    drag_points_per_height: float = 2.0
    drag_max_points: float = 14.0

    # --- stack overhang ----------------------------------------------------
    overhang_free_ratio: float = 0.30
    overhang_max_points: float = 20.0
    overhang_full_ratio: float = 0.85
    """At 85% overhang the upper carton's centre of mass has left its support
    entirely and the stack is held only by friction."""

    # --- abruptness --------------------------------------------------------
    jerk_free: float = 12.0
    jerk_full: float = 36.0
    jerk_max_points: float = 16.0

    # --- repetition (applied by the engine, which sees the whole shift) ----
    repeat_track_points: float = 8.0
    repeat_type_points: float = 6.0
    repeat_max_points: float = 18.0


DEFAULT_WEIGHTS = RiskWeights()


def _ramp(value: float, free: float, full: float, max_points: float) -> float:
    """Linear ramp from 0 points at `free` to `max_points` at `full`."""
    if full <= free:
        return max_points if value > free else 0.0
    if value <= free:
        return 0.0
    return min(max_points, (value - free) / (full - free) * max_points)


class RiskScorer:
    """Scores one event at a time. Shift-level context is added by the engine."""

    def __init__(self, weights: RiskWeights = DEFAULT_WEIGHTS):
        self.w = weights

    def score(self, event: BehaviorEvent) -> tuple[float, List[RiskFactor]]:
        """Return (risk_score, factors) for a single event in isolation."""
        base = self.w.base_scores.get(event.event_type, 30.0)
        factors: List[RiskFactor] = [
            RiskFactor("base", base, f"{event.event_type.replace('_', ' ')} baseline")
        ]

        m = event.metrics
        handler = {
            EVENT_DROP: self._drop_factors,
            EVENT_THROW: self._throw_factors,
            EVENT_DRAG: self._drag_factors,
            EVENT_IMPROPER_STACK: self._stack_factors,
            EVENT_ROUGH_HANDLING: self._rough_factors,
        }.get(event.event_type)

        if handler:
            factors.extend(handler(m))

        proximity = self._proximity_factor(event)
        if proximity:
            factors.append(proximity)

        total = sum(f.points for f in factors)
        return max(0.0, min(100.0, total)), factors

    # ------------------------------------------------------- per-event type

    def _drop_factors(self, m: Dict[str, float]) -> List[RiskFactor]:
        out: List[RiskFactor] = []

        speed = float(m.get("impact_speed_mps", 0.0))
        pts = _ramp(
            speed,
            self.w.impact_free_speed_mps,
            self.w.impact_free_speed_mps + self.w.impact_max_points / self.w.impact_points_per_mps,
            self.w.impact_max_points,
        )
        if pts > 0:
            out.append(RiskFactor("impact_energy", pts, f"impact at {speed:.1f} m/s"))

        height = float(m.get("fall_distance_heights", 0.0))
        pts = _ramp(height, self.w.fall_free_heights,
                    self.w.fall_free_heights + self.w.fall_max_points / self.w.fall_points_per_height,
                    self.w.fall_max_points)
        if pts > 0:
            out.append(RiskFactor("fall_height", pts, f"fell {height:.1f} box-heights"))

        return out

    def _throw_factors(self, m: Dict[str, float]) -> List[RiskFactor]:
        out: List[RiskFactor] = []

        speed = float(m.get("launch_speed_mps", 0.0))
        pts = _ramp(
            speed,
            self.w.impact_free_speed_mps,
            self.w.impact_free_speed_mps + self.w.impact_max_points / self.w.impact_points_per_mps,
            self.w.impact_max_points,
        )
        if pts > 0:
            out.append(RiskFactor("launch_energy", pts, f"launched at {speed:.1f} m/s"))

        travel = float(m.get("horizontal_travel_heights", 0.0))
        pts = _ramp(travel, 1.0, 6.0, 10.0)
        if pts > 0:
            out.append(RiskFactor("throw_distance", pts, f"travelled {travel:.1f} box-widths"))

        return out

    def _drag_factors(self, m: Dict[str, float]) -> List[RiskFactor]:
        out: List[RiskFactor] = []

        distance = float(m.get("drag_distance_heights", 0.0))
        pts = _ramp(distance, self.w.drag_free_heights,
                    self.w.drag_free_heights + self.w.drag_max_points / self.w.drag_points_per_height,
                    self.w.drag_max_points)
        if pts > 0:
            out.append(RiskFactor("drag_extent", pts, f"dragged {distance:.1f} box-widths"))

        duration = float(m.get("drag_duration_s", 0.0))
        pts = _ramp(duration, 1.0, 6.0, 8.0)
        if pts > 0:
            out.append(RiskFactor("drag_duration", pts, f"sustained {duration:.1f}s"))

        return out

    def _stack_factors(self, m: Dict[str, float]) -> List[RiskFactor]:
        out: List[RiskFactor] = []

        overhang = float(m.get("max_overhang_ratio", 0.0))
        pts = _ramp(overhang, self.w.overhang_free_ratio,
                    self.w.overhang_full_ratio, self.w.overhang_max_points)
        if pts > 0:
            out.append(RiskFactor("overhang", pts, f"{overhang * 100:.0f}% overhang"))

        lean = float(m.get("max_lean_ratio", 0.0))
        pts = _ramp(lean, 0.18, 0.60, 10.0)
        if pts > 0:
            out.append(RiskFactor("lean", pts, f"visible lean ({lean * 100:.0f}%)"))

        duration = float(m.get("stack_duration_s", 0.0))
        pts = _ramp(duration, 2.0, 30.0, 8.0)
        if pts > 0:
            out.append(
                RiskFactor("stack_persistence", pts, f"left standing {duration:.0f}s")
            )

        return out

    def _rough_factors(self, m: Dict[str, float]) -> List[RiskFactor]:
        out: List[RiskFactor] = []

        jerk = float(m.get("peak_jerk", 0.0))
        pts = _ramp(jerk, self.w.jerk_free, self.w.jerk_full, self.w.jerk_max_points)
        if pts > 0:
            out.append(RiskFactor("abruptness", pts, f"jerk {jerk:.0f} h/s^2"))

        speed = float(m.get("peak_speed", 0.0))
        pts = _ramp(speed, 1.3, 5.0, 8.0)
        if pts > 0:
            out.append(RiskFactor("handling_speed", pts, f"peak {speed:.1f} h/s"))

        return out

    # ------------------------------------------------------------ proximity

    def _proximity_factor(self, event: BehaviorEvent) -> Optional[RiskFactor]:
        """Escalate when a worker was within reach of the mishandled load.

        Absent proximity data is treated as 'unknown', not 'safe'. No points are
        added, because inventing risk from missing data is as wrong as ignoring
        it - but the distinction is preserved in the event metrics so a reviewer
        knows the difference.
        """
        gap = event.metrics.get("nearest_person_heights")
        if gap is None:
            return None

        gap = float(gap)
        if gap > self.w.proximity_threshold_heights:
            return None

        pts = (
            (self.w.proximity_threshold_heights - gap)
            / max(1e-6, self.w.proximity_threshold_heights)
            * self.w.proximity_max_points
        )
        return RiskFactor(
            "worker_proximity", pts, f"worker {gap:.1f} box-heights away"
        )
