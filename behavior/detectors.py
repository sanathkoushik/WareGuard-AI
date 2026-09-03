"""
WareGuard AI - Behavior Detectors (Phase 2)

Five heuristic detectors, each keyed on a physical signature that separates it
from the others. The separations are the interesting part:

  DROP    vertical fall + free-fall-fraction acceleration + abrupt stop
  THROW   a drop that also carries horizontal momentum (so: ballistic, not fallen)
  DRAG    floor contact + horizontal travel + almost no vertical motion
  STACK   two cargo boxes vertically adjacent with the upper one overhanging
  ROUGH   high speed *and* high jerk while a worker is within reach

A detector returns candidate `BehaviorEvent`s with a calibrated confidence; it
does not decide precedence between overlapping candidates. That is the engine's
job (see engine.py), because a single fall is legitimately both "a drop" and
"an object that was airborne" and only one of those belongs in a report.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .features import TrackFeatures, Kinematics, nearest_person_gap, percentile
from .schema import BehaviorEvent, SceneContext, Track, TrackPoint
from .thresholds import DEFAULT_THRESHOLDS, Thresholds

# Event type vocabulary. Kept as constants so the risk engine, dashboard and
# assistant never disagree about spelling.
EVENT_DROP = "drop"
EVENT_THROW = "throw"
EVENT_DRAG = "drag"
EVENT_IMPROPER_STACK = "improper_stack"
EVENT_ROUGH_HANDLING = "rough_handling"

ALL_EVENT_TYPES = [
    EVENT_DROP,
    EVENT_THROW,
    EVENT_DRAG,
    EVENT_IMPROPER_STACK,
    EVENT_ROUGH_HANDLING,
]


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def find_runs(
    kins: Sequence[Kinematics],
    predicate: Callable[[Kinematics], bool],
    min_length: int = 2,
    max_break: int = 1,
) -> List[Tuple[int, int]]:
    """Contiguous index ranges where `predicate` holds.

    `max_break` tolerates that many consecutive failing samples inside a run,
    so one mis-sized bounding box mid-fall does not split a single drop into
    two half-drops. Returned as inclusive (start, end) index pairs.
    """
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    last_true: Optional[int] = None
    breaks = 0

    for i, k in enumerate(kins):
        if predicate(k):
            if start is None:
                start = i
            last_true = i
            breaks = 0
        elif start is not None:
            breaks += 1
            if breaks > max_break:
                if last_true is not None and (last_true - start + 1) >= min_length:
                    runs.append((start, last_true))
                start, last_true, breaks = None, None, 0

    if start is not None and last_true is not None and (last_true - start + 1) >= min_length:
        runs.append((start, last_true))

    return runs


def saturate(value: float, threshold: float, full: float) -> float:
    """Map `value` to 0..1: 0 at the threshold, 1 at `full` and beyond."""
    if full <= threshold:
        return 1.0 if value >= threshold else 0.0
    if value <= threshold:
        return 0.0
    return min(1.0, (value - threshold) / (full - threshold))


def blend_confidence(
    factors: Sequence[float],
    features: TrackFeatures,
    floor: float = 0.45,
) -> float:
    """Combine evidence strength with track quality into a final confidence.

    Two independent things degrade a claim, and they are kept separate on
    purpose:
      - how far past threshold the evidence sits (`factors`)
      - how trustworthy the underlying track is (continuity, detector score)

    An unmistakable drop seen on a flickering track should not report 0.95.
    """
    if not factors:
        strength = 0.0
    else:
        strength = sum(factors) / len(factors)

    evidence = floor + (1.0 - floor) * strength

    continuity = features.track.continuity()
    det_conf = (
        sum(p.confidence for p in features.track.points) / len(features.track.points)
        if features.track.points else 0.5
    )
    # Detector confidence is a weak signal, so it can only pull the result
    # down by a limited amount rather than dominating it.
    quality = (0.55 + 0.45 * continuity) * (0.70 + 0.30 * min(1.0, det_conf / 0.6))

    return round(max(0.0, min(1.0, evidence * quality)), 3)


def _seconds(kins: Sequence[Kinematics], i: int, j: int) -> float:
    return max(0.0, kins[j].timestamp - kins[i].timestamp)


def _attach_person_context(
    event: BehaviorEvent,
    features: TrackFeatures,
    person_tracks: Sequence[Track],
    kins: Sequence[Kinematics],
    i: int,
    j: int,
) -> None:
    """Record the closest worker across the episode, in object-heights."""
    gaps: List[float] = []
    nearest_ids: List[int] = []
    for k in kins[i : j + 1]:
        gap = nearest_person_gap(
            k.point, k.frame, person_tracks, features.reference_height
        )
        if gap is not None:
            gaps.append(gap)
    if gaps:
        event.metrics["nearest_person_heights"] = round(min(gaps), 3)
    for t in person_tracks:
        for k in kins[i : j + 1]:
            if t.at_frame(k.frame) is not None:
                nearest_ids.append(t.track_id)
                break
    if nearest_ids:
        event.related_track_ids = sorted(set(nearest_ids))


class BehaviorDetector:
    """Base class. Subclasses implement `detect`."""

    event_type: str = "unknown"

    def __init__(self, thresholds: Thresholds = DEFAULT_THRESHOLDS):
        self.t = thresholds

    def detect(
        self,
        features_map: Dict[int, TrackFeatures],
        context: SceneContext,
        person_tracks: Sequence[Track],
    ) -> List[BehaviorEvent]:
        raise NotImplementedError

    def _cargo(self, features_map: Dict[int, TrackFeatures]) -> List[TrackFeatures]:
        return [
            f for f in features_map.values()
            if f.track.is_cargo and f.usable(self.t)
        ]


# --------------------------------------------------------------------------
# 1. Drop
# --------------------------------------------------------------------------

class DropDetector(BehaviorDetector):
    """An object falls under gravity and stops abruptly.

    The abrupt stop is what makes it a *drop* rather than a *lowering*. A
    worker setting a carton down decelerates it smoothly over ~0.5s; a dropped
    carton loses most of its speed in one or two frames when it hits the floor.
    Requiring that collapse is what keeps this detector off careful handling.
    """

    event_type = EVENT_DROP

    def detect(self, features_map, context, person_tracks):
        events: List[BehaviorEvent] = []
        g_norm = context.gravity_in_heights_per_s2()

        for f in self._cargo(features_map):
            kins = f.kinematics
            # A descent is any sustained downward motion; thresholds are
            # applied to the episode as a whole, not sample by sample.
            runs = find_runs(
                kins,
                lambda k: k.vy > 0.15 * self.t.drop_min_peak_fall_speed,
                min_length=2,
                max_break=1,
            )

            for i, j in runs:
                segment = kins[i : j + 1]
                peak_vy = max(k.vy for k in segment)
                peak_idx = i + max(range(len(segment)), key=lambda n: segment[n].vy)
                fall_distance = (
                    segment[-1].point.cy - segment[0].point.cy
                ) / f.reference_height
                duration = _seconds(kins, i, j)
                peak_ay = max(k.ay for k in segment)
                gravity_ratio = peak_ay / g_norm if g_norm > 0 else 0.0

                if peak_vy < self.t.drop_min_peak_fall_speed:
                    continue
                if fall_distance < self.t.drop_min_fall_distance:
                    continue
                if duration < self.t.drop_min_fall_duration:
                    continue
                if gravity_ratio < self.t.drop_min_gravity_ratio:
                    continue

                impact = self._find_impact(kins, peak_idx)
                if impact is None:
                    continue
                impact_idx, impact_ratio = impact

                factors = [
                    saturate(peak_vy, self.t.drop_min_peak_fall_speed,
                             self.t.drop_min_peak_fall_speed * 3.0),
                    saturate(fall_distance, self.t.drop_min_fall_distance,
                             self.t.drop_min_fall_distance * 3.0),
                    saturate(gravity_ratio, self.t.drop_min_gravity_ratio, 1.0),
                    saturate(1.0 - impact_ratio, 1.0 - self.t.drop_impact_speed_ratio, 0.95),
                ]

                impact_speed_mps = context.heights_per_s_to_mps(peak_vy)
                ev = BehaviorEvent(
                    event_type=self.event_type,
                    track_id=f.track_id,
                    class_name=f.class_name,
                    start_frame=kins[i].frame,
                    end_frame=kins[impact_idx].frame,
                    start_time=round(kins[i].timestamp, 3),
                    end_time=round(kins[impact_idx].timestamp, 3),
                    confidence=blend_confidence(factors, f),
                    metrics={
                        "peak_fall_speed": round(peak_vy, 3),
                        "impact_speed_mps": round(impact_speed_mps, 2),
                        "fall_distance_heights": round(fall_distance, 3),
                        "fall_duration_s": round(duration, 3),
                        "gravity_ratio": round(gravity_ratio, 3),
                        "impact_speed_ratio": round(impact_ratio, 3),
                        "ground_clearance_at_start": round(
                            kins[i].ground_clearance, 3
                        ),
                    },
                    description=(
                        f"{f.class_name} #{f.track_id} fell "
                        f"{fall_distance:.1f} box-heights in {duration:.2f}s "
                        f"(~{impact_speed_mps:.1f} m/s at impact, "
                        f"{gravity_ratio * 100:.0f}% of free fall) and stopped abruptly"
                    ),
                )
                _attach_person_context(ev, f, person_tracks, kins, i, impact_idx)
                events.append(ev)

        return events

    def _find_impact(
        self, kins: Sequence[Kinematics], peak_idx: int
    ) -> Optional[Tuple[int, float]]:
        """Locate the frame where speed collapses after the fastest fall.

        Returns (index, speed_ratio) or None when the object never stopped
        abruptly - i.e. it was lowered, or it left the frame still moving.
        """
        peak_speed = kins[peak_idx].speed
        if peak_speed <= 1e-6:
            return None

        deadline = kins[peak_idx].timestamp + self.t.drop_impact_window
        for n in range(peak_idx + 1, len(kins)):
            if kins[n].timestamp > deadline:
                break
            ratio = kins[n].speed / peak_speed
            if ratio <= self.t.drop_impact_speed_ratio:
                return n, ratio
        return None


# --------------------------------------------------------------------------
# 2. Throw
# --------------------------------------------------------------------------

class ThrowDetector(BehaviorDetector):
    """Cargo travels ballistically: airborne, fast, and carrying sideways speed.

    The horizontal component is the whole distinction from a drop. A box that
    leaves someone's hands and lands two metres away was thrown; a box that
    lands at their feet was dropped. Both are unsafe, but a supervisor reads
    them very differently, so they are reported as different behaviors.
    """

    event_type = EVENT_THROW

    def detect(self, features_map, context, person_tracks):
        events: List[BehaviorEvent] = []

        for f in self._cargo(features_map):
            kins = f.kinematics
            airborne = find_runs(
                kins,
                lambda k: (
                    k.ground_clearance > self.t.drag_floor_contact_tolerance
                    and abs(k.vx) > 0.3 * self.t.throw_min_horizontal_speed
                ),
                min_length=2,
                max_break=1,
            )

            for i, j in airborne:
                segment = kins[i : j + 1]
                duration = _seconds(kins, i, j)
                peak_vx = max(abs(k.vx) for k in segment)
                launch_speed = max(k.speed for k in segment)
                horizontal_travel = abs(
                    segment[-1].point.cx - segment[0].point.cx
                ) / f.reference_height

                # Free flight: an unsupported object accelerates downward at g.
                # A carried or shoved box is held, so its vertical acceleration
                # is ~0 no matter how fast it travels horizontally. Without
                # this test, every carry across the frame reads as a throw.
                g_norm = context.gravity_in_heights_per_s2()
                median_ay = percentile([k.ay for k in segment], 0.5)
                gravity_ratio = median_ay / g_norm if g_norm > 0 else 0.0

                if duration < self.t.throw_min_airborne_duration:
                    continue
                if peak_vx < self.t.throw_min_horizontal_speed:
                    continue
                if launch_speed < self.t.throw_min_launch_speed:
                    continue
                if gravity_ratio < self.t.throw_min_gravity_ratio:
                    continue

                # A ballistic arc reverses its vertical velocity: up, then
                # down. Not required (a flat hurl never rises) but it is
                # strong corroboration when present.
                vys = [k.vy for k in segment]
                has_apex = any(
                    vys[n] < 0 and vys[n + 1] > 0 for n in range(len(vys) - 1)
                )

                factors = [
                    saturate(peak_vx, self.t.throw_min_horizontal_speed,
                             self.t.throw_min_horizontal_speed * 2.5),
                    saturate(launch_speed, self.t.throw_min_launch_speed,
                             self.t.throw_min_launch_speed * 2.5),
                    saturate(horizontal_travel, 0.5, 4.0),
                    saturate(gravity_ratio, self.t.throw_min_gravity_ratio, 1.0),
                    1.0 if has_apex else 0.35,
                ]

                ev = BehaviorEvent(
                    event_type=self.event_type,
                    track_id=f.track_id,
                    class_name=f.class_name,
                    start_frame=kins[i].frame,
                    end_frame=kins[j].frame,
                    start_time=round(kins[i].timestamp, 3),
                    end_time=round(kins[j].timestamp, 3),
                    confidence=blend_confidence(factors, f),
                    metrics={
                        "peak_horizontal_speed": round(peak_vx, 3),
                        "launch_speed": round(launch_speed, 3),
                        "launch_speed_mps": round(
                            context.heights_per_s_to_mps(launch_speed), 2
                        ),
                        "horizontal_travel_heights": round(horizontal_travel, 3),
                        "airborne_duration_s": round(duration, 3),
                        "gravity_ratio": round(gravity_ratio, 3),
                        "ballistic_arc": 1.0 if has_apex else 0.0,
                    },
                    description=(
                        f"{f.class_name} #{f.track_id} travelled "
                        f"{horizontal_travel:.1f} box-widths through the air over "
                        f"{duration:.2f}s at ~"
                        f"{context.heights_per_s_to_mps(launch_speed):.1f} m/s"
                        + (" along a ballistic arc" if has_apex else "")
                    ),
                )
                _attach_person_context(ev, f, person_tracks, kins, i, j)
                events.append(ev)

        return events


# --------------------------------------------------------------------------
# 3. Drag
# --------------------------------------------------------------------------

class DragDetector(BehaviorDetector):
    """Cargo pushed or pulled along the floor instead of being carried.

    Defined by what is *absent* as much as what is present: horizontal travel
    with the bottom edge pinned to the floor plane and no vertical motion. That
    absence is what separates dragging from carrying, since a carried box also
    moves horizontally - but well above the floor.
    """

    event_type = EVENT_DRAG

    def detect(self, features_map, context, person_tracks):
        events: List[BehaviorEvent] = []

        for f in self._cargo(features_map):
            kins = f.kinematics
            runs = find_runs(
                kins,
                lambda k: (
                    abs(k.vx) >= self.t.drag_min_horizontal_speed
                    and abs(k.vy) <= self.t.drag_max_vertical_speed
                    and k.ground_clearance <= self.t.drag_floor_contact_tolerance
                ),
                min_length=3,
                max_break=2,
            )

            for i, j in runs:
                segment = kins[i : j + 1]
                duration = _seconds(kins, i, j)
                distance = abs(
                    segment[-1].point.cx - segment[0].point.cx
                ) / f.reference_height
                mean_vx = sum(abs(k.vx) for k in segment) / len(segment)

                if duration < self.t.drag_min_duration:
                    continue
                if distance < self.t.drag_min_distance:
                    continue

                factors = [
                    saturate(duration, self.t.drag_min_duration,
                             self.t.drag_min_duration * 4.0),
                    saturate(distance, self.t.drag_min_distance,
                             self.t.drag_min_distance * 4.0),
                    saturate(mean_vx, self.t.drag_min_horizontal_speed,
                             self.t.drag_min_horizontal_speed * 3.0),
                ]

                direction = "right" if segment[-1].point.cx > segment[0].point.cx else "left"
                ev = BehaviorEvent(
                    event_type=self.event_type,
                    track_id=f.track_id,
                    class_name=f.class_name,
                    start_frame=kins[i].frame,
                    end_frame=kins[j].frame,
                    start_time=round(kins[i].timestamp, 3),
                    end_time=round(kins[j].timestamp, 3),
                    confidence=blend_confidence(factors, f),
                    metrics={
                        "drag_distance_heights": round(distance, 3),
                        "drag_duration_s": round(duration, 3),
                        "mean_horizontal_speed": round(mean_vx, 3),
                        "mean_ground_clearance": round(
                            sum(k.ground_clearance for k in segment) / len(segment), 3
                        ),
                    },
                    description=(
                        f"{f.class_name} #{f.track_id} dragged {direction} along the "
                        f"floor for {distance:.1f} box-widths over {duration:.1f}s"
                    ),
                )
                _attach_person_context(ev, f, person_tracks, kins, i, j)
                events.append(ev)

        return events


# --------------------------------------------------------------------------
# 4. Improper stacking
# --------------------------------------------------------------------------

class ImproperStackDetector(BehaviorDetector):
    """An upper carton's centre of mass sits too far past its support.

    Works on pairs of cargo tracks that are vertically adjacent and
    horizontally overlapping. Overhang is measured as centre offset over the
    lower box's width, which is the quantity that actually predicts toppling -
    once the upper centre of mass passes the lower footprint's edge, the stack
    is held up only by friction.

    Axis-aligned boxes carry no rotation, so tilt is approximated by how much
    the upper box's aspect ratio departs from its own median.
    """

    event_type = EVENT_IMPROPER_STACK

    def detect(self, features_map, context, person_tracks):
        cargo = self._cargo(features_map)
        if len(cargo) < 2:
            return []

        # frame -> [(features, kinematics)]
        by_frame: Dict[int, List[Tuple[TrackFeatures, Kinematics]]] = {}
        for f in cargo:
            for k in f.kinematics:
                by_frame.setdefault(k.frame, []).append((f, k))

        # (lower_id, upper_id) -> list of (frame, timestamp, overhang, lean)
        pair_hits: Dict[Tuple[int, int], List[Tuple[int, float, float, float]]] = {}

        for frame in sorted(by_frame):
            entries = by_frame[frame]
            for a in range(len(entries)):
                for b in range(len(entries)):
                    if a == b:
                        continue
                    upper_f, upper_k = entries[a]
                    lower_f, lower_k = entries[b]
                    up, lo = upper_k.point, lower_k.point

                    if up.cy >= lo.cy:  # "upper" must actually be higher
                        continue

                    stacked = self._is_stacked(up, lo)
                    if not stacked:
                        continue

                    overhang = abs(up.cx - lo.cx) / lo.width
                    lean = upper_k.aspect_excess

                    if (
                        overhang > self.t.stack_max_overhang_ratio
                        or lean > self.t.stack_min_lean_ratio
                    ):
                        pair_hits.setdefault(
                            (lower_f.track_id, upper_f.track_id), []
                        ).append((frame, upper_k.timestamp, overhang, lean))

        events: List[BehaviorEvent] = []
        for (lower_id, upper_id), hits in pair_hits.items():
            for episode in self._split_episodes(hits):
                duration = episode[-1][1] - episode[0][1]
                if duration < self.t.stack_min_duration:
                    continue

                max_overhang = max(h[2] for h in episode)
                max_lean = max(h[3] for h in episode)
                upper_f = features_map[upper_id]

                factors = [
                    saturate(max_overhang, self.t.stack_max_overhang_ratio, 0.85),
                    saturate(duration, self.t.stack_min_duration,
                             self.t.stack_min_duration * 6.0),
                ]
                if max_lean > self.t.stack_min_lean_ratio:
                    factors.append(
                        saturate(max_lean, self.t.stack_min_lean_ratio, 0.60)
                    )

                ev = BehaviorEvent(
                    event_type=self.event_type,
                    track_id=upper_id,
                    class_name=upper_f.class_name,
                    start_frame=episode[0][0],
                    end_frame=episode[-1][0],
                    start_time=round(episode[0][1], 3),
                    end_time=round(episode[-1][1], 3),
                    confidence=blend_confidence(factors, upper_f),
                    related_track_ids=[lower_id],
                    metrics={
                        "max_overhang_ratio": round(max_overhang, 3),
                        "max_lean_ratio": round(max_lean, 3),
                        "stack_duration_s": round(duration, 3),
                        "supporting_track_id": float(lower_id),
                    },
                    description=(
                        f"{upper_f.class_name} #{upper_id} stacked on #{lower_id} with "
                        f"{max_overhang * 100:.0f}% overhang for {duration:.1f}s"
                        + (f" and visible lean" if max_lean > self.t.stack_min_lean_ratio else "")
                    ),
                )
                events.append(ev)

        return events

    def _is_stacked(self, upper: TrackPoint, lower: TrackPoint) -> bool:
        """Vertically adjacent and horizontally overlapping."""
        vertical_gap = abs(upper.y2 - lower.y1) / lower.height
        if vertical_gap > self.t.stack_vertical_gap_tolerance:
            return False

        overlap = min(upper.x2, lower.x2) - max(upper.x1, lower.x1)
        if overlap <= 0:
            return False
        return (overlap / lower.width) >= self.t.stack_min_horizontal_overlap

    @staticmethod
    def _split_episodes(
        hits: List[Tuple[int, float, float, float]], max_frame_gap: int = 5
    ) -> List[List[Tuple[int, float, float, float]]]:
        """Group frame hits into continuous episodes."""
        if not hits:
            return []
        hits = sorted(hits, key=lambda h: h[0])
        episodes = [[hits[0]]]
        for h in hits[1:]:
            if h[0] - episodes[-1][-1][0] <= max_frame_gap:
                episodes[-1].append(h)
            else:
                episodes.append([h])
        return episodes


# --------------------------------------------------------------------------
# 5. Rough handling
# --------------------------------------------------------------------------

class RoughHandlingDetector(BehaviorDetector):
    """Abrupt, forceful cargo movement while a worker is within reach.

    Keyed on **jerk** rather than speed. A box moving quickly on a conveyor is
    normal; a box whose speed changes violently is being yanked, slammed, or
    swung. Requiring a nearby worker attributes the motion to handling and
    keeps forklift and conveyor movement out of the report.
    """

    event_type = EVENT_ROUGH_HANDLING

    def detect(self, features_map, context, person_tracks):
        events: List[BehaviorEvent] = []

        for f in self._cargo(features_map):
            kins = f.kinematics
            runs = find_runs(
                kins,
                lambda k: (
                    k.speed >= self.t.rough_min_speed
                    and abs(k.jerk) >= self.t.rough_min_jerk
                ),
                min_length=2,
                max_break=1,
            )

            for i, j in runs:
                segment = kins[i : j + 1]
                duration = _seconds(kins, i, j)
                if duration < self.t.rough_min_duration:
                    continue

                gaps = [
                    g for g in (
                        nearest_person_gap(
                            k.point, k.frame, person_tracks, f.reference_height
                        )
                        for k in segment
                    ) if g is not None
                ]
                if not gaps:
                    continue  # no worker visible: not attributable to handling
                min_gap = min(gaps)
                if min_gap > self.t.rough_person_proximity:
                    continue

                peak_speed = max(k.speed for k in segment)
                peak_jerk = max(abs(k.jerk) for k in segment)

                factors = [
                    saturate(peak_speed, self.t.rough_min_speed,
                             self.t.rough_min_speed * 2.5),
                    saturate(peak_jerk, self.t.rough_min_jerk,
                             self.t.rough_min_jerk * 3.0),
                    saturate(self.t.rough_person_proximity - min_gap, 0.0,
                             self.t.rough_person_proximity),
                ]

                ev = BehaviorEvent(
                    event_type=self.event_type,
                    track_id=f.track_id,
                    class_name=f.class_name,
                    start_frame=kins[i].frame,
                    end_frame=kins[j].frame,
                    start_time=round(kins[i].timestamp, 3),
                    end_time=round(kins[j].timestamp, 3),
                    confidence=blend_confidence(factors, f),
                    metrics={
                        "peak_speed": round(peak_speed, 3),
                        "peak_jerk": round(peak_jerk, 3),
                        "nearest_person_heights": round(min_gap, 3),
                        "duration_s": round(duration, 3),
                    },
                    description=(
                        f"{f.class_name} #{f.track_id} handled abruptly "
                        f"(jerk {peak_jerk:.1f} h/s^2) with a worker "
                        f"{min_gap:.1f} box-heights away"
                    ),
                )
                _attach_person_context(ev, f, person_tracks, kins, i, j)
                events.append(ev)

        return events


DETECTOR_CLASSES = [
    DropDetector,
    ThrowDetector,
    DragDetector,
    ImproperStackDetector,
    RoughHandlingDetector,
]
