"""
WareGuard AI - Kinematic Feature Extraction (Phase 2)

Turns raw `Track` geometry into the normalised motion signals the behavior
detectors reason about.

Two things happen here that do not happen in detection/tracker.py, and both
matter for correctness:

1. **Derivatives are taken against timestamps, not frame indices.** The
   upstream tracker divides by `max(1, frame_delta)`, which silently reports a
   box that vanished for 10 frames as moving slowly rather than as unobserved.
   Central differences over real elapsed time give the true rate.

2. **Positions are smoothed before differentiating.** Bounding-box corners
   jitter by a few pixels every frame even on a stationary object. Raw
   differencing turns that jitter into phantom acceleration spikes, which is
   exactly the signal a drop detector keys on. A 3-frame moving average removes
   it while preserving a real impact.

Everything leaves here in **object-heights per second**, so thresholds are
resolution- and frame-rate-independent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .schema import SceneContext, Track, TrackPoint
from .thresholds import DEFAULT_THRESHOLDS, Thresholds


# --------------------------------------------------------------------------
# Small numeric helpers (stdlib only - no numpy dependency by design)
# --------------------------------------------------------------------------

def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. `q` in [0, 1]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def moving_average(values: Sequence[float], window: int) -> List[float]:
    """Centred moving average that shrinks the window at the edges.

    Edge-shrinking (rather than padding) keeps the first and last samples
    honest instead of dragging them toward a fabricated boundary value.
    """
    n = len(values)
    if n == 0 or window <= 1:
        return list(values)

    half = window // 2
    out: List[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def central_difference(values: Sequence[float], times: Sequence[float]) -> List[float]:
    """d(values)/d(times) using central differences, one-sided at the ends.

    Returns 0.0 wherever the time delta is degenerate, so a duplicated
    timestamp can never produce an infinite velocity.
    """
    n = len(values)
    if n < 2:
        return [0.0] * n

    out: List[float] = []
    for i in range(n):
        if i == 0:
            dt = times[1] - times[0]
            dv = values[1] - values[0]
        elif i == n - 1:
            dt = times[n - 1] - times[n - 2]
            dv = values[n - 1] - values[n - 2]
        else:
            dt = times[i + 1] - times[i - 1]
            dv = values[i + 1] - values[i - 1]
        out.append(dv / dt if abs(dt) > 1e-9 else 0.0)
    return out


# --------------------------------------------------------------------------
# Per-frame kinematics
# --------------------------------------------------------------------------

@dataclass
class Kinematics:
    """Normalised motion state for one track at one observed frame.

    Sign convention follows image coordinates: +y is **downward**, so a falling
    object has positive `vy`. This is stated explicitly because getting it
    backwards silently inverts every drop detection.
    """

    frame: int
    timestamp: float
    point: TrackPoint

    vx: float = 0.0  # heights/s, +right
    vy: float = 0.0  # heights/s, +down
    speed: float = 0.0  # heights/s
    ax: float = 0.0  # heights/s^2
    ay: float = 0.0  # heights/s^2, +downward acceleration
    jerk: float = 0.0  # heights/s^2, rate of change of |speed|

    ground_clearance: float = 0.0  # heights above the floor plane; 0 == resting
    aspect_excess: float = 0.0  # (aspect - median aspect) / median aspect

    @property
    def is_descending(self) -> bool:
        return self.vy > 0.0


@dataclass
class TrackFeatures:
    """A track plus its computed kinematics, indexed for fast lookup."""

    track: Track
    kinematics: List[Kinematics] = field(default_factory=list)
    reference_height: float = 1.0
    floor_y: float = 0.0

    def __len__(self) -> int:
        return len(self.kinematics)

    @property
    def track_id(self) -> int:
        return self.track.track_id

    @property
    def class_name(self) -> str:
        return self.track.class_name

    def by_frame(self) -> Dict[int, Kinematics]:
        return {k.frame: k for k in self.kinematics}

    def at(self, frame: int) -> Optional[Kinematics]:
        for k in self.kinematics:
            if k.frame == frame:
                return k
        return None

    def peak(self, attr: str) -> float:
        if not self.kinematics:
            return 0.0
        return max(getattr(k, attr) for k in self.kinematics)

    def usable(self, thresholds: Thresholds = DEFAULT_THRESHOLDS) -> bool:
        """Whether this track can support a kinematic claim at all.

        A track with 3 sightings scattered across 40 frames is not evidence of
        anything, and saying so here is cheaper than filtering false events
        downstream.
        """
        return (
            len(self.kinematics) >= thresholds.min_track_points
            and self.track.continuity() >= thresholds.min_track_continuity
        )


def compute_kinematics(
    track: Track,
    context: SceneContext,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    floor_y: Optional[float] = None,
) -> TrackFeatures:
    """Smooth, differentiate and normalise one track."""
    points = track.points
    ref_h = max(1e-6, track.median_height)
    floor = floor_y if floor_y is not None else float(context.height)

    features = TrackFeatures(
        track=track, reference_height=ref_h, floor_y=floor,
    )

    if not points:
        return features

    times = [p.timestamp for p in points]
    # Degenerate timestamps (all zero) fall back to frame-derived time.
    if max(times) - min(times) < 1e-9 and len(points) > 1:
        spf = context.seconds_per_frame
        times = [p.frame * spf for p in points]

    win = max(1, thresholds.smoothing_window)
    xs = moving_average([p.cx for p in points], win)
    ys = moving_average([p.cy for p in points], win)

    # px/s -> heights/s
    vx_px = central_difference(xs, times)
    vy_px = central_difference(ys, times)
    vx = [v / ref_h for v in vx_px]
    vy = [v / ref_h for v in vy_px]

    # Light smoothing of velocity before the second derivative, otherwise a
    # single mis-sized box produces a huge phantom acceleration.
    vx_s = moving_average(vx, win)
    vy_s = moving_average(vy, win)
    ax = central_difference(vx_s, times)
    ay = central_difference(vy_s, times)

    speeds = [math.hypot(a, b) for a, b in zip(vx, vy)]
    jerk = central_difference(moving_average(speeds, win), times)

    aspects = [p.aspect_ratio for p in points]
    median_aspect = sorted(aspects)[len(aspects) // 2] if aspects else 1.0
    median_aspect = max(1e-6, median_aspect)

    for i, p in enumerate(points):
        features.kinematics.append(
            Kinematics(
                frame=p.frame,
                timestamp=times[i],
                point=p,
                vx=vx[i],
                vy=vy[i],
                speed=speeds[i],
                ax=ax[i],
                ay=ay[i],
                jerk=jerk[i],
                ground_clearance=(floor - p.ground_y) / ref_h,
                aspect_excess=(p.aspect_ratio - median_aspect) / median_aspect,
            )
        )

    return features


def estimate_floor_y(
    tracks: Sequence[Track],
    context: SceneContext,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> float:
    """Estimate the y of the floor plane from where objects come to rest.

    Uses a high percentile of observed bottom edges rather than the maximum:
    one badly-sized box at the frame edge would otherwise push the floor off
    the bottom of the image and make every object permanently 'airborne'.

    People are included as a fallback because they stand on the same floor, but
    cargo is preferred - a worker's feet are frequently cut off by the frame.
    """
    cargo_bottoms = [
        p.ground_y for t in tracks if t.is_cargo for p in t.points
    ]
    if len(cargo_bottoms) >= 5:
        return percentile(cargo_bottoms, thresholds.floor_percentile)

    person_bottoms = [
        p.ground_y for t in tracks if t.is_person for p in t.points
    ]
    if len(person_bottoms) >= 5:
        return percentile(person_bottoms, thresholds.floor_percentile)

    all_bottoms = [p.ground_y for t in tracks for p in t.points]
    if all_bottoms:
        return percentile(all_bottoms, thresholds.floor_percentile)

    # Nothing observed: assume the floor sits at the conventional horizon used
    # by the synthetic generator (65% down the frame).
    return context.height * 0.85


def build_features(
    tracks: Sequence[Track],
    context: SceneContext,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> Dict[int, TrackFeatures]:
    """Compute kinematics for every track against a shared floor estimate."""
    floor = context.floor_y if context.floor_y is not None else estimate_floor_y(
        tracks, context, thresholds
    )
    context.floor_y = floor
    return {
        t.track_id: compute_kinematics(t, context, thresholds, floor_y=floor)
        for t in tracks
    }


def nearest_person_gap(
    point: TrackPoint,
    frame: int,
    person_tracks: Sequence[Track],
    reference_height: float,
) -> Optional[float]:
    """Edge-to-edge distance to the closest worker, in object-heights.

    None when no worker was detected in that frame - which is meaningfully
    different from 'a worker was far away', and the risk engine treats it so.
    """
    best: Optional[float] = None
    for t in person_tracks:
        p = t.at_frame(frame)
        if p is None:
            continue
        gap = point.gap_to(p) / max(1e-6, reference_height)
        if best is None or gap < best:
            best = gap
    return best
