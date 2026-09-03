"""
WareGuard AI - Behavior Thresholds (Phase 2)

Every tunable in the behavior engine lives here, and every one of them is
expressed in **object-heights per second** or **seconds** - never in pixels or
frames.

Why that matters: a 1080p camera and a 720p camera watching the same dropped
carton produce completely different pixel velocities. Normalising by the
object's own median bounding-box height cancels the pixel scale, and dividing
by the frame interval cancels the frame rate. The same numbers below then hold
for the synthetic clip, a 4K dock camera, and a 15fps CCTV feed.

`Thresholds` is a dataclass rather than module constants so the dashboard can
expose these as sliders and the engine can be run with several profiles.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class Thresholds:
    # ---------------------------------------------------------------- global
    smoothing_window: int = 3
    """Frames in the moving average applied to centres before differentiating.
    3 kills detector jitter without blunting a real impact spike."""

    min_track_points: int = 4
    """Below this a track cannot support any kinematic claim at all."""

    min_track_continuity: float = 0.30
    """Fraction of frames in the track's span it must actually be seen in.
    Guards against a 'track' that is really four unrelated flickers."""

    # ------------------------------------------------------------------ drop
    drop_min_peak_fall_speed: float = 1.20
    """Heights/s downward at the fastest point of the fall. A 40cm carton
    falling freely for 0.25s reaches ~2.4 m/s == 6 heights/s, so 1.2 is a
    deliberately permissive floor that still excludes a controlled set-down."""

    drop_min_fall_distance: float = 0.60
    """Heights of net downward travel. Stops a wobble being called a drop."""

    drop_min_fall_duration: float = 0.10
    """Seconds of sustained descent before the fall is considered real."""

    drop_min_gravity_ratio: float = 0.30
    """Peak downward acceleration as a fraction of g (computed per scene by
    SceneContext.gravity_in_heights_per_s2). Full free fall is 1.0; a carton
    sliding off a stack accelerates at a fraction of g and still counts."""

    drop_impact_speed_ratio: float = 0.40
    """Impact is declared when speed collapses to this fraction of its peak."""

    drop_impact_window: float = 0.20
    """Seconds after peak speed within which that collapse must happen. A
    sudden stop is an impact; a gradual slowdown is a controlled placement."""

    # ------------------------------------------------------------------ drag
    drag_min_horizontal_speed: float = 0.35
    """Heights/s sideways - slow enough to catch a shove, fast enough to
    exclude a box jittering in place."""

    drag_max_vertical_speed: float = 0.25
    """Heights/s vertically. Dragging is motion *along* the floor; anything
    lifting or falling faster than this is a different behavior."""

    drag_floor_contact_tolerance: float = 0.35
    """Heights between the object's bottom edge and the estimated floor plane
    for it to count as 'on the ground'."""

    drag_min_duration: float = 0.60
    """Seconds. A brief slide is not a drag; sustained floor travel is."""

    drag_min_distance: float = 1.00
    """Heights of horizontal travel over the episode."""

    # ----------------------------------------------------------------- throw
    throw_min_horizontal_speed: float = 0.90
    """Heights/s. This is what separates a throw from a drop - a dropped box
    goes straight down, a thrown box carries horizontal momentum."""

    throw_min_airborne_duration: float = 0.15
    """Seconds off the floor plane."""

    throw_min_launch_speed: float = 1.50
    """Heights/s total speed at release."""

    throw_min_gravity_ratio: float = 0.30
    """Median vertical acceleration during flight, as a fraction of g.

    This is the discriminator that makes 'throw' mean something. Horizontal
    speed at height is not enough - a *carried* box and a *shoved* box both
    move fast, high off the floor, and would otherwise read as throws. The
    difference is support: a thrown box is in free flight and accelerates
    downward at g, while a box still in someone's hands accelerates vertically
    at roughly zero. Median (not peak) is used so the landing frame, where the
    fall abruptly stops, cannot distort the measurement.
    """

    # ------------------------------------------------------------- stacking
    stack_vertical_gap_tolerance: float = 0.30
    """Heights between the upper box's bottom and the lower box's top for the
    two to be considered stacked at all."""

    stack_min_horizontal_overlap: float = 0.20
    """Fraction of the lower box's width that must overlap horizontally."""

    stack_max_overhang_ratio: float = 0.30
    """Centre offset as a fraction of the lower box's width. Above this the
    upper carton's centre of mass is heading past its support - the standard
    'improper stack' failure. 0.30 is conservative; warehouse guidance
    typically flags overhang beyond a third of the footprint."""

    stack_min_duration: float = 0.50
    """Seconds the misaligned stack must persist. Filters the moment a box is
    passing over another on its way somewhere else."""

    stack_min_lean_ratio: float = 0.18
    """Growth in width/height aspect versus the object's own median, used as a
    proxy for tilt when no rotation is available from an axis-aligned box."""

    # ------------------------------------------------- rough handling / jerk
    rough_min_speed: float = 1.30
    """Heights/s peak speed during the episode."""

    rough_min_jerk: float = 12.00
    """Heights/s^2 change in speed. Rough handling is defined by abruptness,
    not by top speed - a fast smooth carry is fine, a sharp yank is not.

    Calibrated against the two reference behaviors rather than guessed: a
    controlled cosine set-down peaks near 6 h/s^2, and a deliberate shove
    (1600 px/s^2 at a 40cm box scale) sustains ~23. Sitting at 12 leaves
    roughly a factor of two of headroom on both sides. An earlier value of 6
    sat directly on top of careful placement and flagged the control case."""

    rough_person_proximity: float = 1.50
    """Heights between cargo and the nearest worker for the motion to be
    attributed to handling rather than to a forklift or a conveyor."""

    rough_min_duration: float = 0.10

    # ------------------------------------------------ person proximity/scene
    person_proximity_heights: float = 1.20
    """Used by the risk engine: cargo this close to a worker at the moment of
    an event escalates severity, because a dropped box near a person is an
    injury risk and not just a damaged-goods risk."""

    floor_percentile: float = 0.90
    """Quantile of cargo bottom-edges used to estimate the floor plane. The
    highest bottom-edge is unreliable (one bad box ruins it); the 90th
    percentile is the floor as far as the goods are concerned."""

    # --------------------------------------------------------------- output
    min_event_confidence: float = 0.35
    """Events below this are dropped rather than shown to a supervisor."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Thresholds":
        """Build from a partial dict; unknown keys are ignored."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


DEFAULT_THRESHOLDS = Thresholds()


# Sensitive profile: for footage where the camera is far away, motion is small,
# and recall matters more than precision (supervisor triages the list anyway).
SENSITIVE_THRESHOLDS = Thresholds(
    drop_min_peak_fall_speed=0.80,
    drop_min_fall_distance=0.40,
    drop_min_gravity_ratio=0.20,
    drag_min_horizontal_speed=0.25,
    drag_min_duration=0.40,
    drag_min_distance=0.60,
    rough_min_jerk=8.00,
    min_event_confidence=0.25,
    min_track_points=3,
)

# Strict profile: for a clean, close camera where a false alarm costs trust.
STRICT_THRESHOLDS = Thresholds(
    drop_min_peak_fall_speed=1.80,
    drop_min_fall_distance=0.90,
    drop_min_gravity_ratio=0.45,
    drag_min_duration=1.00,
    drag_min_distance=1.50,
    rough_min_jerk=16.00,
    min_event_confidence=0.55,
    min_track_continuity=0.50,
)

PROFILES = {
    "default": DEFAULT_THRESHOLDS,
    "sensitive": SENSITIVE_THRESHOLDS,
    "strict": STRICT_THRESHOLDS,
}
