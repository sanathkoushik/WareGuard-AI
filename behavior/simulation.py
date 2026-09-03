"""
WareGuard AI - Ground-Truth Track Simulator (Phase 2)

Emits `Track` objects directly, with no video and no detector in the loop.

Why this exists: the behavior engine is only as good as the tracks it is fed,
and on the current synthetic clip yolov8n produces 4-frame tracks with zero
velocity. Validating drop/drag/throw logic against that data is impossible -
not because the logic is wrong, but because there is no motion in the input to
read. This module supplies motion that is correct by construction, so a failing
detector is unambiguously a bug in the detector.

The kinematics are real, not hand-waved. Falls integrate 9.81 m/s^2 through the
pixel scale implied by the box size, so a simulated drop and a filmed drop
produce the same normalised signature - which is the only reason tuning against
simulated data transfers to real footage at all.

`noise_px` and `dropout` deliberately degrade the output so the detectors can be
tested against the flickering, jittery tracks a real detector actually produces.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .schema import SceneContext, Track, tracks_from_rows

GRAVITY_MPS2 = 9.81


@dataclass
class SimConfig:
    fps: float = 30.0
    width: int = 1280
    height: int = 720
    floor_y: float = 620.0
    box_w: float = 90.0
    box_h: float = 70.0
    person_w: float = 70.0
    person_h: float = 200.0
    cargo_height_m: float = 0.40

    noise_px: float = 0.0
    """Uniform jitter added to every bbox corner, in pixels. Real detectors
    wobble by 2-5px on a stationary object."""

    dropout: float = 0.0
    """Probability that any given frame's detection is missing entirely."""

    seed: int = 7

    @property
    def px_per_m(self) -> float:
        return self.box_h / self.cargo_height_m

    @property
    def gravity_px(self) -> float:
        """Free-fall acceleration in px/s^2 for this scene's scale."""
        return GRAVITY_MPS2 * self.px_per_m

    @property
    def spf(self) -> float:
        return 1.0 / self.fps


class SceneBuilder:
    """Accumulates detection rows, then hands back tracks + context."""

    def __init__(self, config: Optional[SimConfig] = None):
        self.cfg = config or SimConfig()
        self.rows: List[Dict[str, Any]] = []
        self._rng = random.Random(self.cfg.seed)

    # ------------------------------------------------------------- plumbing

    def emit(
        self,
        frame: int,
        track_id: int,
        class_name: str,
        cx: float,
        cy: float,
        w: float,
        h: float,
        confidence: float = 0.92,
        allow_dropout: bool = True,
    ) -> None:
        """Record one sighting, applying configured noise and dropout."""
        if allow_dropout and self.cfg.dropout > 0.0:
            if self._rng.random() < self.cfg.dropout:
                return

        j = self.cfg.noise_px
        def jitter() -> float:
            return self._rng.uniform(-j, j) if j > 0 else 0.0

        x1 = cx - w / 2.0 + jitter()
        y1 = cy - h / 2.0 + jitter()
        x2 = cx + w / 2.0 + jitter()
        y2 = cy + h / 2.0 + jitter()

        self.rows.append({
            "frame": frame,
            "timestamp": frame * self.cfg.spf,
            "track_id": track_id,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": (x1, y1, x2, y2),
        })

    def build(self, video_name: str = "simulated_scene") -> Tuple[List[Track], SceneContext]:
        last_frame = max((r["frame"] for r in self.rows), default=0)
        context = SceneContext(
            fps=self.cfg.fps,
            width=self.cfg.width,
            height=self.cfg.height,
            floor_y=self.cfg.floor_y,
            source="ground_truth",
            video_name=video_name,
            assumed_cargo_height_m=self.cfg.cargo_height_m,
            duration_seconds=(last_frame + 1) * self.cfg.spf,
        )
        return tracks_from_rows(self.rows), context

    # ------------------------------------------------------------ scenarios

    def add_worker(
        self,
        track_id: int,
        start_frame: int,
        end_frame: int,
        x_at: Callable[[float], float],
        confidence: float = 0.88,
    ) -> None:
        """A worker standing or walking, feet planted on the floor plane."""
        cfg = self.cfg
        for frame in range(start_frame, end_frame + 1):
            t = (frame - start_frame) * cfg.spf
            cx = x_at(t)
            cy = cfg.floor_y - cfg.person_h / 2.0
            self.emit(frame, track_id, "person", cx, cy,
                      cfg.person_w, cfg.person_h, confidence)

    def add_normal_carry(
        self, track_id: int, start_frame: int, x_start: float = 200.0,
        x_end: float = 700.0, duration_s: float = 2.5,
    ) -> int:
        """Control case: a box carried at waist height and set down gently.

        This must produce **zero** events. A detector that fires here is
        useless in a real warehouse, where careful handling is the common case.
        """
        cfg = self.cfg
        n = int(duration_s * cfg.fps)
        carry_cy = cfg.floor_y - 220.0
        rest_cy = cfg.floor_y - cfg.box_h / 2.0

        for i in range(n + 1):
            t = i / cfg.fps
            p = i / max(1, n)
            cx = x_start + (x_end - x_start) * p
            if p < 0.65:
                cy = carry_cy
            else:
                # Smooth cosine ease-down: decelerating placement, not a fall.
                q = (p - 0.65) / 0.35
                cy = carry_cy + (rest_cy - carry_cy) * (1 - math.cos(q * math.pi)) / 2.0
            self.emit(start_frame + i, track_id, "box", cx, cy, cfg.box_w, cfg.box_h)

        return start_frame + n

    def add_drop(
        self, track_id: int, start_frame: int, cx: float = 500.0,
        hold_s: float = 0.4, release_height_px: float = 300.0,
        settle_s: float = 0.8,
    ) -> int:
        """Held, released, free fall under gravity, hard stop on the floor."""
        cfg = self.cfg
        frame = start_frame
        rest_cy = cfg.floor_y - cfg.box_h / 2.0
        start_cy = cfg.floor_y - release_height_px - cfg.box_h / 2.0

        for _ in range(int(hold_s * cfg.fps)):
            self.emit(frame, track_id, "box", cx, start_cy, cfg.box_w, cfg.box_h)
            frame += 1

        fall_px = rest_cy - start_cy
        fall_time = math.sqrt(max(0.0, 2.0 * fall_px / cfg.gravity_px))
        n_fall = max(2, int(fall_time * cfg.fps))
        for i in range(1, n_fall + 1):
            t = i / cfg.fps
            cy = min(rest_cy, start_cy + 0.5 * cfg.gravity_px * t * t)
            self.emit(frame, track_id, "box", cx, cy, cfg.box_w, cfg.box_h)
            frame += 1

        # Landed and motionless - this is the abrupt stop the detector needs.
        for _ in range(int(settle_s * cfg.fps)):
            self.emit(frame, track_id, "box", cx, rest_cy, cfg.box_w, cfg.box_h)
            frame += 1

        return frame - 1

    def add_drag(
        self, track_id: int, start_frame: int, x_start: float = 300.0,
        x_end: float = 800.0, duration_s: float = 2.0,
    ) -> int:
        """Box pushed along the floor: horizontal travel, no lift."""
        cfg = self.cfg
        n = int(duration_s * cfg.fps)
        cy = cfg.floor_y - cfg.box_h / 2.0
        for i in range(n + 1):
            p = i / max(1, n)
            cx = x_start + (x_end - x_start) * p
            self.emit(start_frame + i, track_id, "box", cx, cy, cfg.box_w, cfg.box_h)
        return start_frame + n

    def add_throw(
        self, track_id: int, start_frame: int, x_start: float = 250.0,
        launch_vx: float = 320.0, launch_vy: float = -380.0,
        release_height_px: float = 180.0, settle_s: float = 0.5,
    ) -> int:
        """Ballistic arc: horizontal momentum plus an upward launch.

        Deliberately also satisfies the drop signature, so the engine's
        precedence rule has something real to resolve.
        """
        cfg = self.cfg
        frame = start_frame
        rest_cy = cfg.floor_y - cfg.box_h / 2.0
        start_cy = cfg.floor_y - release_height_px - cfg.box_h / 2.0

        t = 0.0
        cx, cy = x_start, start_cy
        while cy < rest_cy and frame - start_frame < int(4 * cfg.fps):
            t += cfg.spf
            cx = x_start + launch_vx * t
            cy = start_cy + launch_vy * t + 0.5 * cfg.gravity_px * t * t
            if cy >= rest_cy:
                cy = rest_cy
            self.emit(frame, track_id, "box", cx, cy, cfg.box_w, cfg.box_h)
            frame += 1

        for _ in range(int(settle_s * cfg.fps)):
            self.emit(frame, track_id, "box", cx, rest_cy, cfg.box_w, cfg.box_h)
            frame += 1

        return frame - 1

    def add_improper_stack(
        self, lower_id: int, upper_id: int, start_frame: int,
        cx: float = 640.0, overhang_ratio: float = 0.45, duration_s: float = 2.5,
    ) -> int:
        """Upper carton resting on a lower one with its centre well past support."""
        cfg = self.cfg
        n = int(duration_s * cfg.fps)
        lower_cy = cfg.floor_y - cfg.box_h / 2.0
        lower_top = cfg.floor_y - cfg.box_h
        upper_cy = lower_top - cfg.box_h / 2.0
        upper_cx = cx + overhang_ratio * cfg.box_w

        for i in range(n + 1):
            f = start_frame + i
            self.emit(f, lower_id, "box", cx, lower_cy, cfg.box_w, cfg.box_h)
            self.emit(f, upper_id, "box", upper_cx, upper_cy, cfg.box_w, cfg.box_h)
        return start_frame + n

    def add_rough_handling(
        self, track_id: int, worker_id: int, start_frame: int,
        cx: float = 400.0, accel_px_s2: float = 1600.0, shove_s: float = 0.30,
    ) -> int:
        """A sharp shove: high jerk at chest height with a worker alongside.

        Held above the floor so it cannot be mistaken for a drag, and kept
        short so it cannot be mistaken for a carry.
        """
        cfg = self.cfg
        n = int(shove_s * cfg.fps)
        cy = cfg.floor_y - 240.0
        frame = start_frame

        # Still, then violently accelerated.
        for _ in range(int(0.4 * cfg.fps)):
            self.emit(frame, track_id, "box", cx, cy, cfg.box_w, cfg.box_h)
            self.emit(frame, worker_id, "person", cx - 90.0,
                      cfg.floor_y - cfg.person_h / 2.0, cfg.person_w, cfg.person_h, 0.88)
            frame += 1

        for i in range(1, n + 1):
            t = i / cfg.fps
            x = cx + 0.5 * accel_px_s2 * t * t
            self.emit(frame, track_id, "box", x, cy, cfg.box_w, cfg.box_h)
            self.emit(frame, worker_id, "person", cx - 90.0,
                      cfg.floor_y - cfg.person_h / 2.0, cfg.person_w, cfg.person_h, 0.88)
            frame += 1

        return frame - 1


# --------------------------------------------------------------------------
# Prebuilt scenes
# --------------------------------------------------------------------------

def build_scenario(name: str, config: Optional[SimConfig] = None) -> Tuple[List[Track], SceneContext]:
    """One isolated behavior, for unit tests and threshold tuning."""
    b = SceneBuilder(config)

    if name == "normal_carry":
        b.add_normal_carry(track_id=1, start_frame=0)
        b.add_worker(2, 0, 80, lambda t: 200.0 + 180.0 * t)
    elif name == "drop":
        b.add_drop(track_id=1, start_frame=0)
        b.add_worker(2, 0, 60, lambda t: 430.0)
    elif name == "drag":
        b.add_drag(track_id=1, start_frame=0)
        b.add_worker(2, 0, 60, lambda t: 260.0 + 240.0 * t)
    elif name == "throw":
        b.add_throw(track_id=1, start_frame=0)
        b.add_worker(2, 0, 60, lambda t: 220.0)
    elif name == "improper_stack":
        b.add_improper_stack(lower_id=1, upper_id=2, start_frame=0)
    elif name == "rough_handling":
        b.add_rough_handling(track_id=1, worker_id=2, start_frame=0)
    else:
        raise ValueError(
            f"Unknown scenario '{name}'. Available: normal_carry, drop, drag, "
            "throw, improper_stack, rough_handling"
        )

    return b.build(video_name=f"sim_{name}")


def build_demo_scene(config: Optional[SimConfig] = None) -> Tuple[List[Track], SceneContext]:
    """A full simulated unloading shift containing every behavior in sequence.

    Each episode is separated in time so the events are individually
    attributable, which makes this the right thing to demo and the right
    thing to regression-test the whole engine against.
    """
    b = SceneBuilder(config)
    cfg = b.cfg
    fps = int(cfg.fps)

    # A worker present throughout, moving between stations.
    end_of_scene = int(16 * fps)
    b.add_worker(100, 0, end_of_scene, lambda t: 250.0 + 60.0 * math.sin(t * 0.6))

    f = 0
    f = b.add_normal_carry(track_id=1, start_frame=f) + fps // 2
    f = b.add_drop(track_id=2, start_frame=f, cx=520.0) + fps // 2
    f = b.add_drag(track_id=3, start_frame=f, x_start=300.0, x_end=820.0) + fps // 2
    f = b.add_throw(track_id=4, start_frame=f, x_start=240.0) + fps // 2
    f = b.add_improper_stack(lower_id=5, upper_id=6, start_frame=f) + fps // 2
    f = b.add_rough_handling(track_id=7, worker_id=101, start_frame=f)

    return b.build(video_name="simulated_unloading_shift")
