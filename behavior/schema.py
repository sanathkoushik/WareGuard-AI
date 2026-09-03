"""
WareGuard AI - Behavior Schema Module (Phase 2)

The single data contract between *where tracks come from* and *what we infer
from them*. Detection output (YOLOv8 + ByteTrack), ground-truth tracks emitted
by the synthetic generator, and hand-written test fixtures all enter here and
become the same `Track` objects.

Nothing in this package imports torch, ultralytics, cv2, numpy or pandas. The
behavior engine stays runnable and unit-testable on a machine with no vision
stack installed.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Class names that count as handled goods rather than people or scenery.
# ByteTrack/COCO aliases are included because yolov8n reports raw COCO names.
CARGO_CLASSES = {
    "box", "package", "item", "carton", "crate",
    "suitcase", "backpack", "handbag", "sports ball",
}
PERSON_CLASSES = {"person", "worker"}


@dataclass
class TrackPoint:
    """One observation of one object in one frame."""

    frame: int
    timestamp: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float = 1.0
    class_name: str = "unknown"

    # Kinematics as reported upstream. features.py recomputes these with
    # smoothing; these are kept for provenance and as a fallback.
    reported_velocity: Tuple[float, float] = (0.0, 0.0)
    reported_speed: float = 0.0
    reported_acceleration_y: float = 0.0

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(1e-6, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(1e-6, self.bbox[3] - self.bbox[1])

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def center(self) -> Tuple[float, float]:
        return (self.cx, self.cy)

    @property
    def ground_y(self) -> float:
        """Bottom edge - where the object meets the floor plane."""
        return self.bbox[3]

    @property
    def top_y(self) -> float:
        return self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Width / height. A carton that squashes on impact widens this."""
        return self.width / self.height

    def iou(self, other: "TrackPoint") -> float:
        ax1, ay1, ax2, ay2 = self.bbox
        bx1, by1, bx2, by2 = other.bbox
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def gap_to(self, other: "TrackPoint") -> float:
        """Shortest edge-to-edge distance in px; 0 when the boxes overlap."""
        ax1, ay1, ax2, ay2 = self.bbox
        bx1, by1, bx2, by2 = other.bbox
        dx = max(bx1 - ax2, ax1 - bx2, 0.0)
        dy = max(by1 - ay2, ay1 - by2, 0.0)
        return math.hypot(dx, dy)


@dataclass
class Track:
    """The full observed life of one object, ordered by frame."""

    track_id: int
    class_name: str
    points: List[TrackPoint] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    @property
    def is_cargo(self) -> bool:
        return self.class_name.lower() in CARGO_CLASSES

    @property
    def is_person(self) -> bool:
        return self.class_name.lower() in PERSON_CLASSES

    @property
    def first_frame(self) -> int:
        return self.points[0].frame if self.points else -1

    @property
    def last_frame(self) -> int:
        return self.points[-1].frame if self.points else -1

    @property
    def duration_frames(self) -> int:
        if not self.points:
            return 0
        return self.last_frame - self.first_frame + 1

    @property
    def duration_seconds(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].timestamp - self.points[0].timestamp

    @property
    def median_height(self) -> float:
        """Reference scale for this object. Median resists a few bad boxes."""
        if not self.points:
            return 1.0
        hs = sorted(p.height for p in self.points)
        return hs[len(hs) // 2]

    @property
    def median_width(self) -> float:
        if not self.points:
            return 1.0
        ws = sorted(p.width for p in self.points)
        return ws[len(ws) // 2]

    def add(self, point: TrackPoint) -> None:
        self.points.append(point)
        if point.class_name and point.class_name != "unknown":
            self.class_name = point.class_name

    def sort(self) -> None:
        self.points.sort(key=lambda p: p.frame)

    def at_frame(self, frame: int) -> Optional[TrackPoint]:
        """Exact-frame lookup. None when the track was not seen that frame."""
        return self.frame_index().get(frame)

    def window(self, start_frame: int, end_frame: int) -> List[TrackPoint]:
        return [p for p in self.points if start_frame <= p.frame <= end_frame]

    def frame_index(self) -> Dict[int, TrackPoint]:
        return {p.frame: p for p in self.points}

    def has_gaps(self) -> bool:
        """True when the detector lost the object mid-track."""
        return len(self.points) < self.duration_frames

    def continuity(self) -> float:
        """Fraction of frames in the track's span where it was actually seen.

        A low value means the detector was flickering. Behavior confidence is
        scaled by this, so a drop inferred from 3 sightings across 40 frames is
        reported honestly instead of as a certainty.
        """
        if self.duration_frames <= 0:
            return 0.0
        return min(1.0, len(self.points) / self.duration_frames)


@dataclass
class SceneContext:
    """What the detectors need to know about the scene as a whole.

    Thresholds are expressed in object-heights and seconds, never raw pixels,
    so the same numbers hold for 720p and 4K footage at any frame rate.
    """

    fps: float = 30.0
    width: int = 1280
    height: int = 720
    floor_y: Optional[float] = None  # y of the floor plane in px
    source: str = "unknown"  # "yolo" | "ground_truth" | "fixture"
    video_name: str = ""
    duration_seconds: Optional[float] = None
    """Length of the source footage. Used by the risk engine to turn an event
    count into a rate; without it, a 5-second clip and a 5-minute clip with the
    same three events would score identically."""

    # Physical assumption used to convert pixel motion into g-equivalents.
    # A standard warehouse carton is ~40cm tall; override per deployment.
    assumed_cargo_height_m: float = 0.40

    @property
    def seconds_per_frame(self) -> float:
        return 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0

    def gravity_in_heights_per_s2(self) -> float:
        """Free-fall acceleration expressed in object-heights per second^2.

        If a box of height H_m metres appears H_px pixels tall, then
        9.81 m/s^2 == 9.81 * (H_px / H_m) px/s^2 == 9.81 / H_m heights/s^2.
        The pixel scale cancels - which is exactly why these detectors work on
        any resolution without recalibration.
        """
        return 9.81 / max(0.05, self.assumed_cargo_height_m)

    def heights_per_s_to_mps(self, v_heights_per_s: float) -> float:
        """Convert a normalised speed back into metres/second for reporting."""
        return v_heights_per_s * self.assumed_cargo_height_m


@dataclass
class BehaviorEvent:
    """One inferred handling behavior, ready for risk scoring."""

    event_type: str
    track_id: int
    class_name: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float
    metrics: Dict[str, float] = field(default_factory=dict)
    related_track_ids: List[int] = field(default_factory=list)
    description: str = ""
    event_id: str = ""

    # Filled in by the risk engine (Phase 3); left empty by Phase 2.
    risk_score: Optional[float] = None
    severity: Optional[str] = None
    risk_factors: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration_seconds"] = round(self.duration_seconds, 3)
        return d


# --------------------------------------------------------------------------
# Loaders - every input format collapses into List[Track] here.
# --------------------------------------------------------------------------

def tracks_from_rows(rows: Iterable[Dict[str, Any]]) -> List[Track]:
    """Build tracks from detection dicts (pipeline output or CSV rows)."""
    by_id: Dict[int, Track] = {}

    for row in rows:
        try:
            track_id = int(row.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        if track_id < 0:
            continue

        bbox = row.get("bbox")
        if bbox is None:
            bbox = (
                float(row.get("bbox_x1", 0.0)),
                float(row.get("bbox_y1", 0.0)),
                float(row.get("bbox_x2", 0.0)),
                float(row.get("bbox_y2", 0.0)),
            )
        bbox_t: Tuple[float, float, float, float] = (
            float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),
        )

        velocity = row.get("velocity")
        if velocity is None:
            velocity = (
                float(row.get("velocity_x", 0.0)),
                float(row.get("velocity_y", 0.0)),
            )

        class_name = str(row.get("class_name", "unknown"))
        point = TrackPoint(
            frame=int(row.get("frame", 0)),
            timestamp=float(row.get("timestamp", 0.0)),
            bbox=bbox_t,
            confidence=float(row.get("confidence", 1.0)),
            class_name=class_name,
            reported_velocity=(float(velocity[0]), float(velocity[1])),
            reported_speed=float(row.get("speed", 0.0)),
            reported_acceleration_y=float(row.get("acceleration_y", 0.0)),
        )

        if track_id not in by_id:
            by_id[track_id] = Track(track_id=track_id, class_name=class_name)
        by_id[track_id].add(point)

    tracks = list(by_id.values())
    for t in tracks:
        t.sort()
    tracks.sort(key=lambda t: (t.first_frame, t.track_id))
    return tracks


def load_tracks_from_csv(path: Path | str) -> List[Track]:
    """Read a detections_<video>.csv produced by utils/log_exporter.py."""
    with open(Path(path), "r", encoding="utf-8", newline="") as f:
        return tracks_from_rows(list(csv.DictReader(f)))


def load_tracks_from_json(path: Path | str) -> Tuple[List[Track], SceneContext]:
    """Read a detections_<video>.json and recover the scene context with it."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    detections = payload.get("detections", [])
    meta = payload.get("video_metadata", {}) or {}
    resolution = meta.get("resolution") or [1280, 720]

    fps = float(meta.get("fps") or 30.0)
    context = SceneContext(
        fps=fps if fps > 0 else 30.0,
        width=int(resolution[0]),
        height=int(resolution[1]),
        source="yolo",
        video_name=str(meta.get("video_name", path.stem)),
        duration_seconds=(
            float(meta["duration_seconds"]) if meta.get("duration_seconds") else None
        ),
    )
    return tracks_from_rows(detections), context
