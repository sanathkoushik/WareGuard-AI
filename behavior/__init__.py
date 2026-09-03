"""
WareGuard AI - Behavior Detection Engine (Phase 2)

Infers warehouse handling behaviors - drop, throw, drag, improper stacking and
rough handling - from object tracks.

The package is deliberately source-agnostic and dependency-free: it consumes a
track schema rather than a video, and imports nothing beyond the Python
standard library. That means it runs identically against YOLOv8 output, against
ground-truth tracks from `behavior.simulation`, and against real CCTV, and it
can be unit-tested on a machine with no vision stack installed.

Typical use:

    from behavior import BehaviorEngine
    report = BehaviorEngine().analyze_json("data/logs/detections_clip.json")
    for event in report.events:
        print(event.event_id, event.event_type, event.description)
"""
from .detectors import (
    ALL_EVENT_TYPES,
    EVENT_DRAG,
    EVENT_DROP,
    EVENT_IMPROPER_STACK,
    EVENT_ROUGH_HANDLING,
    EVENT_THROW,
    DragDetector,
    DropDetector,
    ImproperStackDetector,
    RoughHandlingDetector,
    ThrowDetector,
)
from .engine import BehaviorEngine, BehaviorReport, analyze_tracks
from .features import build_features, compute_kinematics, estimate_floor_y
from .schema import (
    BehaviorEvent,
    SceneContext,
    Track,
    TrackPoint,
    load_tracks_from_csv,
    load_tracks_from_json,
    tracks_from_rows,
)
from .simulation import SimConfig, build_demo_scene, build_scenario
from .thresholds import DEFAULT_THRESHOLDS, PROFILES, Thresholds

__all__ = [
    "BehaviorEngine",
    "BehaviorReport",
    "analyze_tracks",
    "BehaviorEvent",
    "SceneContext",
    "Track",
    "TrackPoint",
    "Thresholds",
    "DEFAULT_THRESHOLDS",
    "PROFILES",
    "tracks_from_rows",
    "load_tracks_from_csv",
    "load_tracks_from_json",
    "build_features",
    "compute_kinematics",
    "estimate_floor_y",
    "build_scenario",
    "build_demo_scene",
    "SimConfig",
    "ALL_EVENT_TYPES",
    "EVENT_DROP",
    "EVENT_THROW",
    "EVENT_DRAG",
    "EVENT_IMPROPER_STACK",
    "EVENT_ROUGH_HANDLING",
    "DropDetector",
    "ThrowDetector",
    "DragDetector",
    "ImproperStackDetector",
    "RoughHandlingDetector",
]
