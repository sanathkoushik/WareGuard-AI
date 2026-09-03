"""
WareGuard AI - Behavior Engine (Phase 2)

Runs every detector over a set of tracks, resolves the overlaps between them,
and returns a clean, ordered event list.

Overlap resolution is the part worth reading. A single mishandled carton fires
several detectors at once - a thrown box is also falling, and it is also being
handled abruptly. Reporting all three would triple the incident count and
destroy a supervisor's trust in the numbers. So overlapping claims on the same
track over the same moment collapse to the most specific one, and the
suppressed alternatives are preserved on the survivor rather than discarded, so
nothing is silently lost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .detectors import (
    ALL_EVENT_TYPES,
    DETECTOR_CLASSES,
    EVENT_DRAG,
    EVENT_DROP,
    EVENT_IMPROPER_STACK,
    EVENT_ROUGH_HANDLING,
    EVENT_THROW,
    BehaviorDetector,
)
from .features import build_features
from .schema import (
    BehaviorEvent,
    SceneContext,
    Track,
    load_tracks_from_csv,
    load_tracks_from_json,
)
from .thresholds import DEFAULT_THRESHOLDS, PROFILES, Thresholds

# Higher wins when two claims describe the same moment of the same object.
# A throw is a more specific statement than a drop, which is more specific
# than "handled abruptly".
EVENT_PRECEDENCE: Dict[str, int] = {
    EVENT_THROW: 50,
    EVENT_DROP: 40,
    EVENT_IMPROPER_STACK: 30,
    EVENT_DRAG: 20,
    EVENT_ROUGH_HANDLING: 10,
}


@dataclass
class BehaviorReport:
    """Everything Phase 2 produces, ready for Phase 3 to score."""

    events: List[BehaviorEvent] = field(default_factory=list)
    context: SceneContext = field(default_factory=SceneContext)
    thresholds: Thresholds = field(default_factory=lambda: DEFAULT_THRESHOLDS)

    total_tracks: int = 0
    cargo_tracks: int = 0
    person_tracks: int = 0
    usable_tracks: int = 0
    suppressed_events: int = 0

    @property
    def event_counts(self) -> Dict[str, int]:
        counts = {t: 0 for t in ALL_EVENT_TYPES}
        for e in self.events:
            counts[e.event_type] = counts.get(e.event_type, 0) + 1
        return counts

    def by_type(self, event_type: str) -> List[BehaviorEvent]:
        return [e for e in self.events if e.event_type == event_type]

    def by_track(self, track_id: int) -> List[BehaviorEvent]:
        return [e for e in self.events if e.track_id == track_id]

    def data_quality_warning(self) -> Optional[str]:
        """A human-readable warning when the tracks cannot support analysis.

        This exists because silence is ambiguous: zero events should mean "the
        handling was clean", not "the detector never saw anything". Those two
        cases look identical in an empty list, and conflating them is how a
        safety tool ends up trusted when it should not be.
        """
        if self.total_tracks == 0:
            return "No tracks were supplied - nothing to analyse."
        if self.cargo_tracks == 0:
            return (
                f"{self.total_tracks} tracks supplied but none were classified as "
                "cargo. Check the class mapping in config.WAREHOUSE_CLASSES."
            )
        if self.usable_tracks == 0:
            return (
                f"{self.cargo_tracks} cargo tracks found, but none survived long "
                f"enough to analyse (need >= {self.thresholds.min_track_points} "
                f"sightings at >= {self.thresholds.min_track_continuity:.0%} "
                "continuity). The tracker is losing objects between frames, so "
                "an empty event list here means 'no usable data', not 'no risk'."
            )
        if self.usable_tracks < max(1, self.cargo_tracks // 4):
            return (
                f"Only {self.usable_tracks} of {self.cargo_tracks} cargo tracks were "
                "usable; results are partial."
            )
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene": {
                "video_name": self.context.video_name,
                "fps": self.context.fps,
                "resolution": [self.context.width, self.context.height],
                "floor_y": self.context.floor_y,
                "source": self.context.source,
            },
            "summary": {
                "total_events": len(self.events),
                "event_counts": self.event_counts,
                "total_tracks": self.total_tracks,
                "cargo_tracks": self.cargo_tracks,
                "person_tracks": self.person_tracks,
                "usable_tracks": self.usable_tracks,
                "suppressed_events": self.suppressed_events,
                "data_quality_warning": self.data_quality_warning(),
            },
            "thresholds": self.thresholds.to_dict(),
            "events": [e.to_dict() for e in self.events],
        }


class BehaviorEngine:
    """Phase 2 entry point."""

    def __init__(
        self,
        thresholds: Thresholds | str = DEFAULT_THRESHOLDS,
        detectors: Optional[Sequence[BehaviorDetector]] = None,
    ):
        if isinstance(thresholds, str):
            if thresholds not in PROFILES:
                raise ValueError(
                    f"Unknown threshold profile '{thresholds}'. "
                    f"Choose from: {', '.join(PROFILES)}"
                )
            thresholds = PROFILES[thresholds]

        self.thresholds = thresholds
        self.detectors = list(detectors) if detectors is not None else [
            cls(thresholds) for cls in DETECTOR_CLASSES
        ]

    # ------------------------------------------------------------------ run

    def analyze(
        self, tracks: Sequence[Track], context: Optional[SceneContext] = None
    ) -> BehaviorReport:
        context = context or SceneContext()
        features_map = build_features(tracks, context, self.thresholds)
        person_tracks = [t for t in tracks if t.is_person]

        raw: List[BehaviorEvent] = []
        for detector in self.detectors:
            raw.extend(detector.detect(features_map, context, person_tracks))

        kept = [e for e in raw if e.confidence >= self.thresholds.min_event_confidence]
        confidence_filtered = len(raw) - len(kept)

        resolved = self._resolve_overlaps(kept)
        overlap_suppressed = len(kept) - len(resolved)

        resolved.sort(key=lambda e: (e.start_time, e.track_id, e.event_type))
        for n, e in enumerate(resolved, start=1):
            e.event_id = f"EVT-{n:04d}"

        cargo = [t for t in tracks if t.is_cargo]
        usable = sum(
            1 for t in cargo
            if t.track_id in features_map and features_map[t.track_id].usable(self.thresholds)
        )

        return BehaviorReport(
            events=resolved,
            context=context,
            thresholds=self.thresholds,
            total_tracks=len(tracks),
            cargo_tracks=len(cargo),
            person_tracks=len(person_tracks),
            usable_tracks=usable,
            suppressed_events=confidence_filtered + overlap_suppressed,
        )

    # ------------------------------------------------------- convenience IO

    def analyze_csv(
        self, csv_path: Path | str, context: Optional[SceneContext] = None
    ) -> BehaviorReport:
        tracks = load_tracks_from_csv(csv_path)
        ctx = context or SceneContext(video_name=Path(csv_path).stem, source="yolo")
        return self.analyze(tracks, ctx)

    def analyze_json(self, json_path: Path | str) -> BehaviorReport:
        tracks, context = load_tracks_from_json(json_path)
        return self.analyze(tracks, context)

    # ---------------------------------------------------------- overlap fix

    def _resolve_overlaps(self, events: Sequence[BehaviorEvent]) -> List[BehaviorEvent]:
        """Collapse competing claims about the same object at the same time.

        Two events conflict when they share a track and their time ranges
        overlap. The winner is the more specific behavior (by precedence), and
        confidence breaks a tie. The loser's type is recorded on the winner as
        `also_matched`, so a reviewer can still see that the moment tripped
        several signatures.
        """
        ordered = sorted(
            events,
            key=lambda e: (
                -EVENT_PRECEDENCE.get(e.event_type, 0),
                -e.confidence,
                e.start_time,
            ),
        )

        kept: List[BehaviorEvent] = []
        for candidate in ordered:
            conflict = None
            for winner in kept:
                if winner.track_id != candidate.track_id:
                    continue
                if self._overlaps(winner, candidate):
                    conflict = winner
                    break

            if conflict is None:
                kept.append(candidate)
            else:
                also = conflict.metrics.get("also_matched_count", 0.0)
                conflict.metrics["also_matched_count"] = also + 1.0
                note = f"also matched {candidate.event_type} ({candidate.confidence:.2f})"
                conflict.description = (
                    f"{conflict.description}; {note}"
                    if note not in conflict.description else conflict.description
                )

        return kept

    @staticmethod
    def _overlaps(a: BehaviorEvent, b: BehaviorEvent) -> bool:
        """Inclusive time-range intersection, with a small tolerance.

        The tolerance matters: a drop's impact frame and a rough-handling spike
        are the same physical instant but can land one frame apart after
        smoothing, and without slack they would both survive.
        """
        tol = 0.10
        return not (a.end_time + tol < b.start_time or b.end_time + tol < a.start_time)


def analyze_tracks(
    tracks: Sequence[Track],
    context: Optional[SceneContext] = None,
    profile: str = "default",
) -> BehaviorReport:
    """One-line convenience wrapper used by the dashboard and the CLI."""
    return BehaviorEngine(thresholds=profile).analyze(tracks, context)
