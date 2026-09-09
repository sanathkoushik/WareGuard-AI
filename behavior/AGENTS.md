# Behavior Detection Engine

Infers handling behaviors (drop / throw / drag / improper_stack /
rough_handling) from object tracks. Does NOT know what a video is, does NOT
score risk (see `../risk/`) — it only turns tracks into a clean, ordered
`BehaviorEvent` list. Full integration contract: `../docs/behavior-risk.md`.

## Entry Points

- `engine.py` — `BehaviorEngine`, the orchestrator. `analyze()` /
  `analyze_json()` is the public surface; everything else in this package is
  an implementation detail reached through it.
- `schema.py` — the `Track`/`TrackPoint`/`BehaviorEvent` data contract and
  the JSON/CSV loaders. This is the seam: anything producing tracks in this
  shape (YOLOv8+ByteTrack, `simulation.py`, hand-written fixtures) works.
- `simulation.py` — ground-truth track generator; lets the whole engine run
  and be tested with no video or vision stack.
- `thresholds.py` — `Thresholds` dataclass, every field a plain float.
  Profiles: `"default"`, `"sensitive"`, `"strict"`.

## Contracts & Invariants

- Standard-library only. No torch/ultralytics/cv2/numpy/pandas imports in
  this package — see root `AGENTS.md`.
- Every threshold is in **object-heights per second**, never pixels or
  frames — normalized by the object's own median bbox height (cancels pixel
  scale) and elapsed time (cancels frame rate). This is why the same numbers
  work on 720p and 4K. Enforced by `TestScaleInvariance` in
  `../tests/test_behavior.py`.
- Physical speeds (m/s) come from `SceneContext.assumed_cargo_height_m`
  (default 0.40 m), not from a hardcoded pixel scale.
- Overlapping detector claims on the same track/moment collapse to the
  **most specific one**: `throw > drop > stack > drag > rough`. Suppressed
  alternatives are appended to the survivor's `description`, never discarded
  — see the overlap-resolution logic in `engine.py`.

## Detector signatures (the non-obvious parts)

| Behavior | Signature | Gotcha |
|---|---|---|
| `drop` | vertical fall + **abrupt stop** | without the abrupt-stop check, every careful set-down scores as a drop |
| `throw` | fall with horizontal momentum, **unsupported** (vertical accel ≈ g) | horizontal speed alone isn't enough — a carried box and a shoved box both move fast; the difference is free-fall acceleration |
| `drag` | floor contact + horizontal travel + ~no vertical motion | |
| `improper_stack` | box vertically adjacent to another, overhanging its support | |
| `rough_handling` | high speed **and** high jerk, worker in reach | keyed on jerk, not speed — threshold (12 h/s²) sits between a controlled set-down (~6) and a deliberate shove (~23) |

## Patterns

Adding/tuning a detector:
1. New behavior type → add to `DETECTOR_CLASSES` / `ALL_EVENT_TYPES` in
   `detectors.py`, implement as a `BehaviorDetector`.
2. New tunable → add a field to `Thresholds` in `thresholds.py` (plain
   float, object-heights/sec) — sliders/UI can drive it directly.
3. Changed detector priority → update the overlap-resolution order in
   `engine.py`, not in the individual detector.

## Anti-patterns

- Don't add a pixel-space or frame-count threshold — breaks scale invariance
  silently at other resolutions/framerates.
- Don't report a `drop` without an abrupt-stop check, or a `throw` without
  checking vertical acceleration ≈ g — both degrade to false positives on
  ordinary careful handling.
- Don't let overlapping detector claims all surface as separate events —
  route through the engine's overlap resolution.

## Pitfalls

- On `data/logs/detections_sample_warehouse.json`, yolov8n produces mostly
  unusable tracks (46 tracks, longest survives 4 frames, 37 misclassified
  `sports ball`) — the engine will stay quiet on that log. This is a
  detection-layer limitation, not a bug here; use `simulation.py` or real
  footage to exercise this package.

## Related Context

- Root: `../AGENTS.md`
- Risk scoring (consumes `BehaviorEvent`): `../risk/`
- Full API contract (dashboard + assistant integration): `../docs/behavior-risk.md`
