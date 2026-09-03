"""
WareGuard AI - Behavior & Risk Analysis CLI (Phases 2 + 3)

Runs the behavior engine and the risk engine over object tracks and writes
scored safety events.

This is deliberately decoupled from run_detection.py. Detection is expensive and
needs torch; behavior analysis is cheap, pure-Python, and re-runnable in
milliseconds. Separating them means thresholds can be re-tuned over an existing
log a hundred times without touching a GPU.

Examples
--------
    # Analyse an existing detection log produced by run_detection.py
    python run_analysis.py --logs data/logs/detections_sample_warehouse.json

    # Prove the engine end-to-end with no video and no vision stack
    python run_analysis.py --simulate demo

    # One isolated behavior, useful when tuning a single detector
    python run_analysis.py --simulate drop --profile sensitive

    # Test resilience against a flickering detector
    python run_analysis.py --simulate demo --noise 4 --dropout 0.2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from behavior import BehaviorEngine, SceneContext, Track
from behavior.schema import load_tracks_from_csv, load_tracks_from_json
from behavior.simulation import SimConfig, build_demo_scene, build_scenario
from behavior.thresholds import PROFILES
from risk import RiskEngine
from risk.export import export_events_csv, export_events_json

LOGS_DIR = Path(__file__).resolve().parent / "data" / "logs"

SCENARIOS = ["demo", "normal_carry", "drop", "drag", "throw", "improper_stack", "rough_handling"]

SEVERITY_MARK = {
    "Critical": "[CRIT]",
    "High": "[HIGH]",
    "Medium": "[MED ]",
    "Low": "[LOW ]",
}


def load_tracks(args) -> Tuple[List[Track], SceneContext, str]:
    """Resolve the CLI arguments into tracks plus a scene context."""
    if args.simulate:
        cfg = SimConfig(noise_px=args.noise, dropout=args.dropout, seed=args.seed)
        if args.simulate == "demo":
            tracks, ctx = build_demo_scene(cfg)
        else:
            tracks, ctx = build_scenario(args.simulate, cfg)
        return tracks, ctx, f"sim_{args.simulate}"

    path = Path(args.logs)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    if path.suffix.lower() == ".json":
        tracks, ctx = load_tracks_from_json(path)
    elif path.suffix.lower() == ".csv":
        tracks = load_tracks_from_csv(path)
        ctx = SceneContext(video_name=path.stem, source="yolo")
    else:
        raise ValueError(f"Expected a .json or .csv detection log, got: {path.suffix}")

    stem = path.stem.replace("detections_", "")
    return tracks, ctx, stem


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="WareGuard AI - behavior detection and risk scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--logs", help="Detection log (.json or .csv) from run_detection.py")
    source.add_argument("--simulate", choices=SCENARIOS,
                        help="Run on ground-truth simulated tracks instead of a video")

    parser.add_argument("--profile", default="default", choices=list(PROFILES),
                        help="Threshold profile (default: default)")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Simulation only: bbox jitter in px")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Simulation only: probability a frame's detection is missing")
    parser.add_argument("--seed", type=int, default=7, help="Simulation RNG seed")
    parser.add_argument("--cargo-height", type=float, default=None,
                        help="Real-world height of a carton in metres (default 0.40)")
    parser.add_argument("--output-dir", default=str(LOGS_DIR),
                        help="Where to write events JSON/CSV")
    parser.add_argument("--no-save", action="store_true", help="Print only, write nothing")
    parser.add_argument("--quiet", action="store_true", help="Summary only, no event list")

    args = parser.parse_args(argv)

    try:
        tracks, context, stem = load_tracks(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.cargo_height:
        context.assumed_cargo_height_m = args.cargo_height

    report = BehaviorEngine(thresholds=args.profile).analyze(tracks, context)
    assessment = RiskEngine().assess(report)
    summary = assessment.summary

    print()
    print("=" * 72)
    print(f" WareGuard AI - {context.video_name or stem}")
    print("=" * 72)
    print(f" source        : {context.source}")
    print(f" profile       : {args.profile}")
    print(f" tracks        : {report.total_tracks} total, {report.cargo_tracks} cargo, "
          f"{report.person_tracks} worker, {report.usable_tracks} usable")
    print(f" floor plane   : y={context.floor_y:.0f}px" if context.floor_y else " floor plane   : n/a")
    print()

    warning = report.data_quality_warning()
    if warning:
        print(" !! DATA QUALITY")
        for line in _wrap(warning, 68):
            print(f"    {line}")
        print()

    print(f" {summary.headline()}")
    if summary.total_events and not summary.rate_is_reliable:
        print(f"    (rate of {summary.events_per_minute:.1f}/min is extrapolated from "
              f"{summary.duration_seconds:.0f}s - treat as indicative only)")
    print()

    if summary.total_events:
        print(f" risk index    : {summary.shift_risk_index:.0f}/100  ({summary.shift_severity})")
        print(f" worst event   : {summary.max_risk_score:.0f}/100")
        by_sev = " ".join(f"{k}={v}" for k, v in summary.events_by_severity.items() if v)
        by_type = " ".join(f"{k}={v}" for k, v in summary.events_by_type.items() if v)
        print(f" by severity   : {by_sev}")
        print(f" by type       : {by_type}")
        if summary.repeat_offender_tracks:
            print(f" repeat loads  : {summary.repeat_offender_tracks}")
        print()

    if summary.total_events and not args.quiet:
        print("-" * 72)
        print(" EVENTS (highest priority first)")
        print("-" * 72)
        for e in assessment.ranked():
            mark = SEVERITY_MARK.get(e.severity or "", "[    ]")
            print(f" {mark} {e.event_id}  {e.event_type:15} "
                  f"risk={e.risk_score:5.1f}  conf={e.confidence:.2f}  "
                  f"t={e.start_time:6.2f}-{e.end_time:6.2f}s")
            for line in _wrap(e.description, 64):
                print(f"         {line}")
            if e.risk_factors:
                print(f"         why: {'; '.join(e.risk_factors)}")
            print()

    if not args.no_save:
        out_dir = Path(args.output_dir)
        json_path = export_events_json(assessment, out_dir / f"events_{stem}.json")
        csv_path = export_events_csv(assessment, out_dir / f"events_{stem}.csv")
        print("-" * 72)
        print(f" wrote {json_path}")
        print(f" wrote {csv_path}")
        print()

    return 0


def _wrap(text: str, width: int) -> List[str]:
    """Minimal word wrap - avoids a textwrap import for two call sites."""
    words, lines, current = text.split(), [], ""
    for w in words:
        if current and len(current) + 1 + len(w) > width:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}".strip()
    if current:
        lines.append(current)
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
