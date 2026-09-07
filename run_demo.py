"""
WareGuard AI - One-Command End-to-End Demo (Phase 6)

Runs the whole pipeline in one command and prints each stage as it happens:

    tracks  ->  behavior detection  ->  risk scoring  ->  export  ->  assistant

    python run_demo.py

By default it runs on **simulated ground-truth tracks**, which is the reliable
demo path: no video decode, no YOLO weights, no GPU, and no third-party
packages at all. It exercises the same behavior and risk engines that real
footage would, so what you see is the real analysis stack - only the tracks are
synthetic.

Why simulated by default
------------------------
The bundled `sample_warehouse.mp4` is drawn with OpenCV primitives, and a
COCO-trained YOLOv8 does not recognise flat-shaded rectangles as cargo. On that
clip the detector yields fragmented tracks (roughly 1 usable cargo track in 32),
which is not enough continuous motion to measure a fall speed or a drag
distance from. The behavior engine correctly reports that as *unanalysable*
rather than as a safe shift.

That limitation is real and is not hidden: `--logs` runs this same chain over a
detection log so you can see exactly what the pipeline says about real
detector output.

Examples
--------
    python run_demo.py                          # simulated end-to-end demo
    python run_demo.py --scenario drop          # one isolated behavior
    python run_demo.py --logs data/logs/detections_sample_warehouse.json
    python run_demo.py --report shift_report.md # also write a shift report
    python run_demo.py --noise 4 --dropout 0.2  # degraded tracking
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "data" / "logs"

SCENARIOS = [
    "demo", "normal_carry", "drop", "drag", "throw",
    "improper_stack", "rough_handling",
]

DEMO_QUESTIONS = [
    "What was the worst event this shift?",
    "How many critical incidents?",
    "What types of unsafe behavior occurred?",
    "Any repeat offenders?",
    "Is this data reliable?",
]

_RULE = "=" * 74
_THIN = "-" * 74


def _step(number: int, title: str) -> None:
    print()
    print(f"[{number}/5] {title}")
    print(_THIN)


def _fail(message: str, hint: str = "") -> int:
    print(f"\nERROR: {message}", file=sys.stderr)
    if hint:
        print(f"       {hint}", file=sys.stderr)
    return 2


def build_report(assessment, source_label: str) -> str:
    """A concise Markdown shift report.

    Deliberately a *rendering* of the existing assessment - it computes no new
    figures. JSON and CSV exports already exist in risk.export for machines;
    this is the version a supervisor or a judge reads.
    """
    s = assessment.summary
    lines: List[str] = [
        "# WareGuard AI - Shift Safety Report",
        "",
        f"**Source:** {source_label}",
        f"**Duration:** {s.duration_seconds / 60.0:.1f} min",
        "",
        "## Summary",
        "",
        f"{s.headline()}",
        "",
    ]

    if s.data_quality_warning:
        lines += [
            "> **Data quality warning**",
            f"> {s.data_quality_warning}",
            ">",
            "> Reliable safety conclusions cannot be drawn from this footage.",
            "> This is not evidence that the shift was safe.",
            "",
        ]

    if s.total_events:
        lines += [
            "| Metric | Value |",
            "| --- | --- |",
            f"| Shift risk index | {s.shift_risk_index:.0f}/100 ({s.shift_severity}) |",
            f"| Total events | {s.total_events} |",
            f"| Worst single event | {s.max_risk_score:.0f}/100 |",
            f"| Mean risk score | {s.mean_risk_score:.0f}/100 |",
        ]
        rate = f"{s.events_per_minute:.1f}/min"
        if not s.rate_is_reliable:
            rate += " (extrapolated from a short clip - indicative only)"
        lines.append(f"| Event rate | {rate} |")
        if s.repeat_offender_tracks:
            lines.append(
                "| Repeat-offender loads | "
                + ", ".join(f"#{t}" for t in s.repeat_offender_tracks)
                + " |"
            )
        lines += ["", "## Events (highest priority first)", ""]

        for event in assessment.ranked():
            lines += [
                f"### {event.event_id} - {event.event_type.replace('_', ' ').title()}"
                f" ({event.severity})",
                "",
                f"- **Risk score:** {event.risk_score:.0f}/100",
                f"- **Detection confidence:** {event.confidence:.0%}",
                f"- **When:** {event.start_time:.2f}s - {event.end_time:.2f}s "
                f"(frames {event.start_frame}-{event.end_frame})",
                f"- **Track:** #{event.track_id}",
                f"- **What happened:** {event.description}",
            ]
            if event.risk_factors:
                lines.append(
                    "- **Why this score:** " + "; ".join(event.risk_factors)
                )
            lines.append("")
    else:
        lines += ["No safety events were produced for this shift.", ""]

    lines += [
        "---",
        "",
        "Generated by `run_demo.py`. Risk scores come from the Phase 3 engine; "
        "this report renders them and computes nothing of its own.",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="WareGuard AI - end-to-end demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--scenario", choices=SCENARIOS, default="demo",
        help="Simulated scenario to run (default: demo - all five behaviors)",
    )
    source.add_argument(
        "--logs",
        help="Run on a real detection log instead of simulated tracks",
    )
    parser.add_argument("--profile", default="default",
                        help="Behavior threshold profile: default | sensitive | strict")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Simulation only: bbox jitter in px")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Simulation only: probability a detection is missing")
    parser.add_argument("--report", help="Also write a Markdown shift report here")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write events JSON/CSV")
    parser.add_argument("--quick", action="store_true",
                        help="Skip the assistant Q&A stage")
    args = parser.parse_args(argv)

    print()
    print(_RULE)
    print(" WareGuard AI - End-to-End Demo")
    print(_RULE)

    # ---------------------------------------------------------- 1. tracks
    _step(1, "Source tracks")
    try:
        if args.logs:
            from behavior.schema import load_tracks_from_json

            log_path = Path(args.logs)
            if not log_path.exists():
                return _fail(
                    f"Detection log not found: {log_path}",
                    "Run: python run_detection.py --input data/raw_videos/<video>.mp4",
                )
            tracks, context = load_tracks_from_json(log_path)
            source_label = f"detection log `{log_path.name}` (real YOLO output)"
            print(f"  loaded {len(tracks)} tracks from {log_path.name}")
            print("  NOTE: real detector output - track quality is reported below.")
        else:
            from behavior.simulation import SimConfig, build_demo_scene, build_scenario

            cfg = SimConfig(noise_px=args.noise, dropout=args.dropout)
            if args.scenario == "demo":
                tracks, context = build_demo_scene(cfg)
            else:
                tracks, context = build_scenario(args.scenario, cfg)
            source_label = f"simulated scenario `{args.scenario}` (ground-truth tracks)"
            print(f"  generated {len(tracks)} ground-truth tracks "
                  f"({args.scenario} scenario)")
            if args.noise or args.dropout:
                print(f"  degraded with noise={args.noise}px dropout={args.dropout:.0%}")
            print("  NOTE: tracks are simulated. The behavior and risk engines")
            print("        below are the real ones used on real footage.")
    except ImportError as exc:
        return _fail(f"Could not import the analysis engines: {exc}",
                     "Run this from the repository root.")
    except (OSError, ValueError) as exc:
        return _fail(f"Could not load tracks: {exc}")

    # -------------------------------------------------- 2. behavior engine
    _step(2, "Behavior detection (Phase 2)")
    try:
        from behavior import BehaviorEngine

        report = BehaviorEngine(thresholds=args.profile).analyze(tracks, context)
    except ValueError as exc:
        return _fail(str(exc), "Valid profiles: default, sensitive, strict")

    print(f"  tracks: {report.total_tracks} total, {report.cargo_tracks} cargo, "
          f"{report.person_tracks} worker, {report.usable_tracks} usable")
    print(f"  events detected: {len(report.events)}")
    for event in report.events:
        print(f"    - {event.event_type:15} t={event.start_time:6.2f}s "
              f"conf={event.confidence:.2f}")

    warning = report.data_quality_warning()
    if warning:
        print()
        print("  !! DATA QUALITY")
        print(f"     {warning}")
        print("     Zero events here means 'not measurable', NOT 'no risk'.")

    # ------------------------------------------------------ 3. risk engine
    _step(3, "Risk scoring (Phase 3)")
    from risk import RiskEngine

    assessment = RiskEngine().assess(report)
    summary = assessment.summary
    print(f"  {summary.headline()}")
    if summary.total_events:
        print(f"  risk index : {summary.shift_risk_index:.0f}/100 "
              f"({summary.shift_severity})")
        print(f"  worst event: {summary.max_risk_score:.0f}/100")
        for event in assessment.ranked():
            print(f"    [{event.severity:8}] {event.event_id} "
                  f"risk={event.risk_score:5.1f} {event.description[:52]}")

    # ----------------------------------------------------------- 4. export
    _step(4, "Export (Phase 3)")
    stem = (
        Path(args.logs).stem.replace("detections_", "")
        if args.logs else f"sim_{args.scenario}"
    )
    events_json = None
    if args.no_save:
        print("  skipped (--no-save)")
    else:
        from risk.export import export_events_csv, export_events_json

        try:
            events_json = export_events_json(assessment, LOGS_DIR / f"events_{stem}.json")
            events_csv = export_events_csv(assessment, LOGS_DIR / f"events_{stem}.csv")
            print(f"  wrote {events_json}")
            print(f"  wrote {events_csv}")
        except OSError as exc:
            print(f"  WARNING: could not write exports: {exc}")

    if args.report:
        try:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(build_report(assessment, source_label),
                                   encoding="utf-8")
            print(f"  wrote {report_path}")
        except OSError as exc:
            print(f"  WARNING: could not write report: {exc}")

    # -------------------------------------------------------- 5. assistant
    _step(5, "AI safety assistant (Phase 5)")
    if args.quick:
        print("  skipped (--quick)")
    else:
        try:
            from assistant import WarehouseAssistant
            from risk.export import assessment_to_assistant_context

            assistant = WarehouseAssistant(
                assessment_to_assistant_context(assessment)
            )
            if not assistant.client.available:
                print("  [offline mode - no API key set, using the heuristic")
                print("   responder. Every answer below is derived from the")
                print("   event data above, not from a language model.]")
            print()
            for question in DEMO_QUESTIONS:
                print(f"  Q: {question}")
                for line in str(assistant.ask(question)).splitlines():
                    print(f"     {line}")
                print()
        except ImportError as exc:
            print(f"  WARNING: assistant unavailable: {exc}")

    print(_RULE)
    print(" Demo complete.")
    if events_json:
        print(f" Explore in the dashboard:  streamlit run dashboard/app.py")
        print(f" Ask more questions:        python run_assistant.py --logs {events_json}")
    print(_RULE)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
