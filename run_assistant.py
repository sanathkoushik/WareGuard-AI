"""
WareGuard AI - Supervisor Q&A CLI (Phase 5)

Ask natural-language questions about a shift's scored events, from a saved
log or straight from a simulated scenario (no video needed, same as
run_analysis.py).

Examples
--------
    # Ask one question about an existing events log and exit
    python run_assistant.py --logs data/logs/events_sim_demo.json --ask "What was the worst event?"

    # Interactive session over the same log
    python run_assistant.py --logs data/logs/events_sim_demo.json

    # Ask about a fresh simulated shift, no log file needed
    python run_assistant.py --simulate demo --ask "Any repeat offenders?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from assistant import (
    InvalidEventsLog,
    WarehouseAssistant,
    load_context_from_events_log,
)
from behavior import BehaviorEngine
from behavior.simulation import SimConfig, build_demo_scene, build_scenario
from risk import RiskEngine
from risk.export import assessment_to_assistant_context

SCENARIOS = ["demo", "normal_carry", "drop", "drag", "throw", "improper_stack", "rough_handling"]


def _context_from_simulation(scenario: str, seed: int) -> Dict[str, Any]:
    cfg = SimConfig(seed=seed)
    tracks, ctx = build_demo_scene(cfg) if scenario == "demo" else build_scenario(scenario, cfg)
    report = BehaviorEngine().analyze(tracks, ctx)
    assessment = RiskEngine().assess(report)
    return assessment_to_assistant_context(assessment)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="WareGuard AI - ask questions about a shift's scored events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--logs", help="events_*.json written by risk.export.export_events_json")
    source.add_argument("--simulate", choices=SCENARIOS,
                        help="Ask about a simulated shift instead of a saved log")
    parser.add_argument("--seed", type=int, default=7, help="Simulation RNG seed")
    parser.add_argument("--ask", help="Ask one question and exit (default: interactive)")

    args = parser.parse_args(argv)

    if args.simulate:
        context = _context_from_simulation(args.simulate, args.seed)
    else:
        path = Path(args.logs)
        if not path.exists():
            print(f"error: log file not found: {path}", file=sys.stderr)
            print(f"       generate one with: python run_analysis.py --logs "
                  f"data/logs/detections_<video>.json", file=sys.stderr)
            return 2
        try:
            context = load_context_from_events_log(path)
        except InvalidEventsLog as exc:
            # A truncated or hand-edited log should tell the operator what is
            # wrong, not dump a traceback at them mid-shift.
            print(f"error: {exc}", file=sys.stderr)
            return 2

    assistant = WarehouseAssistant(context)

    if not assistant.client.available:
        print(
            "[offline mode - no WAREGUARD_LLM_API_KEY/OPENAI_API_KEY set, "
            "using the built-in heuristic responder]",
            file=sys.stderr,
        )

    if args.ask:
        print(assistant.ask(args.ask))
        return 0

    print(f"WareGuard AI assistant - {context['shift']['headline']}")
    print("Ask a question, or 'exit' to quit.")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            return 0
        print(assistant.ask(question))


if __name__ == "__main__":
    raise SystemExit(main())
