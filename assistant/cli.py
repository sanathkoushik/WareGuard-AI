"""
WareGuard AI - Assistant CLI (Phase 5)

Ask questions about an analysed shift from the terminal.

    # interactive
    python -m assistant.cli --events data/logs/events_sim_demo.json

    # one-shot
    python -m assistant.cli --events data/logs/events_sim_demo.json \
        --ask "why was the shift rated critical?"

    # run the built-in demo question set (useful for a live walkthrough)
    python -m assistant.cli --events data/logs/events_sim_demo.json --demo

Works with no API key. Set WAREGUARD_LLM_API_KEY to enable optional LLM
phrasing; `--no-llm` disables it even when a key is present.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .context import InvalidEventsPayload, ShiftContext
from .engine import SUPPORTED_QUESTIONS, SafetyAssistant
from .llm import LLMConfig

_RULE = "=" * 72


def _print_answer(question: str, answer) -> None:
    print()
    print(f"Q: {question}")
    print("-" * 72)
    print(answer.text)
    if answer.notes:
        print()
        print(f"   [{'; '.join(answer.notes)}]")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="WareGuard AI - ask questions about an analysed shift",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--events", "-e", required=True,
        help="Path to an events_<stem>.json produced by run_analysis.py",
    )
    parser.add_argument("--ask", "-a", help="Ask one question and exit")
    parser.add_argument(
        "--demo", action="store_true",
        help="Run the built-in supervisor question set",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Force deterministic answers even if an API key is configured",
    )
    args = parser.parse_args(argv)

    try:
        context = ShiftContext.from_file(args.events)
    except InvalidEventsPayload as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    assistant = SafetyAssistant(context, use_llm=not args.no_llm)

    print()
    print(_RULE)
    print(" WareGuard AI - Safety Assistant")
    print(_RULE)
    print(f" shift    : {context.video_name or Path(args.events).stem}")
    print(f" events   : {context.total_events}")
    print(f" state    : {context.state}"
          f"{'' if context.is_reliable else '  (conclusions NOT reliable)'}")
    if context.shift_severity and context.shift_risk_index is not None:
        print(f" risk     : {context.shift_risk_index:.0f}/100 "
              f"({context.shift_severity})")
    config = LLMConfig.from_env() if not args.no_llm else LLMConfig()
    print(f" phrasing : {config.describe()}")
    print(_RULE)

    if args.ask:
        _print_answer(args.ask, assistant.ask(args.ask))
        return 0

    if args.demo:
        for question in SUPPORTED_QUESTIONS:
            _print_answer(question, assistant.ask(question))
        return 0

    print()
    print("Ask a question, or type 'help' for examples. Ctrl-C or 'quit' to exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            return 0
        if question.lower() in {"help", "?"}:
            print("\nExamples:")
            for q in SUPPORTED_QUESTIONS:
                print(f"  - {q}")
            continue
        _print_answer(question, assistant.ask(question))


if __name__ == "__main__":
    raise SystemExit(main())
