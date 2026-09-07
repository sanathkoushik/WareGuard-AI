"""
WareGuard AI - Safety Assistant Panel (Phase 6)

Puts the Phase 5 assistant in the dashboard, so a supervisor can ask about a
shift in the same place they review its events.

Existing behaviour is reused, not reimplemented: this module builds the
assistant's context with `risk.export.assessment_to_assistant_context` and asks
questions through `assistant.WarehouseAssistant`. It contains no question
handling, no scoring and no fallback logic of its own.

Follows the same conventions as `dashboard/events_panel.py`: Streamlit is
imported defensively so the module stays importable and testable without a
Streamlit runtime, and it adds no dependencies.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:  # pragma: no cover - trivial import guard
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:  # pragma: no cover
    st = None  # type: ignore[assignment]
    _HAS_STREAMLIT = False

# Starter questions. Chosen to exercise the answers a supervisor actually
# needs, and to be answerable from event data alone.
PRESET_QUESTIONS: List[str] = [
    "What was the worst event this shift?",
    "How many critical incidents?",
    "What types of unsafe behavior occurred?",
    "Any repeat offenders?",
    "Is this data reliable?",
]

HISTORY_KEY = "wg_assistant_history"
_MAX_HISTORY = 20


def build_context_from_assessment(assessment) -> Optional[Dict[str, Any]]:
    """Compact assistant context from a live RiskAssessment, or None."""
    if assessment is None:
        return None
    try:
        from risk.export import assessment_to_assistant_context

        return assessment_to_assistant_context(assessment)
    except (ImportError, AttributeError, KeyError, TypeError):
        return None


def build_context_from_log(events_path: Path | str) -> Optional[Dict[str, Any]]:
    """Compact assistant context from a saved events log, or None."""
    try:
        from assistant import InvalidEventsLog, load_context_from_events_log

        return load_context_from_events_log(events_path)
    except Exception:
        # InvalidEventsLog covers the expected failures; the broad catch keeps
        # a malformed log from taking down the whole dashboard page.
        return None


def _require_streamlit() -> None:
    if not _HAS_STREAMLIT:
        raise RuntimeError(
            "streamlit is not installed - assistant_panel render functions "
            "need it. Install with: pip install streamlit"
        )


def _ask(assistant, question: str) -> str:
    """Ask, converting any failure into a readable message rather than a crash."""
    try:
        return str(assistant.ask(question))
    except Exception as exc:  # pragma: no cover - defensive
        return (
            f"The assistant could not answer that ({type(exc).__name__}: {exc}). "
            "The event data above is unaffected."
        )


def render_assistant_tab(
    assessment=None,
    events_path: Optional[Path | str] = None,
) -> None:
    """Entry point for the "Safety Assistant" tab.

    Args:
        assessment: a live RiskAssessment from the dashboard, if one was
            computed for the selected video.
        events_path: fallback events_<stem>.json to load when no live
            assessment is available.
    """
    _require_streamlit()

    st.subheader("Safety Assistant")

    context = build_context_from_assessment(assessment)
    source = "the current analysis"
    if context is None and events_path is not None:
        context = build_context_from_log(events_path)
        source = f"`{Path(events_path).name}`"

    if context is None:
        st.info(
            "**No analysis available to ask about yet.**\n\n"
            "Run the detection pipeline from the sidebar, or generate an "
            "events log:\n\n"
            "`python run_analysis.py --logs data/logs/detections_<video>.json`\n\n"
            "You can also explore the bundled simulated shift:\n\n"
            "`python run_assistant.py --logs data/logs/events_sim_demo.json`"
        )
        return

    shift = context.get("shift", {}) or {}
    warning = shift.get("data_quality_warning")
    total_events = shift.get("total_events", 0)

    # Data quality before anything else, matching the events panel. A blocked
    # shift must never be answered as though its silence meant safety.
    if warning and not total_events:
        st.warning(
            "**Answers below are limited - the analysis could not run.**\n\n"
            f"{warning}\n\n"
            "The assistant will report this rather than describe incidents, "
            "because none could be measured. Treat the shift as *unknown*, "
            "not *safe*.",
            icon="⚠️",
        )
    elif warning:
        st.warning(f"**Partial analysis.** {warning}", icon="⚠️")

    try:
        from assistant import WarehouseAssistant

        assistant = WarehouseAssistant(context)
    except ImportError as exc:
        st.error(f"Assistant unavailable: {exc}")
        return

    online = bool(getattr(assistant.client, "available", False))
    st.caption(
        (
            "Connected to a language model for phrasing. Answers are still "
            "checked against the event data, and a reply containing figures "
            "not present in it is discarded."
            if online else
            "Offline mode - no API key configured. Every answer is derived "
            "directly from the event data by the built-in responder."
        )
        + f" Asking about {source}."
    )

    st.markdown("**Common questions**")
    cols = st.columns(len(PRESET_QUESTIONS))
    asked: Optional[str] = None
    for i, question in enumerate(PRESET_QUESTIONS):
        # Short button labels; the full question is the tooltip.
        label = question.split("?")[0].replace("What was the ", "").replace(
            "What types of ", ""
        ).replace("How many ", "").strip().title()
        if cols[i].button(label[:22], key=f"wg_preset_{i}", help=question,
                          use_container_width=True):
            asked = question

    typed = st.chat_input("Ask about this shift...") if hasattr(st, "chat_input") \
        else st.text_input("Ask about this shift", key="wg_assistant_input")
    if typed:
        asked = typed

    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []

    if asked:
        answer = _ask(assistant, asked)
        st.session_state[HISTORY_KEY].insert(0, (asked, answer))
        del st.session_state[HISTORY_KEY][_MAX_HISTORY:]

    history = st.session_state.get(HISTORY_KEY, [])
    if not history:
        st.info("Pick a question above, or type your own.")
        return

    st.markdown("---")
    for question, answer in history:
        st.markdown(f"**Q: {question}**")
        st.markdown(answer)
        st.markdown("")

    if len(history) > 1 and st.button("Clear history", key="wg_assistant_clear"):
        st.session_state[HISTORY_KEY] = []
