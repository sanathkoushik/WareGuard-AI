"""
WareGuard AI - AI Assistant package (Phase 5)
LLM Q&A interface over warehouse event logs.

Standalone from the vision stack: works from a saved events_*.json log, so a
supervisor can ask about last night's shift without a GPU or a live video
pipeline running. Falls back to a heuristic responder with no LLM configured,
so it works with nothing installed beyond the standard library too.

Typical use:

    from assistant import WarehouseAssistant, load_context_from_events_log

    context = load_context_from_events_log("data/logs/events_sim_demo.json")
    assistant = WarehouseAssistant(context)
    print(assistant.ask("What was the worst event this shift?"))
"""
from .client import LLMClient
from .context import InvalidEventsLog, build_context, load_context_from_events_log
from .qa import WarehouseAssistant, is_grounded

__all__ = [
    "WarehouseAssistant",
    "LLMClient",
    "build_context",
    "load_context_from_events_log",
    "InvalidEventsLog",
    "is_grounded",
]
