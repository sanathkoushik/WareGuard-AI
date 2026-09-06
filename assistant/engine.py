"""
WareGuard AI - Safety Assistant Engine (Phase 5)

Answers a supervisor's questions about an analysed shift using only the
Phase 2/3 event data.

Three guarantees, in priority order:

**1. Nothing is invented.** Every figure in every answer is read from the
payload. Where a field is absent the answer says so rather than substituting a
plausible value. There is no code path that produces an incident, score, track
or timestamp that is not in the data.

**2. Unreliable data is never dressed up as safety.** When the analysis is
`blocked` or `unavailable`, every answer leads with that. "No events were found"
and "no events could be measured" are different statements, and conflating them
is the failure mode that would make this tool dangerous.

**3. Recommendations are derived, not imagined.** The "what should I focus on"
answer names the actual highest-priority event and its actual scoring factors.
It never offers generic warehouse advice that the data does not support.

The optional LLM layer rephrases these answers; it never produces them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .context import (
    STATE_BLOCKED,
    STATE_CLEAN,
    STATE_PARTIAL,
    STATE_UNAVAILABLE,
    InvalidEventsPayload,
    ShiftContext,
    label_for_type,
)
from .intents import (
    INTENT_COUNT,
    INTENT_DATA_QUALITY,
    INTENT_EVENT_TYPES,
    INTENT_FOCUS,
    INTENT_HIGHEST_RISK,
    INTENT_LIST_EVENTS,
    INTENT_MOST_DANGEROUS,
    INTENT_REPEAT_OFFENDER,
    INTENT_SUMMARY,
    INTENT_TIME_QUERY,
    INTENT_UNKNOWN,
    INTENT_WHY_SEVERITY,
    Question,
    parse_question,
)
from .llm import LLMClient

SUPPORTED_QUESTIONS = [
    "What were the most dangerous incidents?",
    "Why was the shift rated Critical?",
    "Which track had the most incidents?",
    "What types of unsafe behavior occurred?",
    "What should the supervisor focus on?",
    "How many critical incidents occurred?",
    "What happened around 8 seconds?",
    "Which event had the highest risk score?",
    "Can I trust this analysis?",
    "List all events.",
]

# Prefixed to every answer when the data cannot support a safety conclusion.
_UNRELIABLE_PREFIX = (
    "WARNING: Reliable conclusions cannot be drawn from this footage - the tracking "
    "data was insufficient for behavior analysis. This is NOT evidence that the "
    "shift was safe."
)


@dataclass
class Answer:
    """One response, with the provenance needed to audit it."""

    text: str
    intent: str
    question: str = ""
    state: str = STATE_UNAVAILABLE
    reliable: bool = True
    supporting_event_ids: List[str] = field(default_factory=list)
    used_llm: bool = False
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "intent": self.intent,
            "text": self.text,
            "state": self.state,
            "reliable": self.reliable,
            "supporting_event_ids": self.supporting_event_ids,
            "used_llm": self.used_llm,
            "notes": self.notes,
        }


class SafetyAssistant:
    """Deterministic Q&A over one analysed shift."""

    def __init__(
        self,
        context: ShiftContext,
        llm: Optional[LLMClient] = None,
        use_llm: bool = True,
    ):
        self.context = context
        self.llm = llm if llm is not None else LLMClient()
        self.use_llm = use_llm

    # ------------------------------------------------------------ factories

    @classmethod
    def from_file(cls, path: Path | str, **kwargs) -> "SafetyAssistant":
        return cls(ShiftContext.from_file(path), **kwargs)

    @classmethod
    def from_payload(cls, payload: Any, **kwargs) -> "SafetyAssistant":
        return cls(ShiftContext.from_payload(payload), **kwargs)

    # ----------------------------------------------------------------- ask

    def ask(self, question: str) -> Answer:
        """Answer one question. Never raises on unrecognised input."""
        parsed = parse_question(question)
        ctx = self.context
        state = ctx.state

        handler = {
            INTENT_HIGHEST_RISK: self._answer_highest_risk,
            INTENT_MOST_DANGEROUS: self._answer_most_dangerous,
            INTENT_WHY_SEVERITY: self._answer_why_severity,
            INTENT_REPEAT_OFFENDER: self._answer_repeat_offender,
            INTENT_EVENT_TYPES: self._answer_event_types,
            INTENT_COUNT: self._answer_count,
            INTENT_TIME_QUERY: self._answer_time_query,
            INTENT_FOCUS: self._answer_focus,
            INTENT_SUMMARY: self._answer_summary,
            INTENT_DATA_QUALITY: self._answer_data_quality,
            INTENT_LIST_EVENTS: self._answer_list_events,
        }.get(parsed.intent)

        if handler is None:
            return self._finalize(self._answer_unknown(parsed), parsed)

        # Data quality outranks the question. The only intent allowed to run
        # normally on unreliable data is the one asking *about* that data.
        if not ctx.is_reliable and parsed.intent != INTENT_DATA_QUALITY:
            return self._finalize(self._answer_unreliable(parsed), parsed)

        return self._finalize(handler(parsed), parsed)

    def _finalize(self, answer: Answer, parsed: Question) -> Answer:
        answer.question = parsed.raw
        answer.state = self.context.state
        answer.reliable = self.context.is_reliable

        if self.use_llm and self.llm is not None and self.llm.enabled:
            rephrased = self.llm.rephrase(
                parsed.raw, self.context.facts(), answer.text
            )
            if rephrased:
                answer.text = rephrased
                answer.used_llm = True
            elif self.llm.last_error:
                answer.notes.append(f"llm: {self.llm.last_error}")

        return answer

    # ------------------------------------------------------- fallback paths

    def _answer_unknown(self, parsed: Question) -> Answer:
        """Say plainly that the question is not supported. Never guess."""
        lines: List[str] = []

        # An unrecognised question asked against unanalysable data still needs
        # the caveat. Without it, "I can't answer that" reads as a limitation of
        # the assistant rather than of the footage, and a supervisor could take
        # the silence for reassurance.
        if not self.context.is_reliable:
            lines += [_UNRELIABLE_PREFIX, ""]

        lines += [
            "I can't answer that from this shift's data.",
            "",
            "I can answer questions like:",
        ]
        lines += [f"  - {q}" for q in SUPPORTED_QUESTIONS[:6]]
        return Answer(
            text="\n".join(lines),
            intent=INTENT_UNKNOWN,
            notes=["question not recognised"],
        )

    def _answer_unreliable(self, parsed: Question) -> Answer:
        """The blocked/unavailable path. Refuses to answer as if data existed."""
        ctx = self.context
        lines = [_UNRELIABLE_PREFIX, ""]

        if ctx.state == STATE_UNAVAILABLE:
            lines.append(
                "No analysis is loaded for this shift. Generate one with:"
            )
            lines.append(
                "  python run_analysis.py --logs data/logs/detections_<video>.json"
            )
            return Answer(
                text="\n".join(lines),
                intent=parsed.intent,
                notes=["no payload"],
            )

        if ctx.data_quality_warning:
            lines.append(f"Reported issue: {ctx.data_quality_warning}")

        usable, cargo, total = ctx.track_quality
        if usable is not None and cargo is not None:
            lines.append(
                f"Tracking quality: {usable} of {cargo} cargo tracks were usable"
                + (f" ({total} tracks total)." if total is not None else ".")
            )

        lines += [
            "",
            "Behavior detection measures fall speed, drag distance and "
            "acceleration across continuous tracks. When tracks survive only a "
            "few frames there is nothing to measure, so no events can be "
            "produced - regardless of what actually happened in the footage.",
            "",
            "Treat this shift as UNKNOWN. Better source footage or detection "
            "tuning is needed before any safety conclusion is possible.",
        ]
        return Answer(
            text="\n".join(lines),
            intent=parsed.intent,
            notes=["analysis blocked"],
        )

    def _no_events_answer(self, parsed: Question) -> Answer:
        """Genuine zero-incident result - distinct from blocked."""
        ctx = self.context
        duration = ctx.duration_seconds
        window = f" across {duration / 60.0:.1f} min of footage" if duration else ""
        text = (
            f"No unsafe handling was detected{window}.\n\n"
            "Tracking data was sufficient to analyse, and no drop, throw, drag, "
            "improper stack or rough handling met the detection thresholds. "
            "This is a genuine zero-incident result, not a data problem."
        )
        return Answer(text=text, intent=parsed.intent, notes=["zero events"])

    # --------------------------------------------------------- answer paths

    def _answer_highest_risk(self, parsed: Question) -> Answer:
        ctx = self.context
        event = ctx.highest_risk_event()
        if event is None:
            return self._no_events_answer(parsed)

        lines = [f"Highest risk score: {ctx.describe_event(event)}"]
        if event.get("description"):
            lines.append(f"  {event['description']}")

        factors = ctx.factors_for(event)
        if factors:
            lines.append(f"  Scored on: {'; '.join(factors)}")

        confidence = event.get("confidence")
        if confidence is not None:
            try:
                lines.append(f"  Detection confidence: {float(confidence):.0%}")
            except (TypeError, ValueError):
                pass

        top = ctx.top_priority_event()
        if top is not None and top.get("event_id") != event.get("event_id"):
            lines.append("")
            lines.append(
                f"Note: the highest *priority* event for review is "
                f"{ctx.describe_event(top)} - priority weighs risk by detection "
                f"confidence."
            )

        return Answer(
            text="\n".join(lines),
            intent=parsed.intent,
            supporting_event_ids=[str(event.get("event_id", ""))],
        )

    def _answer_most_dangerous(self, parsed: Question) -> Answer:
        ctx = self.context
        events = ctx.events
        if not events:
            return self._no_events_answer(parsed)

        top = events[: min(3, len(events))]
        lines = [
            f"Most dangerous incidents ({len(top)} of {len(events)}, "
            "highest priority first):",
            "",
        ]
        for i, event in enumerate(top, start=1):
            lines.append(f"{i}. {ctx.describe_event(event)}")
            if event.get("description"):
                lines.append(f"   {event['description']}")

        critical = ctx.events_by_severity.get("Critical", 0)
        high = ctx.events_by_severity.get("High", 0)
        if critical or high:
            lines.append("")
            counts = []
            if critical:
                counts.append(f"{critical} Critical")
            if high:
                counts.append(f"{high} High")
            lines.append(f"Severity breakdown: {', '.join(counts)}.")

        return Answer(
            text="\n".join(lines),
            intent=parsed.intent,
            supporting_event_ids=[str(e.get("event_id", "")) for e in top],
        )

    def _answer_why_severity(self, parsed: Question) -> Answer:
        ctx = self.context
        severity = ctx.shift_severity
        index = ctx.shift_risk_index

        if severity is None and index is None:
            return Answer(
                text="This shift has no recorded severity rating in the data.",
                intent=parsed.intent,
                notes=["no severity in payload"],
            )
        if ctx.total_events == 0:
            return self._no_events_answer(parsed)

        lines: List[str] = []
        if index is not None and severity:
            lines.append(f"Shift risk index: {index:.0f}/100 - rated {severity}.")
        elif severity:
            lines.append(f"Shift severity: {severity}.")
        else:
            lines.append(f"Shift risk index: {index:.0f}/100.")

        by_type = ctx.events_by_type
        if by_type:
            contributors = ", ".join(
                f"{label_for_type(t)} x{n}" for t, n in list(by_type.items())[:4]
            )
            lines.append(f"Main contributors: {contributors}.")

        by_sev = ctx.events_by_severity
        if by_sev:
            lines.append(
                "Severity mix: "
                + ", ".join(f"{n} {s}" for s, n in by_sev.items())
                + "."
            )

        worst = ctx.highest_risk_event()
        supporting: List[str] = []
        if worst is not None:
            supporting.append(str(worst.get("event_id", "")))
            lines.append("")
            lines.append(f"Highest-risk incident: {ctx.describe_event(worst)}")
            if worst.get("description"):
                lines.append(f"  {worst['description']}")
            factors = ctx.factors_for(worst)
            if factors:
                lines.append(f"  Scored on: {'; '.join(factors)}")

        if ctx.max_risk_score is not None:
            lines.append("")
            lines.append(
                f"The shift rating is driven mainly by its worst single event "
                f"({ctx.max_risk_score:.0f}/100), not by an average - one "
                f"critical incident is not diluted by a quiet period."
            )

        if ctx.events_per_minute is not None and not ctx.rate_is_reliable:
            lines.append(
                f"Event rate ({ctx.events_per_minute:.1f}/min) is extrapolated "
                f"from a short clip and is indicative only."
            )

        if ctx.state == STATE_PARTIAL and ctx.data_quality_warning:
            lines.append("")
            lines.append(f"WARNING: Partial analysis: {ctx.data_quality_warning}")

        return Answer(
            text="\n".join(lines),
            intent=parsed.intent,
            supporting_event_ids=supporting,
        )

    def _answer_repeat_offender(self, parsed: Question) -> Answer:
        ctx = self.context

        if parsed.track_id is not None:
            events = ctx.events_for_track(parsed.track_id)
            if not events:
                return Answer(
                    text=(
                        f"Track #{parsed.track_id} has no recorded incidents in "
                        f"this shift."
                    ),
                    intent=parsed.intent,
                    notes=["track not found in events"],
                )
            lines = [
                f"Track #{parsed.track_id} - {len(events)} "
                f"incident{'s' if len(events) != 1 else ''}:"
            ]
            for e in events:
                lines.append(f"  - {ctx.describe_event(e)}")
            return Answer(
                text="\n".join(lines),
                intent=parsed.intent,
                supporting_event_ids=[str(e.get("event_id", "")) for e in events],
            )

        if ctx.total_events == 0:
            return self._no_events_answer(parsed)

        counts = ctx.track_incident_counts()
        if not counts:
            return Answer(
                text="No track identifiers are recorded on this shift's events.",
                intent=parsed.intent,
                notes=["no track ids"],
            )

        busiest = ctx.busiest_track()
        track_id, count = busiest  # type: ignore[misc]

        lines: List[str] = []
        if count == 1:
            # Do not manufacture a "repeat offender" when every load had one
            # incident - saying so would misrepresent the pattern.
            lines.append(
                f"No load was involved in more than one incident - all "
                f"{len(counts)} tracks with events had exactly one each."
            )
        else:
            lines.append(
                f"Track #{track_id} was involved in the most incidents "
                f"({count})."
            )
            for e in ctx.events_for_track(track_id):
                lines.append(f"  - {ctx.describe_event(e)}")

        repeats = ctx.repeat_offender_tracks
        if repeats:
            lines.append("")
            lines.append(
                "Loads involved in more than one event: "
                + ", ".join(f"#{t}" for t in repeats)
            )

        lines.append("")
        lines.append(
            "Note: these are object track IDs, not worker identities - "
            "WareGuard does not identify people."
        )

        return Answer(
            text="\n".join(lines),
            intent=parsed.intent,
            supporting_event_ids=[
                str(e.get("event_id", "")) for e in ctx.events_for_track(track_id)
            ],
        )

    def _answer_event_types(self, parsed: Question) -> Answer:
        ctx = self.context
        by_type = ctx.events_by_type
        if not by_type:
            return self._no_events_answer(parsed)

        lines = ["Unsafe handling types detected this shift:", ""]
        for event_type, count in by_type.items():
            examples = ctx.events_of_type(event_type)
            worst = max(
                examples, key=lambda e: float(e.get("risk_score") or 0.0)
            ) if examples else None
            line = f"  - {label_for_type(event_type)} x{count}"
            if worst is not None and worst.get("severity"):
                line += f" (worst: {worst['severity']}"
                if worst.get("risk_score") is not None:
                    try:
                        line += f", risk {float(worst['risk_score']):.0f}"
                    except (TypeError, ValueError):
                        pass
                line += ")"
            lines.append(line)

        return Answer(text="\n".join(lines), intent=parsed.intent)

    def _answer_count(self, parsed: Question) -> Answer:
        ctx = self.context

        if parsed.severity:
            matches = ctx.events_of_severity(parsed.severity)
            noun = f"{parsed.severity} incident"
            text = (
                f"{len(matches)} {noun}{'s' if len(matches) != 1 else ''} "
                f"occurred in this shift."
            )
            if matches:
                text += "\n\n" + "\n".join(
                    f"  - {ctx.describe_event(e)}" for e in matches
                )
            return Answer(
                text=text,
                intent=parsed.intent,
                supporting_event_ids=[str(e.get("event_id", "")) for e in matches],
            )

        if parsed.event_type:
            matches = ctx.events_of_type(parsed.event_type)
            label = label_for_type(parsed.event_type).lower()
            text = (
                f"{len(matches)} {label} "
                f"event{'s' if len(matches) != 1 else ''} occurred in this shift."
            )
            if matches:
                text += "\n\n" + "\n".join(
                    f"  - {ctx.describe_event(e)}" for e in matches
                )
            return Answer(
                text=text,
                intent=parsed.intent,
                supporting_event_ids=[str(e.get("event_id", "")) for e in matches],
            )

        total = ctx.total_events
        if total == 0:
            return self._no_events_answer(parsed)

        lines = [f"{total} safety event{'s' if total != 1 else ''} in total."]
        by_sev = ctx.events_by_severity
        if by_sev:
            lines.append(
                "By severity: " + ", ".join(f"{n} {s}" for s, n in by_sev.items()) + "."
            )
        by_type = ctx.events_by_type
        if by_type:
            lines.append(
                "By type: "
                + ", ".join(f"{label_for_type(t)} x{n}" for t, n in by_type.items())
                + "."
            )
        return Answer(text="\n".join(lines), intent=parsed.intent)

    def _answer_time_query(self, parsed: Question) -> Answer:
        ctx = self.context
        if parsed.timestamp is None:
            return Answer(
                text=(
                    "I couldn't tell which moment you meant. Try naming a time, "
                    "for example: \"what happened around 8 seconds?\""
                ),
                intent=parsed.intent,
                notes=["no timestamp parsed"],
            )

        t = parsed.timestamp
        if ctx.total_events == 0:
            return self._no_events_answer(parsed)

        window = 3.0
        matches = ctx.events_near(t, window)
        if not matches:
            duration = ctx.duration_seconds
            text = (
                f"No events were recorded within +/-{window:.0f}s of {t:.1f}s."
            )
            if duration is not None:
                if t > duration:
                    text += (
                        f"\n\nNote: {t:.1f}s is beyond the end of this footage "
                        f"({duration:.1f}s)."
                    )
                else:
                    text += f" The clip runs {duration:.1f}s in total."
            return Answer(text=text, intent=parsed.intent, notes=["no events in window"])

        lines = [
            f"{len(matches)} event{'s' if len(matches) != 1 else ''} within "
            f"+/-{window:.0f}s of {t:.1f}s:",
            "",
        ]
        for e in matches:
            lines.append(f"  - {ctx.describe_event(e)}")
            if e.get("description"):
                lines.append(f"    {e['description']}")

        return Answer(
            text="\n".join(lines),
            intent=parsed.intent,
            supporting_event_ids=[str(e.get("event_id", "")) for e in matches],
        )

    def _answer_focus(self, parsed: Question) -> Answer:
        """Recommendations derived strictly from the recorded events."""
        ctx = self.context
        events = ctx.events
        if not events:
            return self._no_events_answer(parsed)

        top = events[0]
        lines = [
            f"Start with {ctx.describe_event(top)}.",
        ]
        if top.get("description"):
            lines.append(f"  {top['description']}")
        factors = ctx.factors_for(top)
        if factors:
            lines.append(f"  Why it ranks first: {'; '.join(factors)}")

        # The recurring failure mode is the second thing to act on, and only
        # when the data actually shows one.
        by_type = ctx.events_by_type
        recurring = [(t, n) for t, n in by_type.items() if n > 1]
        if recurring:
            lines.append("")
            lines.append(
                "Recurring pattern: "
                + ", ".join(f"{label_for_type(t)} x{n}" for t, n in recurring)
                + " - a repeated failure is usually a process issue rather than "
                "a one-off."
            )

        repeats = ctx.repeat_offender_tracks
        if repeats:
            lines.append("")
            lines.append(
                "Loads involved in multiple incidents: "
                + ", ".join(f"#{t}" for t in repeats)
                + "."
            )

        # Worker-proximity is the one escalation worth calling out explicitly,
        # and only for events that actually recorded it.
        near_worker = []
        for e in events:
            gap = (e.get("metrics") or {}).get("nearest_person_heights")
            try:
                if gap is not None and float(gap) <= 1.2:
                    near_worker.append(e)
            except (TypeError, ValueError):
                continue
        if near_worker:
            lines.append("")
            lines.append(
                f"{len(near_worker)} event{'s' if len(near_worker) != 1 else ''} "
                f"occurred with a worker within reach of the load - these carry "
                f"injury risk, not just damage risk:"
            )
            for e in near_worker[:3]:
                lines.append(f"  - {ctx.describe_event(e)}")

        if ctx.state == STATE_PARTIAL and ctx.data_quality_warning:
            lines.append("")
            lines.append(
                f"WARNING: Coverage is incomplete ({ctx.data_quality_warning}) - absence "
                f"of an event in a period is not proof nothing happened."
            )

        return Answer(
            text="\n".join(lines),
            intent=parsed.intent,
            supporting_event_ids=[str(top.get("event_id", ""))],
        )

    def _answer_summary(self, parsed: Question) -> Answer:
        ctx = self.context
        if ctx.total_events == 0:
            return self._no_events_answer(parsed)

        lines: List[str] = []
        severity = ctx.shift_severity
        index = ctx.shift_risk_index
        duration = ctx.duration_seconds

        headline = f"{ctx.total_events} safety event{'s' if ctx.total_events != 1 else ''}"
        if severity and index is not None:
            headline = f"{severity} risk shift ({index:.0f}/100): {headline}"
        elif severity:
            headline = f"{severity} risk shift: {headline}"
        if duration:
            headline += f" over {duration / 60.0:.1f} min"
        lines.append(headline + ".")

        by_sev = ctx.events_by_severity
        if by_sev:
            lines.append(
                "Severity: " + ", ".join(f"{n} {s}" for s, n in by_sev.items()) + "."
            )
        by_type = ctx.events_by_type
        if by_type:
            lines.append(
                "Types: "
                + ", ".join(f"{label_for_type(t)} x{n}" for t, n in by_type.items())
                + "."
            )

        top = ctx.top_priority_event()
        if top is not None:
            lines.append("")
            lines.append(f"Review first: {ctx.describe_event(top)}")
            if top.get("description"):
                lines.append(f"  {top['description']}")

        if ctx.state == STATE_PARTIAL and ctx.data_quality_warning:
            lines.append("")
            lines.append(f"WARNING: Partial analysis: {ctx.data_quality_warning}")

        return Answer(
            text="\n".join(lines),
            intent=parsed.intent,
            supporting_event_ids=[str(top.get("event_id", ""))] if top else [],
        )

    def _answer_data_quality(self, parsed: Question) -> Answer:
        """The one intent that runs on unreliable data, since it is about it."""
        ctx = self.context
        state = ctx.state

        if state == STATE_UNAVAILABLE:
            return Answer(
                text=(
                    "No analysis is loaded, so nothing can be said about this "
                    "shift.\n\nGenerate an analysis with:\n"
                    "  python run_analysis.py --logs data/logs/detections_<video>.json"
                ),
                intent=parsed.intent,
                notes=["no payload"],
            )

        if state == STATE_BLOCKED:
            return self._answer_unreliable(parsed)

        usable, cargo, total = ctx.track_quality
        lines: List[str] = []

        if state == STATE_PARTIAL:
            lines.append(
                "WARNING: Partially reliable. Events reported are real, but coverage "
                "is incomplete - absence of an event is not proof that nothing "
                "happened."
            )
            if ctx.data_quality_warning:
                lines.append(f"Reported issue: {ctx.data_quality_warning}")
        elif state == STATE_CLEAN:
            lines.append(
                "Reliable. Tracking data was sufficient to analyse and no "
                "unsafe handling was detected - a genuine zero-incident result."
            )
        else:
            lines.append(
                "Reliable. Tracking data was sufficient to analyse and events "
                "were measured from continuous tracks."
            )

        if usable is not None and cargo is not None:
            lines.append(
                f"Tracking quality: {usable} of {cargo} cargo tracks usable"
                + (f" ({total} total)." if total is not None else ".")
            )

        if ctx.events_per_minute is not None and not ctx.rate_is_reliable:
            lines.append(
                f"Event rate ({ctx.events_per_minute:.1f}/min) is extrapolated "
                f"from a short clip - indicative only."
            )

        return Answer(text="\n".join(lines), intent=parsed.intent)

    def _answer_list_events(self, parsed: Question) -> Answer:
        ctx = self.context
        events = ctx.events

        if parsed.severity:
            events = [e for e in events if e.get("severity") == parsed.severity]
        if parsed.event_type:
            events = [e for e in events if e.get("event_type") == parsed.event_type]

        if not events:
            if parsed.severity or parsed.event_type:
                what = parsed.severity or label_for_type(parsed.event_type)
                return Answer(
                    text=f"No {what} events were recorded in this shift.",
                    intent=parsed.intent,
                )
            return self._no_events_answer(parsed)

        lines = [
            f"{len(events)} event{'s' if len(events) != 1 else ''}, "
            "highest priority first:",
            "",
        ]
        for e in events:
            lines.append(f"  - {ctx.describe_event(e)}")
            if e.get("description"):
                lines.append(f"    {e['description']}")

        return Answer(
            text="\n".join(lines),
            intent=parsed.intent,
            supporting_event_ids=[str(e.get("event_id", "")) for e in events],
        )


def answer_question(
    events_path: Path | str, question: str, use_llm: bool = True
) -> Answer:
    """One-shot convenience helper used by the CLI and the dashboard."""
    try:
        assistant = SafetyAssistant.from_file(events_path, use_llm=use_llm)
    except InvalidEventsPayload as exc:
        return Answer(
            text=(
                f"Could not read the analysis: {exc}\n\n"
                "No conclusions can be drawn about this shift."
            ),
            intent=INTENT_UNKNOWN,
            state=STATE_UNAVAILABLE,
            reliable=False,
            question=question,
            notes=["invalid payload"],
        )
    return assistant.ask(question)
