"""
WareGuard AI - Question Understanding (Phase 5)

Maps a supervisor's plain-English question onto one of a fixed set of intents,
plus any entities it mentions (severity, event type, track id, timestamp).

This is deliberately a keyword-scoring classifier rather than a model. A
supervisor asking "how many critical incidents?" must get the same answer every
time, offline, with no API key and no latency. The optional LLM layer sits
*above* this and only rephrases answers that were already derived from the data
- it never performs the classification and never supplies a fact.

Returning `UNKNOWN` is a first-class outcome, not a failure. An honest "I can't
answer that from this data" beats a confident guess.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------- intents

INTENT_HIGHEST_RISK = "highest_risk"
INTENT_MOST_DANGEROUS = "most_dangerous"
INTENT_WHY_SEVERITY = "why_severity"
INTENT_REPEAT_OFFENDER = "repeat_offender"
INTENT_EVENT_TYPES = "event_types"
INTENT_COUNT = "count"
INTENT_TIME_QUERY = "time_query"
INTENT_FOCUS = "focus"
INTENT_SUMMARY = "summary"
INTENT_DATA_QUALITY = "data_quality"
INTENT_LIST_EVENTS = "list_events"
INTENT_UNKNOWN = "unknown"

ALL_INTENTS = [
    INTENT_HIGHEST_RISK,
    INTENT_MOST_DANGEROUS,
    INTENT_WHY_SEVERITY,
    INTENT_REPEAT_OFFENDER,
    INTENT_EVENT_TYPES,
    INTENT_COUNT,
    INTENT_TIME_QUERY,
    INTENT_FOCUS,
    INTENT_SUMMARY,
    INTENT_DATA_QUALITY,
    INTENT_LIST_EVENTS,
]

# Phrase patterns scored per intent. Longer, more specific phrases score higher
# so "how many critical incidents" resolves to COUNT rather than to the generic
# severity match.
_PATTERNS: List[Tuple[str, str, int]] = [
    # --- highest risk / worst single event
    (INTENT_HIGHEST_RISK, r"\bhighest[- ]risk\b", 6),
    (INTENT_HIGHEST_RISK, r"\bhighest\s+(risk\s+)?(score|event|incident)\b", 6),
    (INTENT_HIGHEST_RISK, r"\bworst\s+(single\s+)?(event|incident)\b", 5),
    (INTENT_HIGHEST_RISK, r"\bmaximum\s+risk\b", 5),
    (INTENT_HIGHEST_RISK, r"\btop\s+risk\b", 4),

    # --- most dangerous (plural / general)
    (INTENT_MOST_DANGEROUS, r"\bmost\s+dangerous\b", 6),
    (INTENT_MOST_DANGEROUS, r"\bmost\s+(serious|severe|critical)\b", 5),
    (INTENT_MOST_DANGEROUS, r"\bdangerous\s+incidents?\b", 4),
    (INTENT_MOST_DANGEROUS, r"\bbiggest\s+(risk|danger|problem)s?\b", 4),

    # --- why this severity
    (INTENT_WHY_SEVERITY, r"\bwhy\b.*\b(rated|scored|classified|considered)\b", 7),
    (INTENT_WHY_SEVERITY, r"\bwhy\b.*\b(critical|high|medium|low)\b", 6),
    (INTENT_WHY_SEVERITY, r"\bwhy\b.*\b(risk|severity|shift|score)\b", 5),
    (INTENT_WHY_SEVERITY, r"\bwhat\s+made\b.*\b(shift|it)\b", 4),
    (INTENT_WHY_SEVERITY, r"\bhow\s+was\b.*\b(rated|scored)\b", 4),
    (INTENT_WHY_SEVERITY, r"\breason\b.*\b(rating|severity|score)\b", 4),

    # --- repeat offenders / per-track
    (INTENT_REPEAT_OFFENDER, r"\brepeat\s+(offender|track|load)s?\b", 7),
    (INTENT_REPEAT_OFFENDER, r"\bwhich\s+(worker|track|load|box|package)\b", 6),
    (INTENT_REPEAT_OFFENDER, r"\bmost\s+incidents?\b", 5),
    (INTENT_REPEAT_OFFENDER, r"\bwho\b.*\bmost\b", 4),
    (INTENT_REPEAT_OFFENDER, r"\btrack\s+#?\d+\b", 4),
    (INTENT_REPEAT_OFFENDER, r"\bper[- ]track\b", 4),

    # --- what kinds of behavior
    (INTENT_EVENT_TYPES, r"\bwhat\s+(types?|kinds?|sorts?)\b", 7),
    (INTENT_EVENT_TYPES, r"\btypes?\s+of\s+(unsafe\s+)?(behaviou?r|event|incident)", 6),
    (INTENT_EVENT_TYPES, r"\bbreakdown\s+by\s+type\b", 5),
    (INTENT_EVENT_TYPES, r"\bwhat\s+(unsafe\s+)?behaviou?rs?\b", 5),

    # --- counting
    (INTENT_COUNT, r"\bhow\s+many\b", 7),
    (INTENT_COUNT, r"\bnumber\s+of\b", 6),
    (INTENT_COUNT, r"\bcount\s+of\b", 5),
    (INTENT_COUNT, r"\bhow\s+much\b", 3),

    # --- time window
    (INTENT_TIME_QUERY, r"\b(what|anything)\s+happened\b.*\b(at|around|near)\b", 7),
    (INTENT_TIME_QUERY, r"\b(at|around|near)\s+\d+(\.\d+)?\s*(s\b|sec|second|min)", 6),
    (INTENT_TIME_QUERY, r"\bbetween\s+\d+.*\band\s+\d+", 5),
    (INTENT_TIME_QUERY, r"\btimestamp\b", 4),

    # --- what to do about it
    (INTENT_FOCUS, r"\bwhat\s+should\b.*\b(focus|do|prioriti[sz]e|look)\b", 7),
    (INTENT_FOCUS, r"\b(supervisor|manager)\b.*\b(focus|do|action|priorit)", 6),
    (INTENT_FOCUS, r"\brecommend(ation)?s?\b", 5),
    (INTENT_FOCUS, r"\bwhat\s+to\s+(fix|address|do)\b", 5),
    (INTENT_FOCUS, r"\bnext\s+steps?\b", 4),
    (INTENT_FOCUS, r"\bpriorit(y|ies|ise|ize)\b", 4),

    # --- data quality
    (INTENT_DATA_QUALITY, r"\bdata\s+quality\b", 7),
    (INTENT_DATA_QUALITY, r"\b(can|should)\s+(i|we)\s+trust\b", 6),
    (INTENT_DATA_QUALITY, r"\b(is|are)\s+(this|the|these)\b.*\breliable\b", 6),
    (INTENT_DATA_QUALITY, r"\btracking\s+(data|quality)\b", 5),
    (INTENT_DATA_QUALITY, r"\bhow\s+(reliable|confident)\b", 5),
    (INTENT_DATA_QUALITY, r"\bwhy\s+(are\s+there\s+)?no\s+(events|incidents)\b", 6),

    # --- list everything
    (INTENT_LIST_EVENTS, r"\blist\s+(all\s+)?(the\s+)?(events|incidents)\b", 7),
    (INTENT_LIST_EVENTS, r"\bshow\s+(me\s+)?(all\s+)?(the\s+)?(events|incidents)\b", 6),
    (INTENT_LIST_EVENTS, r"\ball\s+(the\s+)?(events|incidents)\b", 5),

    # --- overview
    (INTENT_SUMMARY, r"\b(summar(y|ise|ize)|overview|brief)\b", 6),
    (INTENT_SUMMARY, r"\bwhat\s+happened\b", 4),
    (INTENT_SUMMARY, r"\bhow\s+(was|did)\s+the\s+shift\b", 5),
    (INTENT_SUMMARY, r"\btell\s+me\s+about\b", 4),
    # "was the shift safe?" is the question a supervisor actually asks. Routing
    # it to the summary means it is answered from the data on a good shift, and
    # routed through the unreliable-data guard on a bad one - rather than
    # falling through to "I can't answer that", which reads as reassurance.
    (INTENT_SUMMARY, r"\b(was|were|is|are)\b.*\bsafe\b", 5),
    (INTENT_SUMMARY, r"\bany\s+(incidents?|problems?|issues?|events?)\b", 4),
]

# Entity vocabularies
_SEVERITY_WORDS = {
    "critical": "Critical",
    "high": "High",
    "medium": "Medium",
    "moderate": "Medium",
    "low": "Low",
}

_TYPE_WORDS = {
    "drop": "drop",
    "dropped": "drop",
    "dropping": "drop",
    "fall": "drop",
    "fell": "drop",
    "throw": "throw",
    "thrown": "throw",
    "throwing": "throw",
    "threw": "throw",
    "drag": "drag",
    "dragged": "drag",
    "dragging": "drag",
    "stack": "improper_stack",
    "stacking": "improper_stack",
    "stacked": "improper_stack",
    "overhang": "improper_stack",
    "rough": "rough_handling",
    "handling": "rough_handling",
    "yank": "rough_handling",
}


@dataclass
class Question:
    """A parsed question: one intent plus whatever entities were mentioned."""

    raw: str
    intent: str = INTENT_UNKNOWN
    score: int = 0
    severity: Optional[str] = None
    event_type: Optional[str] = None
    track_id: Optional[int] = None
    timestamp: Optional[float] = None
    scores: Dict[str, int] = field(default_factory=dict)

    @property
    def understood(self) -> bool:
        return self.intent != INTENT_UNKNOWN


def _extract_timestamp(text: str) -> Optional[float]:
    """Pull a time reference out of the question, normalised to seconds."""
    minute_match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(?:min|minute)s?\b", text
    )
    if minute_match:
        try:
            return float(minute_match.group(1)) * 60.0
        except ValueError:
            pass

    patterns = [
        r"\b(?:at|around|near|about)\s+(\d+(?:\.\d+)?)\s*(?:s\b|sec|second)",
        r"\b(\d+(?:\.\d+)?)\s*(?:s\b|sec|seconds?\b)",
        r"\b(?:at|around|near)\s+(\d+(?:\.\d+)?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _extract_track_id(text: str) -> Optional[int]:
    match = re.search(r"\b(?:track|load|box|id)\s*#?\s*(\d+)\b", text)
    if not match:
        match = re.search(r"#(\d+)\b", text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def parse_question(text: str) -> Question:
    """Classify a question and extract its entities."""
    raw = (text or "").strip()
    question = Question(raw=raw)
    if not raw:
        return question

    lowered = raw.lower()

    scores: Dict[str, int] = {}
    for intent, pattern, weight in _PATTERNS:
        if re.search(pattern, lowered):
            scores[intent] = scores.get(intent, 0) + weight
    question.scores = scores

    # Entities are extracted regardless of intent - a count question and a list
    # question both benefit from knowing which severity was named.
    for word, severity in _SEVERITY_WORDS.items():
        if re.search(rf"\b{word}\b", lowered):
            question.severity = severity
            break

    for word, event_type in _TYPE_WORDS.items():
        if re.search(rf"\b{word}\b", lowered):
            question.event_type = event_type
            break

    question.track_id = _extract_track_id(lowered)
    question.timestamp = _extract_timestamp(lowered)

    if scores:
        best = max(scores.items(), key=lambda kv: kv[1])
        question.intent, question.score = best[0], best[1]

    # A bare time reference ("what about 8s?") is a time query even when no
    # time-query phrase matched, provided nothing stronger did.
    if question.timestamp is not None and question.score < 5:
        question.intent = INTENT_TIME_QUERY
        question.score = max(question.score, 5)

    return question
