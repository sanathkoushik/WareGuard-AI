"""
WareGuard AI - Risk Assessment Engine (Phase 3)

Scores the behavior events from Phase 2 into Low / Medium / High / Critical
severities with an explainable breakdown, and rolls them up into a shift-level
risk picture.

Like `behavior`, this package is standard-library only.

Typical use:

    from behavior import BehaviorEngine
    from risk import RiskEngine

    report = BehaviorEngine().analyze_json("data/logs/detections_clip.json")
    assessment = RiskEngine().assess(report)

    print(assessment.summary.headline())
    for event in assessment.ranked(limit=5):
        print(event.severity, event.risk_score, event.description)
        print("  because:", "; ".join(event.risk_factors))
"""
from .engine import RiskAssessment, RiskEngine, ShiftSummary, assess_report
from .scoring import (
    DEFAULT_WEIGHTS,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_ORDER,
    RiskFactor,
    RiskScorer,
    RiskWeights,
    severity_for,
)

__all__ = [
    "RiskEngine",
    "RiskAssessment",
    "ShiftSummary",
    "assess_report",
    "RiskScorer",
    "RiskWeights",
    "RiskFactor",
    "DEFAULT_WEIGHTS",
    "severity_for",
    "SEVERITY_LOW",
    "SEVERITY_MEDIUM",
    "SEVERITY_HIGH",
    "SEVERITY_CRITICAL",
    "SEVERITY_ORDER",
]
