"""
WareGuard AI - Cross-Shift Leaderboard Panel

A fleet-level view over every shift already scored in data/logs - "which
shift was riskiest", "who are the repeat offenders across footage" - built on
top of the same per-shift analysis events_panel.py uses. This module owns no
analysis logic of its own; it only aggregates `RiskAssessment.to_dict()`
payloads that `events_panel.load_or_run_analysis` already knows how to load
or compute.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:  # pragma: no cover - trivial import guard
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:  # pragma: no cover
    st = None  # type: ignore[assignment]
    _HAS_STREAMLIT = False

try:  # streamlit run puts dashboard/ on sys.path, not the repo root
    from dashboard.events_panel import load_or_run_analysis, get_summary, data_quality_state
except ImportError:
    from events_panel import load_or_run_analysis, get_summary, data_quality_state


def collect_shift_rows(logs_dir: Path | str, profile: str = "default") -> List[Dict[str, Any]]:
    """One row per `detections_*.json` in logs_dir with usable analysis."""
    rows: List[Dict[str, Any]] = []
    for detections_path in sorted(Path(logs_dir).glob("detections_*.json")):
        stem = detections_path.stem[len("detections_"):]
        payload = load_or_run_analysis(detections_path, profile)
        if data_quality_state(payload) in ("unavailable", "blocked"):
            continue

        summary = get_summary(payload)
        by_sev = summary.get("events_by_severity") or {}
        rows.append({
            "Shift": stem,
            "Risk Index": round(float(summary.get("shift_risk_index", 0.0) or 0.0), 1),
            "Severity": summary.get("shift_severity", "-"),
            "Events": int(summary.get("total_events", 0) or 0),
            "Critical": int(by_sev.get("Critical", 0) or 0),
            "High": int(by_sev.get("High", 0) or 0),
            "Repeat offenders": len(summary.get("repeat_offender_tracks") or []),
            "Duration (min)": round(float(summary.get("duration_seconds", 0) or 0) / 60.0, 1),
        })

    rows.sort(key=lambda r: -r["Risk Index"])
    return rows


def render_fleet_tab(logs_dir: Path | str, profile: str = "default") -> None:
    """Entry point for the cross-shift leaderboard section."""
    if not _HAS_STREAMLIT:
        raise RuntimeError("streamlit is not installed - fleet_panel render needs it.")

    st.caption(
        "Every scored shift found in `data/logs`, ranked by shift risk index - "
        "spot the riskiest shift or a pattern repeating across footage without "
        "opening each video one at a time."
    )

    rows = collect_shift_rows(logs_dir, profile)
    if not rows:
        st.info(
            "No scored shifts yet. Run the detection pipeline on at least one "
            "video (sidebar) to populate this view."
        )
        return

    total_critical = sum(r["Critical"] for r in rows)
    total_high = sum(r["High"] for r in rows)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shifts analyzed", len(rows))
    c2.metric("Riskiest shift", rows[0]["Shift"], f"{rows[0]['Risk Index']:.0f}/100")
    c3.metric("Critical events (all shifts)", total_critical)
    c4.metric("High events (all shifts)", total_high)

    st.markdown("##### Shift risk index, highest first")
    chart_data = {
        "Shift": [r["Shift"] for r in rows],
        "Risk Index": [r["Risk Index"] for r in rows],
    }
    try:
        st.bar_chart(chart_data, x="Shift", y="Risk Index", height=260)
    except Exception:
        # Older Streamlit builds do not accept x/y with a dict payload.
        st.bar_chart({"Risk Index": chart_data["Risk Index"]}, height=260)

    st.markdown("##### All shifts")
    st.dataframe(rows, use_container_width=True, hide_index=True)
