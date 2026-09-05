"""
WareGuard AI - Safety Events & Risk Panel (Phase 4)

Renders the Phase 2 behavior events and Phase 3 risk scores inside the existing
Streamlit dashboard.

Design constraints this module holds to:

**It owns no analysis logic.** Every number shown here comes from
`behavior.BehaviorEngine` and `risk.RiskEngine`. This module loads, arranges and
renders - it never re-derives a threshold, a score or a severity. If a figure
looks wrong, the bug is in Phase 2/3, not here.

**It opens no video.** Event cards expose `start_frame` / `start_time` and write
a requested seek into `st.session_state`; the existing scrubber in `app.py`
(L222-234) remains the only thing that touches OpenCV.

**It adds no dependencies.** Streamlit only - no pandas, no plotly, no numpy.
Charts are fed plain dicts and lists, which `st.bar_chart` accepts.

**Streamlit is imported defensively.** The load/normalise half of this module is
pure data work and stays importable on a machine with no Streamlit installed,
so it can be exercised from a plain `python -c` or a unit test the same way
`behavior/` and `risk/` can. Render functions fail loudly if called without it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Repo root on sys.path so `behavior` / `risk` resolve regardless of whether
# Streamlit was launched from the repo root or from inside dashboard/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:  # pragma: no cover - trivial import guard
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:  # pragma: no cover
    st = None  # type: ignore[assignment]
    _HAS_STREAMLIT = False


# --------------------------------------------------------------------------
# Presentation constants
# --------------------------------------------------------------------------

SEVERITY_STYLE: Dict[str, Dict[str, str]] = {
    "Critical": {"icon": "🔴", "color": "#f85149", "bg": "rgba(248, 81, 73, 0.15)"},
    "High":     {"icon": "🟠", "color": "#db6d28", "bg": "rgba(219, 109, 40, 0.15)"},
    "Medium":   {"icon": "🟡", "color": "#e3b341", "bg": "rgba(227, 179, 65, 0.15)"},
    "Low":      {"icon": "🟢", "color": "#3fb950", "bg": "rgba(63, 185, 80, 0.15)"},
}
_UNKNOWN_STYLE = {"icon": "⚪", "color": "#8b949e", "bg": "rgba(139, 148, 158, 0.15)"}

EVENT_LABEL: Dict[str, str] = {
    "drop": "Drop",
    "throw": "Throw",
    "drag": "Drag",
    "improper_stack": "Improper Stack",
    "rough_handling": "Rough Handling",
}

# Friendly labels for the motion metrics Phase 2 attaches to each event.
# Anything not listed is hidden rather than dumped raw at the user.
METRIC_LABEL: Dict[str, str] = {
    "impact_speed_mps": "Impact speed",
    "peak_fall_speed": "Peak fall speed",
    "fall_distance_heights": "Fall distance",
    "fall_duration_s": "Fall duration",
    "gravity_ratio": "Fraction of free fall",
    "launch_speed_mps": "Launch speed",
    "peak_horizontal_speed": "Peak horizontal speed",
    "horizontal_travel_heights": "Horizontal travel",
    "airborne_duration_s": "Airborne",
    "drag_distance_heights": "Drag distance",
    "drag_duration_s": "Drag duration",
    "mean_horizontal_speed": "Mean drag speed",
    "max_overhang_ratio": "Overhang",
    "max_lean_ratio": "Lean",
    "stack_duration_s": "Stack persisted",
    "peak_jerk": "Peak jerk",
    "peak_speed": "Peak speed",
    "nearest_person_heights": "Nearest worker",
}

_METRIC_UNIT: Dict[str, str] = {
    "impact_speed_mps": " m/s",
    "launch_speed_mps": " m/s",
    "peak_fall_speed": " h/s",
    "peak_horizontal_speed": " h/s",
    "mean_horizontal_speed": " h/s",
    "peak_speed": " h/s",
    "peak_jerk": " h/s²",
    "fall_distance_heights": " box-heights",
    "horizontal_travel_heights": " box-widths",
    "drag_distance_heights": " box-widths",
    "nearest_person_heights": " box-heights",
    "fall_duration_s": " s",
    "drag_duration_s": " s",
    "airborne_duration_s": " s",
    "stack_duration_s": " s",
}

# Metrics rendered as percentages rather than raw ratios.
_PERCENT_METRICS = {"gravity_ratio", "max_overhang_ratio", "max_lean_ratio"}

# Bookkeeping keys that are not motion metrics and must not reach the UI.
_HIDDEN_METRICS = {"priority_score", "also_matched_count", "supporting_track_id"}

SEEK_FRAME_KEY = "wg_seek_frame"
SEEK_TIME_KEY = "wg_seek_time"
"""Session-state keys an event card writes when the user asks to jump to it.

`app.py`'s existing scrubber can read `st.session_state[SEEK_FRAME_KEY]` to seed
its slider. Nothing in this module opens a video itself.
"""

DEMO_EVENTS_FILENAME = "events_sim_demo.json"


# --------------------------------------------------------------------------
# Data layer - importable without Streamlit
# --------------------------------------------------------------------------

def _cache_data(func: Callable) -> Callable:
    """Apply `st.cache_data` when Streamlit is available, else pass through.

    Analysis is cheap (pure-Python, milliseconds) but re-runs on every widget
    interaction without this, and Streamlit reruns the whole script on each
    click.
    """
    if _HAS_STREAMLIT and hasattr(st, "cache_data"):
        return st.cache_data(show_spinner=False)(func)
    return func


def events_path_for(detection_json_path: Path | str) -> Path:
    """Map a detection log to the events file `run_analysis.py` would write.

    `detections_<stem>.json` -> `events_<stem>.json`, matching the convention
    already used by the CLI so the dashboard and the CLI never disagree about
    where results live.
    """
    p = Path(detection_json_path)
    stem = p.stem
    if stem.startswith("detections_"):
        stem = stem[len("detections_"):]
    return p.parent / f"events_{stem}.json"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _file_signature(path: Path) -> float:
    """mtime used as a cache key so a re-run of detection invalidates results."""
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


def _run_analysis(detection_json_path: str, profile: str) -> Optional[Dict[str, Any]]:
    """Run Phase 2 + Phase 3 over a detection log via their public APIs."""
    try:
        from behavior import BehaviorEngine
        from risk import RiskEngine
    except ImportError:
        return None

    try:
        report = BehaviorEngine(thresholds=profile).analyze_json(detection_json_path)
        return RiskEngine().assess(report).to_dict()
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _load_or_run_uncached(
    detection_json_path: str,
    profile: str,
    _events_sig: float,
    _detection_sig: float,
) -> Optional[Dict[str, Any]]:
    """Prefer a precomputed events file; fall back to running the engines.

    The two `_sig` arguments are unused at runtime - they exist so the cache
    key changes when either file is rewritten.
    """
    events_file = events_path_for(detection_json_path)
    payload = _read_json(events_file)
    if payload is not None and "summary" in payload:
        payload = dict(payload)
        payload["_origin"] = "file"
        payload["_origin_path"] = str(events_file)
        return payload

    if not Path(detection_json_path).exists():
        return None

    payload = _run_analysis(detection_json_path, profile)
    if payload is None:
        return None
    payload = dict(payload)
    payload["_origin"] = "computed"
    payload["_origin_path"] = str(detection_json_path)
    return payload


_load_or_run_cached = _cache_data(_load_or_run_uncached)


def load_or_run_analysis(
    detection_json_path: Path | str,
    profile: str = "default",
) -> Optional[Dict[str, Any]]:
    """Return the events payload for a detection log, or None if unavailable.

    Loads `events_<stem>.json` when present, otherwise runs the existing
    Phase 2/3 engines over the detection log. The returned dict is exactly
    `RiskAssessment.to_dict()` - the same shape `run_analysis.py` writes - so
    the file and computed paths are indistinguishable downstream.
    """
    detection_json_path = str(detection_json_path)
    events_file = events_path_for(detection_json_path)
    return _load_or_run_cached(
        detection_json_path,
        profile,
        _file_signature(events_file),
        _file_signature(Path(detection_json_path)),
    )


def load_demo_events(logs_dir: Path | str) -> Optional[Dict[str, Any]]:
    """Load the committed simulated-shift events, if present."""
    payload = _read_json(Path(logs_dir) / DEMO_EVENTS_FILENAME)
    if payload is None:
        return None
    payload = dict(payload)
    payload["_origin"] = "demo"
    return payload


# ---------------------------- payload accessors ---------------------------
# Every accessor tolerates a missing key. A payload written by an older build,
# or hand-edited, must degrade to a blank field rather than crash the tab.

def get_summary(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return (payload or {}).get("summary", {}) or {}


def get_events(payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events = (payload or {}).get("events", [])
    return events if isinstance(events, list) else []


def get_factors(payload: Optional[Dict[str, Any]], event_id: str) -> List[Dict[str, Any]]:
    factors = (payload or {}).get("risk_factors", {}) or {}
    found = factors.get(event_id, [])
    return found if isinstance(found, list) else []


def sort_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank by priority (severity x confidence), then severity, then time.

    Priority is what a supervisor should review first; `risk_score` alone would
    float an uncertain event above a confirmed one.
    """
    def key(e: Dict[str, Any]):
        metrics = e.get("metrics") or {}
        priority = metrics.get("priority_score")
        if priority is None:
            priority = (e.get("risk_score") or 0.0) * (e.get("confidence") or 0.0)
        return (-float(priority), -float(e.get("risk_score") or 0.0),
                float(e.get("start_time") or 0.0))

    return sorted(events, key=key)


def data_quality_state(payload: Optional[Dict[str, Any]]) -> str:
    """Classify the result so the UI never conflates two different meanings.

    Returns one of:
      "unavailable" - no payload at all
      "blocked"     - a quality warning and zero events: analysis could not run
                      meaningfully. This is NOT a clean shift.
      "partial"     - a quality warning but some events were still found
      "clean"       - no warning, no events: a genuine zero-incident result
      "ok"          - no warning, events present
    """
    if payload is None:
        return "unavailable"

    # A payload with no summary block is not an assessment - it is a malformed
    # or truncated file. Falling through would classify it as "clean" and paint
    # the dashboard green on the strength of a corrupt file.
    if not isinstance(payload.get("summary"), dict):
        return "unavailable"

    summary = get_summary(payload)
    warning = summary.get("data_quality_warning")
    has_events = len(get_events(payload)) > 0

    if warning and not has_events:
        return "blocked"
    if warning:
        return "partial"
    if not has_events:
        return "clean"
    return "ok"


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------

def severity_style(severity: Optional[str]) -> Dict[str, str]:
    return SEVERITY_STYLE.get(severity or "", _UNKNOWN_STYLE)


def event_label(event_type: Optional[str]) -> str:
    if not event_type:
        return "Unknown"
    return EVENT_LABEL.get(event_type, event_type.replace("_", " ").title())


def format_metric(key: str, value: Any) -> Optional[str]:
    """Render one metric, or None when it should not be shown."""
    if key in _HIDDEN_METRICS or key not in METRIC_LABEL:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if key in _PERCENT_METRICS:
        return f"{number * 100:.0f}%"
    return f"{number:.2f}{_METRIC_UNIT.get(key, '')}"


def visible_metrics(event: Dict[str, Any]) -> List[tuple]:
    """(label, formatted_value) pairs worth showing on a card."""
    metrics = event.get("metrics") or {}
    out = []
    for key, value in metrics.items():
        rendered = format_metric(key, value)
        if rendered is not None:
            out.append((METRIC_LABEL[key], rendered))
    return out


def timeline_chart_data(summary: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Reshape `summary.timeline` into plain lists for st.bar_chart."""
    buckets = summary.get("timeline") or []
    labels: List[str] = []
    counts: List[int] = []
    peaks: List[float] = []
    for b in buckets:
        try:
            start = float(b.get("start_s", 0.0))
            end = float(b.get("end_s", 0.0))
        except (TypeError, ValueError):
            continue
        labels.append(f"{start:.0f}-{end:.0f}s")
        counts.append(int(b.get("events", 0) or 0))
        peaks.append(float(b.get("max_risk", 0.0) or 0.0))
    return {"window": labels, "events": counts, "peak risk": peaks}


# --------------------------------------------------------------------------
# Render layer - requires Streamlit
# --------------------------------------------------------------------------

def _require_streamlit() -> None:
    if not _HAS_STREAMLIT:
        raise RuntimeError(
            "streamlit is not installed - events_panel render functions need it. "
            "Install with: pip install streamlit"
        )


_STYLE_FLAG = "_wg_events_css"


def _inject_css() -> None:
    """Add only severity-chip styling.

    Deliberately does not define `.metric-card` / `.metric-value` /
    `.metric-label` - those come from app.py and this block is injected after
    it, so redefining them would silently override the app's own design.
    """
    if st.session_state.get(_STYLE_FLAG):
        return
    st.markdown(
        """
        <style>
        .wg-chip {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.03em;
        }
        .wg-event-head {
            font-size: 1.0rem;
            font-weight: 600;
            color: #c9d1d9;
        }
        .wg-muted {
            color: #8b949e;
            font-size: 0.82rem;
        }
        .wg-factor {
            color: #c9d1d9;
            font-size: 0.86rem;
            padding: 2px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state[_STYLE_FLAG] = True


def _metric_card(label: str, value: Any, color: Optional[str] = None) -> None:
    """Reuses app.py's `.metric-card` classes so styling stays consistent."""
    style = f' style="color:{color}"' if color else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value"{style}>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_header(payload: Optional[Dict[str, Any]]) -> None:
    """Six KPI tiles: risk index, severity, totals, and the severity split."""
    _require_streamlit()
    summary = get_summary(payload)
    events = get_events(payload)
    by_sev = summary.get("events_by_severity") or {}

    index = summary.get("shift_risk_index")
    severity = summary.get("shift_severity") or "-"
    style = severity_style(summary.get("shift_severity"))

    critical = int(by_sev.get("Critical", 0) or 0)
    high = int(by_sev.get("High", 0) or 0)
    medium = int(by_sev.get("Medium", 0) or 0)
    low = int(by_sev.get("Low", 0) or 0)

    # A blocked analysis must not render a confident "0" in the risk tiles -
    # zero-because-unmeasured and zero-because-safe look identical otherwise.
    blocked = data_quality_state(payload) == "blocked"
    index_display = "n/a" if blocked or index is None else f"{float(index):.0f}"
    severity_display = "Unknown" if blocked else severity

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        _metric_card("Shift Risk Index", index_display,
                     None if blocked else style["color"])
    with c2:
        _metric_card("Shift Severity",
                     f"{'' if blocked else style['icon']} {severity_display}".strip(),
                     None if blocked else style["color"])
    with c3:
        _metric_card("Total Events", len(events))
    with c4:
        _metric_card("Critical", critical,
                     SEVERITY_STYLE["Critical"]["color"] if critical else None)
    with c5:
        _metric_card("High", high,
                     SEVERITY_STYLE["High"]["color"] if high else None)
    with c6:
        _metric_card("Medium / Low", f"{medium} / {low}")


def render_data_quality(payload: Optional[Dict[str, Any]]) -> str:
    """Surface analysis trustworthiness before anything else. Returns the state.

    The distinction this function exists to protect: an empty event list means
    two completely different things depending on whether the tracks were
    analysable. Rendering both as a green "all clear" would be the single most
    dangerous thing this dashboard could do.
    """
    _require_streamlit()
    state = data_quality_state(payload)
    summary = get_summary(payload)
    warning = summary.get("data_quality_warning")
    behavior = (payload or {}).get("behavior_summary", {}) or {}

    if state == "unavailable":
        st.info(
            "**No analysis available for this video.**\n\n"
            "Run the detection pipeline first, then generate events with:\n\n"
            "`python run_analysis.py --logs data/logs/detections_<video>.json`"
        )
        return state

    if state == "blocked":
        st.warning(
            "⚠️ **Analysis unavailable — insufficient tracking data.**\n\n"
            f"{warning}\n\n"
            "**This is not a clean shift.** No safety events could be produced "
            "because the object tracks were too short or too fragmented to "
            "measure motion from. Behavior detection needs continuous tracks to "
            "compute fall speed, drag distance and acceleration; without them "
            "there is nothing to analyse.\n\n"
            "Treat this as *unknown*, not *safe*.",
            icon="⚠️",
        )
        usable = behavior.get("usable_tracks")
        cargo = behavior.get("cargo_tracks")
        total = behavior.get("total_tracks")
        if usable is not None and cargo is not None:
            st.caption(
                f"Tracking quality: {usable} of {cargo} cargo tracks usable "
                f"({total} tracks total). Fixing this requires better source "
                f"footage or detection tuning — it is not a dashboard issue."
            )
        return state

    if state == "partial":
        st.warning(
            f"⚠️ **Partial analysis.** {warning}\n\n"
            "The events below are real, but coverage is incomplete — absence of "
            "an event in a given period is not evidence that nothing happened.",
            icon="⚠️",
        )
        return state

    if state == "clean":
        st.success(
            "✅ **No unsafe handling detected.**\n\n"
            "Tracking data was sufficient to analyse and no drop, throw, drag, "
            "improper stack or rough handling was found. This is a genuine "
            "zero-incident result, not a data problem.",
            icon="✅",
        )
        return state

    # state == "ok": nothing to warn about. The KPI header renders immediately
    # below and carries the same figures, so no banner is emitted here.
    #
    # Note: `ShiftSummary.headline()` would be the natural one-liner, but it is
    # a method and is not present in `ShiftSummary.to_dict()`, so it never
    # reaches the JSON the dashboard loads. Reproducing its wording here would
    # duplicate Phase 3 presentation logic, so it is deliberately left out.
    return state


def render_timeline(payload: Optional[Dict[str, Any]]) -> None:
    """Event activity over the clip, from `summary.timeline`."""
    _require_streamlit()
    summary = get_summary(payload)
    data = timeline_chart_data(summary)
    if not data["window"]:
        return

    st.markdown("##### Event activity over time")
    try:
        st.bar_chart(data, x="window", y="events", height=220)
    except Exception:
        # Older Streamlit builds do not accept x/y with a dict payload.
        st.bar_chart({"events": data["events"]}, height=220)

    if any(data["peak risk"]):
        st.caption(
            "Peak risk score per window: "
            + ", ".join(
                f"{w} → {p:.0f}"
                for w, p in zip(data["window"], data["peak risk"]) if p
            )
        )


def render_event_card(
    event: Dict[str, Any],
    factors: Optional[List[Dict[str, Any]]] = None,
    allow_seek: bool = True,
) -> None:
    """One expandable event card."""
    _require_streamlit()

    severity = event.get("severity")
    style = severity_style(severity)
    event_id = event.get("event_id") or f"track-{event.get('track_id', '?')}"
    label = event_label(event.get("event_type"))
    risk = event.get("risk_score")
    confidence = event.get("confidence")
    start_t = event.get("start_time")
    end_t = event.get("end_time")
    start_frame = event.get("start_frame")

    risk_txt = f"{float(risk):.0f}" if risk is not None else "-"
    time_txt = (
        f"{float(start_t):.2f}s – {float(end_t):.2f}s"
        if start_t is not None and end_t is not None else "-"
    )

    title = (
        f"{style['icon']}  {label}  ·  {severity or 'Unknown'}  ·  "
        f"risk {risk_txt}  ·  {time_txt}"
    )

    with st.expander(title, expanded=False):
        st.markdown(
            f"<span class='wg-chip' style=\"background:{style['bg']};"
            f"color:{style['color']};border:1px solid {style['color']}\">"
            f"{severity or 'Unknown'}</span>"
            f"&nbsp;&nbsp;<span class='wg-muted'>{event_id}</span>",
            unsafe_allow_html=True,
        )

        description = event.get("description")
        if description:
            st.markdown(f"**{description}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk score", risk_txt)
        c2.metric(
            "Confidence",
            f"{float(confidence):.0%}" if confidence is not None else "-",
        )
        c3.metric("Track ID", event.get("track_id", "-"))
        priority = (event.get("metrics") or {}).get("priority_score")
        c4.metric("Priority", f"{float(priority):.0f}" if priority is not None else "-")

        related = event.get("related_track_ids") or []
        st.markdown(
            f"<span class='wg-muted'>Frames {event.get('start_frame', '?')}–"
            f"{event.get('end_frame', '?')} &nbsp;·&nbsp; {time_txt}"
            + (f" &nbsp;·&nbsp; related tracks: {', '.join(str(t) for t in related)}"
               if related else "")
            + "</span>",
            unsafe_allow_html=True,
        )

        metrics = visible_metrics(event)
        if metrics:
            st.markdown("**Motion measurements**")
            cols = st.columns(min(4, len(metrics)))
            for i, (name, value) in enumerate(metrics):
                cols[i % len(cols)].markdown(
                    f"<div class='wg-muted'>{name}</div>"
                    f"<div class='wg-event-head'>{value}</div>",
                    unsafe_allow_html=True,
                )

        factor_lines: List[str] = []
        if factors:
            for f in factors:
                points = f.get("points")
                detail = f.get("detail", f.get("code", ""))
                sign = "+" if (points or 0) >= 0 else ""
                factor_lines.append(f"{detail} ({sign}{float(points or 0):.0f})")
        elif event.get("risk_factors"):
            factor_lines = [str(x) for x in event["risk_factors"]]

        if factor_lines:
            # Rendered inline, not in a nested expander: Streamlit forbids an
            # expander inside an expander and raises StreamlitAPIException.
            st.markdown("**Why this score**")
            for line in factor_lines:
                st.markdown(
                    f"<div class='wg-factor'>• {line}</div>",
                    unsafe_allow_html=True,
                )
            st.caption(
                "A baseline plus named contributions — the score always equals "
                "the sum of these factors."
            )

        if allow_seek and start_frame is not None:
            if st.button("🎯 Jump to this event", key=f"seek_{event_id}"):
                st.session_state[SEEK_FRAME_KEY] = int(start_frame)
                if start_t is not None:
                    st.session_state[SEEK_TIME_KEY] = float(start_t)
                st.info(
                    f"Seek requested at **frame {int(start_frame)}** "
                    f"({time_txt.split('–')[0].strip()}). Open the "
                    f"**Video Playback & HUD** tab and set the frame scrubber "
                    f"to {int(start_frame)}."
                )


def render_events_list(payload: Optional[Dict[str, Any]]) -> None:
    """Severity filter plus the ranked event cards."""
    _require_streamlit()
    events = get_events(payload)
    if not events:
        return

    ranked = sort_events(events)
    present = [s for s in ("Critical", "High", "Medium", "Low")
               if any(e.get("severity") == s for e in ranked)]

    selected = present
    if len(present) > 1:
        selected = st.multiselect(
            "Filter by severity", options=present, default=present,
            key="wg_sev_filter",
        )

    shown = [e for e in ranked if e.get("severity") in selected] if selected else []

    st.markdown(
        f"##### {len(shown)} event{'s' if len(shown) != 1 else ''} "
        "· highest priority first"
    )
    if not shown:
        st.caption("No events match the selected severities.")
        return

    for event in shown:
        render_event_card(event, get_factors(payload, event.get("event_id", "")))


def render_events_tab(
    detection_json_path: Path | str,
    annotated_video_path: Optional[Path | str] = None,
    profile: str = "default",
    logs_dir: Optional[Path | str] = None,
) -> None:
    """Entry point for the "Safety Events & Risk" tab.

    Args:
        detection_json_path: the `detections_<stem>.json` for the selected video.
        annotated_video_path: accepted for signature stability; this module does
            not open it. Seeking is handed to app.py's existing scrubber via
            session state.
        profile: behavior threshold profile ("default" | "sensitive" | "strict").
        logs_dir: where to look for the simulated demo events. Defaults to the
            detection log's own directory.
    """
    _require_streamlit()
    _inject_css()

    st.subheader("🚨 Safety Events & Risk")

    payload = load_or_run_analysis(detection_json_path, profile)
    state = data_quality_state(payload)

    # 1. Data quality first, and always about the real selected video.
    render_data_quality(payload)

    # 2. Header. Suppressed entirely when there is no payload at all.
    if payload is not None:
        st.markdown("")
        render_risk_header(payload)

    # 3. When the real video yields nothing usable, offer the committed
    #    simulated shift so the panel can still be demonstrated. Opt-in and
    #    explicitly labelled - never silently substituted for real footage.
    if state in ("unavailable", "blocked"):
        demo_dir = Path(logs_dir) if logs_dir else Path(detection_json_path).parent
        demo = load_demo_events(demo_dir)
        if demo is None:
            return

        st.markdown("---")
        if not st.checkbox(
            "Show simulated demo shift instead (ground-truth tracks)",
            key="wg_use_demo",
            help="Loads events_sim_demo.json — simulated data, not this video.",
        ):
            return

        st.caption(
            "⚙️ Showing **simulated** ground-truth tracks from "
            "`events_sim_demo.json` — not derived from the selected video."
        )
        payload = demo
        state = data_quality_state(payload)
        st.markdown("")
        render_risk_header(payload)

    st.markdown("")

    summary = get_summary(payload)
    if summary.get("total_events"):
        if not summary.get("rate_is_reliable", True):
            st.caption(
                f"Event rate of {float(summary.get('events_per_minute', 0)):.1f}/min "
                f"is extrapolated from "
                f"{float(summary.get('duration_seconds', 0)):.0f}s of footage — "
                "indicative only."
            )
        repeats = summary.get("repeat_offender_tracks") or []
        if repeats:
            st.caption(
                "Loads involved in more than one event: "
                + ", ".join(f"#{t}" for t in repeats)
            )

    render_timeline(payload)
    st.markdown("---")
    render_events_list(payload)

    origin = payload.get("_origin")
    if origin == "computed":
        st.caption(
            "Computed live from the detection log. Persist it with "
            "`python run_analysis.py --logs "
            f"{Path(detection_json_path).name}`."
        )
    elif origin == "file":
        st.caption(f"Loaded from `{Path(payload.get('_origin_path', '')).name}`.")
