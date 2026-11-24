"""
Alerts page — Filterable alert timeline with severity badges and action items.
Monitors drift anomaly alerts, model degradation notices, and recommended remediation.
"""

import streamlit as st
from dashboard.components import (
    page_header,
    section_header,
    alert_card,
    empty_state,
    kpi_card,
    CHART_COLORS,
)
from src.alert import get_all_alerts, clear_alerts


def render() -> None:
    """Render the proactive alert intelligence center."""
    page_header(
        "Alert Intelligence Center",
        "Automated drift warnings, performance degradation notices, and remediation recommendations",
    )

    alerts = get_all_alerts()

    if not alerts:
        empty_state(
            "🔔",
            "Zero Active Alerts",
            "The system automatically logs alerts when statistical drift or performance degradation breaches configured thresholds. Run drift detection from 'Upload & Run' to begin.",
        )
        return

    # ── Summary KPI Metrics Row ───────────────────────────────────────────
    total_count = len(alerts)
    critical_count = sum(1 for a in alerts if a.get("severity", "").upper() == "CRITICAL")
    warning_count = sum(1 for a in alerts if a.get("severity", "").upper() == "WARNING")
    info_count = sum(1 for a in alerts if a.get("severity", "").upper() == "INFO")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Alerts Logged", str(total_count), "🔔", CHART_COLORS["blue"])
    with c2:
        kpi_card("Critical Anomalies", str(critical_count), "🚨", CHART_COLORS["red"])
    with c3:
        kpi_card("Warnings", str(warning_count), "⚠️", CHART_COLORS["yellow"])
    with c4:
        kpi_card("Informational", str(info_count), "ℹ️", CHART_COLORS["cyan"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filters & Search Controls ─────────────────────────────────────────
    section_header("Filter & Search Alerts", "🔍")
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=["INFO", "WARNING", "CRITICAL"],
            default=["INFO", "WARNING", "CRITICAL"],
            key="alert_severity_filter",
            help="Select one or more alert severity tiers to display.",
        )

    with col2:
        search_query = st.text_input(
            "Search alerts by keyword",
            placeholder="Search message, batch name, or recommended action...",
            key="alert_keyword_search",
        )

    with col3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All Alerts", type="secondary", key="clear_all_alerts_btn", width="stretch"):
            clear_alerts()
            st.toast("✅ All alerts cleared successfully.")
            st.rerun()

    # ── Filter Execution ──────────────────────────────────────────────────
    filtered_alerts = [a for a in alerts if a.get("severity", "INFO").upper() in severity_filter]

    if search_query:
        query_lower = search_query.strip().lower()
        filtered_alerts = [
            a for a in filtered_alerts
            if query_lower in a.get("message", "").lower()
            or query_lower in str(a.get("batch", "")).lower()
            or query_lower in a.get("action", "").lower()
        ]

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Alert Timeline Stream ─────────────────────────────────────────────
    section_header(f"Telemetry Stream — Showing {len(filtered_alerts)} of {total_count} Alerts", "📋")

    if not filtered_alerts:
        st.markdown("""
        <div style="text-align:center;padding:32px 16px;background:rgba(255,255,255,0.02);border:1px dashed rgba(255,255,255,0.1);border-radius:12px;color:#94a3b8;">
            <div style="font-size:1.8rem;margin-bottom:6px;">🔍</div>
            <div style="font-weight:700;color:#cbd5e1;">No alerts match the active filter criteria.</div>
            <div style="font-size:0.8rem;margin-top:4px;">Try expanding the severity filter or clearing the search query.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for alert_item in reversed(filtered_alerts):
            alert_card(alert_item)
