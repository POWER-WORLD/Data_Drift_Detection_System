"""
Alerts page — Filterable alert timeline with severity badges.
"""

import streamlit as st
from dashboard.components import page_header, section_header, alert_card, empty_state
from src.alert import get_all_alerts, clear_alerts


def render():
    """Render the alerts dashboard page."""
    page_header("Alert Center", "Drift warnings, model degradation notices, and recommended actions")

    alerts = get_all_alerts()

    if not alerts:
        empty_state("🔔", "No Alerts Yet",
                     "Alerts are generated automatically when drift detection finds issues. "
                     "Run an analysis from Upload & Run to get started.")
        return

    # ── Filter & Controls ─────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        filter_sev = st.multiselect(
            "Filter by severity",
            ["INFO", "WARNING", "CRITICAL"],
            default=["INFO", "WARNING", "CRITICAL"],
            key="alert_filter",
        )
    with col2:
        search = st.text_input("Search alerts", placeholder="Type to filter...", key="alert_search")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All", type="secondary", key="clear_alerts"):
            clear_alerts()
            st.rerun()

    # ── Filter alerts ─────────────────────────────────────────────────────
    filtered = [a for a in alerts if a.get("severity", "INFO") in filter_sev]
    if search:
        search_lower = search.lower()
        filtered = [a for a in filtered if
                    search_lower in a.get("message", "").lower() or
                    search_lower in a.get("batch", "").lower()]

    # ── Stats row ─────────────────────────────────────────────────────────
    total = len(alerts)
    critical = sum(1 for a in alerts if a.get("severity") == "CRITICAL")
    warning = sum(1 for a in alerts if a.get("severity") == "WARNING")
    info = sum(1 for a in alerts if a.get("severity") == "INFO")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Alerts", total)
    c2.metric("🚨 Critical", critical)
    c3.metric("⚠️ Warnings", warning)
    c4.metric("ℹ️ Info", info)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Alert Timeline ────────────────────────────────────────────────────
    section_header(f"Showing {len(filtered)} of {total} alerts", "📋")

    for a in reversed(filtered):
        alert_card(a)
