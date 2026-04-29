"""
Overview dashboard page — KPI cards, risk gauge, drift summary, and insights panel.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from dashboard.components import (
    page_header, section_header, kpi_card, risk_badge,
    feature_item, empty_state, plotly_layout, CHART_COLORS
)


def render(drift_report: dict = None, performance: dict = None, confidence_stats: dict = None):
    """Render the overview dashboard."""
    page_header("Dashboard Overview", "Real-time drift monitoring and model health at a glance")

    if not drift_report:
        empty_state("📊", "No Analysis Data", "Run drift detection to populate the dashboard.")
        return

    dataset = drift_report.get("dataset_drift", {})
    risk = dataset.get("risk_level", "LOW")
    drifted = dataset.get("drifted_count", 0)
    total = dataset.get("total_features", 0)
    drift_pct = dataset.get("drifted_pct", 0)

    # ── KPI Row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        risk_colors = {"LOW": "#10b981", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}
        kpi_card("Risk Level", risk, "🛡️", risk_colors.get(risk, "#3b82f6"))
    with c2:
        kpi_card("Drifted Features", f"{drifted} / {total}", "📊", "#3b82f6")
    with c3:
        health = "—"
        health_color = "#64748b"
        if performance:
            health = performance.get("overall_health", "—")
            health_color = {"HEALTHY": "#10b981", "WARNING": "#f59e0b", "CRITICAL": "#ef4444"}.get(health, "#64748b")
        kpi_card("Model Health", health, "🤖", health_color)
    with c4:
        conf = "—"
        conf_color = "#64748b"
        if confidence_stats:
            conf_val = confidence_stats.get("mean_confidence", 0)
            conf = f"{conf_val:.1%}"
            conf_color = "#10b981" if conf_val > 0.7 else "#f59e0b"
        kpi_card("Avg Confidence", conf, "🎯", conf_color)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-Column Layout ─────────────────────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        # Risk Gauge
        section_header("Dataset Risk Score", "📈")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=drift_pct,
            number=dict(suffix="%", font=dict(size=40, color="#f0f4f8")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#475569", tickfont=dict(color="#64748b")),
                bar=dict(color=risk_colors.get(risk, "#3b82f6"), thickness=0.7),
                bgcolor="rgba(255,255,255,0.03)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 20], color="rgba(16,185,129,0.08)"),
                    dict(range=[20, 50], color="rgba(245,158,11,0.08)"),
                    dict(range=[50, 100], color="rgba(239,68,68,0.08)"),
                ],
                threshold=dict(line=dict(color="#ef4444", width=2), thickness=0.8, value=50),
            ),
        ))
        layout = plotly_layout(height=280)
        layout["margin"] = dict(t=20, b=20, l=30, r=30)
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True, key="risk_gauge")

        # Feature Drift Breakdown
        section_header("Feature Drift Breakdown", "🔍")
        num_results = drift_report.get("feature_drift", {}).get("numerical", {})
        cat_results = drift_report.get("feature_drift", {}).get("categorical", {})

        # PSI bar chart
        psi_data = []
        for col, tests in num_results.items():
            psi = tests.get("psi", {})
            psi_data.append({"Feature": col, "PSI": psi.get("value", 0), "Severity": psi.get("severity", "LOW")})

        if psi_data:
            psi_df = pd.DataFrame(psi_data).sort_values("PSI", ascending=True)
            sev_colors = {"LOW": CHART_COLORS["green"], "MODERATE": CHART_COLORS["yellow"], "SEVERE": CHART_COLORS["red"]}
            colors = [sev_colors.get(s, CHART_COLORS["blue"]) for s in psi_df["Severity"]]

            fig2 = go.Figure(go.Bar(
                y=psi_df["Feature"], x=psi_df["PSI"],
                orientation="h",
                marker_color=colors,
                text=psi_df["PSI"].round(4),
                textposition="outside",
                textfont=dict(size=11, color="#94a3b8"),
            ))
            fig2.add_vline(x=0.1, line_dash="dot", line_color=CHART_COLORS["yellow"],
                           annotation_text="Moderate", annotation_font_color="#fbbf24")
            fig2.add_vline(x=0.25, line_dash="dot", line_color=CHART_COLORS["red"],
                           annotation_text="Severe", annotation_font_color="#f87171")
            layout2 = plotly_layout(height=max(250, len(psi_data) * 35 + 80))
            layout2["xaxis"]["title"] = "PSI Value"
            fig2.update_layout(**layout2)
            st.plotly_chart(fig2, use_container_width=True, key="psi_overview")

    with right:
        # Drifted Features List
        section_header("Drifted Features", "⚡")
        drifted_list = dataset.get("drifted_features", [])
        if drifted_list:
            for feat in drifted_list:
                sev = feat.get("severity", "MODERATE")
                badge_cls = "badge-low" if sev == "LOW" else ("badge-high" if sev == "SEVERE" else "badge-medium")
                badge_html = f'<span class="badge {badge_cls}">{sev}</span>'
                feature_item(feat["feature"], badge_html, feat.get("type", "").title())
        else:
            st.markdown("""
            <div style="text-align:center;padding:32px;color:#64748b;">
                <div style="font-size:2rem;margin-bottom:8px;">✅</div>
                <div>No features drifted — distribution is stable.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Performance Summary (compact)
        if performance:
            section_header("Performance Delta", "📉")
            for key in ["accuracy", "precision", "recall", "f1_score"]:
                m = performance.get(key)
                if isinstance(m, dict):
                    delta = m.get("delta", 0)
                    arrow = "↑" if delta >= 0 else "↓"
                    color = "#10b981" if delta >= 0 else "#ef4444"
                    st.markdown(f"""
                    <div class="feature-item">
                        <span class="feature-name">{key.replace('_', ' ').title()}</span>
                        <span style="color:{color};font-weight:700;font-size:0.9rem;">{arrow} {delta:+.4f}</span>
                    </div>
                    """, unsafe_allow_html=True)

        # Missing Data Summary
        missing = drift_report.get("feature_drift", {}).get("missing_data", {})
        miss_count = sum(1 for v in missing.values() if v.get("drift_detected"))
        if miss_count > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Missing Data Alerts", "❓")
            st.warning(f"**{miss_count}** feature(s) show increased missing data in production.")
