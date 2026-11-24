"""
Overview dashboard page — KPI cards, risk gauge, drift summary, and insights panel.
Provides executive-level telemetry on dataset distribution shifts and model degradation.
"""

from typing import Optional
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from dashboard.components import (
    page_header,
    section_header,
    kpi_card,
    feature_item,
    empty_state,
    plotly_layout,
    CHART_COLORS,
)


def render(
    drift_report: Optional[dict] = None,
    performance: Optional[dict] = None,
    confidence_stats: Optional[dict] = None,
) -> None:
    """Render the overview telemetry dashboard."""
    page_header(
        "Dashboard Overview",
        "Real-time drift monitoring, model reliability, and dataset distribution telemetry",
    )

    if not drift_report or not drift_report.get("dataset_drift"):
        empty_state(
            "📊",
            "No Telemetry Data Available",
            "Execute drift detection from the 'Upload & Run' page to populate the executive dashboard.",
        )
        return

    dataset = drift_report.get("dataset_drift", {})
    risk = dataset.get("risk_level", "LOW")
    drifted = dataset.get("drifted_count", 0)
    total = dataset.get("total_features", 0)
    drift_pct = dataset.get("drifted_pct", 0.0)

    risk_colors = {
        "LOW": "#10b981",
        "MEDIUM": "#f59e0b",
        "HIGH": "#ef4444",
    }
    current_risk_color = risk_colors.get(risk, "#3b82f6")

    # ── Executive KPI Row ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Overall Risk", risk, "🛡️", current_risk_color)
    with c2:
        drift_delta = f"{drift_pct:.1f}% shifted"
        kpi_card(
            "Drifted Features",
            f"{drifted} / {total}",
            "📊",
            "#3b82f6",
            delta=drift_delta,
            delta_positive=(drifted == 0),
        )
    with c3:
        health_status = "No Ground Truth"
        health_color = "#94a3b8"
        if performance:
            health_status = performance.get("overall_health", "UNKNOWN")
            health_color = {
                "HEALTHY": "#10b981",
                "WARNING": "#f59e0b",
                "CRITICAL": "#ef4444",
            }.get(health_status, "#94a3b8")
        kpi_card("Model Health", health_status, "🤖", health_color)
    with c4:
        conf_display = "—"
        conf_color = "#94a3b8"
        if confidence_stats:
            conf_val = confidence_stats.get("mean_confidence", 0.0)
            conf_display = f"{conf_val:.1%}"
            conf_color = "#10b981" if conf_val >= 0.70 else ("#f59e0b" if conf_val >= 0.55 else "#ef4444")
        kpi_card("Avg Confidence", conf_display, "🎯", conf_color)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Telemetry Grid ────────────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2])

    with left_col:
        # Risk Score Gauge
        section_header("Dataset Risk Score", "📈")
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=drift_pct,
                number=dict(suffix="%", font=dict(size=38, color="#f8fafc", family="Inter")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#475569", tickfont=dict(color="#94a3b8")),
                    bar=dict(color=current_risk_color, thickness=0.65),
                    bgcolor="rgba(255,255,255,0.03)",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 20], color="rgba(16,185,129,0.1)"),
                        dict(range=[20, 50], color="rgba(245,158,11,0.1)"),
                        dict(range=[50, 100], color="rgba(239,68,68,0.1)"),
                    ],
                    threshold=dict(line=dict(color="#ef4444", width=3), thickness=0.8, value=50),
                ),
            )
        )
        gauge_layout = plotly_layout(height=280)
        gauge_layout["margin"] = dict(t=20, b=20, l=30, r=30)
        fig_gauge.update_layout(**gauge_layout)
        st.plotly_chart(fig_gauge, width="stretch", key="overview_risk_gauge")

        # Feature Drift Breakdown (PSI Analysis)
        section_header("Numerical Feature Drift (PSI)", "🔍")
        num_results = drift_report.get("feature_drift", {}).get("numerical", {})

        psi_records = []
        for col_name, tests in num_results.items():
            psi_info = tests.get("psi", {})
            val = psi_info.get("value")
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                psi_records.append({
                    "Feature": col_name,
                    "PSI": float(val),
                    "Severity": psi_info.get("severity", "LOW"),
                })

        if psi_records:
            psi_df = pd.DataFrame(psi_records).sort_values("PSI", ascending=True)
            sev_colors = {
                "LOW": CHART_COLORS["green"],
                "MODERATE": CHART_COLORS["yellow"],
                "SEVERE": CHART_COLORS["red"],
            }
            colors = [sev_colors.get(s, CHART_COLORS["blue"]) for s in psi_df["Severity"]]

            fig_psi = go.Figure(
                go.Bar(
                    y=psi_df["Feature"],
                    x=psi_df["PSI"],
                    orientation="h",
                    marker_color=colors,
                    text=psi_df["PSI"].round(4),
                    textposition="outside",
                    textfont=dict(size=11, color="#cbd5e1"),
                    cliponaxis=False,
                )
            )
            fig_psi.add_vline(
                x=0.1,
                line_dash="dot",
                line_color=CHART_COLORS["yellow"],
                annotation_text="Moderate (0.10)",
                annotation_font_color="#fbbf24",
                annotation_position="top right",
            )
            fig_psi.add_vline(
                x=0.25,
                line_dash="dot",
                line_color=CHART_COLORS["red"],
                annotation_text="Severe (0.25)",
                annotation_font_color="#f87171",
                annotation_position="top right",
            )
            psi_layout = plotly_layout(height=max(260, len(psi_records) * 32 + 80))
            psi_layout["xaxis"]["title"] = "Population Stability Index (PSI)"
            fig_psi.update_layout(**psi_layout)
            st.plotly_chart(fig_psi, width="stretch", key="overview_psi_chart")
        else:
            st.info("No numerical feature PSI measurements available in this report.")

    with right_col:
        # Drifted Features List
        section_header("Drifted Features Inventory", "⚡")
        drifted_list = dataset.get("drifted_features", [])
        if drifted_list:
            for feat in drifted_list:
                sev = feat.get("severity", "MODERATE").upper()
                badge_cls = (
                    "badge-low"
                    if sev == "LOW"
                    else ("badge-high" if sev in ("SEVERE", "HIGH") else "badge-medium")
                )
                badge_html = f'<span class="badge {badge_cls}">{sev}</span>'
                feature_item(
                    feat.get("feature", "Unknown"),
                    badge_html,
                    feat.get("type", "").title(),
                )
        else:
            st.markdown("""
            <div style="text-align:center;padding:28px 16px;background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.2);border-radius:12px;">
                <div style="font-size:1.8rem;margin-bottom:6px;">✅</div>
                <div style="color:#34d399;font-weight:700;">Zero Feature Drift Detected</div>
                <div style="color:#94a3b8;font-size:0.8rem;margin-top:2px;">All features conform to baseline distributions.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Performance Degradation Delta
        if performance:
            section_header("Performance vs Baseline", "📉")
            metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
            for m_key in metric_keys:
                metric_data = performance.get(m_key)
                if isinstance(metric_data, dict):
                    delta = metric_data.get("delta", 0.0)
                    arrow = "↑" if delta >= 0 else "↓"
                    color = "#10b981" if delta >= 0 else "#ef4444"
                    label = m_key.replace("_", " ").title()
                    curr = metric_data.get("current", 0.0)
                    base = metric_data.get("baseline", 0.0)
                    st.markdown(f"""
                    <div class="feature-item">
                        <div>
                            <span class="feature-name">{label}</span>
                            <span class="feature-type">Base: {base:.3f} → Curr: {curr:.3f}</span>
                        </div>
                        <span style="color:{color};font-weight:700;font-size:0.92rem;">
                            {arrow} {delta:+.4f}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

        # Missing Data Alerts
        missing_dict = drift_report.get("feature_drift", {}).get("missing_data", {})
        flagged_missing = [k for k, v in missing_dict.items() if v.get("drift_detected")]
        if flagged_missing:
            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Missing Data Anomalies", "❓")
            st.warning(
                f"**{len(flagged_missing)} feature(s)** show elevated missing value rates in production: "
                + ", ".join(f"`{f}`" for f in flagged_missing[:5])
                + ("..." if len(flagged_missing) > 5 else "")
            )
