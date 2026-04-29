"""
Model Health page — Performance tracking, confusion matrix, confidence analysis.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dashboard.components import page_header, section_header, kpi_card, empty_state, plotly_layout, CHART_COLORS


def render(performance: dict = None, confidence_stats: dict = None):
    """Render model health dashboard."""
    page_header("Model Health", "Track model performance degradation and prediction confidence")

    if not performance:
        empty_state("🤖", "No Performance Data",
                     "Run drift detection with a batch containing ground truth labels.")
        return

    health = performance.get("overall_health", "HEALTHY")
    health_colors = {"HEALTHY": "#10b981", "WARNING": "#f59e0b", "CRITICAL": "#ef4444"}
    degraded = performance.get("degraded_metrics_count", 0)

    # ── Health Status ─────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        kpi_card("Model Health", health, "💚" if health == "HEALTHY" else "⚠️",
                 health_colors.get(health, "#64748b"))
    with c2:
        kpi_card("Degraded Metrics", str(degraded), "📉",
                 "#ef4444" if degraded > 2 else ("#f59e0b" if degraded > 0 else "#10b981"))
    with c3:
        if confidence_stats:
            low_pct = confidence_stats.get("low_confidence_pct", 0)
            kpi_card("Low Confidence", f"{low_pct:.1f}%", "⚡",
                     "#ef4444" if low_pct > 20 else "#10b981")
        else:
            kpi_card("Low Confidence", "—", "⚡", "#64748b")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics Comparison ────────────────────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        section_header("Baseline vs Production", "📊")
        keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        available = [k for k in keys if k in performance and isinstance(performance[k], dict)]

        if available:
            baseline_vals = [performance[k]["baseline"] for k in available]
            current_vals = [performance[k]["current"] for k in available]
            labels = [k.replace("_", " ").title() for k in available]

            fig = go.Figure()
            fig.add_trace(go.Bar(name="Baseline", x=labels, y=baseline_vals,
                                 marker_color=CHART_COLORS["baseline"], opacity=0.8))
            fig.add_trace(go.Bar(name="Production", x=labels, y=current_vals,
                                 marker_color=CHART_COLORS["production"], opacity=0.8))

            layout = plotly_layout(height=380)
            layout["barmode"] = "group"
            layout["yaxis"]["range"] = [0, 1]
            layout["yaxis"]["title"] = "Score"
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True, key="metrics_cmp")

    with right:
        section_header("Metric Deltas", "📐")
        for key in available:
            m = performance[key]
            delta = m.get("delta", 0)
            arrow = "↑" if delta >= 0 else "↓"
            color = "#10b981" if delta >= 0 else "#ef4444"
            label = key.replace("_", " ").title()
            st.markdown(f"""
            <div class="feature-item">
                <div>
                    <span class="feature-name">{label}</span>
                    <span class="feature-type">Baseline: {m['baseline']:.4f}</span>
                </div>
                <span style="color:{color};font-weight:700;font-size:0.95rem;">{arrow} {delta:+.4f}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confusion Matrix ──────────────────────────────────────────────────
    section_header("Confusion Matrix", "🧮")
    try:
        from src.utils import load_json
        from src.config.settings import MODELS_DIR
        bm = load_json(MODELS_DIR / "baseline_metrics.json")
        cm = bm.get("confusion_matrix")
        if cm and len(cm) == 2:
            labels = ["No Churn", "Churn"]
            fig_cm = px.imshow(
                cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                x=labels, y=labels, text_auto=True,
                color_continuous_scale=["#0a0e1a", "#3b82f6", "#8b5cf6"],
            )
            cm_layout = plotly_layout(height=350)
            cm_layout["margin"] = dict(t=20, b=40, l=40, r=20)
            fig_cm.update_layout(**cm_layout)
            st.plotly_chart(fig_cm, use_container_width=True, key="cm_chart")
        else:
            st.info("Confusion matrix not available.")
    except Exception:
        st.info("Could not load confusion matrix.")

    # ── Confidence Stats ──────────────────────────────────────────────────
    if confidence_stats:
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Prediction Confidence", "🎯")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            kpi_card("Mean Confidence", f"{confidence_stats.get('mean_confidence', 0):.1%}",
                     "📊", CHART_COLORS["blue"])
        with cc2:
            kpi_card("Std Deviation", f"{confidence_stats.get('std_confidence', 0):.4f}",
                     "📏", CHART_COLORS["yellow"])
        with cc3:
            low = confidence_stats.get("low_confidence_pct", 0)
            kpi_card("Low Confidence %", f"{low:.1f}%", "⚡",
                     CHART_COLORS["red"] if low > 20 else CHART_COLORS["green"])
