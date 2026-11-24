"""
Model Health page — Performance tracking, confusion matrix, confidence analysis.
Monitors operational model degradation, classification metric deltas, and uncertainty.
"""

from typing import Optional
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dashboard.components import (
    page_header,
    section_header,
    kpi_card,
    empty_state,
    plotly_layout,
    CHART_COLORS,
)
from src.config.settings import MODELS_DIR
from src.utils import load_json


def render(
    performance: Optional[dict] = None,
    confidence_stats: Optional[dict] = None,
) -> None:
    """Render the model health and degradation telemetry page."""
    page_header(
        "Model Health & Reliability",
        "Track classification performance degradation, confusion matrices, and prediction confidence uncertainty",
    )

    if not performance and not confidence_stats:
        empty_state(
            "🤖",
            "No Model Telemetry Available",
            "Evaluate a production batch that includes ground truth labels ('Churn') to monitor accuracy, precision, recall, and F1 degradation.",
        )
        return

    # ── Overall Health KPIs ───────────────────────────────────────────────
    health_status = performance.get("overall_health", "UNKNOWN") if performance else "NOT_EVALUATED"
    degraded_count = performance.get("degraded_metrics_count", 0) if performance else 0

    health_palette = {
        "HEALTHY": "#10b981",
        "WARNING": "#f59e0b",
        "CRITICAL": "#ef4444",
        "NOT_EVALUATED": "#94a3b8",
        "UNKNOWN": "#94a3b8",
    }
    health_color = health_palette.get(health_status, "#94a3b8")

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card(
            "Model Status",
            health_status,
            "💚" if health_status == "HEALTHY" else ("⚠️" if health_status == "WARNING" else "🚨"),
            health_color,
        )
    with c2:
        deg_color = "#10b981" if degraded_count == 0 else ("#f59e0b" if degraded_count <= 2 else "#ef4444")
        kpi_card("Degraded Metrics", str(degraded_count), "📉", deg_color)
    with c3:
        if confidence_stats:
            low_pct = confidence_stats.get("low_confidence_pct", 0.0)
            low_color = "#10b981" if low_pct < 15.0 else ("#f59e0b" if low_pct < 25.0 else "#ef4444")
            kpi_card("Low Confidence Predictions", f"{low_pct:.1f}%", "⚡", low_color)
        else:
            kpi_card("Low Confidence Predictions", "—", "⚡", "#94a3b8")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Performance Metric Comparisons ────────────────────────────────────
    if performance:
        metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        available_keys = [k for k in metric_keys if k in performance and isinstance(performance[k], dict)]

        if available_keys:
            left_col, right_col = st.columns([3, 2])

            with left_col:
                section_header("Baseline vs Production Performance", "📊")
                baseline_vals = [performance[k].get("baseline", 0.0) for k in available_keys]
                current_vals = [performance[k].get("current", 0.0) for k in available_keys]
                labels = [k.replace("_", " ").title() for k in available_keys]

                fig_cmp = go.Figure()
                fig_cmp.add_trace(
                    go.Bar(
                        name="Baseline",
                        x=labels,
                        y=baseline_vals,
                        marker_color=CHART_COLORS["baseline"],
                        opacity=0.85,
                        text=[f"{v:.3f}" for v in baseline_vals],
                        textposition="outside",
                    )
                )
                fig_cmp.add_trace(
                    go.Bar(
                        name="Production Batch",
                        x=labels,
                        y=current_vals,
                        marker_color=CHART_COLORS["production"],
                        opacity=0.85,
                        text=[f"{v:.3f}" for v in current_vals],
                        textposition="outside",
                    )
                )
                cmp_layout = plotly_layout(title="Standard Evaluation Metrics (0 to 1 scale)", height=380)
                cmp_layout["barmode"] = "group"
                cmp_layout["yaxis"]["range"] = [0, 1.1]
                cmp_layout["yaxis"]["title"] = "Score"
                fig_cmp.update_layout(**cmp_layout)
                st.plotly_chart(fig_cmp, width="stretch", key="metrics_cmp_chart")

            with right_col:
                section_header("Metric Deltas & Trends", "📐")
                for m_key in available_keys:
                    m = performance[m_key]
                    delta = m.get("delta", 0.0)
                    arrow = "↑" if delta >= 0 else "↓"
                    delta_color = "#10b981" if delta >= 0 else "#ef4444"
                    label = m_key.replace("_", " ").title()
                    b_val = m.get("baseline", 0.0)
                    c_val = m.get("current", 0.0)

                    st.markdown(f"""
                    <div class="feature-item">
                        <div>
                            <span class="feature-name">{label}</span>
                            <span class="feature-type">Baseline: {b_val:.4f} → Batch: {c_val:.4f}</span>
                        </div>
                        <span style="color:{delta_color};font-weight:700;font-size:0.95rem;">
                            {arrow} {delta:+.4f}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Confusion Matrix ──────────────────────────────────────────────────
    section_header("Baseline Evaluation Confusion Matrix", "🧮")
    try:
        baseline_file = MODELS_DIR / "baseline_metrics.json"
        if baseline_file.exists():
            bm = load_json(baseline_file)
            cm = bm.get("confusion_matrix")
            if cm and len(cm) == 2 and len(cm[0]) == 2:
                labels = ["No Churn (0)", "Churn (1)"]
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Sample Count"),
                    x=labels,
                    y=labels,
                    text_auto=True,
                    color_continuous_scale=["#0a0e1a", "#1e3a8a", "#3b82f6", "#8b5cf6"],
                )
                cm_layout = plotly_layout(
                    title=f"Baseline Model: {bm.get('model_name', 'Trained Classifier')}",
                    height=340,
                )
                cm_layout["margin"] = dict(t=30, b=40, l=60, r=30)
                fig_cm.update_layout(**cm_layout)
                st.plotly_chart(fig_cm, width="stretch", key="cm_heatmap_chart")
            else:
                st.info("Confusion matrix format is non-standard or missing in baseline metrics.")
        else:
            st.info("Baseline metrics file `models/baseline_metrics.json` not found. Train models to populate.")
    except Exception as exc:
        st.info(f"Could not load confusion matrix: {exc}")

    # ── Prediction Confidence Telemetry ───────────────────────────────────
    if confidence_stats:
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Inference Confidence Telemetry", "🎯")

        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            mean_conf = confidence_stats.get("mean_confidence", 0.0)
            kpi_card(
                "Mean Confidence",
                f"{mean_conf:.1%}",
                "📊",
                CHART_COLORS["blue"],
            )
        with cc2:
            std_conf = confidence_stats.get("std_confidence", 0.0)
            kpi_card(
                "Confidence Std Dev",
                f"{std_conf:.4f}",
                "📏",
                CHART_COLORS["yellow"],
            )
        with cc3:
            low_pct = confidence_stats.get("low_confidence_pct", 0.0)
            kpi_card(
                "Uncertain Samples (<60%)",
                f"{low_pct:.1f}%",
                "⚡",
                CHART_COLORS["red"] if low_pct > 20 else CHART_COLORS["green"],
            )
