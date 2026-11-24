"""
Feature Analysis page — Distribution comparisons, PSI/KS analysis, statistical tables.
Provides interactive statistical drill-downs for numerical, categorical, and missingness drift.
"""

from typing import Optional
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from dashboard.components import (
    page_header,
    section_header,
    empty_state,
    plotly_layout,
    CHART_COLORS,
)


def render(
    drift_report: Optional[dict] = None,
    baseline_df: Optional[pd.DataFrame] = None,
    production_df: Optional[pd.DataFrame] = None,
) -> None:
    """Render the feature drift analysis drill-down page."""
    page_header(
        "Feature Drift Analysis",
        "Deep-dive statistical audit into individual feature distributions, shifts, and hypothesis tests",
    )

    if not drift_report or baseline_df is None or production_df is None:
        empty_state(
            "🔬",
            "No Data Available for Feature Drill-Down",
            "Please run drift analysis from the 'Upload & Run' page to populate distribution comparisons.",
        )
        return

    feature_drift = drift_report.get("feature_drift", {})

    tab1, tab2, tab3 = st.tabs(["📊 Numerical Features", "📝 Categorical Features", "❓ Missingness & Quality"])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1: Numerical Features
    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        num_results = feature_drift.get("numerical", {})
        if not num_results:
            empty_state("📊", "No Numerical Features", "No numerical features were identified or evaluated in this dataset.")
        else:
            section_header("Distribution Comparison", "📉")
            feature_options = list(num_results.keys())
            selected_feature = st.selectbox(
                "Select Numerical Feature:",
                feature_options,
                key="num_feature_selector",
                help="Select a feature to inspect its baseline vs. production empirical distribution overlay.",
            )

            if selected_feature and selected_feature in baseline_df.columns and selected_feature in production_df.columns:
                b_raw = pd.to_numeric(baseline_df[selected_feature], errors="coerce").dropna()
                p_raw = pd.to_numeric(production_df[selected_feature], errors="coerce").dropna()

                # Filter finite values for safety
                b_vals = b_raw[np.isfinite(b_raw)]
                p_vals = p_raw[np.isfinite(p_raw)]

                # Distribution Overlay Chart
                fig_dist = go.Figure()
                fig_dist.add_trace(
                    go.Histogram(
                        x=b_vals,
                        name="Baseline",
                        opacity=0.65,
                        marker_color=CHART_COLORS["baseline"],
                        nbinsx=40,
                        histnorm="probability density",
                    )
                )
                fig_dist.add_trace(
                    go.Histogram(
                        x=p_vals,
                        name="Production",
                        opacity=0.65,
                        marker_color=CHART_COLORS["production"],
                        nbinsx=40,
                        histnorm="probability density",
                    )
                )
                dist_layout = plotly_layout(
                    title=f"{selected_feature} — Empirical Density Overlay (Baseline vs Production)",
                    height=360,
                )
                dist_layout["barmode"] = "overlay"
                dist_layout["xaxis"]["title"] = selected_feature
                dist_layout["yaxis"]["title"] = "Density"
                fig_dist.update_layout(**dist_layout)
                st.plotly_chart(fig_dist, width="stretch", key="dist_overlay_chart")

                # Test Results for Selected Feature
                tests = num_results.get(selected_feature, {})
                ks = tests.get("ks_test", {})
                psi = tests.get("psi", {})
                ms = tests.get("mean_shift", {})
                vc = tests.get("variance_change", {})
                js = tests.get("jensen_shannon", {})

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    ks_drift = ks.get("drift_detected", False)
                    ks_p = ks.get("p_value", 0.0)
                    ks_p_str = f"{ks_p:.4e}" if isinstance(ks_p, float) and ks_p < 0.001 else f"{ks_p:.4f}" if isinstance(ks_p, (int, float)) else str(ks_p)
                    st.metric(
                        "Kolmogorov-Smirnov",
                        f"p = {ks_p_str}",
                        f"{'🔴 Drift' if ks_drift else '🟢 Stable'}",
                        delta_color="normal" if not ks_drift else "inverse",
                    )
                with c2:
                    psi_val = psi.get("value", 0.0)
                    psi_val_str = f"{psi_val:.4f}" if isinstance(psi_val, (int, float)) else str(psi_val)
                    psi_sev = psi.get("severity", "LOW")
                    st.metric("PSI Stability", psi_val_str, f"Severity: {psi_sev}")
                with c3:
                    ms_drift = ms.get("drift_detected", False)
                    ms_pct = ms.get("pct_change", 0.0)
                    ms_str = f"{ms_pct:+.2f}%" if isinstance(ms_pct, (int, float)) else str(ms_pct)
                    st.metric(
                        "Mean Shift",
                        ms_str,
                        f"{'🔴 Drift' if ms_drift else '🟢 Stable'}",
                        delta_color="normal" if not ms_drift else "inverse",
                    )
                with c4:
                    vc_pct = vc.get("pct_change", 0.0)
                    vc_str = f"{vc_pct:+.2f}%" if isinstance(vc_pct, (int, float)) else str(vc_pct)
                    js_val = js.get("value", 0.0)
                    js_str = f"{js_val:.4f}" if isinstance(js_val, (int, float)) else str(js_val)
                    st.metric("Variance Δ / JS", f"Δ {vc_str}", f"JS: {js_str}")

                # Feature Distribution Summary Stats
                b_mean = float(b_vals.mean()) if not b_vals.empty else 0.0
                p_mean = float(p_vals.mean()) if not p_vals.empty else 0.0
                b_std = float(b_vals.std()) if not b_vals.empty else 0.0
                p_std = float(p_vals.std()) if not p_vals.empty else 0.0

                st.caption(
                    f"**Summary Metrics:** Baseline Mean = `{b_mean:.3f}` (± `{b_std:.3f}`) | "
                    f"Production Mean = `{p_mean:.3f}` (± `{p_std:.3f}`) | "
                    f"Sample Counts: Baseline = `{len(b_vals):,}` samples, Production = `{len(p_vals):,}` samples."
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Full Numerical Tests Summary Table
            section_header("All Numerical Statistical Tests", "📋")
            table_rows = []
            for feat_name, t_dict in num_results.items():
                ks_t = t_dict.get("ks_test", {})
                psi_t = t_dict.get("psi", {})
                js_t = t_dict.get("jensen_shannon", {})
                ms_t = t_dict.get("mean_shift", {})
                vc_t = t_dict.get("variance_change", {})

                ks_stat = ks_t.get("statistic")
                ks_pval = ks_t.get("p_value")
                psi_v = psi_t.get("value")
                js_v = js_t.get("value")
                ms_p = ms_t.get("pct_change")
                vc_p = vc_t.get("pct_change")

                table_rows.append({
                    "Feature": feat_name,
                    "KS Stat": f"{ks_stat:.4f}" if isinstance(ks_stat, (int, float)) else "—",
                    "KS p-value": f"{ks_pval:.4e}" if isinstance(ks_pval, (int, float)) and ks_pval < 0.001 else (f"{ks_pval:.4f}" if isinstance(ks_pval, (int, float)) else "—"),
                    "KS Drift": "🚨 YES" if ks_t.get("drift_detected") else "STABLE",
                    "PSI": f"{psi_v:.4f}" if isinstance(psi_v, (int, float)) else "—",
                    "PSI Severity": psi_t.get("severity", "LOW"),
                    "JS Dist": f"{js_v:.4f}" if isinstance(js_v, (int, float)) else "—",
                    "Mean Δ%": f"{ms_p:+.2f}%" if isinstance(ms_p, (int, float)) else "—",
                    "Variance Δ%": f"{vc_p:+.2f}%" if isinstance(vc_p, (int, float)) else "—",
                })

            if table_rows:
                st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2: Categorical Features
    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        cat_results = feature_drift.get("categorical", {})
        if not cat_results:
            empty_state("📝", "No Categorical Features", "No categorical features were detected in this dataset.")
        else:
            section_header("Category Distribution Analysis", "📊")
            cat_options = list(cat_results.keys())
            selected_cat = st.selectbox(
                "Select Categorical Feature:",
                cat_options,
                key="cat_feature_selector",
                help="Compare relative category frequencies between baseline and production batches.",
            )

            if selected_cat and selected_cat in baseline_df.columns and selected_cat in production_df.columns:
                b_counts = baseline_df[selected_cat].astype(str).value_counts(normalize=True).reset_index()
                b_counts.columns = ["Category", "Frequency"]
                b_counts["Batch"] = "Baseline"

                p_counts = production_df[selected_cat].astype(str).value_counts(normalize=True).reset_index()
                p_counts.columns = ["Category", "Frequency"]
                p_counts["Batch"] = "Production"

                merged_counts = pd.concat([b_counts, p_counts], ignore_index=True)

                fig_cat = px.bar(
                    merged_counts,
                    x="Category",
                    y="Frequency",
                    color="Batch",
                    barmode="group",
                    color_discrete_map={
                        "Baseline": CHART_COLORS["baseline"],
                        "Production": CHART_COLORS["production"],
                    },
                )
                cat_layout = plotly_layout(
                    title=f"{selected_cat} — Normalized Category Frequencies",
                    height=350,
                )
                cat_layout["yaxis"]["title"] = "Relative Frequency"
                cat_layout["xaxis"]["title"] = selected_cat
                fig_cat.update_layout(**cat_layout)
                st.plotly_chart(fig_cat, width="stretch", key="cat_freq_chart")

            st.markdown("<br>", unsafe_allow_html=True)

            # Chi-Square and New Categories Table
            section_header("Chi-Square & Novel Categories", "📋")
            cat_table_rows = []
            for feat_name, t_dict in cat_results.items():
                chi = t_dict.get("chi_square", {})
                nc = t_dict.get("new_categories", {})
                js = t_dict.get("jensen_shannon", {})

                chi_stat = chi.get("statistic")
                chi_pval = chi.get("p_value")
                new_cats = nc.get("new_categories", [])

                cat_table_rows.append({
                    "Feature": feat_name,
                    "χ² Stat": f"{chi_stat:.3f}" if isinstance(chi_stat, (int, float)) else "—",
                    "χ² p-val": f"{chi_pval:.4f}" if isinstance(chi_pval, (int, float)) else "—",
                    "χ² Drift": "🚨 YES" if chi.get("drift_detected") else "STABLE",
                    "JS Divergence": f"{js.get('value', 0.0):.4f}" if isinstance(js.get("value"), (int, float)) else "—",
                    "New Categories Count": nc.get("new_count", 0),
                    "Discovered Categories": ", ".join(f"'{c}'" for c in new_cats) if new_cats else "None",
                })

            if cat_table_rows:
                st.dataframe(pd.DataFrame(cat_table_rows), hide_index=True, width="stretch")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3: Missingness & Quality
    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        missing_info = feature_drift.get("missing_data", {})
        if not missing_info:
            empty_state("❓", "No Missingness Telemetry", "Missing data analysis is not available for this run.")
        else:
            section_header("Missing Data Rate Comparison", "❓")
            miss_table = []
            for feat_name, info in missing_info.items():
                b_pct = info.get("baseline_missing_pct", 0.0)
                p_pct = info.get("production_missing_pct", 0.0)
                inc = info.get("increase", 0.0)
                drift_flag = info.get("drift_detected", False)

                miss_table.append({
                    "Feature": feat_name,
                    "Baseline Missing %": f"{b_pct:.2f}%",
                    "Production Missing %": f"{p_pct:.2f}%",
                    "Delta Increase %": f"{inc:+.2f}%",
                    "Status": "⚠️ ELEVATED" if drift_flag else "HEALTHY",
                    "_inc_val": inc,
                })

            miss_df = pd.DataFrame(miss_table)
            display_cols = ["Feature", "Baseline Missing %", "Production Missing %", "Delta Increase %", "Status"]
            st.dataframe(miss_df[display_cols], hide_index=True, width="stretch")

            # Chart of features with positive missing data increase
            flagged = miss_df[miss_df["_inc_val"] > 0].sort_values("_inc_val", ascending=True)
            if not flagged.empty:
                st.markdown("<br>", unsafe_allow_html=True)
                section_header("Missing Data Increase Impact", "📈")
                fig_miss = go.Figure(
                    go.Bar(
                        y=flagged["Feature"],
                        x=flagged["_inc_val"],
                        orientation="h",
                        marker_color=CHART_COLORS["yellow"],
                        text=flagged["_inc_val"].round(2).astype(str) + "%",
                        textposition="outside",
                        textfont=dict(size=11, color="#cbd5e1"),
                        cliponaxis=False,
                    )
                )
                miss_layout = plotly_layout(
                    title="Missingness Growth (%) in Production",
                    height=max(220, len(flagged) * 32 + 80),
                )
                miss_layout["xaxis"]["title"] = "Percentage Point Increase (%)"
                fig_miss.update_layout(**miss_layout)
                st.plotly_chart(fig_miss, width="stretch", key="missing_growth_chart")
            else:
                st.success("✅ Zero features have an increased rate of missing values in production.")
