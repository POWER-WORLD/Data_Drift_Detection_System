"""
Feature Analysis page — Distribution comparisons, PSI/KS analysis, statistical tables.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from dashboard.components import page_header, section_header, empty_state, plotly_layout, CHART_COLORS


def render(drift_report: dict, baseline_df: pd.DataFrame, production_df: pd.DataFrame):
    """Render the feature drift analysis page."""
    page_header("Feature Drift Analysis", "Deep-dive into individual feature distributions and statistical tests")

    feature_drift = drift_report.get("feature_drift", {})

    tab1, tab2, tab3 = st.tabs(["📊 Numerical", "📝 Categorical", "❓ Missing Data"])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1: Numerical Features
    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        num_results = feature_drift.get("numerical", {})
        if not num_results:
            empty_state("📊", "No Numerical Results", "No numerical features were analyzed.")
        else:
            # Distribution Comparison
            section_header("Distribution Comparison", "📉")
            sel = st.selectbox("Select feature:", list(num_results.keys()), key="num_sel")

            if sel:
                b_vals = pd.to_numeric(baseline_df[sel], errors="coerce").dropna()
                p_vals = pd.to_numeric(production_df[sel], errors="coerce").dropna()

                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=b_vals, name="Baseline", opacity=0.65,
                    marker_color=CHART_COLORS["baseline"], nbinsx=40,
                ))
                fig.add_trace(go.Histogram(
                    x=p_vals, name="Production", opacity=0.65,
                    marker_color=CHART_COLORS["production"], nbinsx=40,
                ))
                layout = plotly_layout(title=f"{sel} — Distribution Overlay", height=350)
                layout["barmode"] = "overlay"
                layout["xaxis"]["title"] = sel
                layout["yaxis"]["title"] = "Count"
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True, key="dist_overlay")

                # Compact test results for selected feature
                tests = num_results[sel]
                c1, c2, c3 = st.columns(3)
                ks = tests.get("ks_test", {})
                psi = tests.get("psi", {})
                ms = tests.get("mean_shift", {})

                with c1:
                    drift_icon = "🔴" if ks.get("drift_detected") else "🟢"
                    st.metric("KS p-value", f"{ks.get('p_value', '—')}", f"{drift_icon} {'Drift' if ks.get('drift_detected') else 'Stable'}")
                with c2:
                    st.metric("PSI", f"{psi.get('value', '—')}", psi.get("severity", "—"))
                with c3:
                    st.metric("Mean Shift", f"{ms.get('pct_change', '—')}%", f"{'Drift' if ms.get('drift_detected') else 'Stable'}")

            st.markdown("<br>", unsafe_allow_html=True)

            # Full statistics table
            section_header("All Statistical Tests", "📋")
            rows = []
            for col, tests in num_results.items():
                ks = tests.get("ks_test", {})
                psi = tests.get("psi", {})
                js = tests.get("jensen_shannon", {})
                ms = tests.get("mean_shift", {})
                vc = tests.get("variance_change", {})
                rows.append({
                    "Feature": col,
                    "KS Stat": ks.get("statistic", "—"),
                    "KS p-val": ks.get("p_value", "—"),
                    "KS Drift": "✅" if ks.get("drift_detected") else "—",
                    "PSI": psi.get("value", "—"),
                    "PSI Sev": psi.get("severity", "—"),
                    "JS Dist": js.get("value", "—"),
                    "Mean Δ%": ms.get("pct_change", "—"),
                    "Var Δ%": vc.get("pct_change", "—"),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2: Categorical Features
    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        cat_results = feature_drift.get("categorical", {})
        if not cat_results:
            empty_state("📝", "No Categorical Results", "No categorical features were analyzed.")
        else:
            section_header("Category Distribution", "📊")
            sel_cat = st.selectbox("Select feature:", list(cat_results.keys()), key="cat_sel")

            if sel_cat:
                b_vc = baseline_df[sel_cat].astype(str).value_counts(normalize=True).reset_index()
                b_vc.columns = ["Category", "Frequency"]
                b_vc["Source"] = "Baseline"
                p_vc = production_df[sel_cat].astype(str).value_counts(normalize=True).reset_index()
                p_vc.columns = ["Category", "Frequency"]
                p_vc["Source"] = "Production"
                combined = pd.concat([b_vc, p_vc])

                fig3 = px.bar(combined, x="Category", y="Frequency", color="Source", barmode="group",
                              color_discrete_map={"Baseline": CHART_COLORS["baseline"],
                                                  "Production": CHART_COLORS["production"]})
                layout3 = plotly_layout(title=f"{sel_cat} — Category Frequencies", height=350)
                layout3["yaxis"]["title"] = "Relative Frequency"
                fig3.update_layout(**layout3)
                st.plotly_chart(fig3, use_container_width=True, key="cat_freq")

            st.markdown("<br>", unsafe_allow_html=True)

            section_header("Chi-Square & New Categories", "📋")
            cat_rows = []
            for col, tests in cat_results.items():
                chi = tests.get("chi_square", {})
                nc = tests.get("new_categories", {})
                cat_rows.append({
                    "Feature": col,
                    "χ² Stat": chi.get("statistic", "—"),
                    "χ² p-val": chi.get("p_value", "—"),
                    "χ² Drift": "✅" if chi.get("drift_detected") else "—",
                    "New Cats": nc.get("new_count", 0),
                    "Values": str(nc.get("new_categories", [])),
                })
            st.dataframe(pd.DataFrame(cat_rows), hide_index=True, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3: Missing Data
    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        missing = feature_drift.get("missing_data", {})
        if not missing:
            empty_state("❓", "No Missing Data Info", "Missing data analysis not available.")
        else:
            section_header("Missing Data Comparison", "❓")
            miss_rows = []
            for col, info in missing.items():
                miss_rows.append({
                    "Feature": col,
                    "Baseline %": info.get("baseline_missing_pct", 0),
                    "Production %": info.get("production_missing_pct", 0),
                    "Increase %": info.get("increase", 0),
                    "Drift": "✅" if info.get("drift_detected") else "—",
                })
            miss_df = pd.DataFrame(miss_rows)
            st.dataframe(miss_df, hide_index=True, use_container_width=True)

            # Bar chart for increased missing
            flagged = miss_df[miss_df["Increase %"] > 0].sort_values("Increase %", ascending=True)
            if not flagged.empty:
                fig4 = go.Figure(go.Bar(
                    y=flagged["Feature"], x=flagged["Increase %"],
                    orientation="h",
                    marker_color=CHART_COLORS["yellow"],
                    text=flagged["Increase %"].round(2),
                    textposition="outside",
                    textfont=dict(size=11, color="#94a3b8"),
                ))
                layout4 = plotly_layout(height=max(200, len(flagged) * 30 + 80))
                layout4["xaxis"]["title"] = "Missing % Increase"
                fig4.update_layout(**layout4)
                st.plotly_chart(fig4, use_container_width=True, key="missing_bar")
