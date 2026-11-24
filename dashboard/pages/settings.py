"""
Settings page — Drift thresholds, display preferences, system info.
"""

import streamlit as st
from dashboard.components import page_header, section_header
from src.config.settings import DRIFT_THRESHOLDS, MODELS_DIR, PRODUCTION_DIR, REPORTS_DIR, LOG_FILE


def render():
    """Render the settings page."""
    page_header("Settings", "Configure drift thresholds, review system info, and manage preferences")

    tab1, tab2, tab3 = st.tabs(["⚙️ Thresholds", "📂 System Info", "🎨 Display"])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1: Drift Thresholds
    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        section_header("Statistical Test Thresholds", "📊")
        st.caption("These thresholds determine when a feature is flagged as drifted.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Numerical Tests**")
            ks = st.slider("KS p-value threshold", 0.01, 0.20, float(DRIFT_THRESHOLDS["ks_pvalue"]),
                           0.01, key="s_ks", help="Lower = more strict")
            psi_m = st.slider("PSI moderate threshold", 0.05, 0.30, float(DRIFT_THRESHOLDS["psi_moderate"]),
                              0.01, key="s_psi_m")
            psi_s = st.slider("PSI severe threshold", 0.15, 0.50, float(DRIFT_THRESHOLDS["psi_severe"]),
                              0.01, key="s_psi_s")
            mean_s = st.slider("Mean shift %", 0.05, 0.50, float(DRIFT_THRESHOLDS["mean_shift_pct"]),
                               0.01, key="s_mean")
            var_c = st.slider("Variance change %", 0.10, 0.80, float(DRIFT_THRESHOLDS["variance_change_pct"]),
                              0.01, key="s_var")

        with c2:
            st.markdown("**Categorical Tests**")
            chi2 = st.slider("Chi² p-value threshold", 0.01, 0.20, float(DRIFT_THRESHOLDS["chi2_pvalue"]),
                             0.01, key="s_chi2")
            js_m = st.slider("JS moderate threshold", 0.05, 0.30, float(DRIFT_THRESHOLDS["js_moderate"]),
                             0.01, key="s_js_m")
            js_s = st.slider("JS severe threshold", 0.15, 0.50, float(DRIFT_THRESHOLDS["js_severe"]),
                             0.01, key="s_js_s")

            st.markdown("**Dataset Risk Levels**")
            d_mod = st.slider("Moderate risk (% features)", 0.10, 0.50,
                               float(DRIFT_THRESHOLDS["drifted_features_pct_moderate"]), 0.05, key="s_dm")
            d_sev = st.slider("High risk (% features)", 0.30, 0.80,
                               float(DRIFT_THRESHOLDS["drifted_features_pct_severe"]), 0.05, key="s_ds")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Apply Thresholds", type="primary", key="apply_thresh"):
            DRIFT_THRESHOLDS.update({
                "ks_pvalue": ks, "psi_moderate": psi_m, "psi_severe": psi_s,
                "mean_shift_pct": mean_s, "variance_change_pct": var_c,
                "chi2_pvalue": chi2, "js_moderate": js_m, "js_severe": js_s,
                "drifted_features_pct_moderate": d_mod, "drifted_features_pct_severe": d_sev,
            })
            st.success("✅ Thresholds updated for current session.")
            st.info("Note: Changes apply to the current session only. Edit `src/config/settings.py` for permanent changes.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2: System Info
    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        section_header("System Paths", "📂")
        paths = {
            "Models Directory": str(MODELS_DIR),
            "Production Batches": str(PRODUCTION_DIR),
            "Reports Directory": str(REPORTS_DIR),
            "Log File": str(LOG_FILE),
        }
        for label, path in paths.items():
            st.markdown(f"""
            <div class="feature-item">
                <span class="feature-name">{label}</span>
                <span style="font-family:monospace;font-size:0.78rem;color:#64748b;">{path}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Model Artifacts", "🤖")
        model_path = MODELS_DIR / "best_model.joblib"
        prep_path = MODELS_DIR / "preprocessor.joblib"
        metrics_path = MODELS_DIR / "baseline_metrics.json"

        artifacts = [
            ("Best Model", model_path),
            ("Preprocessor", prep_path),
            ("Baseline Metrics", metrics_path),
        ]
        for name, path in artifacts:
            exists = path.exists()
            icon = "✅" if exists else "❌"
            size = f"{path.stat().st_size / 1024:.0f} KB" if exists else "Not found"
            st.markdown(f"""
            <div class="feature-item">
                <span class="feature-name">{icon} {name}</span>
                <span style="font-size:0.8rem;color:#64748b;">{size}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Batch Files", "📊")
        batch_files = sorted(PRODUCTION_DIR.glob("*.csv"))
        if batch_files:
            for b in batch_files:
                size = f"{b.stat().st_size / 1024:.0f} KB"
                st.markdown(f"""
                <div class="feature-item">
                    <span class="feature-name">📄 {b.name}</span>
                    <span style="font-size:0.8rem;color:#64748b;">{size}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No batch files found.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3: Display Preferences
    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        section_header("Display Preferences", "🎨")
        st.info("Theme is controlled by `.streamlit/config.toml`. The dashboard uses a premium dark theme by default.")

        section_header("About", "ℹ️")
        st.markdown("""
        <div style="color:#94a3b8;font-size:0.88rem;line-height:1.8;">
            <strong style="color:#f0f4f8;">Data Drift Detection System v2.0</strong><br>
            Production-grade ML monitoring dashboard.<br><br>
            <strong style="color:#94a3b8;">Stack:</strong> Python · Streamlit · scikit-learn · XGBoost · Plotly<br>
            <strong style="color:#94a3b8;">Tests:</strong> 73 passing (pytest)<br>
            <strong style="color:#94a3b8;">Models:</strong> 4 algorithms with cross-validation<br>
            <strong style="color:#94a3b8;">Stats:</strong> 8 drift detection tests
        </div>
        """, unsafe_allow_html=True)
