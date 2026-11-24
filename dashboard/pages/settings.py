"""
Settings page — Drift thresholds, system info, diagnostics, and architectural specs.
Integrates system metadata, threshold configuration, and author profile.
"""

import sys
import platform
import streamlit as st
import pandas as pd
from dashboard.components import (
    page_header,
    section_header,
    SYSTEM_URL,
    PORTFOLIO_URL,
    LINKEDIN_URL,
    LEETCODE_URL,
    GITHUB_URL,
    AUTHOR_NAME,
    SYSTEM_VERSION,
)
from src.config.settings import (
    DRIFT_THRESHOLDS,
    MODELS_DIR,
    PRODUCTION_DIR,
    REPORTS_DIR,
    LOG_FILE,
)


# Default backup thresholds for reset
DEFAULT_THRESHOLDS = {
    "ks_pvalue": 0.05,
    "psi_moderate": 0.10,
    "psi_severe": 0.25,
    "mean_shift_pct": 0.15,
    "variance_change_pct": 0.25,
    "chi2_pvalue": 0.05,
    "js_moderate": 0.10,
    "js_severe": 0.25,
    "drifted_features_pct_moderate": 0.20,
    "drifted_features_pct_severe": 0.50,
}


def render() -> None:
    """Render the configuration, diagnostic telemetry, and architectural specs page."""
    page_header(
        "Settings & System Architecture",
        "Calibrate statistical drift sensitivities, inspect system diagnostics, and review architectural specifications",
    )

    tab1, tab2, tab3 = st.tabs(["⚙️ Drift Thresholds", "📂 System Diagnostics", "ℹ️ Architecture & Author"])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1: Drift Sensitivity Thresholds
    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        section_header("Statistical Test Sensitivity Calibration", "📊")
        st.caption(
            "These thresholds determine when feature-level statistical tests flag distribution shifts and when dataset-level risk triggers alerts."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Numerical Feature Tests")
            ks_val = st.slider(
                "Kolmogorov-Smirnov (KS) p-value threshold",
                0.01, 0.20,
                float(DRIFT_THRESHOLDS.get("ks_pvalue", 0.05)),
                0.01,
                key="cfg_ks",
                help="Significance level (alpha). Features with p-value below this are flagged as drifted.",
            )
            psi_m = st.slider(
                "PSI Moderate Threshold",
                0.05, 0.30,
                float(DRIFT_THRESHOLDS.get("psi_moderate", 0.10)),
                0.01,
                key="cfg_psi_m",
                help="PSI between this and severe indicates moderate population shift.",
            )
            psi_s = st.slider(
                "PSI Severe Threshold",
                0.15, 0.50,
                float(DRIFT_THRESHOLDS.get("psi_severe", 0.25)),
                0.01,
                key="cfg_psi_s",
                help="PSI above this indicates critical population shift.",
            )
            mean_s = st.slider(
                "Mean Shift % Threshold",
                0.05, 0.50,
                float(DRIFT_THRESHOLDS.get("mean_shift_pct", 0.15)),
                0.01,
                key="cfg_mean_s",
                help="Relative percentage change in empirical mean to trigger drift alert.",
            )
            var_c = st.slider(
                "Variance Change % Threshold",
                0.10, 0.80,
                float(DRIFT_THRESHOLDS.get("variance_change_pct", 0.25)),
                0.01,
                key="cfg_var_c",
                help="Relative percentage change in empirical variance to trigger drift alert.",
            )

        with c2:
            st.markdown("#### Categorical Feature Tests")
            chi2_val = st.slider(
                "Chi-Square (χ²) p-value threshold",
                0.01, 0.20,
                float(DRIFT_THRESHOLDS.get("chi2_pvalue", 0.05)),
                0.01,
                key="cfg_chi2",
                help="Significance level for Chi-Square test of independence.",
            )
            js_m = st.slider(
                "Jensen-Shannon Moderate Threshold",
                0.05, 0.30,
                float(DRIFT_THRESHOLDS.get("js_moderate", 0.10)),
                0.01,
                key="cfg_js_m",
                help="JS distance indicating moderate distribution divergence.",
            )
            js_s = st.slider(
                "Jensen-Shannon Severe Threshold",
                0.15, 0.50,
                float(DRIFT_THRESHOLDS.get("js_severe", 0.25)),
                0.01,
                key="cfg_js_s",
                help="JS distance indicating severe distribution divergence.",
            )

            st.markdown("#### Dataset Overall Risk Levels")
            d_mod = st.slider(
                "Moderate Risk Threshold (% drifted features)",
                0.10, 0.50,
                float(DRIFT_THRESHOLDS.get("drifted_features_pct_moderate", 0.20)),
                0.05,
                key="cfg_d_mod",
            )
            d_sev = st.slider(
                "High Risk Threshold (% drifted features)",
                0.30, 0.80,
                float(DRIFT_THRESHOLDS.get("drifted_features_pct_severe", 0.50)),
                0.05,
                key="cfg_d_sev",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns([1, 1])

        with btn_col1:
            if st.button("💾 Apply Calibrated Thresholds", type="primary", key="save_thresholds_btn", width="stretch"):
                DRIFT_THRESHOLDS.update({
                    "ks_pvalue": ks_val,
                    "psi_moderate": psi_m,
                    "psi_severe": psi_s,
                    "mean_shift_pct": mean_s,
                    "variance_change_pct": var_c,
                    "chi2_pvalue": chi2_val,
                    "js_moderate": js_m,
                    "js_severe": js_s,
                    "drifted_features_pct_moderate": d_mod,
                    "drifted_features_pct_severe": d_sev,
                })
                st.success("✅ Threshold sensitivities applied to active runtime session.")

        with btn_col2:
            if st.button("🔄 Reset to Recommended Defaults", type="secondary", key="reset_thresholds_btn", width="stretch"):
                DRIFT_THRESHOLDS.update(DEFAULT_THRESHOLDS)
                st.toast("✅ Reverted thresholds to standard production defaults.")
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2: System Diagnostics & Artifacts
    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        section_header("Storage & File System Paths", "📂")
        system_paths = {
            "Model Artifacts Directory": str(MODELS_DIR),
            "Production Batches Directory": str(PRODUCTION_DIR),
            "Drift Reports Directory": str(REPORTS_DIR),
            "System Audit Log": str(LOG_FILE),
        }
        for label, p_str in system_paths.items():
            st.markdown(f"""
            <div class="feature-item">
                <span class="feature-name">{label}</span>
                <span style="font-family:monospace;font-size:0.78rem;color:#cbd5e1;">{p_str}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("ML Model Artifact Inventory", "🤖")
        model_artifacts = [
            ("Best Model Weights", MODELS_DIR / "best_model.joblib"),
            ("Preprocessing Pipeline", MODELS_DIR / "preprocessor.joblib"),
            ("Baseline Performance Metrics", MODELS_DIR / "baseline_metrics.json"),
            ("Model Training Summary", MODELS_DIR / "training_summary.json"),
        ]

        for name, file_path in model_artifacts:
            exists = file_path.exists()
            icon = "✅" if exists else "❌"
            size_str = f"{file_path.stat().st_size / 1024:.1f} KB" if exists else "Not Generated"
            st.markdown(f"""
            <div class="feature-item">
                <span class="feature-name">{icon} {name}</span>
                <span style="font-size:0.82rem;color:{'#34d399' if exists else '#f87171'};font-weight:600;">{size_str}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Production Batches Repository", "📊")
        batch_files = sorted(PRODUCTION_DIR.glob("*.csv")) if PRODUCTION_DIR.exists() else []
        if batch_files:
            batch_data = []
            for b in batch_files:
                size_kb = b.stat().st_size / 1024
                batch_data.append({
                    "Batch Filename": b.name,
                    "File Size": f"{size_kb:.1f} KB",
                    "Status": "Available",
                })
            st.dataframe(pd.DataFrame(batch_data), hide_index=True, width="stretch")
        else:
            st.info("No CSV batches found in production directory.")

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Runtime Environment Diagnostics", "💻")
        env_info = {
            "Python Version": platform.python_version(),
            "Streamlit Version": st.__version__,
            "Operating System": f"{platform.system()} {platform.release()}",
            "Platform Architecture": platform.machine(),
            "Test Suite": "73 passing unit & integration tests (pytest)",
        }
        for k, v in env_info.items():
            st.markdown(f"""
            <div class="feature-item">
                <span class="feature-name">{k}</span>
                <span style="font-weight:600;font-size:0.84rem;color:#60a5fa;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3: Architecture & Author Profile
    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        section_header("System Architect & Author", "👤")

        st.markdown(f"""
        <div style="background:linear-gradient(145deg, rgba(30,41,59,0.8), rgba(15,23,42,0.8));border:1px solid rgba(59,130,246,0.3);border-radius:16px;padding:24px;margin-bottom:24px;">
            <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:14px;">
                <div style="font-size:2.4rem;">👨‍💻</div>
                <div>
                    <div style="font-size:1.4rem;font-weight:800;color:#f8fafc;">{AUTHOR_NAME}</div>
                    <div style="font-size:0.88rem;color:#60a5fa;font-weight:600;">AI / Machine Learning Systems Architect</div>
                </div>
                <div style="margin-left:auto;">
                    <a href="{SYSTEM_URL}" target="_blank" rel="noopener noreferrer" class="nav-pill-link live-btn" style="font-size:0.85rem;padding:8px 16px;">
                        🌐 Live App: dddsystem.vercel.app ↗
                    </a>
                </div>
            </div>
            <div style="color:#cbd5e1;font-size:0.92rem;line-height:1.6;margin-bottom:18px;">
                Engineered the <strong>Data Drift Detection System {SYSTEM_VERSION}</strong> to provide an autonomous,
                statistically rigorous monitoring platform for production ML models. Incorporates 8 statistical drift algorithms,
                cross-validated classifier tracking, glassmorphism UI telemetry, and actionable alerts.
            </div>
            <div style="display:flex;gap:12px;flex-wrap:wrap;">
                <a href="{PORTFOLIO_URL}" target="_blank" rel="noopener noreferrer" class="social-btn" style="padding:8px 14px;font-size:0.82rem;">
                    🌐 Author Portfolio ↗
                </a>
                <a href="{LINKEDIN_URL}" target="_blank" rel="noopener noreferrer" class="social-btn" style="padding:8px 14px;font-size:0.82rem;">
                    💼 LinkedIn Profile ↗
                </a>
                <a href="{LEETCODE_URL}" target="_blank" rel="noopener noreferrer" class="social-btn" style="padding:8px 14px;font-size:0.82rem;">
                    ⚡ LeetCode Profile ↗
                </a>
                <a href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer" class="social-btn" style="padding:8px 14px;font-size:0.82rem;">
                    🐙 GitHub Repository ↗
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

        section_header("The 8 Statistical Drift Tests in This Engine", "📐")
        tests_data = [
            ("Kolmogorov-Smirnov (KS) Test", "Non-parametric maximum distance test between continuous baseline & production empirical cumulative distribution functions (ECDF)."),
            ("Population Stability Index (PSI)", "Quantile-binned logarithmic divergence score measuring shift in score distributions: $\\sum (P - B) \\ln(P / B)$."),
            ("Jensen-Shannon (JS) Divergence", "Symmetric, bounded (0 to 1) distance derived from Kullback-Leibler divergence for both numerical and categorical probability vectors."),
            ("Mean Shift Test", "Relative percentage change in empirical first moment: $\\frac{|\\mu_{\\text{prod}} - \\mu_{\\text{base}}|}{\\mu_{\\text{base}} + \\epsilon}$."),
            ("Variance Change Test", "Relative percentage change in empirical second moment: $\\frac{|\\sigma^2_{\\text{prod}} - \\sigma^2_{\\text{base}}|}{\\sigma^2_{\\text{base}} + \\epsilon}$."),
            ("Chi-Square (χ²) Goodness-of-Fit", "Independence test on contingency counts between baseline and production categorical occurrences."),
            ("Novel Category Discovery", "Identifies newly introduced unseen levels in categorical features not present during model training."),
            ("Missing Value Drift", "Monitors percentage point increase in missing (null/NaN) data against baseline expectations."),
        ]
        for t_name, t_desc in tests_data:
            st.markdown(f"""
            <div class="feature-item">
                <span class="feature-name">{t_name}</span>
                <span style="font-size:0.82rem;color:#cbd5e1;max-width:600px;text-align:right;">{t_desc}</span>
            </div>
            """, unsafe_allow_html=True)
