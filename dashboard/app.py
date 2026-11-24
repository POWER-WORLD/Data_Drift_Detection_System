"""
Data Drift Detection System — Main Dashboard Application.
Premium enterprise-grade Streamlit monitoring application with glassmorphism UI,
deep-dive statistical audits, model degradation tracking, and author attribution.
"""

import sys
from pathlib import Path

# Ensure project root is available in module path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dashboard.styles import get_custom_css
from dashboard.components import (
    render_top_nav,
    render_sidebar_author,
    render_footer,
    empty_state,
    risk_badge,
    SYSTEM_URL,
    PORTFOLIO_URL,
    GITHUB_URL,
    AUTHOR_NAME,
    SYSTEM_VERSION,
)

# ── Streamlit Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="Data Drift Detection — ML Intelligence & Monitoring",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject Global Design System CSS ───────────────────────────────────────
st.markdown(get_custom_css(), unsafe_allow_html=True)

# ── Initialize Session State Defaults ─────────────────────────────────────
SESSION_DEFAULTS = {
    "analysis_complete": False,
    "drift_report": None,
    "performance": None,
    "confidence_stats": None,
    "baseline_df": None,
    "production_df": None,
    "batch_name": None,
    "report_paths": {},
    "nav_page": "🏠  Overview",
}
for key, val in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


def render_welcome() -> None:
    """Welcome hero screen when no analysis batch has been evaluated."""
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-title">ML Data Drift Detection Engine</div>
        <div class="hero-desc">
            Autonomous data distribution monitoring, 8 statistical drift algorithms,
            real-time model degradation tracking, and proactive alert intelligence.
        </div>
        <div class="hero-cta-group">
            <a href="{SYSTEM_URL}" target="_blank" rel="noopener noreferrer" class="hero-btn hero-btn-primary">
                🌐 Visit Live Web System ↗
            </a>
            <a href="{PORTFOLIO_URL}" target="_blank" rel="noopener noreferrer" class="hero-btn hero-btn-secondary">
                👨‍💻 Architect: {AUTHOR_NAME} ↗
            </a>
            <a href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer" class="hero-btn hero-btn-secondary">
                🐙 View on GitHub ↗
            </a>
        </div>
        <div class="hero-features">
            <div class="hero-feat">
                <span class="hero-feat-icon">📊</span>
                <div class="hero-feat-text">8 Statistical Drift Tests</div>
            </div>
            <div class="hero-feat">
                <span class="hero-feat-icon">🤖</span>
                <div class="hero-feat-text">4 Cross-Validated ML Models</div>
            </div>
            <div class="hero-feat">
                <span class="hero-feat-icon">🔔</span>
                <div class="hero-feat-text">Autonomous Alert Center</div>
            </div>
            <div class="hero-feat">
                <span class="hero-feat-icon">📈</span>
                <div class="hero-feat-text">Interactive Plotly Telemetry</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👈 Navigate to **Upload & Run** in the sidebar to upload a new batch or run analysis on existing test batches.")


# ── Top Utility Navbar ────────────────────────────────────────────────────
render_top_nav()

# ── Sidebar Navigation & Profile ──────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-box">
        <div class="sidebar-brand-icon">🔷</div>
        <div class="sidebar-brand-title">Drift Detector</div>
        <div class="sidebar-brand-sub">ML Monitoring Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding:0 8px 12px;">
        <a href="{SYSTEM_URL}" target="_blank" rel="noopener noreferrer" class="nav-pill-link live-btn" style="width:100%;text-align:center;justify-content:center;display:flex;">
            🌐 Production System ↗
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin:8px 0;border-top:1px solid rgba(255,255,255,0.06);'></div>", unsafe_allow_html=True)

    nav_options = [
        "🏠  Overview",
        "🔬  Drift Analysis",
        "🤖  Model Health",
        "🔔  Alerts",
        "🚀  Upload & Run",
        "⚙️  Settings",
    ]
    page = st.radio(
        "Navigation Menu",
        nav_options,
        index=nav_options.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in nav_options else 0,
        label_visibility="collapsed",
    )
    st.session_state["nav_page"] = page

    st.markdown("<div style='margin:12px 0;border-top:1px solid rgba(255,255,255,0.06);'></div>", unsafe_allow_html=True)

    # Status summary panel
    batch_name = st.session_state.get("batch_name")
    drift_report = st.session_state.get("drift_report")
    risk_level = None
    if drift_report:
        risk_level = drift_report.get("dataset_drift", {}).get("risk_level", "LOW")

    if batch_name:
        st.markdown(f"""
        <div class="sidebar-card">
            <div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:1.4px;color:#94a3b8;font-weight:700;margin-bottom:6px;">
                Active Batch
            </div>
            <div style="font-size:0.86rem;font-weight:700;color:#f8fafc;word-break:break-all;margin-bottom:8px;">
                {batch_name}
            </div>
            <div>{risk_badge(risk_level)}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sidebar-card">
            <div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:1.4px;color:#94a3b8;font-weight:700;margin-bottom:4px;">
                Telemetry Status
            </div>
            <div style="color:#64748b;font-size:0.82rem;">No analysis batch loaded</div>
        </div>
        """, unsafe_allow_html=True)

    # Author card in sidebar
    render_sidebar_author()

    # Sidebar footer
    st.markdown(f"""
    <div class="sidebar-footer-info">
        <div>Enterprise ML Engine {SYSTEM_VERSION}</div>
        <div>Streamlit & Python 3.13</div>
    </div>
    """, unsafe_allow_html=True)


# ── Page Routing ──────────────────────────────────────────────────────────
if "🏠" in page:
    if st.session_state.get("analysis_complete") and st.session_state.get("drift_report"):
        from dashboard.pages.overview import render as render_overview
        render_overview(
            drift_report=st.session_state.get("drift_report"),
            performance=st.session_state.get("performance"),
            confidence_stats=st.session_state.get("confidence_stats"),
        )
    else:
        render_welcome()

elif "🔬" in page:
    if st.session_state.get("analysis_complete") and st.session_state.get("drift_report"):
        from dashboard.pages.feature_analysis import render as render_feature_analysis
        render_feature_analysis(
            drift_report=st.session_state.get("drift_report"),
            baseline_df=st.session_state.get("baseline_df"),
            production_df=st.session_state.get("production_df"),
        )
    else:
        empty_state(
            "🔬",
            "No Drift Analysis Available",
            "Upload production data and execute drift detection from the 'Upload & Run' page to view statistical tests and distribution shifts.",
        )

elif "🤖" in page:
    if st.session_state.get("analysis_complete"):
        from dashboard.pages.model_health import render as render_model_health
        render_model_health(
            performance=st.session_state.get("performance"),
            confidence_stats=st.session_state.get("confidence_stats"),
        )
    else:
        empty_state(
            "🤖",
            "Model Health Telemetry Unavailable",
            "Evaluate a production batch with ground-truth labels to inspect accuracy, precision, recall, F1, and prediction confidence.",
        )

elif "🔔" in page:
    from dashboard.pages.alerts import render as render_alerts
    render_alerts()

elif "🚀" in page:
    from dashboard.pages.upload import render as render_upload
    render_upload()

elif "⚙️" in page:
    from dashboard.pages.settings import render as render_settings
    render_settings()


# ── Global Page Footer ────────────────────────────────────────────────────
render_footer()
