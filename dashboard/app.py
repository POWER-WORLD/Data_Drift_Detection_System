"""
Data Drift Detection System — Main Dashboard Application.
Premium SaaS-grade Streamlit dashboard with glassmorphism UI.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dashboard.styles import get_custom_css
from dashboard.components import empty_state, risk_badge

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drift Detector — ML Monitoring",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject Design System CSS ─────────────────────────────────────────────
st.markdown(get_custom_css(), unsafe_allow_html=True)


def render_welcome():
    """Welcome hero screen when no analysis has been run."""
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">ML Drift Detection</div>
        <div class="hero-desc">
            Monitor data distribution changes, track model health, and get actionable
            alerts — all from a single, intelligent dashboard.
        </div>
        <div class="hero-features">
            <div class="hero-feat">
                <span class="hero-feat-icon">📊</span>
                <div class="hero-feat-text">8 Statistical Tests</div>
            </div>
            <div class="hero-feat">
                <span class="hero-feat-icon">🤖</span>
                <div class="hero-feat-text">4 ML Algorithms</div>
            </div>
            <div class="hero-feat">
                <span class="hero-feat-icon">🔔</span>
                <div class="hero-feat-text">Smart Alerts</div>
            </div>
            <div class="hero-feat">
                <span class="hero-feat-icon">📈</span>
                <div class="hero-feat-text">Rich Visualizations</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👈 Navigate to **Upload & Run** in the sidebar to start your first analysis.")


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div style="padding:20px 16px 8px;text-align:center;">
        <div style="font-size:2rem;margin-bottom:4px;">🔷</div>
        <div style="font-size:1.2rem;font-weight:800;background:linear-gradient(135deg,#3b82f6,#8b5cf6);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-0.5px;">
            Drift Detector
        </div>
        <div style="font-size:0.7rem;color:#64748b;letter-spacing:1px;text-transform:uppercase;margin-top:2px;">
            ML Monitoring Suite
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin:16px 0;border-top:1px solid rgba(255,255,255,0.06);'></div>",
                unsafe_allow_html=True)

    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠  Overview", "🔬  Drift Analysis", "🤖  Model Health", "🔔  Alerts", "🚀  Upload & Run", "⚙️  Settings"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='margin:16px 0;border-top:1px solid rgba(255,255,255,0.06);'></div>",
                unsafe_allow_html=True)

    # Status panel
    batch = st.session_state.get("batch_name", None)
    risk = None
    drift_report = st.session_state.get("drift_report")
    if drift_report:
        risk = drift_report.get("dataset_drift", {}).get("risk_level", "LOW")

    if batch:
        st.markdown(f"""
        <div class="sidebar-status">
            <div class="status-label">Current Batch</div>
            <div class="status-batch">{batch}</div>
            <div style="margin-top:8px;">{risk_badge(risk) if risk else ''}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sidebar-status">
            <div class="status-label">Status</div>
            <div style="color:#64748b;font-size:0.82rem;">No analysis run yet</div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="position:fixed;bottom:16px;left:16px;right:0;width:248px;">
        <div style="font-size:0.65rem;color:#475569;padding:0 16px;">
            Built with Streamlit & Python<br>
            Data Drift Detection System v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Page Routing ──────────────────────────────────────────────────────────
if "🏠" in page:
    if st.session_state.get("analysis_complete"):
        from dashboard.pages.overview import render
        render(
            drift_report=st.session_state.get("drift_report"),
            performance=st.session_state.get("performance"),
            confidence_stats=st.session_state.get("confidence_stats"),
        )
    else:
        render_welcome()

elif "🔬" in page:
    if st.session_state.get("analysis_complete"):
        from dashboard.pages.feature_analysis import render
        render(
            drift_report=st.session_state.get("drift_report"),
            baseline_df=st.session_state.get("baseline_df"),
            production_df=st.session_state.get("production_df"),
        )
    else:
        empty_state("🔬", "No Drift Analysis Available",
                     "Run drift detection from the Upload & Run page to see feature-level analysis.")

elif "🤖" in page:
    if st.session_state.get("analysis_complete"):
        from dashboard.pages.model_health import render
        render(
            performance=st.session_state.get("performance"),
            confidence_stats=st.session_state.get("confidence_stats"),
        )
    else:
        empty_state("🤖", "Model Health Unavailable",
                     "Train a model and run drift detection to view performance metrics.")

elif "🔔" in page:
    from dashboard.pages.alerts import render
    render()

elif "🚀" in page:
    from dashboard.pages.upload import render
    render()

elif "⚙️" in page:
    from dashboard.pages.settings import render
    render()
