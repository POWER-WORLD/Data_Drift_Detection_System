"""
Premium reusable UI components for the Data Drift Detection Dashboard.
Design-system-aligned components with accessibility, micro-interactions,
author attribution, and seamless navigation.
"""

from typing import Any, Optional
import streamlit as st

# ── Global System & Author Links Standardization ───────────────────────────
SYSTEM_URL: str = "https://dddsystem.vercel.app"
PORTFOLIO_URL: str = "https://pawankumar.info/"
LINKEDIN_URL: str = "https://www.linkedin.com/in/pawankumar101/"
LEETCODE_URL: str = "https://leetcode.com/u/pk3253702/"
GITHUB_URL: str = "https://github.com/POWER-WORLD"
AUTHOR_NAME: str = "Pawan Kumar"
SYSTEM_VERSION: str = "v2.0 Enterprise"

# Standard chart color palette
CHART_COLORS: dict[str, str] = {
    "blue": "#3b82f6",
    "purple": "#8b5cf6",
    "cyan": "#06b6d4",
    "green": "#10b981",
    "yellow": "#f59e0b",
    "red": "#ef4444",
    "pink": "#ec4899",
    "baseline": "#60a5fa",
    "production": "#f472b6",
    "card_bg": "#111827",
    "surface": "#1e293b",
}


def page_header(title: str, subtitle: str = "", badge_text: Optional[str] = None) -> None:
    """Render a gradient page title with optional subtitle and badge."""
    badge_html = f'<span class="badge badge-info" style="margin-left:12px;vertical-align:middle;">{badge_text}</span>' if badge_text else ""
    st.markdown(f"""
    <div class="page-title-container">
        <div class="page-title">{title}{badge_html}</div>
        {f'<div class="page-subtitle">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def section_header(text: str, icon: str = "") -> None:
    """Render a styled section divider with icon."""
    icon_html = f'<span class="section-icon">{icon}</span> ' if icon else ""
    st.markdown(f'<div class="section-header">{icon_html}{text}</div>', unsafe_allow_html=True)


def kpi_card(
    label: str,
    value: Any,
    icon: str = "",
    color: str = "#3b82f6",
    delta: Optional[str] = None,
    delta_positive: bool = True,
) -> None:
    """
    Render an accessible KPI metric card with optional delta badge.

    Args:
        label: Card label (uppercase).
        value: Main display value.
        icon: Emoji or icon representation.
        color: Accent color for the value.
        delta: Optional delta text (e.g. "+2.3%").
        delta_positive: Whether the delta represents a positive change.
    """
    delta_html = ""
    if delta is not None:
        cls = "positive" if delta_positive else "negative"
        delta_html = f'<div class="kpi-delta {cls}">{delta}</div>'

    formatted_val = "—" if value is None else str(value)

    st.markdown(f"""
    <div class="kpi-card">
        <span class="kpi-icon" aria-hidden="true">{icon}</span>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color: {color};">{formatted_val}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def risk_badge(risk_level: Optional[str]) -> str:
    """Return styled risk badge HTML string."""
    lvl = (risk_level or "LOW").upper()
    config = {
        "LOW": ("badge-low", "green", "Low Risk"),
        "MEDIUM": ("badge-medium", "yellow", "Medium Risk"),
        "HIGH": ("badge-high", "red", "High Risk"),
        "HEALTHY": ("badge-healthy", "green", "Healthy"),
        "WARNING": ("badge-warning", "yellow", "Warning"),
        "CRITICAL": ("badge-critical", "red", "Critical"),
    }
    badge_cls, dot_cls, display = config.get(lvl, ("badge-info", "green", lvl))
    return f'<span class="badge {badge_cls}"><span class="badge-dot {dot_cls}"></span>{display}</span>'


def alert_card(alert: dict) -> None:
    """Render a premium alert notification card."""
    severity = alert.get("severity", "INFO").upper()
    css_class = f"alert-{severity.lower()}" if severity.lower() in ("info", "warning", "critical") else "alert-info"
    icon_map = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}
    icon = icon_map.get(severity, "ℹ️")

    batch_str = f" • Batch: {alert.get('batch')}" if alert.get("batch") else ""

    st.markdown(f"""
    <div class="alert-card {css_class}">
        <div class="alert-header">
            <div class="alert-title">{icon} [{severity}]{batch_str}</div>
            <div class="alert-time">{alert.get('timestamp', '')}</div>
        </div>
        <div class="alert-body">{alert.get('message', '')}</div>
        <div class="alert-action">↳ Action: {alert.get('action', 'No action required')}</div>
    </div>
    """, unsafe_allow_html=True)


def empty_state(icon: str, title: str, description: str) -> None:
    """Render a professional empty state placeholder."""
    st.markdown(f"""
    <div class="empty-state">
        <span class="empty-icon" aria-hidden="true">{icon}</span>
        <div class="empty-title">{title}</div>
        <div class="empty-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def panel(title: str, icon: str = "") -> None:
    """Render the opening tag of a styled content panel."""
    icon_html = f"{icon} " if icon else ""
    st.markdown(f"""
    <div class="panel">
        <div class="panel-header">{icon_html}{title}</div>
    """, unsafe_allow_html=True)


def panel_end() -> None:
    """Close a content panel block."""
    st.markdown("</div>", unsafe_allow_html=True)


def feature_item(name: str, badge_html: str, value_text: str = "") -> None:
    """Render a single feature row in an audit list."""
    val = f'<span style="font-size:0.82rem;color:#cbd5e1;font-weight:500;">{value_text}</span>' if value_text else ""
    st.markdown(f"""
    <div class="feature-item">
        <div>
            <span class="feature-name">{name}</span>
        </div>
        <div style="display:flex;align-items:center;gap:12px;">
            {val}
            {badge_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def plotly_layout(title: str = "", height: int = 380) -> dict:
    """Return a standardized Plotly layout dict conforming to the design system."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", color="#cbd5e1", size=12),
        height=height,
        autosize=True,
        title=dict(text=title, font=dict(size=14, color="#f8fafc")) if title else None,
        margin=dict(t=40 if title else 20, b=40, l=50, r=20),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.08)",
            tickfont=dict(size=11, color="#94a3b8"),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.08)",
            tickfont=dict(size=11, color="#94a3b8"),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#cbd5e1"),
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="rgba(255,255,255,0.15)",
            font=dict(family="Inter", color="#f8fafc", size=12),
        ),
    )


def render_top_nav() -> None:
    """Render the top utility navbar with live link and author attribution."""
    st.markdown(f"""
    <div class="top-nav-bar">
        <div class="top-nav-brand">
            <span>🔷</span>
            <span>Data Drift Detection Engine</span>
            <span class="badge badge-info" style="font-size:0.68rem;padding:2px 8px;">{SYSTEM_VERSION}</span>
        </div>
        <div class="top-nav-links">
            <a href="{SYSTEM_URL}" target="_blank" rel="noopener noreferrer" class="nav-pill-link live-btn">
                🌐 Live System: dddsystem.vercel.app ↗
            </a>
            <a href="{PORTFOLIO_URL}" target="_blank" rel="noopener noreferrer" class="nav-pill-link">
                👨‍💻 {AUTHOR_NAME} ↗
            </a>
            <a href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer" class="nav-pill-link">
                🐙 GitHub ↗
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_author() -> None:
    """Render author info and social profiles inside the sidebar."""
    st.markdown(f"""
    <div class="sidebar-author-box">
        <div class="author-title-row">
            <div style="font-size:1.4rem;">👨‍💻</div>
            <div>
                <div class="author-name">{AUTHOR_NAME}</div>
                <div class="author-role">AI / ML Systems Architect</div>
            </div>
        </div>
        <div style="font-size:0.75rem;color:#94a3b8;margin-top:4px;line-height:1.4;">
            Enterprise ML monitoring & drift intelligence platform.
        </div>
        <div class="social-links-grid">
            <a href="{PORTFOLIO_URL}" target="_blank" rel="noopener noreferrer" class="social-btn" title="Portfolio">
                🌐 Portfolio
            </a>
            <a href="{LINKEDIN_URL}" target="_blank" rel="noopener noreferrer" class="social-btn" title="LinkedIn">
                💼 LinkedIn
            </a>
            <a href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer" class="social-btn" title="GitHub">
                🐙 GitHub
            </a>
            <a href="{LEETCODE_URL}" target="_blank" rel="noopener noreferrer" class="social-btn" title="LeetCode">
                ⚡ LeetCode
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_footer() -> None:
    """Render a comprehensive, enterprise-grade SaaS footer."""
    st.markdown(f"""
    <div class="saas-footer">
        <div class="footer-content">
            <div class="footer-left">
                <div class="footer-brand">
                    <span>🔷</span>
                    <span>Data Drift Detection System {SYSTEM_VERSION}</span>
                </div>
                <div class="footer-desc">
                    Autonomous data distribution monitoring, statistical drift detection,
                    and model reliability tracking for mission-critical ML pipelines.
                </div>
            </div>
            <div class="footer-links">
                <a href="{SYSTEM_URL}" target="_blank" rel="noopener noreferrer" class="footer-link-item">
                    🌐 Live Production App
                </a>
                <a href="{PORTFOLIO_URL}" target="_blank" rel="noopener noreferrer" class="footer-link-item">
                    👤 {AUTHOR_NAME}
                </a>
                <a href="{LINKEDIN_URL}" target="_blank" rel="noopener noreferrer" class="footer-link-item">
                    💼 LinkedIn
                </a>
                <a href="{LEETCODE_URL}" target="_blank" rel="noopener noreferrer" class="footer-link-item">
                    ⚡ LeetCode
                </a>
                <a href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer" class="footer-link-item">
                    🐙 GitHub Repository
                </a>
            </div>
        </div>
        <div class="footer-bottom-row">
            <div>
                © 2026 {AUTHOR_NAME}. All rights reserved. Architected with Python, Streamlit, Scikit-learn, XGBoost & Plotly.
            </div>
            <div>
                <a href="{SYSTEM_URL}" target="_blank" rel="noopener noreferrer" style="color:#60a5fa;text-decoration:none;">
                    dddsystem.vercel.app
                </a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
