"""
Premium reusable UI components for the Data Drift Detection Dashboard.
Design-system-aligned components with accessibility and micro-interactions.
"""

import streamlit as st
from typing import Any, Optional


def page_header(title: str, subtitle: str = ""):
    """Render a gradient page title with optional subtitle."""
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def section_header(text: str, icon: str = ""):
    """Render a styled section divider with icon."""
    icon_html = f'<span class="section-icon">{icon}</span>' if icon else ""
    st.markdown(f'<div class="section-header">{icon_html}{text}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: Any, icon: str = "", color: str = "#3b82f6",
             delta: Optional[str] = None, delta_positive: bool = True):
    """
    Render a premium KPI metric card with optional delta badge.

    Args:
        label: Card label (uppercase).
        value: Main display value.
        icon: Emoji icon.
        color: Accent color for the value.
        delta: Optional delta text (e.g. "+2.3%").
        delta_positive: Whether the delta is positive.
    """
    delta_html = ""
    if delta:
        cls = "positive" if delta_positive else "negative"
        delta_html = f'<div class="kpi-delta {cls}">{delta}</div>'

    st.markdown(f"""
    <div class="kpi-card">
        <span class="kpi-icon">{icon}</span>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color: {color};">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def risk_badge(risk_level: str) -> str:
    """Return styled risk badge HTML string."""
    config = {
        "LOW": ("badge-low", "green", "Low Risk"),
        "MEDIUM": ("badge-medium", "yellow", "Medium Risk"),
        "HIGH": ("badge-high", "red", "High Risk"),
        "HEALTHY": ("badge-healthy", "green", "Healthy"),
        "WARNING": ("badge-warning", "yellow", "Warning"),
        "CRITICAL": ("badge-critical", "red", "Critical"),
    }
    badge_cls, dot_cls, display = config.get(risk_level, ("badge-info", "green", risk_level))
    return f'<span class="badge {badge_cls}"><span class="badge-dot {dot_cls}"></span>{display}</span>'


def alert_card(alert: dict):
    """Render a premium alert notification card."""
    severity = alert.get("severity", "INFO").lower()
    css_class = f"alert-{severity}" if severity in ("info", "warning", "critical") else "alert-info"
    icon_map = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
    icon = icon_map.get(severity, "ℹ️")

    st.markdown(f"""
    <div class="alert-card {css_class}">
        <div class="alert-header">
            <div class="alert-title">{icon} [{alert.get('severity', 'INFO')}] {alert.get('batch', '')}</div>
            <div class="alert-time">{alert.get('timestamp', '')}</div>
        </div>
        <div class="alert-body">{alert.get('message', '')}</div>
        <div class="alert-action">↳ {alert.get('action', 'No action required')}</div>
    </div>
    """, unsafe_allow_html=True)


def empty_state(icon: str, title: str, description: str):
    """Render a professional empty state placeholder."""
    st.markdown(f"""
    <div class="empty-state">
        <span class="empty-icon">{icon}</span>
        <div class="empty-title">{title}</div>
        <div class="empty-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def panel(title: str, icon: str = ""):
    """Context manager for a styled content panel. Use: panel("Title"); content..."""
    icon_html = f"{icon} " if icon else ""
    st.markdown(f"""
    <div class="panel">
        <div class="panel-header">{icon_html}{title}</div>
    """, unsafe_allow_html=True)


def panel_end():
    """Close a panel block."""
    st.markdown("</div>", unsafe_allow_html=True)


def feature_item(name: str, badge_html: str, value_text: str = ""):
    """Render a single feature list row."""
    val = f'<span style="font-size:0.82rem;color:#94a3b8;">{value_text}</span>' if value_text else ""
    st.markdown(f"""
    <div class="feature-item">
        <div>
            <span class="feature-name">{name}</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
            {val}
            {badge_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def plotly_layout(title: str = "", height: int = 380) -> dict:
    """Return a standardized Plotly layout dict matching the design system."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
        height=height,
        title=dict(text=title, font=dict(size=14, color="#f0f4f8")) if title else None,
        margin=dict(t=40 if title else 20, b=40, l=50, r=20),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=11),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#94a3b8"),
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(family="Inter", color="#f0f4f8", size=12),
        ),
    )


# Color palette for charts
CHART_COLORS = {
    "blue": "#3b82f6",
    "purple": "#8b5cf6",
    "cyan": "#06b6d4",
    "green": "#10b981",
    "yellow": "#f59e0b",
    "red": "#ef4444",
    "pink": "#ec4899",
    "baseline": "#60a5fa",
    "production": "#f472b6",
}
