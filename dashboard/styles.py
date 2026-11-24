"""
Data Drift Detection System — Premium Design System CSS.
Production SaaS-grade dark theme with glassmorphism, micro-animations,
accessible color contrast, and fully responsive layouts.
"""


def get_custom_css() -> str:
    """Return the complete design system stylesheet."""
    return """
    <style>
    /* ══════════════════════════════════════════════════════════════════════
       DESIGN TOKENS & SYSTEM VARIABLES
       ══════════════════════════════════════════════════════════════════════ */
    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #111827;
        --bg-surface: #1e293b;
        --bg-card: rgba(17, 24, 39, 0.85);
        --bg-card-hover: rgba(30, 41, 59, 0.95);
        --bg-glass: rgba(255, 255, 255, 0.04);
        --bg-glass-hover: rgba(255, 255, 255, 0.08);

        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-medium: rgba(255, 255, 255, 0.16);
        --border-accent: rgba(59, 130, 246, 0.35);

        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --text-dim: #64748b;
        --text-accent: #60a5fa;

        --accent-blue: #3b82f6;
        --accent-indigo: #6366f1;
        --accent-purple: #8b5cf6;
        --accent-cyan: #06b6d4;
        --accent-green: #10b981;
        --accent-yellow: #f59e0b;
        --accent-red: #ef4444;
        --accent-pink: #ec4899;

        --gradient-brand: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        --gradient-accent: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
        --gradient-card: linear-gradient(145deg, rgba(17, 24, 39, 0.9) 0%, rgba(15, 23, 42, 0.75) 100%);
        --gradient-sidebar: linear-gradient(180deg, #0c1220 0%, #111827 50%, #0b1120 100%);
        --gradient-hero: radial-gradient(circle at 50% 20%, rgba(59, 130, 246, 0.15) 0%, rgba(10, 14, 26, 0) 70%);

        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.4);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.45);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.6);
        --shadow-glow-blue: 0 0 24px rgba(59, 130, 246, 0.25);
        --shadow-glow-purple: 0 0 24px rgba(139, 92, 246, 0.25);

        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        --radius-full: 9999px;

        --transition-fast: 0.15s cubic-bezier(0.4, 0, 0.2, 1);
        --transition-base: 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        --transition-slow: 0.4s cubic-bezier(0.4, 0, 0.2, 1);

        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --font-mono: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
    }

    /* ══════════════════════════════════════════════════════════════════════
       BASE & ACCESSIBILITY
       ══════════════════════════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    .stApp {
        font-family: var(--font-sans) !important;
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    .stApp > header { background: transparent !important; }

    .block-container {
        max-width: 1440px !important;
        padding: 1.5rem 2.5rem 3.5rem !important;
    }

    :focus-visible {
        outline: 2px solid var(--accent-blue) !important;
        outline-offset: 2px !important;
    }

    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: var(--radius-full);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }

    /* ══════════════════════════════════════════════════════════════════════
       TOP UTILITY BAR / NAVIGATION HEADER
       ══════════════════════════════════════════════════════════════════════ */
    .top-nav-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 10px 18px;
        margin-bottom: 24px;
        box-shadow: var(--shadow-sm);
    }

    .top-nav-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    .top-nav-links {
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }

    .nav-pill-link {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-full);
        color: var(--text-secondary);
        font-size: 0.8rem;
        font-weight: 600;
        text-decoration: none !important;
        transition: var(--transition-fast);
    }

    .nav-pill-link:hover {
        background: var(--bg-glass-hover);
        color: #ffffff !important;
        border-color: var(--accent-blue);
        transform: translateY(-1px);
    }

    .nav-pill-link.live-btn {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2));
        border-color: rgba(59, 130, 246, 0.5);
        color: #93c5fd !important;
    }

    .nav-pill-link.live-btn:hover {
        background: var(--gradient-brand);
        color: #ffffff !important;
        box-shadow: var(--shadow-glow-blue);
    }

    /* ══════════════════════════════════════════════════════════════════════
       SIDEBAR ENHANCEMENTS
       ══════════════════════════════════════════════════════════════════════ */
    section[data-testid="stSidebar"] {
        background: var(--gradient-sidebar) !important;
        border-right: 1px solid var(--border-subtle) !important;
        width: 300px !important;
    }

    section[data-testid="stSidebar"] .stRadio > label {
        display: none !important;
    }

    section[data-testid="stSidebar"] .stRadio > div {
        gap: 4px !important;
    }

    section[data-testid="stSidebar"] .stRadio > div > label {
        padding: 10px 16px !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: var(--transition-fast) !important;
        cursor: pointer !important;
        margin: 0 6px !important;
        border: 1px solid transparent !important;
    }

    section[data-testid="stSidebar"] .stRadio > div > label:hover {
        background: var(--bg-glass-hover) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-subtle) !important;
    }

    section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
    section[data-testid="stSidebar"] .stRadio > div [data-testid="stMarkdownContainer"] {
        color: var(--text-primary) !important;
    }

    .sidebar-brand-box {
        padding: 16px 12px 10px;
        text-align: center;
    }

    .sidebar-brand-icon {
        font-size: 2.2rem;
        margin-bottom: 4px;
        filter: drop-shadow(0 0 12px rgba(59, 130, 246, 0.4));
    }

    .sidebar-brand-title {
        font-size: 1.25rem;
        font-weight: 800;
        background: var(--gradient-brand);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }

    .sidebar-brand-sub {
        font-size: 0.72rem;
        color: var(--text-muted);
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-top: 3px;
        font-weight: 600;
    }

    .sidebar-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 14px 16px;
        margin: 10px 8px;
        transition: var(--transition-fast);
    }
    .sidebar-card:hover {
        background: var(--bg-glass-hover);
        border-color: var(--border-medium);
    }

    .sidebar-author-box {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.6));
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 14px;
        margin: 14px 8px;
    }

    .author-title-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
    }

    .author-name {
        font-size: 0.88rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    .author-role {
        font-size: 0.72rem;
        color: var(--accent-cyan);
        font-weight: 500;
    }

    .social-links-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px;
        margin-top: 10px;
    }

    .social-btn {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 10px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        color: var(--text-secondary) !important;
        font-size: 0.75rem;
        font-weight: 600;
        text-decoration: none !important;
        transition: var(--transition-fast);
    }

    .social-btn:hover {
        background: var(--bg-glass-hover);
        color: #ffffff !important;
        border-color: var(--accent-blue);
    }

    .sidebar-footer-info {
        padding: 12px 14px;
        text-align: center;
        font-size: 0.72rem;
        color: var(--text-dim);
        line-height: 1.6;
        margin-top: 16px;
    }

    /* ══════════════════════════════════════════════════════════════════════
       KPI CARDS — Glassmorphism & High-Contrast Metrics
       ══════════════════════════════════════════════════════════════════════ */
    .kpi-card {
        background: var(--gradient-card);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 22px 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: var(--transition-base);
        box-shadow: var(--shadow-sm);
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: var(--gradient-brand);
        opacity: 0;
        transition: var(--transition-base);
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: var(--border-medium);
        box-shadow: var(--shadow-md);
    }
    .kpi-card:hover::before { opacity: 1; }

    .kpi-icon {
        font-size: 1.8rem;
        margin-bottom: 6px;
        display: block;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 4px 0;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.4px;
        font-weight: 700;
    }
    .kpi-delta {
        font-size: 0.8rem;
        font-weight: 700;
        margin-top: 6px;
        padding: 2px 8px;
        border-radius: var(--radius-full);
        display: inline-block;
    }
    .kpi-delta.positive { background: rgba(16, 185, 129, 0.15); color: #34d399; }
    .kpi-delta.negative { background: rgba(239, 68, 68, 0.15); color: #f87171; }

    /* ══════════════════════════════════════════════════════════════════════
       SECTION & PAGE HEADERS
       ══════════════════════════════════════════════════════════════════════ */
    .page-title-container {
        margin-bottom: 24px;
    }

    .page-title {
        font-size: 2rem;
        font-weight: 800;
        background: var(--gradient-brand);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }

    .page-subtitle {
        color: var(--text-muted);
        font-size: 0.95rem;
        font-weight: 400;
        line-height: 1.5;
    }

    .section-header {
        font-size: 1.18rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 28px 0 14px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border-subtle);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-header .section-icon {
        font-size: 1.15rem;
        color: var(--accent-blue);
    }

    /* ══════════════════════════════════════════════════════════════════════
       ALERT NOTIFICATION CARDS
       ══════════════════════════════════════════════════════════════════════ */
    .alert-card {
        border-radius: var(--radius-md);
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 4px solid;
        background: rgba(17, 24, 39, 0.7);
        border-top: 1px solid var(--border-subtle);
        border-right: 1px solid var(--border-subtle);
        border-bottom: 1px solid var(--border-subtle);
        transition: var(--transition-fast);
    }
    .alert-card:hover {
        background: rgba(30, 41, 59, 0.8);
        border-color: var(--border-medium);
    }
    .alert-info { border-left-color: var(--accent-blue); }
    .alert-warning { border-left-color: var(--accent-yellow); }
    .alert-critical { border-left-color: var(--accent-red); }

    .alert-card .alert-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .alert-card .alert-title {
        font-weight: 700;
        font-size: 0.92rem;
        color: var(--text-primary);
    }
    .alert-card .alert-time {
        font-size: 0.74rem;
        color: var(--text-muted);
        font-family: var(--font-mono);
    }
    .alert-card .alert-body {
        font-size: 0.86rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }
    .alert-card .alert-action {
        font-size: 0.8rem;
        color: var(--accent-cyan);
        font-weight: 600;
        margin-top: 8px;
    }

    /* ══════════════════════════════════════════════════════════════════════
       SEVERITY BADGES & PILLS
       ══════════════════════════════════════════════════════════════════════ */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: var(--radius-full);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .badge-low, .badge-healthy { background: rgba(16, 185, 129, 0.16); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-medium, .badge-warning { background: rgba(245, 158, 11, 0.16); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-high, .badge-critical { background: rgba(239, 68, 68, 0.16); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-info { background: rgba(59, 130, 246, 0.16); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }

    .badge-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
        animation: pulse-dot 2s infinite;
    }
    .badge-dot.green { background: #34d399; }
    .badge-dot.yellow { background: #fbbf24; }
    .badge-dot.red { background: #f87171; }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.45; transform: scale(0.85); }
    }

    /* ══════════════════════════════════════════════════════════════════════
       CONTENT PANELS & FEATURE ITEMS
       ══════════════════════════════════════════════════════════════════════ */
    .panel {
        background: var(--gradient-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 22px;
        backdrop-filter: blur(10px);
        margin-bottom: 16px;
        box-shadow: var(--shadow-sm);
    }
    .panel-header {
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .feature-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 14px;
        border-radius: var(--radius-sm);
        transition: var(--transition-fast);
        border-bottom: 1px solid var(--border-subtle);
    }
    .feature-item:hover {
        background: var(--bg-glass-hover);
    }
    .feature-item:last-child {
        border-bottom: none;
    }
    .feature-name {
        font-weight: 600;
        font-size: 0.9rem;
        color: var(--text-primary);
    }
    .feature-type {
        font-size: 0.74rem;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-left: 8px;
    }

    /* ══════════════════════════════════════════════════════════════════════
       EMPTY STATE
       ══════════════════════════════════════════════════════════════════════ */
    .empty-state {
        text-align: center;
        padding: 60px 24px;
        color: var(--text-muted);
        background: rgba(17, 24, 39, 0.4);
        border: 1px dashed var(--border-medium);
        border-radius: var(--radius-lg);
        margin: 20px 0;
    }
    .empty-state .empty-icon {
        font-size: 3.6rem;
        margin-bottom: 16px;
        opacity: 0.6;
        display: block;
    }
    .empty-state .empty-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 8px;
    }
    .empty-state .empty-desc {
        font-size: 0.9rem;
        max-width: 440px;
        margin: 0 auto;
        line-height: 1.6;
        color: var(--text-secondary);
    }

    /* ══════════════════════════════════════════════════════════════════════
       HERO WELCOME BANNER
       ══════════════════════════════════════════════════════════════════════ */
    .hero-section {
        text-align: center;
        padding: 50px 20px 40px;
        position: relative;
        background: var(--gradient-hero);
        border-radius: var(--radius-xl);
        border: 1px solid var(--border-subtle);
        margin-bottom: 30px;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        background: var(--gradient-brand);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 12px;
        letter-spacing: -1px;
    }
    .hero-desc {
        color: var(--text-secondary);
        font-size: 1.15rem;
        max-width: 620px;
        margin: 0 auto 28px;
        line-height: 1.7;
    }
    .hero-cta-group {
        display: flex;
        justify-content: center;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 36px;
    }
    .hero-btn {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 12px 26px;
        border-radius: var(--radius-md);
        font-weight: 700;
        font-size: 0.95rem;
        text-decoration: none !important;
        transition: var(--transition-base);
        cursor: pointer;
    }
    .hero-btn-primary {
        background: var(--gradient-brand);
        color: #ffffff !important;
        box-shadow: var(--shadow-glow-blue);
    }
    .hero-btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.4);
    }
    .hero-btn-secondary {
        background: var(--bg-surface);
        border: 1px solid var(--border-medium);
        color: var(--text-primary) !important;
    }
    .hero-btn-secondary:hover {
        background: var(--bg-card-hover);
        border-color: var(--accent-blue);
        transform: translateY(-2px);
    }
    .hero-features {
        display: flex;
        justify-content: center;
        gap: 32px;
        margin-top: 30px;
        flex-wrap: wrap;
    }
    .hero-feat {
        text-align: center;
        max-width: 170px;
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 16px 12px;
        transition: var(--transition-fast);
    }
    .hero-feat:hover {
        background: var(--bg-glass-hover);
        border-color: var(--border-accent);
        transform: translateY(-2px);
    }
    .hero-feat-icon {
        font-size: 1.8rem;
        margin-bottom: 6px;
        display: block;
    }
    .hero-feat-text {
        font-size: 0.85rem;
        color: var(--text-primary);
        font-weight: 600;
    }

    /* ══════════════════════════════════════════════════════════════════════
       GLOBAL FOOTER
       ══════════════════════════════════════════════════════════════════════ */
    .saas-footer {
        margin-top: 60px;
        padding-top: 24px;
        border-top: 1px solid var(--border-subtle);
        background: rgba(10, 14, 26, 0.6);
        border-radius: var(--radius-lg);
        padding: 28px 24px 20px;
    }
    .footer-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 20px;
    }
    .footer-left {
        max-width: 480px;
    }
    .footer-brand {
        font-size: 1rem;
        font-weight: 800;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .footer-desc {
        font-size: 0.82rem;
        color: var(--text-muted);
        margin-top: 4px;
        line-height: 1.5;
    }
    .footer-links {
        display: flex;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
    }
    .footer-link-item {
        color: var(--text-secondary) !important;
        font-size: 0.82rem;
        font-weight: 600;
        text-decoration: none !important;
        transition: var(--transition-fast);
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .footer-link-item:hover {
        color: var(--accent-blue) !important;
    }
    .footer-bottom-row {
        margin-top: 18px;
        padding-top: 14px;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.74rem;
        color: var(--text-dim);
        flex-wrap: wrap;
        gap: 10px;
    }

    /* ══════════════════════════════════════════════════════════════════════
       STREAMLIT WIDGET OVERRIDES & CONSISTENCY
       ══════════════════════════════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: var(--bg-glass);
        border-radius: var(--radius-md);
        padding: 6px;
        border: 1px solid var(--border-subtle);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-sm) !important;
        padding: 8px 22px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: var(--text-secondary) !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent-blue) !important;
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    .stDataFrame {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-subtle) !important;
    }

    .stButton > button[kind="primary"] {
        background: var(--gradient-brand) !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        font-weight: 700 !important;
        padding: 10px 24px !important;
        font-size: 0.92rem !important;
        transition: var(--transition-base) !important;
        box-shadow: var(--shadow-glow-blue) !important;
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 24px rgba(59, 130, 246, 0.4) !important;
    }

    .stButton > button[kind="secondary"] {
        border: 1px solid var(--border-medium) !important;
        border-radius: var(--radius-md) !important;
        font-weight: 600 !important;
        background: rgba(255, 255, 255, 0.03) !important;
        color: var(--text-secondary) !important;
        transition: var(--transition-fast) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: var(--accent-blue) !important;
        color: var(--text-primary) !important;
        background: var(--bg-glass-hover) !important;
    }

    .stSelectbox > div > div {
        border-radius: var(--radius-sm) !important;
        background-color: var(--bg-surface) !important;
        border-color: var(--border-subtle) !important;
    }

    .stSlider > div > div > div {
        border-radius: var(--radius-full) !important;
    }

    .stExpander {
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        background: var(--bg-glass) !important;
    }

    div[data-testid="stMetric"] {
        background: var(--gradient-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 16px;
    }

    .stDownloadButton > button {
        border: 1px solid var(--border-medium) !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        background: var(--bg-glass) !important;
        color: var(--text-primary) !important;
        transition: var(--transition-fast) !important;
    }
    .stDownloadButton > button:hover {
        border-color: var(--accent-blue) !important;
        color: var(--accent-blue) !important;
        background: var(--bg-glass-hover) !important;
    }

    /* Hide unnecessary default Streamlit banners */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ══════════════════════════════════════════════════════════════════════
       RESPONSIVE BREAKPOINTS
       ══════════════════════════════════════════════════════════════════════ */
    @media (max-width: 1024px) {
        .block-container { padding: 1.25rem 1.5rem 3rem !important; }
        .hero-title { font-size: 2.4rem; }
        .hero-features { gap: 20px; }
        .kpi-value { font-size: 1.8rem; }
    }

    @media (max-width: 768px) {
        .block-container { padding: 1rem 1rem 2.5rem !important; }
        .hero-title { font-size: 2rem; }
        .hero-desc { font-size: 1rem; }
        .top-nav-bar { flex-direction: column; align-items: flex-start; gap: 10px; }
        .footer-content { flex-direction: column; align-items: flex-start; }
        .kpi-value { font-size: 1.6rem; }
    }

    @media (max-width: 480px) {
        .hero-title { font-size: 1.6rem; }
        .social-links-grid { grid-template-columns: 1fr; }
    }
    </style>
    """
