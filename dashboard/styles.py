"""
Premium Design System CSS for the Data Drift Detection Dashboard.
Modern SaaS-grade dark theme with glassmorphism, micro-animations, and responsive layout.
"""


def get_custom_css() -> str:
    """Return the complete design system CSS."""
    return """
    <style>
    /* ══════════════════════════════════════════════════════════════════════
       DESIGN TOKENS
       ══════════════════════════════════════════════════════════════════════ */
    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #111827;
        --bg-card: rgba(17, 24, 39, 0.7);
        --bg-card-hover: rgba(17, 24, 39, 0.9);
        --bg-glass: rgba(255, 255, 255, 0.03);
        --bg-glass-hover: rgba(255, 255, 255, 0.06);

        --border-subtle: rgba(255, 255, 255, 0.06);
        --border-medium: rgba(255, 255, 255, 0.10);
        --border-accent: rgba(99, 179, 237, 0.3);

        --text-primary: #f0f4f8;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --text-accent: #63b3ed;

        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-cyan: #06b6d4;
        --accent-green: #10b981;
        --accent-yellow: #f59e0b;
        --accent-red: #ef4444;
        --accent-pink: #ec4899;

        --gradient-brand: linear-gradient(135deg, #3b82f6, #8b5cf6);
        --gradient-card: linear-gradient(145deg, rgba(17,24,39,0.8), rgba(15,23,42,0.6));
        --gradient-sidebar: linear-gradient(180deg, #0c1220 0%, #111827 50%, #0f172a 100%);

        --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
        --shadow-lg: 0 8px 32px rgba(0,0,0,0.5);
        --shadow-glow-blue: 0 0 20px rgba(59,130,246,0.15);
        --shadow-glow-purple: 0 0 20px rgba(139,92,246,0.15);

        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;

        --transition-fast: 0.15s cubic-bezier(0.4, 0, 0.2, 1);
        --transition-base: 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        --transition-slow: 0.4s cubic-bezier(0.4, 0, 0.2, 1);

        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    }

    /* ══════════════════════════════════════════════════════════════════════
       FONTS
       ══════════════════════════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ══════════════════════════════════════════════════════════════════════
       GLOBAL
       ══════════════════════════════════════════════════════════════════════ */
    .stApp {
        font-family: var(--font-sans) !important;
        background: var(--bg-primary) !important;
    }

    .stApp > header { background: transparent !important; }

    .block-container {
        max-width: 1400px !important;
        padding: 2rem 2.5rem 4rem !important;
    }

    /* ══════════════════════════════════════════════════════════════════════
       SIDEBAR
       ══════════════════════════════════════════════════════════════════════ */
    section[data-testid="stSidebar"] {
        background: var(--gradient-sidebar) !important;
        border-right: 1px solid var(--border-subtle);
        width: 280px !important;
    }

    section[data-testid="stSidebar"] .stRadio > label {
        display: none !important;
    }

    section[data-testid="stSidebar"] .stRadio > div {
        gap: 2px !important;
    }

    section[data-testid="stSidebar"] .stRadio > div > label {
        padding: 10px 16px !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        transition: var(--transition-fast) !important;
        cursor: pointer !important;
        margin: 0 8px !important;
    }

    section[data-testid="stSidebar"] .stRadio > div > label:hover {
        background: var(--bg-glass-hover) !important;
        color: var(--text-primary) !important;
    }

    section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
    section[data-testid="stSidebar"] .stRadio > div [data-testid="stMarkdownContainer"] {
        color: var(--text-primary) !important;
    }

    /* ══════════════════════════════════════════════════════════════════════
       METRIC CARDS — Premium Glassmorphism
       ══════════════════════════════════════════════════════════════════════ */
    .kpi-card {
        background: var(--gradient-card);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: var(--transition-base);
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
        box-shadow: var(--shadow-lg);
    }
    .kpi-card:hover::before { opacity: 1; }

    .kpi-icon {
        font-size: 1.8rem;
        margin-bottom: 8px;
        display: block;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 6px 0;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    .kpi-delta {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 6px;
        padding: 2px 8px;
        border-radius: 99px;
        display: inline-block;
    }
    .kpi-delta.positive { background: rgba(16,185,129,0.12); color: #34d399; }
    .kpi-delta.negative { background: rgba(239,68,68,0.12); color: #f87171; }

    /* ══════════════════════════════════════════════════════════════════════
       SECTION HEADERS
       ══════════════════════════════════════════════════════════════════════ */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 28px 0 14px 0;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border-subtle);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-header .section-icon {
        font-size: 1.1rem;
        opacity: 0.8;
    }

    /* ══════════════════════════════════════════════════════════════════════
       PAGE TITLE
       ══════════════════════════════════════════════════════════════════════ */
    .page-title {
        font-size: 1.8rem;
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
        font-size: 0.9rem;
        margin-bottom: 24px;
        font-weight: 400;
    }

    /* ══════════════════════════════════════════════════════════════════════
       ALERT CARDS
       ══════════════════════════════════════════════════════════════════════ */
    .alert-card {
        border-radius: var(--radius-md);
        padding: 16px 20px;
        margin-bottom: 10px;
        border-left: 4px solid;
        background: var(--bg-glass);
        transition: var(--transition-fast);
    }
    .alert-card:hover { background: var(--bg-glass-hover); }
    .alert-info { border-color: var(--accent-blue); }
    .alert-warning { border-color: var(--accent-yellow); }
    .alert-critical { border-color: var(--accent-red); }
    .alert-card .alert-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .alert-card .alert-title {
        font-weight: 700;
        font-size: 0.9rem;
        color: var(--text-primary);
    }
    .alert-card .alert-time {
        font-size: 0.72rem;
        color: var(--text-muted);
    }
    .alert-card .alert-body {
        font-size: 0.82rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }
    .alert-card .alert-action {
        font-size: 0.78rem;
        color: var(--accent-blue);
        font-weight: 600;
        margin-top: 6px;
    }

    /* ══════════════════════════════════════════════════════════════════════
       SEVERITY BADGES
       ══════════════════════════════════════════════════════════════════════ */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 99px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-low { background: rgba(16,185,129,0.12); color: #34d399; }
    .badge-medium { background: rgba(245,158,11,0.12); color: #fbbf24; }
    .badge-high { background: rgba(239,68,68,0.12); color: #f87171; }
    .badge-healthy { background: rgba(16,185,129,0.12); color: #34d399; }
    .badge-warning { background: rgba(245,158,11,0.12); color: #fbbf24; }
    .badge-critical { background: rgba(239,68,68,0.12); color: #f87171; }
    .badge-info { background: rgba(59,130,246,0.12); color: #60a5fa; }

    .badge-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        display: inline-block;
        animation: pulse-dot 2s infinite;
    }
    .badge-dot.green { background: #34d399; }
    .badge-dot.yellow { background: #fbbf24; }
    .badge-dot.red { background: #f87171; }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.85); }
    }

    /* ══════════════════════════════════════════════════════════════════════
       CONTENT PANELS
       ══════════════════════════════════════════════════════════════════════ */
    .panel {
        background: var(--gradient-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 24px;
        backdrop-filter: blur(8px);
        margin-bottom: 16px;
    }
    .panel-header {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ══════════════════════════════════════════════════════════════════════
       EMPTY STATE
       ══════════════════════════════════════════════════════════════════════ */
    .empty-state {
        text-align: center;
        padding: 60px 24px;
        color: var(--text-muted);
    }
    .empty-state .empty-icon {
        font-size: 3.5rem;
        margin-bottom: 16px;
        opacity: 0.4;
        display: block;
    }
    .empty-state .empty-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-secondary);
        margin-bottom: 8px;
    }
    .empty-state .empty-desc {
        font-size: 0.88rem;
        max-width: 400px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ══════════════════════════════════════════════════════════════════════
       FEATURE LIST ITEMS
       ══════════════════════════════════════════════════════════════════════ */
    .feature-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 14px;
        border-radius: var(--radius-sm);
        transition: var(--transition-fast);
        border-bottom: 1px solid var(--border-subtle);
    }
    .feature-item:hover { background: var(--bg-glass-hover); }
    .feature-item:last-child { border-bottom: none; }
    .feature-name {
        font-weight: 600;
        font-size: 0.88rem;
        color: var(--text-primary);
    }
    .feature-type {
        font-size: 0.72rem;
        color: var(--text-muted);
        text-transform: uppercase;
    }

    /* ══════════════════════════════════════════════════════════════════════
       WELCOME HERO
       ══════════════════════════════════════════════════════════════════════ */
    .hero-section {
        text-align: center;
        padding: 80px 20px 60px;
        position: relative;
    }
    .hero-title {
        font-size: 2.8rem;
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
        font-size: 1.1rem;
        max-width: 560px;
        margin: 0 auto 32px;
        line-height: 1.7;
    }
    .hero-cta {
        display: inline-block;
        padding: 12px 28px;
        background: var(--gradient-brand);
        color: white;
        border-radius: var(--radius-md);
        font-weight: 700;
        font-size: 0.95rem;
        text-decoration: none;
        box-shadow: var(--shadow-glow-blue);
        transition: var(--transition-base);
    }
    .hero-cta:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(59,130,246,0.3);
    }
    .hero-features {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-top: 48px;
        flex-wrap: wrap;
    }
    .hero-feat {
        text-align: center;
        max-width: 160px;
    }
    .hero-feat-icon {
        font-size: 1.8rem;
        margin-bottom: 8px;
        display: block;
    }
    .hero-feat-text {
        font-size: 0.82rem;
        color: var(--text-secondary);
        font-weight: 500;
    }

    /* ══════════════════════════════════════════════════════════════════════
       STREAMLIT OVERRIDES
       ══════════════════════════════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--bg-glass);
        border-radius: var(--radius-md);
        padding: 4px;
        border: 1px solid var(--border-subtle);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-sm) !important;
        padding: 8px 20px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        color: var(--text-secondary) !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent-blue) !important;
        color: white !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    .stDataFrame { border-radius: var(--radius-md) !important; }

    .stButton > button[kind="primary"] {
        background: var(--gradient-brand) !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        font-weight: 700 !important;
        padding: 10px 24px !important;
        font-size: 0.9rem !important;
        transition: var(--transition-base) !important;
        box-shadow: var(--shadow-glow-blue) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 24px rgba(59,130,246,0.35) !important;
    }

    .stButton > button[kind="secondary"] {
        border: 1px solid var(--border-medium) !important;
        border-radius: var(--radius-md) !important;
        font-weight: 600 !important;
        background: transparent !important;
        color: var(--text-secondary) !important;
    }

    .stSelectbox > div > div { border-radius: var(--radius-sm) !important; }
    .stSlider > div > div > div { border-radius: 99px !important; }

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
        transition: var(--transition-fast) !important;
    }
    .stDownloadButton > button:hover {
        border-color: var(--accent-blue) !important;
        color: var(--accent-blue) !important;
    }

    /* Hide defaults */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ══════════════════════════════════════════════════════════════════════
       SIDEBAR STATUS PANEL
       ══════════════════════════════════════════════════════════════════════ */
    .sidebar-status {
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 16px;
        margin: 0 8px;
    }
    .sidebar-status .status-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--text-muted);
        font-weight: 700;
        margin-bottom: 8px;
    }
    .sidebar-status .status-batch {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 4px;
    }

    /* ══════════════════════════════════════════════════════════════════════
       RESPONSIVE
       ══════════════════════════════════════════════════════════════════════ */
    @media (max-width: 768px) {
        .block-container { padding: 1rem 1rem 3rem !important; }
        .hero-title { font-size: 2rem; }
        .hero-features { gap: 20px; }
        .kpi-value { font-size: 1.6rem; }
    }
    </style>
    """
