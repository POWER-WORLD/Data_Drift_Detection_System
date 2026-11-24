"""
Configuration settings for the Data Drift Detection System.
Contains all configurable parameters: paths, thresholds, model settings, feature definitions.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BASELINE_DIR = DATA_DIR / "baseline"
PRODUCTION_DIR = DATA_DIR / "production"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for d in [BASELINE_DIR, PRODUCTION_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Dataset Settings
# ──────────────────────────────────────────────────────────────────────────────
BASELINE_FILE = BASELINE_DIR / "telco_churn.csv"
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

# Feature categories
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ──────────────────────────────────────────────────────────────────────────────
# Drift Detection Thresholds
# ──────────────────────────────────────────────────────────────────────────────
DRIFT_THRESHOLDS = {
    # Kolmogorov-Smirnov test
    "ks_pvalue": 0.05,          # p-value below this = drift detected

    # Population Stability Index
    "psi_moderate": 0.10,       # PSI above this = moderate drift
    "psi_severe": 0.25,         # PSI above this = severe drift

    # Jensen-Shannon Distance
    "js_moderate": 0.05,
    "js_severe": 0.15,

    # Chi-Square test
    "chi2_pvalue": 0.05,        # p-value below this = drift detected

    # Mean shift (relative change)
    "mean_shift_pct": 0.10,     # 10% relative change

    # Variance change (relative change)
    "variance_change_pct": 0.20,  # 20% relative change

    # Dataset-level drift
    "drifted_features_pct_moderate": 0.20,  # 20% of features drifted = moderate
    "drifted_features_pct_severe": 0.50,    # 50% of features drifted = severe
}

# ──────────────────────────────────────────────────────────────────────────────
# Model Training Settings
# ──────────────────────────────────────────────────────────────────────────────
MODEL_SETTINGS = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
}

# Model performance degradation thresholds
PERFORMANCE_THRESHOLDS = {
    "accuracy_drop": 0.05,    # 5% accuracy drop = warning
    "recall_drop": 0.05,
    "precision_drop": 0.05,
    "f1_drop": 0.05,
    "auc_drop": 0.05,
}

# ──────────────────────────────────────────────────────────────────────────────
# Alert Severity Levels
# ──────────────────────────────────────────────────────────────────────────────
ALERT_LEVELS = {
    "INFO": "ℹ️",
    "WARNING": "⚠️",
    "CRITICAL": "🔴",
    "RETRAIN": "🔄",
}

# ──────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────────────────────────
LOG_FILE = LOGS_DIR / "drift_system.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ──────────────────────────────────────────────────────────────────────────────
# Dashboard Settings
# ──────────────────────────────────────────────────────────────────────────────
DASHBOARD_SETTINGS = {
    "page_title": "Data Drift Detection System",
    "page_icon": "📊",
    "layout": "wide",
}
