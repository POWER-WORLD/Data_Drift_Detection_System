"""
Configuration settings for the Data Drift Detection System.

Defines project-level paths, dataset parameters, drift detection thresholds,
model hyperparameters, performance degradation limits, and alert configurations.
Adheres to PEP 8, strict type-hinting, and Global Naming Standardization.
"""

from pathlib import Path
from typing import Any, Dict, List

# ──────────────────────────────────────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
BASELINE_DIR: Path = DATA_DIR / "baseline"
PRODUCTION_DIR: Path = DATA_DIR / "production"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Ensure all runtime output directories exist (graceful fallback for read-only serverless environments)
for directory in (BASELINE_DIR, PRODUCTION_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR):
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError):
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Dataset Definitions
# ──────────────────────────────────────────────────────────────────────────────
BASELINE_FILE: Path = BASELINE_DIR / "telco_churn.csv"
BASELINE_DATA_PATH: Path = BASELINE_FILE
BASE_DIR: Path = PROJECT_ROOT
TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"

NUMERICAL_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURES: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ──────────────────────────────────────────────────────────────────────────────
# Drift Detection Thresholds
# ──────────────────────────────────────────────────────────────────────────────
DRIFT_THRESHOLDS: Dict[str, float] = {
    # Kolmogorov-Smirnov test (numerical)
    "ks_pvalue": 0.05,
    # Population Stability Index (numerical)
    "psi_moderate": 0.10,
    "psi_severe": 0.25,
    # Jensen-Shannon Distance (numerical)
    "js_moderate": 0.05,
    "js_severe": 0.15,
    # Chi-Square test (categorical)
    "chi2_pvalue": 0.05,
    # Mean shift (relative percentage change)
    "mean_shift_pct": 0.10,
    # Variance change (relative percentage change)
    "variance_change_pct": 0.20,
    # Dataset-level drift risk boundaries
    "drifted_features_pct_moderate": 0.20,
    "drifted_features_pct_severe": 0.50,
}

# ──────────────────────────────────────────────────────────────────────────────
# Model Training Settings
# ──────────────────────────────────────────────────────────────────────────────
MODEL_SETTINGS: Dict[str, Any] = {
    "test_size": 0.20,
    "random_state": 42,
    "cv_folds": 5,
}

# Model performance degradation thresholds
PERFORMANCE_THRESHOLDS: Dict[str, float] = {
    "accuracy_drop": 0.05,
    "recall_drop": 0.05,
    "precision_drop": 0.05,
    "f1_drop": 0.05,
    "auc_drop": 0.05,
}

# ──────────────────────────────────────────────────────────────────────────────
# Alert Severity Levels & UI Presentation
# ──────────────────────────────────────────────────────────────────────────────
ALERT_LEVELS: Dict[str, str] = {
    "INFO": "ℹ️",
    "WARNING": "⚠️",
    "CRITICAL": "🔴",
    "RETRAIN": "🔄",
}

# ──────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────────────────────────
LOG_FILE: Path = LOGS_DIR / "drift_system.log"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

