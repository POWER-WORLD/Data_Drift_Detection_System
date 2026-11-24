"""
Drift Detection Engine for the Data Drift Detection System.

Implements:
- Numerical drift: KS test, PSI, Jensen-Shannon distance, mean shift, variance change
- Categorical drift: Chi-Square test, frequency shift, new category detection
- Dataset-level drift scoring and risk classification

All tests include minimum sample-size guards and cast results to native Python
types for safe JSON serialization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy import stats
from scipy.spatial.distance import jensenshannon

from src.config.settings import (
    DRIFT_THRESHOLDS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    TARGET_COLUMN, ID_COLUMN
)
from src.utils import setup_logger

logger = setup_logger(__name__)

# Minimum samples required for reliable statistical testing
_MIN_SAMPLES = 10


def _safe_bool(value) -> bool:
    """Convert numpy bool to Python bool for JSON serialization."""
    return bool(value)


# ═══════════════════════════════════════════════════════════════════════════════
# NUMERICAL DRIFT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def ks_test(baseline: pd.Series, production: pd.Series) -> Dict[str, Any]:
    """Kolmogorov-Smirnov two-sample test."""
    b = baseline.dropna().values.astype(float)
    p = production.dropna().values.astype(float)

    if len(b) < _MIN_SAMPLES or len(p) < _MIN_SAMPLES:
        logger.warning(f"KS test: insufficient samples (baseline={len(b)}, production={len(p)})")
        return {"test": "Kolmogorov-Smirnov", "statistic": None, "p_value": None,
                "drift_detected": False, "warning": "Insufficient samples"}

    stat, pvalue = stats.ks_2samp(b, p)
    return {
        "test": "Kolmogorov-Smirnov",
        "statistic": round(float(stat), 6),
        "p_value": round(float(pvalue), 6),
        "drift_detected": _safe_bool(pvalue < DRIFT_THRESHOLDS["ks_pvalue"]),
    }


def compute_psi(baseline: pd.Series, production: pd.Series, bins: int = 10) -> Dict[str, Any]:
    """
    Population Stability Index.
    Measures how much the distribution has shifted.
    """
    b = baseline.dropna().values.astype(float)
    p = production.dropna().values.astype(float)

    if len(b) < _MIN_SAMPLES or len(p) < _MIN_SAMPLES:
        return {"test": "PSI", "value": 0.0, "severity": "LOW",
                "drift_detected": False, "warning": "Insufficient samples"}

    # Guard against zero-variance columns
    if np.std(b) == 0 and np.std(p) == 0:
        return {"test": "PSI", "value": 0.0, "severity": "LOW", "drift_detected": False}

    breakpoints = np.linspace(np.min(b), np.max(b), bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    b_counts = np.histogram(b, bins=breakpoints)[0] / len(b)
    p_counts = np.histogram(p, bins=breakpoints)[0] / len(p)

    # Avoid zero division
    b_counts = np.clip(b_counts, 1e-6, None)
    p_counts = np.clip(p_counts, 1e-6, None)

    psi_value = float(np.sum((p_counts - b_counts) * np.log(p_counts / b_counts)))

    if psi_value > DRIFT_THRESHOLDS["psi_severe"]:
        severity = "SEVERE"
    elif psi_value > DRIFT_THRESHOLDS["psi_moderate"]:
        severity = "MODERATE"
    else:
        severity = "LOW"

    return {
        "test": "PSI",
        "value": round(psi_value, 6),
        "severity": severity,
        "drift_detected": _safe_bool(psi_value > DRIFT_THRESHOLDS["psi_moderate"]),
    }


def jensen_shannon_distance(baseline: pd.Series, production: pd.Series, bins: int = 30) -> Dict[str, Any]:
    """Jensen-Shannon Distance between two distributions."""
    b = baseline.dropna().values.astype(float)
    p = production.dropna().values.astype(float)

    if len(b) < _MIN_SAMPLES or len(p) < _MIN_SAMPLES:
        return {"test": "Jensen-Shannon", "value": 0.0, "severity": "LOW",
                "drift_detected": False, "warning": "Insufficient samples"}

    all_data = np.concatenate([b, p])
    breakpoints = np.linspace(np.min(all_data), np.max(all_data), bins + 1)

    b_hist = np.histogram(b, bins=breakpoints)[0].astype(float)
    p_hist = np.histogram(p, bins=breakpoints)[0].astype(float)

    b_hist = b_hist / (b_hist.sum() + 1e-10)
    p_hist = p_hist / (p_hist.sum() + 1e-10)

    js_dist = float(jensenshannon(b_hist, p_hist))

    if js_dist > DRIFT_THRESHOLDS["js_severe"]:
        severity = "SEVERE"
    elif js_dist > DRIFT_THRESHOLDS["js_moderate"]:
        severity = "MODERATE"
    else:
        severity = "LOW"

    return {
        "test": "Jensen-Shannon",
        "value": round(js_dist, 6),
        "severity": severity,
        "drift_detected": _safe_bool(js_dist > DRIFT_THRESHOLDS["js_moderate"]),
    }


def mean_shift_test(baseline: pd.Series, production: pd.Series) -> Dict[str, Any]:
    """Detect shift in mean value."""
    b_vals = baseline.dropna()
    p_vals = production.dropna()

    if len(b_vals) < 2 or len(p_vals) < 2:
        return {"test": "Mean Shift", "baseline_mean": None, "production_mean": None,
                "pct_change": 0.0, "drift_detected": False, "warning": "Insufficient samples"}

    b_mean = float(b_vals.mean())
    p_mean = float(p_vals.mean())
    if abs(b_mean) < 1e-10:
        pct_change = 0.0 if abs(p_mean) < 1e-10 else 100.0
    else:
        pct_change = abs(p_mean - b_mean) / abs(b_mean)

    return {
        "test": "Mean Shift",
        "baseline_mean": round(b_mean, 4),
        "production_mean": round(p_mean, 4),
        "pct_change": round(pct_change * 100, 2),
        "drift_detected": _safe_bool(pct_change > DRIFT_THRESHOLDS["mean_shift_pct"]),
    }


def variance_change_test(baseline: pd.Series, production: pd.Series) -> Dict[str, Any]:
    """Detect change in variance."""
    b_vals = baseline.dropna()
    p_vals = production.dropna()

    if len(b_vals) < 2 or len(p_vals) < 2:
        return {"test": "Variance Change", "baseline_var": None, "production_var": None,
                "pct_change": 0.0, "drift_detected": False, "warning": "Insufficient samples"}

    b_var = float(b_vals.var())
    p_var = float(p_vals.var())
    if abs(b_var) < 1e-10:
        pct_change = 0.0 if abs(p_var) < 1e-10 else 100.0
    else:
        pct_change = abs(p_var - b_var) / abs(b_var)

    return {
        "test": "Variance Change",
        "baseline_var": round(b_var, 4),
        "production_var": round(p_var, 4),
        "pct_change": round(pct_change * 100, 2),
        "drift_detected": _safe_bool(pct_change > DRIFT_THRESHOLDS["variance_change_pct"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORICAL DRIFT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def chi_square_test(baseline: pd.Series, production: pd.Series) -> Dict[str, Any]:
    """Chi-Square test for categorical drift."""
    b = baseline.dropna()
    p = production.dropna()

    if len(b) < _MIN_SAMPLES or len(p) < _MIN_SAMPLES:
        return {"test": "Chi-Square", "statistic": None, "p_value": None,
                "drift_detected": False, "warning": "Insufficient samples"}

    all_cats = sorted(set(b.unique()) | set(p.unique()))
    b_counts = b.value_counts().reindex(all_cats, fill_value=0)
    p_counts = p.value_counts().reindex(all_cats, fill_value=0)

    b_freq = (b_counts / b_counts.sum() * len(p)).values
    p_freq = p_counts.values

    b_freq = np.clip(b_freq, 1e-6, None)

    try:
        stat, pvalue = stats.chisquare(p_freq, f_exp=b_freq)
        return {
            "test": "Chi-Square",
            "statistic": round(float(stat), 6),
            "p_value": round(float(pvalue), 6),
            "drift_detected": _safe_bool(pvalue < DRIFT_THRESHOLDS["chi2_pvalue"]),
        }
    except Exception as e:
        logger.warning(f"Chi-Square test failed: {e}")
        return {"test": "Chi-Square", "statistic": None, "p_value": None,
                "drift_detected": False, "error": str(e)}


def category_frequency_shift(baseline: pd.Series, production: pd.Series) -> Dict[str, Any]:
    """Detect shifts in category frequency distributions."""
    b_freq = baseline.dropna().value_counts(normalize=True)
    p_freq = production.dropna().value_counts(normalize=True)

    all_cats = sorted(set(b_freq.index) | set(p_freq.index))
    shifts = {}
    max_shift = 0.0

    for cat in all_cats:
        b_pct = float(b_freq.get(cat, 0))
        p_pct = float(p_freq.get(cat, 0))
        shift = abs(p_pct - b_pct)
        shifts[str(cat)] = {
            "baseline_pct": round(b_pct * 100, 2),
            "production_pct": round(p_pct * 100, 2),
            "shift": round(shift * 100, 2),
        }
        max_shift = max(max_shift, shift)

    # Use configurable threshold instead of hardcoded 0.10
    threshold = DRIFT_THRESHOLDS.get("mean_shift_pct", 0.10)

    return {
        "test": "Category Frequency Shift",
        "shifts": shifts,
        "max_shift_pct": round(max_shift * 100, 2),
        "drift_detected": _safe_bool(max_shift > threshold),
    }


def new_category_detection(baseline: pd.Series, production: pd.Series) -> Dict[str, Any]:
    """Detect new categories in production that weren't in baseline."""
    b_cats = set(baseline.dropna().unique())
    p_cats = set(production.dropna().unique())
    new_cats = p_cats - b_cats
    missing_cats = b_cats - p_cats

    return {
        "test": "New Category Detection",
        "new_categories": sorted([str(c) for c in new_cats]),
        "missing_categories": sorted([str(c) for c in missing_cats]),
        "new_count": len(new_cats),
        "missing_count": len(missing_cats),
        "drift_detected": _safe_bool(len(new_cats) > 0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE-LEVEL DRIFT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_feature_drift(baseline_df: pd.DataFrame, production_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run all drift tests on each feature.

    Args:
        baseline_df: Raw baseline DataFrame.
        production_df: Raw production DataFrame.

    Returns:
        Comprehensive drift analysis results.
    """
    results: Dict[str, Any] = {"numerical": {}, "categorical": {}, "missing_data": {}}

    # Numerical features
    for col in NUMERICAL_FEATURES:
        if col not in baseline_df.columns or col not in production_df.columns:
            continue
        b = pd.to_numeric(baseline_df[col], errors="coerce")
        p = pd.to_numeric(production_df[col], errors="coerce")

        results["numerical"][col] = {
            "ks_test": ks_test(b, p),
            "psi": compute_psi(b, p),
            "jensen_shannon": jensen_shannon_distance(b, p),
            "mean_shift": mean_shift_test(b, p),
            "variance_change": variance_change_test(b, p),
        }

    # Categorical features
    for col in CATEGORICAL_FEATURES:
        if col not in baseline_df.columns or col not in production_df.columns:
            continue
        b = baseline_df[col].astype(str)
        p = production_df[col].astype(str)

        results["categorical"][col] = {
            "chi_square": chi_square_test(b, p),
            "frequency_shift": category_frequency_shift(b, p),
            "new_categories": new_category_detection(b, p),
        }

    # Missing data analysis
    for col in baseline_df.columns:
        if col in (ID_COLUMN, TARGET_COLUMN):
            continue
        if col not in production_df.columns:
            continue
        b_miss = float(baseline_df[col].isna().mean() * 100)
        p_miss = float(production_df[col].isna().mean() * 100)
        results["missing_data"][col] = {
            "baseline_missing_pct": round(b_miss, 2),
            "production_missing_pct": round(p_miss, 2),
            "increase": round(p_miss - b_miss, 2),
            "drift_detected": _safe_bool((p_miss - b_miss) > 5.0),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET-LEVEL DRIFT SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dataset_drift_score(feature_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate feature-level results into a dataset-level drift score.

    Returns:
        Dictionary with overall drift metrics and risk classification.
    """
    drifted_features = []
    total_features = 0

    for col, tests in feature_results.get("numerical", {}).items():
        total_features += 1
        ks_drift = tests.get("ks_test", {}).get("drift_detected", False)
        psi_drift = tests.get("psi", {}).get("drift_detected", False)
        if ks_drift or psi_drift:
            severity = tests.get("psi", {}).get("severity", "MODERATE")
            drifted_features.append({"feature": col, "type": "numerical", "severity": severity})

    for col, tests in feature_results.get("categorical", {}).items():
        total_features += 1
        chi2_drift = tests.get("chi_square", {}).get("drift_detected", False)
        new_cat_drift = tests.get("new_categories", {}).get("drift_detected", False)
        if chi2_drift or new_cat_drift:
            severity = "SEVERE" if new_cat_drift else "MODERATE"
            drifted_features.append({"feature": col, "type": "categorical", "severity": severity})

    missing_drift_count = sum(
        1 for v in feature_results.get("missing_data", {}).values()
        if v.get("drift_detected", False)
    )

    drifted_pct = len(drifted_features) / total_features if total_features > 0 else 0

    if drifted_pct >= DRIFT_THRESHOLDS["drifted_features_pct_severe"]:
        risk = "HIGH"
    elif drifted_pct >= DRIFT_THRESHOLDS["drifted_features_pct_moderate"]:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "total_features": total_features,
        "drifted_count": len(drifted_features),
        "drifted_pct": round(drifted_pct * 100, 2),
        "drifted_features": drifted_features,
        "missing_data_drift_count": missing_drift_count,
        "risk_level": risk,
        "overall_drift_score": round(drifted_pct, 4),
    }


def run_full_drift_analysis(baseline_df: pd.DataFrame, production_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run complete drift analysis: feature-level + dataset-level.

    Args:
        baseline_df: Raw baseline DataFrame.
        production_df: Raw production DataFrame.

    Returns:
        Full drift report dictionary.
    """
    logger.info("Starting full drift analysis...")
    feature_results = analyze_feature_drift(baseline_df, production_df)
    dataset_score = compute_dataset_drift_score(feature_results)

    report = {
        "feature_drift": feature_results,
        "dataset_drift": dataset_score,
    }
    logger.info(
        f"Drift analysis complete — Risk: {dataset_score['risk_level']}, "
        f"Drifted: {dataset_score['drifted_count']}/{dataset_score['total_features']}"
    )
    return report
