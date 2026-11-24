"""
Drift Detection Engine for the Data Drift Detection System.

Implements statistical and informational tests for both feature-level and dataset-level drift:
- Numerical drift: Two-sample Kolmogorov-Smirnov test, Population Stability Index (PSI),
  Jensen-Shannon Divergence/Distance, Relative Mean Shift, and Relative Variance Change.
- Categorical drift: Pearson's Chi-Square contingency test, Category Frequency Shift,
  and New Unseen Category Detection.
- Dataset-level drift aggregation and risk level classification (LOW, MEDIUM, HIGH).

Guards against small sample sizes, zero variance, NaN/inf values, and guarantees native Python
data types for clean JSON serialization. Follows PEP 8 and Global Naming Standardization.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from src.config.settings import (
    CATEGORICAL_FEATURES,
    DRIFT_THRESHOLDS,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from src.utils import setup_logger

logger = setup_logger(__name__)

# Minimum required samples for statistical significance testing
_MIN_SAMPLES: int = 10


def _safe_bool(value: Any) -> bool:
    """Safely convert truthy/falsy or numpy boolean values to native Python bool."""
    return bool(value)


# ═══════════════════════════════════════════════════════════════════════════════
# NUMERICAL DRIFT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def ks_test(baseline: pd.Series, production: pd.Series) -> Dict[str, Any]:
    """
    Execute two-sample Kolmogorov-Smirnov goodness-of-fit test.

    Args:
        baseline: Baseline feature series.
        production: Production feature series.

    Returns:
        Dictionary containing test metadata, KS statistic, p-value, and drift flag.
    """
    b_clean = pd.to_numeric(baseline, errors="coerce").dropna()
    b = b_clean[np.isfinite(b_clean)].to_numpy(dtype=float)

    p_clean = pd.to_numeric(production, errors="coerce").dropna()
    p = p_clean[np.isfinite(p_clean)].to_numpy(dtype=float)

    if len(b) < _MIN_SAMPLES or len(p) < _MIN_SAMPLES:
        logger.warning(
            "KS test: insufficient samples (baseline=%d, production=%d)",
            len(b), len(p)
        )
        return {
            "test": "Kolmogorov-Smirnov",
            "statistic": None,
            "p_value": None,
            "drift_detected": False,
            "warning": "Insufficient samples",
        }

    try:
        stat, pvalue = stats.ks_2samp(b, p)
        stat_val = float(stat) if not np.isnan(stat) else 0.0
        pval_val = float(pvalue) if not np.isnan(pvalue) else 1.0
        drift_flag = bool(pval_val < DRIFT_THRESHOLDS["ks_pvalue"])
        return {
            "test": "Kolmogorov-Smirnov",
            "statistic": round(stat_val, 6),
            "p_value": round(pval_val, 6),
            "drift_detected": drift_flag,
        }
    except Exception as exc:
        logger.warning("KS test execution error: %s", exc)
        return {
            "test": "Kolmogorov-Smirnov",
            "statistic": None,
            "p_value": None,
            "drift_detected": False,
            "error": str(exc),
        }


def compute_psi(baseline: pd.Series, production: pd.Series, bins: int = 10) -> Dict[str, Any]:
    """
    Calculate Population Stability Index (PSI) between baseline and production distributions.

    Args:
        baseline: Baseline numerical series.
        production: Production numerical series.
        bins: Number of quantiles/buckets (default 10).

    Returns:
        Dictionary containing PSI value, severity rating (LOW/MODERATE/SEVERE), and drift flag.
    """
    b_clean = pd.to_numeric(baseline, errors="coerce").dropna()
    b = b_clean[np.isfinite(b_clean)].to_numpy(dtype=float)

    p_clean = pd.to_numeric(production, errors="coerce").dropna()
    p = p_clean[np.isfinite(p_clean)].to_numpy(dtype=float)

    if len(b) < _MIN_SAMPLES or len(p) < _MIN_SAMPLES:
        return {
            "test": "PSI",
            "value": 0.0,
            "severity": "LOW",
            "drift_detected": False,
            "warning": "Insufficient samples",
        }

    # Guard against zero-variance columns in both baseline and production
    b_min, b_max = np.min(b), np.max(b)
    p_min, p_max = np.min(p), np.max(p)
    if b_min == b_max and p_min == p_max:
        if b_min == p_min:
            return {
                "test": "PSI",
                "value": 0.0,
                "severity": "LOW",
                "drift_detected": False,
            }

    # Use combined range if baseline has zero-variance to prevent identical bin edges
    if b_min == b_max:
        min_bound, max_bound = min(b_min, p_min), max(b_max, p_max)
    else:
        min_bound, max_bound = b_min, b_max

    if min_bound == max_bound:
        return {
            "test": "PSI",
            "value": 0.0,
            "severity": "LOW",
            "drift_detected": False,
        }

    breakpoints = np.linspace(min_bound, max_bound, bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    b_counts = np.histogram(b, bins=breakpoints)[0].astype(float) / len(b)
    p_counts = np.histogram(p, bins=breakpoints)[0].astype(float) / len(p)

    # Avoid zero division and log of zero
    b_counts = np.clip(b_counts, 1e-6, None)
    p_counts = np.clip(p_counts, 1e-6, None)

    raw_psi = np.sum((p_counts - b_counts) * np.log(p_counts / b_counts))
    if np.isnan(raw_psi) or np.isinf(raw_psi):
        psi_value = 0.0
    else:
        psi_value = float(raw_psi)

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
        "drift_detected": bool(psi_value > DRIFT_THRESHOLDS["psi_moderate"]),
    }


def jensen_shannon_distance(
    baseline: pd.Series, production: pd.Series, bins: int = 30
) -> Dict[str, Any]:
    """
    Calculate Jensen-Shannon distance (square root of JS divergence) between distributions.

    Args:
        baseline: Baseline numerical series.
        production: Production numerical series.
        bins: Number of histogram bins (default 30).

    Returns:
        Dictionary containing JS distance (0.0 to 1.0), severity, and drift flag.
    """
    b_clean = pd.to_numeric(baseline, errors="coerce").dropna()
    b = b_clean[np.isfinite(b_clean)].to_numpy(dtype=float)

    p_clean = pd.to_numeric(production, errors="coerce").dropna()
    p = p_clean[np.isfinite(p_clean)].to_numpy(dtype=float)

    if len(b) < _MIN_SAMPLES or len(p) < _MIN_SAMPLES:
        return {
            "test": "Jensen-Shannon",
            "value": 0.0,
            "severity": "LOW",
            "drift_detected": False,
            "warning": "Insufficient samples",
        }

    all_data = np.concatenate([b, p])
    min_val, max_val = np.min(all_data), np.max(all_data)

    if min_val == max_val:
        return {
            "test": "Jensen-Shannon",
            "value": 0.0,
            "severity": "LOW",
            "drift_detected": False,
        }

    breakpoints = np.linspace(min_val, max_val, bins + 1)
    b_hist = np.histogram(b, bins=breakpoints)[0].astype(float)
    p_hist = np.histogram(p, bins=breakpoints)[0].astype(float)

    b_sum = b_hist.sum()
    p_sum = p_hist.sum()
    b_hist = b_hist / (b_sum + 1e-10)
    p_hist = p_hist / (p_sum + 1e-10)

    raw_js = jensenshannon(b_hist, p_hist)
    if np.isnan(raw_js) or np.isinf(raw_js):
        js_dist = 0.0
    else:
        js_dist = float(raw_js)

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
        "drift_detected": bool(js_dist > DRIFT_THRESHOLDS["js_moderate"]),
    }


def mean_shift_test(baseline: pd.Series, production: pd.Series) -> Dict[str, Any]:
    """
    Detect relative percentage shift in mean values between baseline and production.

    Args:
        baseline: Baseline numerical series.
        production: Production numerical series.

    Returns:
        Dictionary containing means, percentage change, and drift flag.
    """
    b_clean = pd.to_numeric(baseline, errors="coerce").dropna()
    b_vals = b_clean[np.isfinite(b_clean)]

    p_clean = pd.to_numeric(production, errors="coerce").dropna()
    p_vals = p_clean[np.isfinite(p_clean)]

    if len(b_vals) < 2 or len(p_vals) < 2:
        return {
            "test": "Mean Shift",
            "baseline_mean": None,
            "production_mean": None,
            "pct_change": 0.0,
            "drift_detected": False,
            "warning": "Insufficient samples",
        }

    b_mean = float(b_vals.mean())
    p_mean = float(p_vals.mean())

    if abs(b_mean) < 1e-10:
        relative_change = 0.0 if abs(p_mean) < 1e-10 else 1.0
    else:
        relative_change = abs(p_mean - b_mean) / abs(b_mean)

    pct_change = round(relative_change * 100.0, 2)
    drift_flag = bool(relative_change > DRIFT_THRESHOLDS["mean_shift_pct"])

    return {
        "test": "Mean Shift",
        "baseline_mean": round(b_mean, 4),
        "production_mean": round(p_mean, 4),
        "pct_change": pct_change,
        "drift_detected": drift_flag,
    }


def variance_change_test(baseline: pd.Series, production: pd.Series) -> Dict[str, Any]:
    """
    Detect relative percentage change in variance between baseline and production.

    Args:
        baseline: Baseline numerical series.
        production: Production numerical series.

    Returns:
        Dictionary containing variances, percentage change, and drift flag.
    """
    b_clean = pd.to_numeric(baseline, errors="coerce").dropna()
    b_vals = b_clean[np.isfinite(b_clean)]

    p_clean = pd.to_numeric(production, errors="coerce").dropna()
    p_vals = p_clean[np.isfinite(p_clean)]

    if len(b_vals) < 2 or len(p_vals) < 2:
        return {
            "test": "Variance Change",
            "baseline_var": None,
            "production_var": None,
            "pct_change": 0.0,
            "drift_detected": False,
            "warning": "Insufficient samples",
        }

    b_var = float(b_vals.var())
    p_var = float(p_vals.var())

    if np.isnan(b_var) or np.isnan(p_var):
        return {
            "test": "Variance Change",
            "baseline_var": None,
            "production_var": None,
            "pct_change": 0.0,
            "drift_detected": False,
            "warning": "Variance could not be determined",
        }

    if abs(b_var) < 1e-10:
        relative_change = 0.0 if abs(p_var) < 1e-10 else 1.0
    else:
        relative_change = abs(p_var - b_var) / abs(b_var)

    pct_change = round(relative_change * 100.0, 2)
    drift_flag = bool(relative_change > DRIFT_THRESHOLDS["variance_change_pct"])

    return {
        "test": "Variance Change",
        "baseline_var": round(b_var, 4),
        "production_var": round(p_var, 4),
        "pct_change": pct_change,
        "drift_detected": drift_flag,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORICAL DRIFT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def chi_square_test(baseline: pd.Series, production: pd.Series) -> Dict[str, Any]:
    """
    Execute Pearson's Chi-Square test for categorical frequency shifts.

    Args:
        baseline: Baseline categorical series.
        production: Production categorical series.

    Returns:
        Dictionary containing Chi-square statistic, p-value, and drift flag.
    """
    b = baseline.dropna().astype(str)
    p = production.dropna().astype(str)

    if len(b) < _MIN_SAMPLES or len(p) < _MIN_SAMPLES:
        return {
            "test": "Chi-Square",
            "statistic": None,
            "p_value": None,
            "drift_detected": False,
            "warning": "Insufficient samples",
        }

    all_cats = sorted(set(b.unique()) | set(p.unique()))
    if len(all_cats) <= 1:
        return {
            "test": "Chi-Square",
            "statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "warning": "Only one unique category present",
        }

    b_counts = b.value_counts().reindex(all_cats, fill_value=0)
    p_counts = p.value_counts().reindex(all_cats, fill_value=0)

    b_total = float(b_counts.sum())
    if b_total == 0.0:
        return {
            "test": "Chi-Square",
            "statistic": None,
            "p_value": None,
            "drift_detected": False,
            "warning": "Empty baseline categories",
        }

    b_freq = (b_counts / b_total * len(p)).values.astype(float)
    p_freq = p_counts.values.astype(float)

    # Re-normalize after clipping to ensure sum(expected) == sum(observed) for SciPy compliance
    b_freq = np.clip(b_freq, 1e-6, None)
    b_freq = (b_freq / b_freq.sum()) * float(p_freq.sum())

    try:
        stat, pvalue = stats.chisquare(p_freq, f_exp=b_freq)
        stat_val = float(stat) if not np.isnan(stat) else 0.0
        pval_val = float(pvalue) if not np.isnan(pvalue) else 1.0
        drift_flag = bool(pval_val < DRIFT_THRESHOLDS["chi2_pvalue"])
        return {
            "test": "Chi-Square",
            "statistic": round(stat_val, 6),
            "p_value": round(pval_val, 6),
            "drift_detected": drift_flag,
        }
    except Exception as exc:
        logger.warning("Chi-Square test computation failed: %s", exc)
        return {
            "test": "Chi-Square",
            "statistic": None,
            "p_value": None,
            "drift_detected": False,
            "error": str(exc),
        }


def category_frequency_shift(baseline: pd.Series, production: pd.Series) -> Dict[str, Any]:
    """
    Compute distribution shift across individual categories.

    Args:
        baseline: Baseline categorical series.
        production: Production categorical series.

    Returns:
        Dictionary mapping each category to baseline %, production %, and delta shift.
    """
    b_clean = baseline.dropna().astype(str)
    p_clean = production.dropna().astype(str)

    if len(b_clean) == 0 or len(p_clean) == 0:
        return {
            "test": "Category Frequency Shift",
            "shifts": {},
            "max_shift_pct": 0.0,
            "drift_detected": False,
            "warning": "Empty series provided",
        }

    b_freq = b_clean.value_counts(normalize=True)
    p_freq = p_clean.value_counts(normalize=True)

    all_cats = sorted(set(b_freq.index) | set(p_freq.index))
    shifts: Dict[str, Dict[str, float]] = {}
    max_shift: float = 0.0

    for cat in all_cats:
        b_pct = float(b_freq.get(cat, 0.0))
        p_pct = float(p_freq.get(cat, 0.0))
        shift = abs(p_pct - b_pct)
        shifts[str(cat)] = {
            "baseline_pct": round(b_pct * 100.0, 2),
            "production_pct": round(p_pct * 100.0, 2),
            "shift": round(shift * 100.0, 2),
        }
        if shift > max_shift:
            max_shift = shift

    threshold = DRIFT_THRESHOLDS.get("mean_shift_pct", 0.10)
    drift_flag = bool(max_shift > threshold)

    return {
        "test": "Category Frequency Shift",
        "shifts": shifts,
        "max_shift_pct": round(max_shift * 100.0, 2),
        "drift_detected": drift_flag,
    }


def new_category_detection(baseline: pd.Series, production: pd.Series) -> Dict[str, Any]:
    """
    Detect new, previously unseen categorical values in production data.

    Args:
        baseline: Baseline categorical series.
        production: Production categorical series.

    Returns:
        Dictionary listing new and missing categories with counts and drift status.
    """
    b_cats = set(baseline.dropna().astype(str).unique())
    p_cats = set(production.dropna().astype(str).unique())

    new_cats = sorted(list(p_cats - b_cats))
    missing_cats = sorted(list(b_cats - p_cats))

    return {
        "test": "New Category Detection",
        "new_categories": new_cats,
        "missing_categories": missing_cats,
        "new_count": int(len(new_cats)),
        "missing_count": int(len(missing_cats)),
        "drift_detected": bool(len(new_cats) > 0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE-LEVEL DRIFT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_feature_drift(
    baseline_df: pd.DataFrame, production_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Run comprehensive statistical drift tests across all numerical and categorical features.

    Args:
        baseline_df: Raw baseline DataFrame.
        production_df: Raw production DataFrame.

    Returns:
        Structured dictionary of feature-level drift metrics.
    """
    results: Dict[str, Any] = {"numerical": {}, "categorical": {}, "missing_data": {}}

    # Numerical feature tests
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

    # Categorical feature tests
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

    # Missingness comparison across all features
    for col in baseline_df.columns:
        if col in (ID_COLUMN, TARGET_COLUMN):
            continue
        if col not in production_df.columns:
            continue
        b_miss = float(baseline_df[col].isna().mean() * 100.0)
        p_miss = float(production_df[col].isna().mean() * 100.0)
        diff = p_miss - b_miss

        results["missing_data"][col] = {
            "baseline_missing_pct": round(b_miss, 2),
            "production_missing_pct": round(p_miss, 2),
            "increase": round(diff, 2),
            "drift_detected": bool(diff > 5.0),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET-LEVEL DRIFT SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dataset_drift_score(feature_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate feature-level test outcomes into an overall dataset drift risk score.

    Args:
        feature_results: Output dictionary from analyze_feature_drift().

    Returns:
        Dictionary summarizing drifted count, drift percentage, and risk level.
    """
    drifted_features: List[Dict[str, str]] = []
    total_features: int = 0

    # Numerical feature evaluations
    for col, tests in feature_results.get("numerical", {}).items():
        total_features += 1
        ks_flag = tests.get("ks_test", {}).get("drift_detected", False)
        psi_flag = tests.get("psi", {}).get("drift_detected", False)
        if ks_flag or psi_flag:
            severity = tests.get("psi", {}).get("severity", "MODERATE")
            drifted_features.append(
                {"feature": col, "type": "numerical", "severity": severity}
            )

    # Categorical feature evaluations
    for col, tests in feature_results.get("categorical", {}).items():
        total_features += 1
        chi2_flag = tests.get("chi_square", {}).get("drift_detected", False)
        new_cat_flag = tests.get("new_categories", {}).get("drift_detected", False)
        if chi2_flag or new_cat_flag:
            severity = "SEVERE" if new_cat_flag else "MODERATE"
            drifted_features.append(
                {"feature": col, "type": "categorical", "severity": severity}
            )

    missing_drift_count = sum(
        1
        for val in feature_results.get("missing_data", {}).values()
        if val.get("drift_detected", False)
    )

    drifted_pct = (len(drifted_features) / total_features) if total_features > 0 else 0.0

    if drifted_pct >= DRIFT_THRESHOLDS["drifted_features_pct_severe"]:
        risk_level = "HIGH"
    elif drifted_pct >= DRIFT_THRESHOLDS["drifted_features_pct_moderate"]:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "total_features": int(total_features),
        "drifted_count": int(len(drifted_features)),
        "drifted_pct": round(drifted_pct * 100.0, 2),
        "drifted_features": drifted_features,
        "missing_data_drift_count": int(missing_drift_count),
        "risk_level": risk_level,
        "overall_drift_score": round(drifted_pct, 4),
    }


def run_full_drift_analysis(
    baseline_df: pd.DataFrame, production_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Execute full end-to-end drift detection workflow: feature-level and dataset-level.

    Args:
        baseline_df: Baseline training DataFrame.
        production_df: Production inference DataFrame.

    Returns:
        Comprehensive drift analysis dictionary.
    """
    logger.info("Executing comprehensive data drift analysis...")
    feature_results = analyze_feature_drift(baseline_df, production_df)
    dataset_score = compute_dataset_drift_score(feature_results)

    report = {
        "feature_drift": feature_results,
        "dataset_drift": dataset_score,
    }
    logger.info(
        "Drift analysis complete — Risk: %s, Drifted features: %d/%d (%.1f%%)",
        dataset_score["risk_level"],
        dataset_score["drifted_count"],
        dataset_score["total_features"],
        dataset_score["drifted_pct"],
    )
    return report
