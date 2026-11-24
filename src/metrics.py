"""
Evaluation metrics module for the Data Drift Detection System.

Computes classification metrics, compares model performance across batches,
and computes prediction confidence statistics.
Guards against single-class edge cases in ROC AUC and confusion matrix calculation.
Follows PEP 8 and Global Naming Standardization.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import setup_logger

logger = setup_logger(__name__)


def evaluate_model(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    y_proba: Optional[Union[np.ndarray, pd.Series, list]] = None,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth binary target labels.
        y_pred: Predicted target labels.
        y_proba: Predicted probabilities for the positive class (optional).
        model_name: Identifier of the model for logging context.

    Returns:
        Dictionary containing accuracy, precision, recall, F1-score, ROC AUC,
        and confusion matrix.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if len(y_true_arr) == 0:
        return {
            "model_name": model_name,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc": None,
            "confusion_matrix": [],
        }

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    # Standardize confusion matrix labels to avoid single-class UserWarning in scikit-learn
    present_labels = set(np.unique(y_true_arr)) | set(np.unique(y_pred_arr))
    if present_labels.issubset({0, 1}):
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).tolist()
    elif len(present_labels) >= 2:
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=sorted(list(present_labels))).tolist()
    else:
        cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()

    metrics: Dict[str, Any] = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "roc_auc": None,
    }

    if y_proba is not None:
        y_proba_arr = np.asarray(y_proba)
        # In scikit-learn 1.8+, single-class y_true returns NaN / UndefinedMetricWarning
        unique_classes = np.unique(y_true_arr[~pd.isna(y_true_arr)])
        if len(unique_classes) < 2:
            metrics["roc_auc"] = None
        else:
            try:
                raw_auc = roc_auc_score(y_true_arr, y_proba_arr)
                if not np.isnan(raw_auc):
                    metrics["roc_auc"] = round(float(raw_auc), 4)
            except Exception as exc:
                metrics["roc_auc"] = None
                logger.warning("Could not compute ROC AUC for %s: %s", model_name, exc)

    logger.info(
        "%s — Accuracy: %.4f | F1: %.4f | AUC: %s",
        model_name,
        metrics["accuracy"],
        metrics["f1_score"],
        str(metrics["roc_auc"]) if metrics["roc_auc"] is not None else "N/A",
    )
    return metrics


def compare_performance(
    baseline_metrics: Dict[str, Any],
    current_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare current model production performance metrics against baseline benchmark values.

    Args:
        baseline_metrics: Dictionary of baseline evaluation metrics.
        current_metrics: Dictionary of current production evaluation metrics.

    Returns:
        Structured dictionary with metric deltas, degradation flags, and
        overall health status ('HEALTHY', 'WARNING', 'CRITICAL').
    """
    comparison: Dict[str, Any] = {}
    tracked_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    for key in tracked_metrics:
        baseline_val = baseline_metrics.get(key)
        current_val = current_metrics.get(key)
        if baseline_val is not None and current_val is not None:
            delta = round(float(current_val) - float(baseline_val), 4)
            comparison[key] = {
                "baseline": float(baseline_val),
                "current": float(current_val),
                "delta": delta,
                "degraded": bool(delta < 0),
            }

    degraded_count = sum(
        1
        for val in comparison.values()
        if isinstance(val, dict) and val.get("degraded", False)
    )

    if degraded_count == 0:
        health_status = "HEALTHY"
    elif degraded_count <= 2:
        health_status = "WARNING"
    else:
        health_status = "CRITICAL"

    comparison["overall_health"] = health_status
    comparison["degraded_metrics_count"] = int(degraded_count)

    if "confusion_matrix" in current_metrics and current_metrics["confusion_matrix"]:
        comparison["confusion_matrix"] = current_metrics["confusion_matrix"]
    elif "confusion_matrix" in baseline_metrics and baseline_metrics["confusion_matrix"]:
        comparison["confusion_matrix"] = baseline_metrics["confusion_matrix"]

    logger.info(
        "Performance comparison complete: %s (%d degraded metric(s))",
        health_status,
        degraded_count,
    )
    return comparison


def get_prediction_confidence_stats(
    y_proba: Union[np.ndarray, pd.Series, list]
) -> Dict[str, float]:
    """
    Compute distribution statistics over prediction certainty.

    Confidence is evaluated as max(p, 1-p) for binary classification.

    Args:
        y_proba: 1-D array of positive class predicted probabilities or 2-D array.

    Returns:
        Dictionary containing mean confidence, standard deviation, and
        low-confidence percentage (< 0.60).
    """
    proba_arr = np.asarray(y_proba, dtype=float)

    if proba_arr.ndim == 2:
        if proba_arr.shape[1] == 2:
            proba_arr = proba_arr[:, 1]
        else:
            proba_arr = proba_arr.ravel()

    # Filter out NaNs
    valid_proba = proba_arr[np.isfinite(proba_arr)]
    if len(valid_proba) == 0:
        return {
            "mean_confidence": 0.0,
            "std_confidence": 0.0,
            "low_confidence_pct": 0.0,
        }

    confidence_scores = np.maximum(valid_proba, 1.0 - valid_proba)
    mean_conf = float(np.mean(confidence_scores))
    std_conf = float(np.std(confidence_scores))
    low_conf_pct = float(np.mean(confidence_scores < 0.60) * 100.0)

    return {
        "mean_confidence": round(mean_conf, 4),
        "std_confidence": round(std_conf, 4),
        "low_confidence_pct": round(low_conf_pct, 2),
    }
