"""
Evaluation metrics module for the Data Drift Detection System.
Provides model evaluation, performance comparison, and prediction confidence analysis.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from src.utils import setup_logger

logger = setup_logger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities for the positive class (optional).
        model_name: Name of the model for logging.

    Returns:
        Dictionary with accuracy, precision, recall, F1, ROC AUC,
        and confusion matrix.
    """
    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
        except ValueError:
            metrics["roc_auc"] = None
            logger.warning(f"Could not compute ROC AUC for {model_name}")
    else:
        metrics["roc_auc"] = None

    logger.info(
        f"{model_name} — Acc:{metrics['accuracy']:.4f} "
        f"F1:{metrics['f1_score']:.4f} AUC:{metrics.get('roc_auc', 'N/A')}"
    )
    return metrics


def compare_performance(
    baseline_metrics: Dict[str, Any],
    current_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare current model performance against baseline metrics.

    Args:
        baseline_metrics: Metrics from baseline model evaluation.
        current_metrics: Metrics from current production evaluation.

    Returns:
        Dictionary with per-metric deltas, degradation flags, and
        an overall health classification (HEALTHY / WARNING / CRITICAL).
    """
    comparison: Dict[str, Any] = {}
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        b = baseline_metrics.get(key)
        c = current_metrics.get(key)
        if b is not None and c is not None:
            delta = round(c - b, 4)
            comparison[key] = {
                "baseline": b,
                "current": c,
                "delta": delta,
                "degraded": delta < 0,
            }

    deg = sum(
        1 for v in comparison.values()
        if isinstance(v, dict) and v.get("degraded")
    )
    comparison["overall_health"] = (
        "HEALTHY" if deg == 0 else ("WARNING" if deg <= 2 else "CRITICAL")
    )
    comparison["degraded_metrics_count"] = deg

    logger.info(f"Performance comparison: {comparison['overall_health']} ({deg} degraded)")
    return comparison


def get_prediction_confidence_stats(y_proba: np.ndarray) -> Dict[str, float]:
    """
    Compute prediction confidence statistics from probability scores.

    Confidence is defined as ``max(p, 1-p)`` for each prediction,
    i.e. how certain the model is about its binary decision.

    Args:
        y_proba: Predicted probabilities for the positive class (1-D array).

    Returns:
        Dictionary with mean confidence, standard deviation, and
        percentage of predictions below the 0.6 confidence threshold.
    """
    conf = np.max(np.column_stack([y_proba, 1 - y_proba]), axis=1)
    return {
        "mean_confidence": round(float(np.mean(conf)), 4),
        "std_confidence": round(float(np.std(conf)), 4),
        "low_confidence_pct": round(float(np.mean(conf < 0.6) * 100), 2),
    }
