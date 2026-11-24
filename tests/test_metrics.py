"""
Unit tests for the model evaluation and metrics module (src/metrics.py).

Tests comprehensive evaluation metrics computation, performance comparison against
baselines, degradation detection, and prediction confidence statistics under standard
and edge conditions.
"""

from typing import Any, Dict
import numpy as np
import pytest

from src.metrics import (
    compare_performance,
    evaluate_model,
    get_prediction_confidence_stats,
)


class TestEvaluateModel:
    """Tests for the evaluate_model function."""

    def test_standard_binary_classification(self) -> None:
        """Verify metric calculation on standard balanced binary predictions."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8, 0.7, 0.85])

        result: Dict[str, Any] = evaluate_model(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            model_name="TestClassifier",
        )

        assert result["model_name"] == "TestClassifier"
        assert 0.0 <= result["accuracy"] <= 1.0
        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"] <= 1.0
        assert 0.0 <= result["f1_score"] <= 1.0
        assert result["roc_auc"] is not None
        assert 0.0 <= result["roc_auc"] <= 1.0
        assert isinstance(result["confusion_matrix"], list)
        assert len(result["confusion_matrix"]) == 2

    def test_perfect_predictions(self) -> None:
        """Verify metric values when predictions match ground truth perfectly."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.05, 0.1, 0.9, 0.95])

        result = evaluate_model(y_true, y_pred, y_proba)

        assert result["accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1_score"] == 1.0
        assert result["roc_auc"] == 1.0

    def test_zero_division_resilience(self) -> None:
        """Verify zero division does not raise exceptions when positive class is never predicted."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0])

        result = evaluate_model(y_true, y_pred, y_proba=None)

        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1_score"] == 0.0
        assert result["roc_auc"] is None

    def test_single_class_roc_auc_fallback(self) -> None:
        """Verify ROC AUC handles single-class ground truth gracefully by returning None or NaN."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        y_proba = np.array([0.9, 0.8, 0.85, 0.95])

        result = evaluate_model(y_true, y_pred, y_proba)
        assert result["roc_auc"] is None or np.isnan(result["roc_auc"])


class TestComparePerformance:
    """Tests for the compare_performance function."""

    def test_healthy_performance_comparison(self) -> None:
        """Verify HEALTHY classification when current metrics meet or exceed baseline."""
        baseline: Dict[str, Any] = {
            "accuracy": 0.8000,
            "precision": 0.7500,
            "recall": 0.7000,
            "f1_score": 0.7241,
            "roc_auc": 0.8500,
        }
        current: Dict[str, Any] = {
            "accuracy": 0.8200,
            "precision": 0.7600,
            "recall": 0.7100,
            "f1_score": 0.7340,
            "roc_auc": 0.8600,
        }

        comparison = compare_performance(baseline, current)

        assert comparison["overall_health"] == "HEALTHY"
        assert comparison["degraded_metrics_count"] == 0
        assert comparison["accuracy"]["delta"] == 0.02
        assert comparison["accuracy"]["degraded"] is False

    def test_warning_performance_comparison(self) -> None:
        """Verify WARNING classification when 1 or 2 metrics degrade."""
        baseline: Dict[str, Any] = {
            "accuracy": 0.8000,
            "precision": 0.7500,
            "recall": 0.7000,
            "f1_score": 0.7241,
            "roc_auc": 0.8500,
        }
        current: Dict[str, Any] = {
            "accuracy": 0.7800,  # degraded
            "precision": 0.7300,  # degraded
            "recall": 0.7200,     # improved
            "f1_score": 0.7300,   # improved
            "roc_auc": 0.8550,    # improved
        }

        comparison = compare_performance(baseline, current)

        assert comparison["overall_health"] == "WARNING"
        assert comparison["degraded_metrics_count"] == 2
        assert comparison["accuracy"]["degraded"] is True
        assert comparison["precision"]["degraded"] is True

    def test_critical_performance_comparison(self) -> None:
        """Verify CRITICAL classification when 3 or more metrics degrade."""
        baseline: Dict[str, Any] = {
            "accuracy": 0.8500,
            "precision": 0.8000,
            "recall": 0.7800,
            "f1_score": 0.7898,
            "roc_auc": 0.9000,
        }
        current: Dict[str, Any] = {
            "accuracy": 0.7000,  # degraded
            "precision": 0.6500,  # degraded
            "recall": 0.6000,     # degraded
            "f1_score": 0.6240,   # degraded
            "roc_auc": 0.7500,    # degraded
        }

        comparison = compare_performance(baseline, current)

        assert comparison["overall_health"] == "CRITICAL"
        assert comparison["degraded_metrics_count"] == 5

    def test_partial_metrics_graceful_handling(self) -> None:
        """Verify missing metric keys in baseline or current do not raise errors."""
        baseline: Dict[str, Any] = {"accuracy": 0.8000}
        current: Dict[str, Any] = {"accuracy": 0.7900, "f1_score": 0.7500}

        comparison = compare_performance(baseline, current)

        assert "accuracy" in comparison
        assert comparison["accuracy"]["degraded"] is True
        assert "f1_score" not in comparison  # Not present in baseline
        assert comparison["degraded_metrics_count"] == 1


class TestGetPredictionConfidenceStats:
    """Tests for the get_prediction_confidence_stats function."""

    def test_confidence_statistics_high_certainty(self) -> None:
        """Verify confidence metrics when predictions are very confident (close to 0 or 1)."""
        y_proba = np.array([0.02, 0.98, 0.05, 0.95])
        stats = get_prediction_confidence_stats(y_proba)

        assert stats["mean_confidence"] >= 0.95
        assert stats["low_confidence_pct"] == 0.0

    def test_confidence_statistics_low_certainty(self) -> None:
        """Verify confidence metrics when predictions are uncertain (near 0.5)."""
        y_proba = np.array([0.51, 0.49, 0.52, 0.48])
        stats = get_prediction_confidence_stats(y_proba)

        assert stats["mean_confidence"] < 0.60
        assert stats["low_confidence_pct"] == 100.0

    def test_confidence_boundary_precision(self) -> None:
        """Verify confidence computation exactness on known values."""
        y_proba = np.array([0.50, 0.60, 0.70, 0.80])
        # confidences: max(p, 1-p) -> [0.50, 0.60, 0.70, 0.80]
        # conf < 0.6: only 0.50 -> 1 out of 4 = 25.0%
        stats = get_prediction_confidence_stats(y_proba)

        assert stats["mean_confidence"] == 0.65
        assert stats["low_confidence_pct"] == 25.0
