"""
Prediction service for the Data Drift Detection System.

Loads champion model artifacts, preprocesses production batches, computes
class predictions and probability distributions, and evaluates performance deltas
against baseline benchmarks when ground truth is provided.
Follows PEP 8 and Global Naming Standardization.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.config.settings import MODELS_DIR
from src.metrics import (
    compare_performance,
    evaluate_model,
    get_prediction_confidence_stats,
)
from src.preprocessing import DataPreprocessor
from src.train import load_trained_model
from src.utils import load_json, setup_logger

logger = setup_logger(__name__)


def predict_batch(df_raw: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute predictions on a production batch and evaluate metrics if ground truth is available.

    Args:
        df_raw: Raw production batch DataFrame.

    Returns:
        Dictionary containing:
            - 'predictions': np.ndarray of class labels
            - 'probabilities': np.ndarray of positive class probabilities
            - 'confidence_stats': confidence distribution metrics
            - 'current_metrics': evaluation metrics (if labels present)
            - 'comparison': performance delta comparison (if labels present)

    Raises:
        FileNotFoundError: If model or preprocessor artifacts are missing from disk.
    """
    if len(df_raw) == 0:
        return {
            "predictions": np.array([]),
            "probabilities": None,
            "confidence_stats": {
                "mean_confidence": 0.0,
                "std_confidence": 0.0,
                "low_confidence_pct": 0.0,
            },
        }

    # Verify and load artifacts
    model = load_trained_model()
    preprocessor = DataPreprocessor.load()

    baseline_metrics_path = MODELS_DIR / "baseline_metrics.json"
    baseline_metrics = (
        load_json(baseline_metrics_path) if baseline_metrics_path.exists() else {}
    )

    # Preprocess incoming production data
    X_processed, y_true = preprocessor.transform(df_raw)

    # Compute model predictions
    y_pred = model.predict(X_processed)
    y_proba = (
        model.predict_proba(X_processed)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    result: Dict[str, Any] = {
        "predictions": y_pred,
        "probabilities": y_proba,
    }

    # Evaluate if ground truth labels are provided
    if y_true is not None and len(y_true) > 0 and len(y_true.dropna()) > 0:
        current_metrics = evaluate_model(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            model_name="Production",
        )
        comparison = compare_performance(baseline_metrics, current_metrics)
        conf_stats = (
            get_prediction_confidence_stats(y_proba)
            if y_proba is not None
            else {}
        )

        result.update(
            {
                "current_metrics": current_metrics,
                "comparison": comparison,
                "confidence_stats": conf_stats,
            }
        )
    else:
        if y_proba is not None:
            result["confidence_stats"] = get_prediction_confidence_stats(y_proba)

    logger.info("Inference completed for %d production instances", len(X_processed))
    return result
