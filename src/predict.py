"""
Prediction service for the Data Drift Detection System.
Loads model artifacts, runs predictions on production batches,
and compares performance against baseline metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from src.train import load_trained_model
from src.preprocessing import DataPreprocessor
from src.metrics import evaluate_model, compare_performance, get_prediction_confidence_stats
from src.utils import setup_logger, load_json
from src.config.settings import MODELS_DIR

logger = setup_logger(__name__)


def predict_batch(df_raw: pd.DataFrame) -> Dict[str, Any]:
    """
    Run predictions on a production batch and evaluate against baseline.

    Args:
        df_raw: Raw production DataFrame.

    Returns:
        Dictionary with predictions, metrics, and comparison results.

    Raises:
        FileNotFoundError: If model or preprocessor artifacts are missing.
    """
    # Load artifacts
    model = load_trained_model()
    preprocessor = DataPreprocessor.load()
    baseline_metrics = load_json(MODELS_DIR / "baseline_metrics.json")

    # Preprocess
    X, y_true = preprocessor.transform(df_raw)

    # Predict
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    result: Dict[str, Any] = {"predictions": y_pred, "probabilities": y_proba}

    # Evaluate if ground truth is available
    if y_true is not None and len(y_true) > 0:
        current_metrics = evaluate_model(y_true, y_pred, y_proba, model_name="Production")
        comparison = compare_performance(baseline_metrics, current_metrics)
        conf_stats = get_prediction_confidence_stats(y_proba) if y_proba is not None else {}
        result.update({
            "current_metrics": current_metrics,
            "comparison": comparison,
            "confidence_stats": conf_stats,
        })
    else:
        # No ground truth — only confidence stats
        if y_proba is not None:
            result["confidence_stats"] = get_prediction_confidence_stats(y_proba)

    logger.info(f"Predictions complete for {len(X)} samples")
    return result
