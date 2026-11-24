"""
Model training module for the Data Drift Detection System.

Trains multiple classification algorithms with class-imbalance weighting:
- Logistic Regression (class_weight='balanced')
- Random Forest (class_weight='balanced')
- Gradient Boosting Classifier
- XGBoost Classifier (scale_pos_weight tuned to class ratio)

Evaluates via Stratified K-Fold cross validation, selects best model based
on mean CV F1-score, and persists production artifacts.
Follows PEP 8 and Global Naming Standardization.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

try:
    from xgboost import XGBClassifier
except (ImportError, ModuleNotFoundError):
    # Fallback surrogate for serverless environments (e.g. Vercel) where xgboost is omitted to keep bundle size lean
    from sklearn.ensemble import HistGradientBoostingClassifier

    class XGBClassifier(HistGradientBoostingClassifier):  # type: ignore[no-redef]
        """Serverless fallback surrogate for XGBClassifier when xgboost is not installed."""

        def __init__(
            self,
            scale_pos_weight: float = 1.0,
            random_state: int = 42,
            **kwargs: Any,
        ) -> None:
            super().__init__(random_state=random_state)
            self.scale_pos_weight = scale_pos_weight

from src.config.settings import MODEL_SETTINGS, MODELS_DIR
from src.metrics import evaluate_model
from src.utils import get_timestamp, save_json, setup_logger

logger = setup_logger(__name__)


def get_models(scale_pos_weight: float = 2.7) -> Dict[str, Any]:
    """
    Construct model estimator dictionary.

    Args:
        scale_pos_weight: Minority class weighting factor for XGBoost (default 2.7).

    Returns:
        Dictionary mapping model names to instantiated estimators.
    """
    seed = int(MODEL_SETTINGS["random_state"])
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=seed,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            verbosity=0,
        ),
    }


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Train and cross-validate candidate models, select champion by CV F1-score, and save artifacts.

    Args:
        X: Preprocessed feature matrix DataFrame.
        y: Binary target Series.

    Returns:
        Dictionary containing champion model, baseline metrics, and full model benchmark results.
    """
    test_size = float(MODEL_SETTINGS["test_size"])
    seed = int(MODEL_SETTINGS["random_state"])
    cv_folds = int(MODEL_SETTINGS["cv_folds"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    logger.info("Dataset split: Train samples=%d, Test samples=%d", len(X_train), len(X_test))

    # Dynamically compute class balance ratio for minority class
    pos_samples = int((y_train == 1).sum())
    neg_samples = int((y_train == 0).sum())
    class_ratio = (neg_samples / pos_samples) if pos_samples > 0 else 2.7

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    model_candidates = get_models(scale_pos_weight=class_ratio)

    all_results: Dict[str, Any] = {}
    best_model: Optional[Any] = None
    best_f1: float = -1.0
    best_name: str = ""

    for name, model in model_candidates.items():
        logger.info("Evaluating candidate model: %s...", name)

        # Cross-validation on training partition
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        logger.info("%s — CV F1: %.4f (± %.4f)", name, cv_mean, cv_std)

        # Fit model on training split and evaluate on holdout test partition
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        metrics = evaluate_model(y_test, y_pred, y_proba, model_name=name)
        metrics["cv_f1_mean"] = round(cv_mean, 4)
        metrics["cv_f1_std"] = round(cv_std, 4)

        all_results[name] = {"model": model, "metrics": metrics}

        # Select model using CV F1 score to minimize overfitting variance
        if cv_mean > best_f1:
            best_f1 = cv_mean
            best_model = model
            best_name = name

    logger.info("Selected champion model: %s (Mean CV F1: %.4f)", best_name, best_f1)

    # Persist champion model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    champion_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, champion_path)
    logger.info("Champion model saved to %s", champion_path)

    # Save baseline metrics
    baseline_metrics = all_results[best_name]["metrics"]
    save_json(baseline_metrics, MODELS_DIR / "baseline_metrics.json")

    # Save comprehensive training benchmark summary
    summary_data = {
        "timestamp": get_timestamp(),
        "best_model": best_name,
        "best_cv_f1": round(best_f1, 4),
        "all_models": {k: v["metrics"] for k, v in all_results.items()},
    }
    save_json(summary_data, MODELS_DIR / "training_summary.json")

    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "baseline_metrics": baseline_metrics,
        "all_results": {k: v["metrics"] for k, v in all_results.items()},
        "X_test": X_test,
        "y_test": y_test,
    }


def load_trained_model(filepath: Optional[Union[str, Path]] = None) -> Any:
    """
    Load a trained model artifact from disk.

    Args:
        filepath: Optional path to joblib model artifact. Defaults to MODELS_DIR / 'best_model.joblib'.

    Returns:
        Loaded scikit-learn or XGBoost model instance.

    Raises:
        FileNotFoundError: If model artifact is missing.
    """
    target_path = Path(filepath) if filepath is not None else (MODELS_DIR / "best_model.joblib")
    if not target_path.exists():
        raise FileNotFoundError(f"Trained model artifact not found at: {target_path}")

    model = joblib.load(target_path)
    logger.info("Successfully loaded model from %s", target_path)
    return model
