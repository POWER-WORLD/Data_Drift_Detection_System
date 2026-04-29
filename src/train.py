"""
Model training module for the Data Drift Detection System.
Trains multiple classifiers with class-weight balancing, evaluates via
stratified cross-validation, selects the best, and saves artifacts.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from src.config.settings import MODEL_SETTINGS, MODELS_DIR
from src.metrics import evaluate_model
from src.utils import setup_logger, save_json, get_timestamp

logger = setup_logger(__name__)


def get_models() -> Dict[str, Any]:
    """
    Return dictionary of model instances to train.
    All tree-based models use class_weight='balanced' (or scale_pos_weight
    for XGBoost) to handle the ~73/27 churn class imbalance.
    """
    rs = MODEL_SETTINGS["random_state"]
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=rs,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=rs,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, random_state=rs,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, eval_metric="logloss",
            scale_pos_weight=2.7,   # ≈ ratio of majority / minority class
            random_state=rs, verbosity=0,
        ),
    }


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Train all models, evaluate them with stratified cross-validation,
    select the best by mean CV F1 score, and save the best model artifact.

    Args:
        X: Preprocessed feature matrix.
        y: Encoded target vector.

    Returns:
        Dictionary with all results and the best model info.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=MODEL_SETTINGS["test_size"],
        random_state=MODEL_SETTINGS["random_state"],
        stratify=y,
    )
    logger.info(f"Train/Test split: {len(X_train)}/{len(X_test)}")

    cv = StratifiedKFold(
        n_splits=MODEL_SETTINGS["cv_folds"],
        shuffle=True,
        random_state=MODEL_SETTINGS["random_state"],
    )

    models = get_models()
    all_results = {}
    best_model = None
    best_f1 = -1
    best_name = ""

    for name, model in models.items():
        logger.info(f"Training {name}...")

        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        logger.info(f"{name} CV F1: {cv_mean:.4f} ± {cv_std:.4f}")

        # Full train + test evaluation
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = evaluate_model(y_test, y_pred, y_proba, model_name=name)
        metrics["cv_f1_mean"] = round(cv_mean, 4)
        metrics["cv_f1_std"] = round(cv_std, 4)

        all_results[name] = {"model": model, "metrics": metrics}

        # Select by CV F1 (more reliable than single-split F1)
        if cv_mean > best_f1:
            best_f1 = cv_mean
            best_model = model
            best_name = name

    logger.info(f"Best model: {best_name} (CV F1={best_f1:.4f})")

    # Save best model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)
    logger.info(f"Best model saved to {model_path}")

    # Save baseline metrics (test-set metrics of the best model)
    baseline_metrics = all_results[best_name]["metrics"]
    save_json(baseline_metrics, MODELS_DIR / "baseline_metrics.json")

    # Save all results summary
    summary = {
        "timestamp": get_timestamp(),
        "best_model": best_name,
        "best_cv_f1": round(best_f1, 4),
        "all_models": {n: r["metrics"] for n, r in all_results.items()},
    }
    save_json(summary, MODELS_DIR / "training_summary.json")

    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "baseline_metrics": baseline_metrics,
        "all_results": {n: r["metrics"] for n, r in all_results.items()},
        "X_test": X_test,
        "y_test": y_test,
    }


def load_trained_model(filepath: Optional[Path] = None):
    """Load saved model from disk."""
    filepath = filepath or (MODELS_DIR / "best_model.joblib")
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    model = joblib.load(filepath)
    logger.info(f"Loaded model from {filepath}")
    return model


if __name__ == "__main__":
    from src.data_loader import load_baseline_data, generate_production_batches
    from src.preprocessing import DataPreprocessor

    # Load and preprocess
    df = load_baseline_data()
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    preprocessor.save()

    # Generate production batches
    generate_production_batches(df)

    # Train and evaluate
    results = train_and_evaluate(X, y)
    print(f"\nBest model: {results['best_model_name']}")
    for name, m in results["all_results"].items():
        print(f"  {name}: F1={m['f1_score']:.4f} CV-F1={m.get('cv_f1_mean','N/A')} AUC={m.get('roc_auc', 'N/A')}")
