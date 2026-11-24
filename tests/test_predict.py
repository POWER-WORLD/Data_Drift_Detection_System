"""
Unit and integration tests for the prediction service (src/predict.py).

Tests batch predictions with and without ground truth, probability confidence analysis,
and error handling when model artifacts are unavailable.
"""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_baseline_data
from src.predict import predict_batch
from src.config.settings import MODELS_DIR, TARGET_COLUMN


@pytest.fixture
def baseline_sample() -> pd.DataFrame:
    """Fixture providing a sample of the baseline dataset."""
    df = load_baseline_data()
    return df.head(100).copy()


class TestPredictBatch:
    """Tests for the predict_batch function."""

    def test_predict_with_ground_truth(self, baseline_sample: pd.DataFrame) -> None:
        """Verify predictions on data containing ground truth labels produce comparison metrics."""
        if not (MODELS_DIR / "best_model.joblib").exists():
            pytest.skip("Model artifact not yet trained")

        result: Dict[str, Any] = predict_batch(baseline_sample)

        assert "predictions" in result
        assert "probabilities" in result
        assert len(result["predictions"]) == len(baseline_sample)
        assert len(result["probabilities"]) == len(baseline_sample)

        # Ground truth was present
        assert "current_metrics" in result
        assert "comparison" in result
        assert "confidence_stats" in result

        metrics = result["current_metrics"]
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0

        comparison = result["comparison"]
        assert comparison["overall_health"] in ("HEALTHY", "WARNING", "CRITICAL")

        conf = result["confidence_stats"]
        assert 0.0 <= conf["mean_confidence"] <= 1.0

    def test_predict_without_ground_truth(self, baseline_sample: pd.DataFrame) -> None:
        """Verify predictions on unlabelled production data succeed without computing accuracy."""
        if not (MODELS_DIR / "best_model.joblib").exists():
            pytest.skip("Model artifact not yet trained")

        df_unlabelled = baseline_sample.drop(columns=[TARGET_COLUMN])
        result = predict_batch(df_unlabelled)

        assert "predictions" in result
        assert "probabilities" in result
        assert len(result["predictions"]) == len(df_unlabelled)
        assert "confidence_stats" in result

        # Ground truth absent — comparison & current_metrics should not be computed
        assert "current_metrics" not in result
        assert "comparison" not in result

    def test_probability_values_in_valid_range(self, baseline_sample: pd.DataFrame) -> None:
        """Verify all predicted probabilities fall strictly between 0 and 1 inclusive."""
        if not (MODELS_DIR / "best_model.joblib").exists():
            pytest.skip("Model artifact not yet trained")

        result = predict_batch(baseline_sample)
        proba = result["probabilities"]
        assert proba is not None
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_missing_model_artifact_raises(self, baseline_sample: pd.DataFrame, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Verify FileNotFoundError is raised when the model artifact does not exist."""
        fake_models_dir = tmp_path / "empty_models"
        fake_models_dir.mkdir()
        monkeypatch.setattr("src.train.MODELS_DIR", fake_models_dir)
        monkeypatch.setattr("src.predict.MODELS_DIR", fake_models_dir)

        with pytest.raises(FileNotFoundError):
            predict_batch(baseline_sample)
