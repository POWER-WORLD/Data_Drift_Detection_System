"""
Tests for model training functionality (src/train.py).

Verifies model architecture initialization, cross-validation evaluation,
champion model selection, baseline metric persistence, and artifact deserialization.
"""

from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
import pytest

from src.config.settings import MODELS_DIR
from src.train import get_models, load_trained_model, train_and_evaluate
from src.utils import load_json


class TestTrainedModelArtifacts:
    """Tests for model training outputs (run after training)."""

    def test_best_model_exists(self) -> None:
        model_path = MODELS_DIR / "best_model.joblib"
        assert model_path.exists(), "Best model artifact not found"

    def test_preprocessor_exists(self) -> None:
        prep_path = MODELS_DIR / "preprocessor.joblib"
        assert prep_path.exists(), "Preprocessor artifact not found"

    def test_baseline_metrics_exist(self) -> None:
        metrics_path = MODELS_DIR / "baseline_metrics.json"
        assert metrics_path.exists(), "Baseline metrics not found"

    def test_baseline_metrics_valid(self) -> None:
        metrics = load_json(MODELS_DIR / "baseline_metrics.json")
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_training_summary_exists(self) -> None:
        summary_path = MODELS_DIR / "training_summary.json"
        assert summary_path.exists()
        summary = load_json(summary_path)
        assert "best_model" in summary
        assert "all_models" in summary
        assert len(summary["all_models"]) == 4  # 4 algorithms


class TestGetModels:
    """Tests for get_models function."""

    def test_returns_four_model_architectures(self) -> None:
        """Verify get_models initializes the 4 core classifiers."""
        models = get_models(scale_pos_weight=2.5)
        expected_keys = {"LogisticRegression", "RandomForest", "GradientBoosting", "XGBoost"}
        assert set(models.keys()) == expected_keys
        assert models["XGBoost"].scale_pos_weight == 2.5


class TestTrainAndEvaluate:
    """Tests for train_and_evaluate training pipeline."""

    def test_train_pipeline_execution_synthetic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify train_and_evaluate trains candidate models, selects champion, and saves artifacts."""
        temp_models_dir = tmp_path / "models"
        monkeypatch.setattr("src.train.MODELS_DIR", temp_models_dir)

        # Create minimal synthetic dataset with 60 samples
        rng = np.random.RandomState(42)
        n = 60
        X = pd.DataFrame({
            "feat_1": rng.normal(0, 1, n),
            "feat_2": rng.normal(5, 2, n),
            "feat_3": rng.choice([0, 1], n),
        })
        y = pd.Series(rng.choice([0, 1], n, p=[0.7, 0.3]), name="Churn")

        result = train_and_evaluate(X, y)

        assert "best_model_name" in result
        assert "best_model" in result
        assert "baseline_metrics" in result
        assert "all_results" in result
        assert len(result["all_results"]) == 4

        # Verify artifacts were written to temp directory
        assert (temp_models_dir / "best_model.joblib").exists()
        assert (temp_models_dir / "baseline_metrics.json").exists()
        assert (temp_models_dir / "training_summary.json").exists()


class TestLoadTrainedModel:
    """Tests for load_trained_model loader."""

    def test_loads_existing_champion_model(self) -> None:
        """Verify loading champion model succeeds and returns fitted estimator."""
        model = load_trained_model()
        assert hasattr(model, "predict")

    def test_missing_model_raises_file_not_found(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError is raised when target artifact does not exist."""
        fake_path = tmp_path / "nonexistent_model.joblib"
        with pytest.raises(FileNotFoundError):
            load_trained_model(fake_path)
