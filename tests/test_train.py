"""
Tests for model training functionality.
"""

import pytest
from src.config.settings import MODELS_DIR
from src.utils import load_json


class TestTrainedModelArtifacts:
    """Tests for model training outputs (run after training)."""

    def test_best_model_exists(self):
        model_path = MODELS_DIR / "best_model.joblib"
        assert model_path.exists(), "Best model artifact not found"

    def test_preprocessor_exists(self):
        prep_path = MODELS_DIR / "preprocessor.joblib"
        assert prep_path.exists(), "Preprocessor artifact not found"

    def test_baseline_metrics_exist(self):
        metrics_path = MODELS_DIR / "baseline_metrics.json"
        assert metrics_path.exists(), "Baseline metrics not found"

    def test_baseline_metrics_valid(self):
        metrics = load_json(MODELS_DIR / "baseline_metrics.json")
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_training_summary_exists(self):
        summary_path = MODELS_DIR / "training_summary.json"
        assert summary_path.exists()
        summary = load_json(summary_path)
        assert "best_model" in summary
        assert "all_models" in summary
        assert len(summary["all_models"]) == 4  # 4 algorithms
