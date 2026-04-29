"""
Integration tests for the Data Drift Detection System.
Tests the full pipeline: load → preprocess → predict → drift → alert → report.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import load_baseline_data, load_production_batch, list_production_batches
from src.preprocessing import DataPreprocessor
from src.drift_detection import run_full_drift_analysis
from src.predict import predict_batch
from src.alert import generate_alerts, get_all_alerts, clear_alerts
from src.report import generate_drift_report
from src.config.settings import PRODUCTION_DIR, MODELS_DIR, REPORTS_DIR


class TestEndToEndPipeline:
    """Full pipeline integration tests."""

    def test_baseline_load_and_preprocess(self):
        """Baseline data can be loaded and preprocessed without errors."""
        df = load_baseline_data()
        assert len(df) > 0
        assert "Churn" in df.columns

        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(df)
        assert X.shape[0] == len(df)
        assert not X.isna().any().any(), "Preprocessed data should have no NaN"
        assert preprocessor.is_fitted

    def test_full_drift_pipeline_no_drift(self):
        """Full drift pipeline on batch with no drift."""
        baseline = load_baseline_data()
        prod_path = PRODUCTION_DIR / "batch_01_no_drift.csv"
        if not prod_path.exists():
            pytest.skip("Production batch not generated")

        production = load_production_batch(prod_path)
        report = run_full_drift_analysis(baseline, production)

        assert "feature_drift" in report
        assert "dataset_drift" in report
        risk = report["dataset_drift"]["risk_level"]
        assert risk in ("LOW", "MEDIUM", "HIGH")

    def test_full_drift_pipeline_severe_drift(self):
        """Severe drift batch should flag higher risk."""
        baseline = load_baseline_data()
        prod_path = PRODUCTION_DIR / "batch_03_severe_drift.csv"
        if not prod_path.exists():
            pytest.skip("Production batch not generated")

        production = load_production_batch(prod_path)
        report = run_full_drift_analysis(baseline, production)

        # Severe batch should detect at least some drift
        drifted = report["dataset_drift"]["drifted_count"]
        assert drifted > 0, "Severe drift batch should have drifted features"

    def test_predict_batch_with_ground_truth(self):
        """Predictions on a batch with Churn labels should return comparison."""
        prod_path = PRODUCTION_DIR / "batch_01_no_drift.csv"
        if not prod_path.exists():
            pytest.skip("Production batch not generated")
        if not (MODELS_DIR / "best_model.joblib").exists():
            pytest.skip("Model not trained")

        production = load_production_batch(prod_path)
        result = predict_batch(production)

        assert "predictions" in result
        assert "comparison" in result
        assert result["comparison"]["overall_health"] in ("HEALTHY", "WARNING", "CRITICAL")

    def test_alert_generation_and_retrieval(self):
        """Alerts can be generated, stored, and retrieved."""
        baseline = load_baseline_data()
        prod_path = PRODUCTION_DIR / "batch_03_severe_drift.csv"
        if not prod_path.exists():
            pytest.skip("Production batch not generated")

        production = load_production_batch(prod_path)
        report = run_full_drift_analysis(baseline, production)

        # Clear existing alerts
        clear_alerts()
        assert get_all_alerts() == []

        # Generate new alerts
        alerts = generate_alerts(report, batch_name="test_batch")
        assert len(alerts) > 0
        assert all("severity" in a for a in alerts)

        # Verify persistence
        stored = get_all_alerts()
        assert len(stored) == len(alerts)

        # Cleanup
        clear_alerts()

    def test_report_generation(self):
        """Reports can be generated in JSON and CSV formats."""
        baseline = load_baseline_data()
        prod_path = PRODUCTION_DIR / "batch_01_no_drift.csv"
        if not prod_path.exists():
            pytest.skip("Production batch not generated")

        production = load_production_batch(prod_path)
        report = run_full_drift_analysis(baseline, production)

        paths = generate_drift_report(report, batch_name="test_integration")
        assert "json" in paths
        assert "csv" in paths
        assert paths["json"].exists()
        assert paths["csv"].exists()


class TestEdgeCasePipeline:
    """Edge case tests for the pipeline."""

    def test_category_drift_batch(self):
        """Category drift batch should detect new categories."""
        baseline = load_baseline_data()
        prod_path = PRODUCTION_DIR / "batch_05_category_drift.csv"
        if not prod_path.exists():
            pytest.skip("Production batch not generated")

        production = load_production_batch(prod_path)
        report = run_full_drift_analysis(baseline, production)

        # Check for new categories in at least one feature
        cat_results = report["feature_drift"]["categorical"]
        has_new = any(
            tests.get("new_categories", {}).get("new_count", 0) > 0
            for tests in cat_results.values()
        )
        assert has_new, "Category drift batch should detect new categories"

    def test_missing_data_batch(self):
        """Missing data batch should flag missing data drift."""
        baseline = load_baseline_data()
        prod_path = PRODUCTION_DIR / "batch_04_missing_data.csv"
        if not prod_path.exists():
            pytest.skip("Production batch not generated")

        production = load_production_batch(prod_path)
        report = run_full_drift_analysis(baseline, production)

        missing_drift = report["dataset_drift"]["missing_data_drift_count"]
        assert missing_drift > 0, "Missing data batch should detect missing data drift"
