"""
End-to-End lifecycle verification tests for all production batches.

Tests each production batch through the complete 7-stage ML pipeline:
1. File Validation (validate_file)
2. Batch Ingestion (load_production_batch)
3. Schema & Data Quality Auditing (validate_schema, validate_data_quality)
4. Data Sanitization (sanitize_dataframe)
5. Statistical Drift Detection (run_full_drift_analysis across 8 statistical tests)
6. Model Inference & Health Comparison (predict_batch)
7. Alert Generation & Report Export (generate_alerts, generate_drift_report)
"""

from pathlib import Path
from typing import Any, Dict
import json
import pytest
import pandas as pd

from src.data_loader import load_baseline_data, load_production_batch
from src.data_validator import (
    sanitize_dataframe,
    validate_data_quality,
    validate_file,
    validate_schema,
)
from src.drift_detection import run_full_drift_analysis
from src.predict import predict_batch
from src.alert import generate_alerts, clear_alerts
from src.report import generate_drift_report
from src.config.settings import PRODUCTION_DIR, MODELS_DIR


@pytest.fixture(scope="module")
def baseline_data() -> pd.DataFrame:
    """Fixture providing loaded baseline dataset."""
    return load_baseline_data()


class TestAllProductionBatchesLifecycle:
    """Parametrized end-to-end pipeline verification across all 5 production batches."""

    BATCH_FILES = [
        "batch_01_no_drift.csv",
        "batch_02_mild_drift.csv",
        "batch_03_severe_drift.csv",
        "batch_04_missing_data.csv",
        "batch_05_category_drift.csv",
    ]

    @pytest.mark.parametrize("batch_filename", BATCH_FILES)
    def test_complete_pipeline_execution(
        self,
        batch_filename: str,
        baseline_data: pd.DataFrame,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Execute full lifecycle for each production batch and verify zero crashes,
        accurate schema compliance, valid predictions, alerts, and exportable reports.
        """
        batch_path = PRODUCTION_DIR / batch_filename
        assert batch_path.exists(), f"Production batch {batch_filename} does not exist"

        # 1. File validation
        validate_file(batch_path)

        # 2. Batch Ingestion
        prod_df = load_production_batch(batch_path)
        assert len(prod_df) > 0
        assert isinstance(prod_df, pd.DataFrame)

        # 3. Schema & Quality Audit
        schema_warnings = validate_schema(prod_df, require_target=False)
        assert isinstance(schema_warnings, list)

        quality_report = validate_data_quality(prod_df)
        assert quality_report["stats"]["rows"] == len(prod_df)
        assert quality_report["stats"]["columns"] == len(prod_df.columns)

        # 4. Data Sanitization
        clean_df = sanitize_dataframe(prod_df)
        assert len(clean_df) > 0

        # 5. Statistical Drift Detection
        drift_report = run_full_drift_analysis(baseline_data, clean_df)
        assert "dataset_drift" in drift_report
        assert "feature_drift" in drift_report
        dataset_drift = drift_report["dataset_drift"]
        assert dataset_drift["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert dataset_drift["total_features"] == 19
        assert 0 <= dataset_drift["drifted_count"] <= 19
        assert 0.0 <= dataset_drift["drifted_pct"] <= 100.0

        # 6. ML Inference & Performance Comparison
        if (MODELS_DIR / "best_model.joblib").exists():
            inference_result = predict_batch(clean_df)
            assert "predictions" in inference_result
            assert len(inference_result["predictions"]) == len(clean_df)
            if inference_result.get("probabilities") is not None:
                assert len(inference_result["probabilities"]) == len(clean_df)
            perf_comparison = inference_result.get("comparison")
        else:
            perf_comparison = None

        # 7. Alert Management
        temp_alerts_file = tmp_path / "alerts.json"
        monkeypatch.setattr("src.alert.ALERTS_FILE", temp_alerts_file)
        clear_alerts()

        alerts = generate_alerts(
            drift_report=drift_report,
            batch_name=batch_filename,
            performance_comparison=perf_comparison,
        )
        assert len(alerts) >= 1
        assert all("severity" in a and "message" in a for a in alerts)

        # 8. Report Export
        temp_reports_dir = tmp_path / "reports"
        monkeypatch.setattr("src.report.REPORTS_DIR", temp_reports_dir)

        report_paths = generate_drift_report(
            drift_report=drift_report,
            batch_name=batch_filename,
            performance=perf_comparison,
            alerts=alerts,
        )
        assert "json" in report_paths and report_paths["json"].exists()
        assert "csv" in report_paths and report_paths["csv"].exists()

        # Validate serialized JSON integrity
        with open(report_paths["json"], "r", encoding="utf-8") as f:
            saved_report = json.load(f)
        assert saved_report["batch_name"] == batch_filename
        assert "dataset_drift" in saved_report

    def test_batch_01_no_drift_characteristics(self, baseline_data: pd.DataFrame) -> None:
        """Verify batch 01 exhibits minimal or low risk drift."""
        prod_df = load_production_batch(PRODUCTION_DIR / "batch_01_no_drift.csv")
        report = run_full_drift_analysis(baseline_data, prod_df)
        assert report["dataset_drift"]["risk_level"] in ("LOW", "MEDIUM")
        assert report["dataset_drift"]["drifted_count"] <= 5

    def test_batch_02_mild_drift_characteristics(self, baseline_data: pd.DataFrame) -> None:
        """Verify batch 02 exhibits mild drift behavior."""
        prod_df = load_production_batch(PRODUCTION_DIR / "batch_02_mild_drift.csv")
        report = run_full_drift_analysis(baseline_data, prod_df)
        assert report["dataset_drift"]["drifted_count"] >= 1

    def test_batch_03_severe_drift_characteristics(self, baseline_data: pd.DataFrame) -> None:
        """Verify batch 03 exhibits significant feature distribution shifts."""
        prod_df = load_production_batch(PRODUCTION_DIR / "batch_03_severe_drift.csv")
        report = run_full_drift_analysis(baseline_data, prod_df)
        assert report["dataset_drift"]["drifted_count"] >= 3
        # Should flag high or medium risk with drifted features
        assert len(report["dataset_drift"]["drifted_features"]) >= 3

    def test_batch_04_missing_data_characteristics(self, baseline_data: pd.DataFrame) -> None:
        """Verify batch 04 exhibits increased missing data and preprocessor imputes successfully."""
        prod_df = load_production_batch(PRODUCTION_DIR / "batch_04_missing_data.csv")
        quality = validate_data_quality(prod_df)
        assert quality["stats"]["rows"] == len(prod_df)

        report = run_full_drift_analysis(baseline_data, prod_df)
        assert report["dataset_drift"]["missing_data_drift_count"] > 0

        # Verify predictions succeed despite missing data
        inference = predict_batch(prod_df)
        assert len(inference["predictions"]) == len(prod_df)

    def test_batch_05_category_drift_characteristics(self, baseline_data: pd.DataFrame) -> None:
        """Verify batch 05 exhibits new/shifted categories and handles inference without error."""
        prod_df = load_production_batch(PRODUCTION_DIR / "batch_05_category_drift.csv")
        report = run_full_drift_analysis(baseline_data, prod_df)

        cat_drift = report["feature_drift"]["categorical"]
        has_new_cats = any(
            tests.get("new_categories", {}).get("new_count", 0) > 0
            for tests in cat_drift.values()
        )
        assert has_new_cats, "Batch 05 should exhibit new categories"

        # Verify inference handles unseen categories cleanly via preprocessor fallback
        inference = predict_batch(prod_df)
        assert len(inference["predictions"]) == len(prod_df)
