"""
Unit tests for report generation module (src/report.py).

Tests generation of JSON, CSV feature drift summaries, alert summaries,
recommendation logic, and metadata structure.
"""

from pathlib import Path
from typing import Any, Dict, List
import json
import pytest
import pandas as pd

import src.report as report_module
from src.report import (
    generate_drift_report,
    _summarize_features,
    _get_recommendation,
)


@pytest.fixture(autouse=True)
def isolate_reports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect REPORTS_DIR to a temporary directory for each test run."""
    temp_reports_dir = tmp_path / "reports"
    monkeypatch.setattr(report_module, "REPORTS_DIR", temp_reports_dir)
    return temp_reports_dir


def _create_sample_drift_report() -> Dict[str, Any]:
    """Helper to create a standard structured drift report dictionary."""
    return {
        "dataset_drift": {
            "total_features": 3,
            "drifted_count": 1,
            "drifted_pct": 33.3,
            "risk_level": "MEDIUM",
            "drifted_features": [
                {"feature": "MonthlyCharges", "type": "numerical", "severity": "SEVERE"}
            ],
        },
        "feature_drift": {
            "numerical": {
                "MonthlyCharges": {
                    "ks_test": {"statistic": 0.35, "p_value": 0.001, "drift_detected": True},
                    "psi": {"value": 0.28, "severity": "SEVERE"},
                    "jensen_shannon": {"value": 0.22, "severity": "MODERATE"},
                    "mean_shift": {"pct_change": 18.5},
                },
                "tenure": {
                    "ks_test": {"statistic": 0.05, "p_value": 0.85, "drift_detected": False},
                    "psi": {"value": 0.02, "severity": "LOW"},
                    "jensen_shannon": {"value": 0.01, "severity": "LOW"},
                    "mean_shift": {"pct_change": 1.2},
                },
            },
            "categorical": {
                "Contract": {
                    "chi_square": {"statistic": 4.5, "p_value": 0.12, "drift_detected": False},
                    "new_categories": {"new_count": 0, "new_categories": []},
                }
            },
        },
    }


class TestGenerateDriftReport:
    """Tests for generate_drift_report function."""

    def test_creates_json_and_csv_reports(self) -> None:
        """Verify that JSON and CSV report files are generated and contain valid data."""
        drift_data = _create_sample_drift_report()
        perf_data = {"overall_health": "HEALTHY", "degraded_metrics_count": 0}
        alerts_data: List[Dict[str, Any]] = [
            {"severity": "WARNING", "message": "Moderate drift detected", "action": "Monitor"}
        ]

        result_paths = generate_drift_report(
            drift_report=drift_data,
            batch_name="batch_test_01",
            performance=perf_data,
            alerts=alerts_data,
        )

        assert "json" in result_paths
        assert "csv" in result_paths
        assert "alerts_csv" in result_paths

        assert result_paths["json"].exists()
        assert result_paths["csv"].exists()
        assert result_paths["alerts_csv"].exists()

        # Validate JSON content
        with open(result_paths["json"], "r", encoding="utf-8") as f:
            saved_json = json.load(f)
        assert saved_json["batch_name"] == "batch_test_01"
        assert saved_json["dataset_drift"]["risk_level"] == "MEDIUM"
        assert saved_json["performance"]["overall_health"] == "HEALTHY"
        assert len(saved_json["alerts"]) == 1
        assert "SCHEDULE RETRAINING" in saved_json["recommended_action"]

        # Validate CSV content
        df_csv = pd.read_csv(result_paths["csv"])
        assert len(df_csv) == 3  # 2 numerical + 1 categorical
        assert set(df_csv["feature"]) == {"MonthlyCharges", "tenure", "Contract"}
        assert "ks_statistic" in df_csv.columns
        assert "chi2_statistic" in df_csv.columns

    def test_report_generation_without_optional_fields(self) -> None:
        """Verify report generation succeeds without performance or alert inputs."""
        drift_data = _create_sample_drift_report()

        result_paths = generate_drift_report(
            drift_report=drift_data,
            batch_name="minimal_batch",
        )

        assert "json" in result_paths
        assert "csv" in result_paths
        assert "alerts_csv" not in result_paths
        assert result_paths["json"].exists()
        assert result_paths["csv"].exists()


class TestReportHelpers:
    """Tests for private helper functions in report module."""

    def test_summarize_features(self) -> None:
        """Verify _summarize_features extracts correct summary counts and lists."""
        drift_data = _create_sample_drift_report()
        summary = _summarize_features(drift_data)

        assert summary["total_features"] == 3
        assert summary["drifted_count"] == 1
        assert summary["drifted_pct"] == 33.3
        assert summary["risk_level"] == "MEDIUM"
        assert summary["drifted_features"] == ["MonthlyCharges"]

    def test_summarize_features_empty_dict(self) -> None:
        """Verify _summarize_features handles empty dictionary input gracefully."""
        summary = _summarize_features({})
        assert summary["total_features"] == 0
        assert summary["drifted_count"] == 0
        assert summary["risk_level"] == "UNKNOWN"

    @pytest.mark.parametrize(
        "risk_level, expected_keyword",
        [
            ("HIGH", "IMMEDIATE ACTION"),
            ("MEDIUM", "SCHEDULE RETRAINING"),
            ("LOW", "NO ACTION NEEDED"),
            ("UNKNOWN", "NO ACTION NEEDED"),
        ],
    )
    def test_get_recommendation(self, risk_level: str, expected_keyword: str) -> None:
        """Verify _get_recommendation produces appropriate guidance per risk level."""
        report = {"dataset_drift": {"risk_level": risk_level}}
        rec = _get_recommendation(report)
        assert expected_keyword in rec
