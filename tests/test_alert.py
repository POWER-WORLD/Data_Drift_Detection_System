"""
Unit tests for the alert management system (src/alert.py).

Tests alert generation rules for dataset-level drift, feature-level severe drift,
missing data drift, model performance degradation, alert persistence, cleanup,
corrupt file handling, and email alert simulation.
"""

from pathlib import Path
from typing import Any, Dict, List
import json
import pytest

import src.alert as alert_module
from src.alert import (
    clear_alerts,
    generate_alerts,
    get_all_alerts,
    simulate_email_alert,
)


@pytest.fixture(autouse=True)
def isolate_alerts_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ALERTS_FILE to a temporary directory for each test run."""
    temp_alerts_file = tmp_path / "test_alerts.json"
    monkeypatch.setattr(alert_module, "ALERTS_FILE", temp_alerts_file)
    return temp_alerts_file


class TestGenerateAlerts:
    """Tests for the generate_alerts function under various drift and performance scenarios."""

    def test_high_risk_dataset_drift_alert(self) -> None:
        """Verify HIGH risk dataset drift generates a CRITICAL alert."""
        drift_report: Dict[str, Any] = {
            "dataset_drift": {
                "risk_level": "HIGH",
                "drifted_count": 8,
                "total_features": 10,
                "drifted_pct": 80.0,
                "drifted_features": [],
            }
        }

        alerts = generate_alerts(drift_report, batch_name="batch_high_risk")

        assert len(alerts) >= 1
        critical_alerts = [a for a in alerts if a["severity"] == "CRITICAL"]
        assert len(critical_alerts) == 1
        assert "SEVERE DRIFT DETECTED" in critical_alerts[0]["message"]
        assert "Immediate model retraining recommended" == critical_alerts[0]["action"]

    def test_medium_risk_dataset_drift_alert(self) -> None:
        """Verify MEDIUM risk dataset drift generates a WARNING alert."""
        drift_report: Dict[str, Any] = {
            "dataset_drift": {
                "risk_level": "MEDIUM",
                "drifted_count": 3,
                "total_features": 10,
                "drifted_pct": 30.0,
                "drifted_features": [],
            }
        }

        alerts = generate_alerts(drift_report, batch_name="batch_medium_risk")

        assert len(alerts) >= 1
        warning_alerts = [a for a in alerts if a["severity"] == "WARNING"]
        assert len(warning_alerts) == 1
        assert "Moderate drift detected" in warning_alerts[0]["message"]

    def test_low_risk_dataset_drift_alert(self) -> None:
        """Verify LOW risk dataset drift generates an INFO alert."""
        drift_report: Dict[str, Any] = {
            "dataset_drift": {
                "risk_level": "LOW",
                "drifted_count": 0,
                "total_features": 10,
                "drifted_pct": 0.0,
                "drifted_features": [],
            }
        }

        alerts = generate_alerts(drift_report, batch_name="batch_low_risk")

        assert len(alerts) >= 1
        info_alerts = [a for a in alerts if a["severity"] == "INFO"]
        assert len(info_alerts) == 1
        assert "No significant drift" in info_alerts[0]["message"]

    def test_feature_level_severe_drift_alerts(self) -> None:
        """Verify feature-level severe drift alerts are properly triggered."""
        drift_report: Dict[str, Any] = {
            "dataset_drift": {
                "risk_level": "LOW",
                "drifted_count": 1,
                "total_features": 10,
                "drifted_pct": 10.0,
                "drifted_features": [
                    {"feature": "MonthlyCharges", "type": "numerical", "severity": "SEVERE"},
                    {"feature": "Contract", "type": "categorical", "severity": "MODERATE"},
                ],
            }
        }

        alerts = generate_alerts(drift_report, batch_name="batch_feat_drift")

        severe_alerts = [a for a in alerts if "MonthlyCharges" in a["message"]]
        assert len(severe_alerts) == 1
        assert severe_alerts[0]["severity"] == "WARNING"
        assert "Investigate data source for 'MonthlyCharges'" == severe_alerts[0]["action"]

    def test_missing_data_drift_alert(self) -> None:
        """Verify missing data increase triggers a WARNING alert."""
        drift_report: Dict[str, Any] = {
            "dataset_drift": {
                "risk_level": "LOW",
                "drifted_count": 0,
                "total_features": 10,
                "drifted_pct": 0.0,
                "drifted_features": [],
                "missing_data_drift_count": 2,
            }
        }

        alerts = generate_alerts(drift_report, batch_name="batch_missing")

        missing_alerts = [a for a in alerts if "Missing data increase" in a["message"]]
        assert len(missing_alerts) == 1
        assert "Check data pipeline integrity" == missing_alerts[0]["action"]

    def test_performance_degradation_alerts(self) -> None:
        """Verify performance comparison degradation generates appropriate alerts."""
        drift_report: Dict[str, Any] = {
            "dataset_drift": {"risk_level": "LOW", "drifted_count": 0, "total_features": 5, "drifted_pct": 0.0}
        }
        critical_perf: Dict[str, Any] = {
            "overall_health": "CRITICAL",
            "degraded_metrics_count": 4,
        }

        alerts = generate_alerts(
            drift_report=drift_report,
            batch_name="batch_perf_critical",
            performance_comparison=critical_perf,
        )

        perf_alerts = [a for a in alerts if "Model performance critically degraded" in a["message"]]
        assert len(perf_alerts) == 1
        assert perf_alerts[0]["severity"] == "CRITICAL"
        assert perf_alerts[0]["action"] == "IMMEDIATE RETRAINING REQUIRED"

        warning_perf: Dict[str, Any] = {
            "overall_health": "WARNING",
            "degraded_metrics_count": 1,
        }
        warning_alerts = generate_alerts(
            drift_report=drift_report,
            batch_name="batch_perf_warning",
            performance_comparison=warning_perf,
        )
        warn_perf_alerts = [a for a in warning_alerts if "Model performance slightly degraded" in a["message"]]
        assert len(warn_perf_alerts) == 1


class TestAlertPersistence:
    """Tests for alert storage, retrieval, and clearing."""

    def test_alert_persistence_and_clearing(self) -> None:
        """Verify alerts persist across calls and can be cleared."""
        drift_report: Dict[str, Any] = {
            "dataset_drift": {"risk_level": "LOW", "drifted_count": 0, "total_features": 2, "drifted_pct": 0.0}
        }

        clear_alerts()
        assert get_all_alerts() == []

        generate_alerts(drift_report, batch_name="batch_a")
        assert len(get_all_alerts()) == 1

        generate_alerts(drift_report, batch_name="batch_b")
        assert len(get_all_alerts()) == 2

        clear_alerts()
        assert get_all_alerts() == []

    def test_corrupt_alerts_file_recovery(self, isolate_alerts_file: Path) -> None:
        """Verify corrupt JSON file in alert storage is handled gracefully and reset."""
        isolate_alerts_file.parent.mkdir(parents=True, exist_ok=True)
        isolate_alerts_file.write_text("{corrupt: json}", encoding="utf-8")

        recovered = get_all_alerts()
        assert recovered == []

    def test_non_list_alerts_file_recovery(self, isolate_alerts_file: Path) -> None:
        """Verify non-list JSON root in alert storage is handled gracefully."""
        isolate_alerts_file.parent.mkdir(parents=True, exist_ok=True)
        isolate_alerts_file.write_text(json.dumps({"error": "not a list"}), encoding="utf-8")

        recovered = get_all_alerts()
        assert recovered == []


class TestSimulateEmailAlert:
    """Tests for the simulate_email_alert function."""

    def test_simulate_email_content_and_headers(self) -> None:
        """Verify simulated email contains necessary headers, severity, and action."""
        alert: Dict[str, Any] = {
            "timestamp": "2026-11-24 12:00:00",
            "batch": "batch_production_01",
            "severity": "CRITICAL",
            "icon": "🔴",
            "message": "Critical drift detected in feature distributions",
            "action": "Immediate retraining recommended",
        }

        email_text = simulate_email_alert(alert)

        assert "To: ml-ops-team@company.com" in email_text
        assert "[CRITICAL] Data Drift Alert - batch_production_01" in email_text
        assert "Critical drift detected" in email_text
        assert "Immediate retraining recommended" in email_text
