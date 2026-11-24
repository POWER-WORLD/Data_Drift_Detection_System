"""
Pytest configuration — shared fixtures and path setup.
Eliminates the need for sys.path.insert in every test file.
"""

import sys
from pathlib import Path
import pytest

# Add project root to Python path once
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.config.settings as settings
import src.report as report_module
import src.alert as alert_module


@pytest.fixture(autouse=True)
def isolate_test_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure all test runs write exclusively to tmp_path, preventing repository pollution."""
    test_reports = tmp_path / "reports"
    test_reports.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(report_module, "REPORTS_DIR", test_reports)
    monkeypatch.setattr(settings, "REPORTS_DIR", test_reports)

    test_logs = tmp_path / "logs"
    test_logs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(alert_module, "ALERTS_FILE", test_logs / "alerts.json")
    monkeypatch.setattr(settings, "LOGS_DIR", test_logs)
