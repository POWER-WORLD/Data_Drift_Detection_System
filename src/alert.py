"""
Alert management system for the Data Drift Detection System.
Generates, stores, and retrieves alerts based on drift analysis results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from src.config.settings import ALERT_LEVELS, LOGS_DIR, PERFORMANCE_THRESHOLDS
from src.utils import setup_logger, get_readable_timestamp

logger = setup_logger(__name__)

ALERTS_FILE = LOGS_DIR / "alerts.json"


def _load_alerts() -> List[Dict[str, Any]]:
    """Load existing alerts from file. Returns empty list on corrupt file."""
    if ALERTS_FILE.exists():
        try:
            with open(ALERTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Corrupt alerts file, resetting: {e}")
            return []
    return []


def _save_alerts(alerts: List[Dict[str, Any]]) -> None:
    """Save alerts to file."""
    ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_FILE, "w") as f:
        json.dump(alerts, f, indent=2, default=str)


def generate_alerts(drift_report: Dict[str, Any], batch_name: str = "Unknown",
                    performance_comparison: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Generate alerts based on drift analysis and performance results.

    Args:
        drift_report: Output from run_full_drift_analysis().
        batch_name: Name of the production batch.
        performance_comparison: Output from compare_performance() (optional).

    Returns:
        List of generated alert dictionaries.
    """
    new_alerts = []
    timestamp = get_readable_timestamp()
    dataset_drift = drift_report.get("dataset_drift", {})
    risk = dataset_drift.get("risk_level", "LOW")

    # Dataset-level drift alert
    if risk == "HIGH":
        new_alerts.append({
            "timestamp": timestamp, "batch": batch_name, "severity": "CRITICAL",
            "icon": ALERT_LEVELS["CRITICAL"],
            "message": f"SEVERE DRIFT DETECTED — {dataset_drift['drifted_count']}/{dataset_drift['total_features']} features drifted ({dataset_drift['drifted_pct']}%)",
            "action": "Immediate model retraining recommended",
        })
    elif risk == "MEDIUM":
        new_alerts.append({
            "timestamp": timestamp, "batch": batch_name, "severity": "WARNING",
            "icon": ALERT_LEVELS["WARNING"],
            "message": f"Moderate drift detected — {dataset_drift['drifted_count']}/{dataset_drift['total_features']} features drifted ({dataset_drift['drifted_pct']}%)",
            "action": "Monitor closely; schedule retraining if trend continues",
        })
    else:
        new_alerts.append({
            "timestamp": timestamp, "batch": batch_name, "severity": "INFO",
            "icon": ALERT_LEVELS["INFO"],
            "message": f"No significant drift — {dataset_drift['drifted_count']}/{dataset_drift['total_features']} features drifted",
            "action": "No action required",
        })

    # Feature-level alerts for severe drift
    for feat_info in dataset_drift.get("drifted_features", []):
        if feat_info.get("severity") == "SEVERE":
            new_alerts.append({
                "timestamp": timestamp, "batch": batch_name, "severity": "WARNING",
                "icon": ALERT_LEVELS["WARNING"],
                "message": f"Severe drift in feature '{feat_info['feature']}' ({feat_info['type']})",
                "action": f"Investigate data source for '{feat_info['feature']}'",
            })

    # Missing data alerts
    missing_count = dataset_drift.get("missing_data_drift_count", 0)
    if missing_count > 0:
        new_alerts.append({
            "timestamp": timestamp, "batch": batch_name, "severity": "WARNING",
            "icon": ALERT_LEVELS["WARNING"],
            "message": f"Missing data increase detected in {missing_count} feature(s)",
            "action": "Check data pipeline integrity",
        })

    # Performance degradation alerts
    if performance_comparison:
        health = performance_comparison.get("overall_health", "HEALTHY")
        if health == "CRITICAL":
            new_alerts.append({
                "timestamp": timestamp, "batch": batch_name, "severity": "CRITICAL",
                "icon": ALERT_LEVELS["RETRAIN"],
                "message": f"Model performance critically degraded ({performance_comparison.get('degraded_metrics_count', 0)} metrics declined)",
                "action": "IMMEDIATE RETRAINING REQUIRED",
            })
        elif health == "WARNING":
            new_alerts.append({
                "timestamp": timestamp, "batch": batch_name, "severity": "WARNING",
                "icon": ALERT_LEVELS["WARNING"],
                "message": "Model performance slightly degraded",
                "action": "Schedule retraining in next cycle",
            })

    # Persist alerts
    existing = _load_alerts()
    existing.extend(new_alerts)
    _save_alerts(existing)

    logger.info(f"Generated {len(new_alerts)} alerts for batch '{batch_name}'")
    return new_alerts


def get_all_alerts() -> List[Dict[str, Any]]:
    """Retrieve all stored alerts."""
    return _load_alerts()


def clear_alerts() -> None:
    """Clear all stored alerts."""
    _save_alerts([])
    logger.info("All alerts cleared")


def simulate_email_alert(alert: Dict[str, Any]) -> str:
    """Simulate sending an email alert (returns formatted string)."""
    email = (
        f"{'='*60}\n"
        f"📧 DRIFT ALERT EMAIL SIMULATION\n"
        f"{'='*60}\n"
        f"To: ml-ops-team@company.com\n"
        f"Subject: [{alert['severity']}] Data Drift Alert - {alert['batch']}\n"
        f"Time: {alert['timestamp']}\n"
        f"{'─'*60}\n"
        f"{alert['icon']} {alert['message']}\n"
        f"Recommended Action: {alert['action']}\n"
        f"{'='*60}\n"
    )
    logger.info(f"Email alert simulated for: {alert['message'][:50]}...")
    return email
