"""
Alert management system for the Data Drift Detection System.

Generates, stores, and retrieves structured alert objects based on drift detection
and model performance degradation. Provides email simulation formatting and
atomic persistence to prevent JSON corruption under concurrent dashboard reads.
Follows PEP 8 and Global Naming Standardization.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.settings import ALERT_LEVELS, LOGS_DIR
from src.utils import get_readable_timestamp, json_default_serializer, setup_logger

logger = setup_logger(__name__)

ALERTS_FILE: Path = LOGS_DIR / "alerts.json"

# In-memory alert store for zero disk write dependence and serverless safety
_IN_MEMORY_ALERTS: List[Dict[str, Any]] = []


def _load_alerts() -> List[Dict[str, Any]]:
    """
    Load existing alerts from persistent storage or in-memory cache with error recovery.

    Returns:
        List of alert dictionaries.
    """
    if ALERTS_FILE.exists():
        try:
            with open(ALERTS_FILE, "r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)
                if isinstance(data, list):
                    return data
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Corrupted alerts file detected, resetting cache: %s", exc)
            return []
        except (OSError, PermissionError) as exc:
            logger.warning("Could not read alerts file (%s), falling back to in-memory: %s", ALERTS_FILE, exc)
    return list(_IN_MEMORY_ALERTS)


def _save_alerts(alerts: List[Dict[str, Any]]) -> None:
    """
    Persist alert list to in-memory store and atomically to file if writable.

    Args:
        alerts: List of alert dictionaries to serialize.
    """
    global _IN_MEMORY_ALERTS
    _IN_MEMORY_ALERTS = list(alerts)

    try:
        ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        temp_file = ALERTS_FILE.with_suffix(f"{ALERTS_FILE.suffix}.tmp")

        with open(temp_file, "w", encoding="utf-8") as file_handle:
            json.dump(alerts, file_handle, indent=2, default=json_default_serializer)

        temp_file.replace(ALERTS_FILE)
    except (OSError, PermissionError) as exc:
        logger.debug("Read-only filesystem: alerts retained in-memory (%s)", exc)


def generate_alerts(
    drift_report: Dict[str, Any],
    batch_name: str = "Unknown",
    performance_comparison: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate actionable alerts from drift analysis and model performance results.

    Args:
        drift_report: Full output dictionary from run_full_drift_analysis().
        batch_name: Identifier name of the evaluated production batch.
        performance_comparison: Output dictionary from compare_performance() (optional).

    Returns:
        List of generated alert records for the current batch.
    """
    new_alerts: List[Dict[str, Any]] = []
    timestamp = get_readable_timestamp()
    dataset_drift = drift_report.get("dataset_drift", {})
    risk_level = dataset_drift.get("risk_level", "LOW")

    # Dataset-level drift alert
    drifted_count = dataset_drift.get("drifted_count", 0)
    total_features = dataset_drift.get("total_features", 0)
    drifted_pct = dataset_drift.get("drifted_pct", 0.0)

    if risk_level == "HIGH":
        new_alerts.append({
            "timestamp": timestamp,
            "batch": batch_name,
            "severity": "CRITICAL",
            "icon": ALERT_LEVELS["CRITICAL"],
            "message": (
                f"SEVERE DRIFT DETECTED — {drifted_count}/{total_features} features "
                f"drifted ({drifted_pct}%)"
            ),
            "action": "Immediate model retraining recommended",
        })
    elif risk_level == "MEDIUM":
        new_alerts.append({
            "timestamp": timestamp,
            "batch": batch_name,
            "severity": "WARNING",
            "icon": ALERT_LEVELS["WARNING"],
            "message": (
                f"Moderate drift detected — {drifted_count}/{total_features} features "
                f"drifted ({drifted_pct}%)"
            ),
            "action": "Monitor closely; schedule retraining if trend continues",
        })
    else:
        new_alerts.append({
            "timestamp": timestamp,
            "batch": batch_name,
            "severity": "INFO",
            "icon": ALERT_LEVELS["INFO"],
            "message": (
                f"No significant drift — {drifted_count}/{total_features} features drifted"
            ),
            "action": "No action required",
        })

    # Feature-level alerts for individual severe drift
    for feat_info in dataset_drift.get("drifted_features", []):
        if feat_info.get("severity") == "SEVERE":
            feature_name = feat_info.get("feature", "unknown")
            feature_type = feat_info.get("type", "numerical")
            new_alerts.append({
                "timestamp": timestamp,
                "batch": batch_name,
                "severity": "WARNING",
                "icon": ALERT_LEVELS["WARNING"],
                "message": f"Severe drift in feature '{feature_name}' ({feature_type})",
                "action": f"Investigate data source for '{feature_name}'",
            })

    # Missing data drift alerts
    missing_drift_count = int(dataset_drift.get("missing_data_drift_count", 0))
    if missing_drift_count > 0:
        new_alerts.append({
            "timestamp": timestamp,
            "batch": batch_name,
            "severity": "WARNING",
            "icon": ALERT_LEVELS["WARNING"],
            "message": f"Missing data increase detected in {missing_drift_count} feature(s)",
            "action": "Check data pipeline integrity",
        })

    # Model performance degradation alerts
    if performance_comparison:
        health_status = performance_comparison.get("overall_health", "HEALTHY")
        degraded_count = performance_comparison.get("degraded_metrics_count", 0)

        if health_status == "CRITICAL":
            new_alerts.append({
                "timestamp": timestamp,
                "batch": batch_name,
                "severity": "CRITICAL",
                "icon": ALERT_LEVELS["RETRAIN"],
                "message": (
                    f"Model performance critically degraded ({degraded_count} metrics declined)"
                ),
                "action": "IMMEDIATE RETRAINING REQUIRED",
            })
        elif health_status == "WARNING":
            new_alerts.append({
                "timestamp": timestamp,
                "batch": batch_name,
                "severity": "WARNING",
                "icon": ALERT_LEVELS["WARNING"],
                "message": "Model performance slightly degraded",
                "action": "Schedule retraining in next cycle",
            })

    # Persist accumulated alerts
    existing_alerts = _load_alerts()
    existing_alerts.extend(new_alerts)
    _save_alerts(existing_alerts)

    logger.info("Generated %d alerts for batch '%s'", len(new_alerts), batch_name)
    return new_alerts


def get_all_alerts() -> List[Dict[str, Any]]:
    """Retrieve all historical stored alerts."""
    return _load_alerts()


def clear_alerts() -> None:
    """Clear all historical alerts from storage and in-memory cache."""
    global _IN_MEMORY_ALERTS
    _IN_MEMORY_ALERTS.clear()
    _save_alerts([])
    logger.info("All stored alerts cleared successfully.")


def simulate_email_alert(alert: Dict[str, Any]) -> str:
    """
    Format a simulated email alert notification.

    Args:
        alert: Alert dictionary.

    Returns:
        Formatted ASCII email template string.
    """
    email_text = (
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
    logger.info("Email alert simulated for: %s...", alert["message"][:50])
    return email_text
