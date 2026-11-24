"""
Report generation module for the Data Drift Detection System.

Generates structured JSON and CSV reports summarizing statistical drift findings,
feature distributions, model performance, and operational alerts.
Follows PEP 8 and Global Naming Standardization.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config.settings import REPORTS_DIR
from src.utils import get_timestamp, json_default_serializer, setup_logger

logger = setup_logger(__name__)


def generate_drift_report(
    drift_report: Dict[str, Any],
    batch_name: str = "Unknown",
    performance: Optional[Dict[str, Any]] = None,
    alerts: Optional[List[Dict[str, Any]]] = None,
    save_to_disk: bool = True,
) -> Dict[str, Path]:
    """
    Generate downloadable JSON and CSV report artifacts.

    Args:
        drift_report: Complete drift analysis dictionary.
        batch_name: Name identifier for the production batch.
        performance: Model performance comparison dictionary (optional).
        alerts: List of generated alerts (optional).
        save_to_disk: If True, writes files to REPORTS_DIR (default True for testing/audit).

    Returns:
        Dictionary mapping report types ('json', 'csv', 'alerts_csv') to Path objects.
    """
    generated_files: Dict[str, Path] = {}
    if not save_to_disk:
        return generated_files

    timestamp = get_timestamp()
    report_directory = REPORTS_DIR / f"report_{timestamp}"
    try:
        report_directory.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as exc:
        logger.warning("Read-only filesystem: skipping report file generation: %s", exc)
        return generated_files

    try:
        # ── JSON Comprehensive Report ─────────────────────────────────────────
        full_report_payload = {
            "report_date": datetime.now().isoformat(),
            "batch_name": batch_name,
            "dataset_drift": drift_report.get("dataset_drift", {}),
            "feature_drift_summary": _summarize_features(drift_report),
            "performance": performance,
            "alerts": alerts,
            "recommended_action": _get_recommendation(drift_report),
        }

        json_path = report_directory / "drift_report.json"
        with open(json_path, "w", encoding="utf-8") as file_handle:
            json.dump(
                full_report_payload,
                file_handle,
                indent=2,
                default=json_default_serializer,
            )
        generated_files["json"] = json_path

        # ── CSV Feature Drift Summary ─────────────────────────────────────────
        summary_rows: List[Dict[str, Any]] = []

        # Numerical feature drift metrics
        numerical_drift = drift_report.get("feature_drift", {}).get("numerical", {})
        for col, tests in numerical_drift.items():
            ks_data = tests.get("ks_test", {})
            psi_data = tests.get("psi", {})
            js_data = tests.get("jensen_shannon", {})
            ms_data = tests.get("mean_shift", {})

            summary_rows.append({
                "feature": col,
                "type": "numerical",
                "ks_statistic": ks_data.get("statistic"),
                "ks_pvalue": ks_data.get("p_value"),
                "ks_drift": ks_data.get("drift_detected"),
                "psi_value": psi_data.get("value"),
                "psi_severity": psi_data.get("severity"),
                "js_distance": js_data.get("value"),
                "js_severity": js_data.get("severity"),
                "mean_shift_pct": ms_data.get("pct_change"),
            })

        # Categorical feature drift metrics
        categorical_drift = drift_report.get("feature_drift", {}).get("categorical", {})
        for col, tests in categorical_drift.items():
            chi2_data = tests.get("chi_square", {})
            nc_data = tests.get("new_categories", {})

            summary_rows.append({
                "feature": col,
                "type": "categorical",
                "chi2_statistic": chi2_data.get("statistic"),
                "chi2_pvalue": chi2_data.get("p_value"),
                "chi2_drift": chi2_data.get("drift_detected"),
                "new_categories_count": nc_data.get("new_count", 0),
                "new_categories": str(nc_data.get("new_categories", [])),
            })

        csv_path = report_directory / "feature_drift_summary.csv"
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
        generated_files["csv"] = csv_path

        # ── Alerts CSV (if alerts exist) ──────────────────────────────────────
        if alerts:
            alerts_path = report_directory / "alerts.csv"
            pd.DataFrame(alerts).to_csv(alerts_path, index=False)
            generated_files["alerts_csv"] = alerts_path

        logger.info("Successfully generated drift report artifacts in %s", report_directory)
    except (OSError, PermissionError) as exc:
        logger.warning("Read-only filesystem: could not write reports: %s", exc)

    return generated_files


def _summarize_features(drift_report: Dict[str, Any]) -> Dict[str, Any]:
    """Extract an executive overview summary of drifted features."""
    dataset_summary = drift_report.get("dataset_drift", {})
    drifted_list = [
        item.get("feature")
        for item in dataset_summary.get("drifted_features", [])
        if isinstance(item, dict) and "feature" in item
    ]
    return {
        "total_features": dataset_summary.get("total_features", 0),
        "drifted_count": dataset_summary.get("drifted_count", 0),
        "drifted_pct": dataset_summary.get("drifted_pct", 0.0),
        "risk_level": dataset_summary.get("risk_level", "UNKNOWN"),
        "drifted_features": drifted_list,
    }


def _get_recommendation(drift_report: Dict[str, Any]) -> str:
    """Generate action recommendation based on drift severity."""
    risk = drift_report.get("dataset_drift", {}).get("risk_level", "LOW")
    if risk == "HIGH":
        return "IMMEDIATE ACTION: Retrain the model with recent data. Investigate root causes of drift in affected features."
    elif risk == "MEDIUM":
        return "SCHEDULE RETRAINING: Monitor the situation closely. If drift persists in next batch, retrain the model."
    else:
        return "NO ACTION NEEDED: Data distributions are stable. Continue routine monitoring."
