"""
Report generation module for the Data Drift Detection System.
Generates CSV and JSON drift reports.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from src.config.settings import REPORTS_DIR
from src.utils import setup_logger, get_timestamp

logger = setup_logger(__name__)


def generate_drift_report(drift_report: Dict[str, Any], batch_name: str = "Unknown",
                          performance: Dict[str, Any] = None,
                          alerts: list = None) -> Dict[str, Path]:
    """
    Generate downloadable drift reports in CSV and JSON formats.

    Args:
        drift_report: Full drift analysis output.
        batch_name: Name of the production batch.
        performance: Performance comparison results (optional).
        alerts: List of generated alerts (optional).

    Returns:
        Dictionary with paths to generated report files.
    """
    ts = get_timestamp()
    report_dir = REPORTS_DIR / f"report_{ts}"
    report_dir.mkdir(parents=True, exist_ok=True)
    generated = {}

    # ── JSON Full Report ──────────────────────────────────────────────────
    full_report = {
        "report_date": datetime.now().isoformat(),
        "batch_name": batch_name,
        "dataset_drift": drift_report.get("dataset_drift", {}),
        "feature_drift_summary": _summarize_features(drift_report),
        "performance": performance,
        "alerts": alerts,
        "recommended_action": _get_recommendation(drift_report),
    }
    json_path = report_dir / "drift_report.json"
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    generated["json"] = json_path

    # ── CSV Feature Summary ───────────────────────────────────────────────
    rows = []
    # Numerical features
    for col, tests in drift_report.get("feature_drift", {}).get("numerical", {}).items():
        ks = tests.get("ks_test", {})
        psi = tests.get("psi", {})
        js = tests.get("jensen_shannon", {})
        ms = tests.get("mean_shift", {})
        rows.append({
            "feature": col, "type": "numerical",
            "ks_statistic": ks.get("statistic"), "ks_pvalue": ks.get("p_value"), "ks_drift": ks.get("drift_detected"),
            "psi_value": psi.get("value"), "psi_severity": psi.get("severity"),
            "js_distance": js.get("value"), "js_severity": js.get("severity"),
            "mean_shift_pct": ms.get("pct_change"),
        })
    # Categorical features
    for col, tests in drift_report.get("feature_drift", {}).get("categorical", {}).items():
        chi2 = tests.get("chi_square", {})
        nc = tests.get("new_categories", {})
        rows.append({
            "feature": col, "type": "categorical",
            "chi2_statistic": chi2.get("statistic"), "chi2_pvalue": chi2.get("p_value"), "chi2_drift": chi2.get("drift_detected"),
            "new_categories_count": nc.get("new_count", 0),
            "new_categories": str(nc.get("new_categories", [])),
        })

    csv_path = report_dir / "feature_drift_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    generated["csv"] = csv_path

    # ── Alerts CSV ────────────────────────────────────────────────────────
    if alerts:
        alerts_path = report_dir / "alerts.csv"
        pd.DataFrame(alerts).to_csv(alerts_path, index=False)
        generated["alerts_csv"] = alerts_path

    logger.info(f"Reports generated in {report_dir}")
    return generated


def _summarize_features(drift_report):
    """Create a brief summary of drifted features."""
    dataset = drift_report.get("dataset_drift", {})
    return {
        "total_features": dataset.get("total_features", 0),
        "drifted_count": dataset.get("drifted_count", 0),
        "drifted_pct": dataset.get("drifted_pct", 0),
        "risk_level": dataset.get("risk_level", "UNKNOWN"),
        "drifted_features": [f["feature"] for f in dataset.get("drifted_features", [])],
    }


def _get_recommendation(drift_report):
    """Generate action recommendation based on drift severity."""
    risk = drift_report.get("dataset_drift", {}).get("risk_level", "LOW")
    if risk == "HIGH":
        return "IMMEDIATE ACTION: Retrain the model with recent data. Investigate root causes of drift in affected features."
    elif risk == "MEDIUM":
        return "SCHEDULE RETRAINING: Monitor the situation closely. If drift persists in next batch, retrain the model."
    else:
        return "NO ACTION NEEDED: Data distributions are stable. Continue routine monitoring."
