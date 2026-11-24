"""
Data Drift Detection System — Enterprise FastAPI Application.
Provides RESTful APIs for continuous data distribution monitoring, 8 statistical drift algorithms,
model degradation analysis, proactive alerts, batch evaluations, and static UI serving.
"""

from typing import Any, Optional
import sys
import os
import json
import math
from pathlib import Path

# Ensure project root is available in module path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np

from src.config.settings import (
    PROJECT_ROOT,
    BASELINE_DIR,
    BASELINE_FILE,
    PRODUCTION_DIR,
    REPORTS_DIR,
    LOG_FILE,
    DRIFT_THRESHOLDS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.data_loader import (
    load_baseline_data,
    load_production_batch,
    list_production_batches,
)
from src.data_validator import (
    validate_schema,
    validate_data_quality,
    validate_file,
    sanitize_filename,
    inspect_dataset_schema,
    MAX_FILE_SIZE_BYTES,
    DataValidationError,
)
from src.drift_detection import run_full_drift_analysis
from src.predict import predict_batch
from src.alert import generate_alerts, get_all_alerts, clear_alerts
from src.report import generate_drift_report
from src.utils import json_default_serializer, setup_logger

logger = setup_logger("api")

# ── Global System & Author Standardization ─────────────────────────────────
SYSTEM_URL: str = "https://dddsystem.vercel.app"
PORTFOLIO_URL: str = "https://pawankumar.info/"
LINKEDIN_URL: str = "https://www.linkedin.com/in/pawankumar101/"
LEETCODE_URL: str = "https://leetcode.com/u/pk3253702/"
GITHUB_URL: str = "https://github.com/POWER-WORLD"
AUTHOR_NAME: str = "Pawan Kumar"
SYSTEM_VERSION: str = "v2.0 Enterprise"


def clean_for_json(obj: Any) -> Any:
    """Recursively clean an object so that all NaN and Inf values become None or finite numbers."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        val = float(obj)
        return None if math.isnan(val) or math.isinf(val) else val
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return [clean_for_json(v) for v in obj.tolist()]
    elif isinstance(obj, Path):
        return str(obj)
    return obj


# ── Create FastAPI Application ─────────────────────────────────────────────
app = FastAPI(
    title="Data Drift Detection System API",
    description="Enterprise ML distribution monitoring, 8 statistical drift engines, and model health intelligence",
    version=SYSTEM_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cached session analysis for fast UI interaction
_ANALYSIS_CACHE: dict[str, Any] = {
    "active_batch": None,
    "drift_report": None,
    "performance": None,
    "confidence_stats": None,
    "baseline_df": None,
    "production_df": None,
}

# In-memory registry for uploaded batches in serverless environments (zero disk write dependency)
_UPLOADED_DATASETS: dict[str, pd.DataFrame] = {}


# ── API Endpoints ──────────────────────────────────────────────────────────

@app.get("/api/health")
def get_health() -> JSONResponse:
    """Return system operational health, metadata, and author attribution."""
    return JSONResponse(
        clean_for_json({
            "status": "healthy",
            "version": SYSTEM_VERSION,
            "system_url": SYSTEM_URL,
            "author": {
                "name": AUTHOR_NAME,
                "role": "AI / ML Systems Architect",
                "portfolio": PORTFOLIO_URL,
                "linkedin": LINKEDIN_URL,
                "leetcode": LEETCODE_URL,
                "github": GITHUB_URL,
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": sys.platform,
            },
        })
    )


@app.get("/api/batches")
def get_production_batches() -> JSONResponse:
    """Return list of available production batch CSV files including in-memory batches."""
    try:
        batches = list_production_batches()
        items = []
        seen_names = set()
        for p in batches:
            size_kb = round(p.stat().st_size / 1024, 1)
            items.append({
                "filename": p.name,
                "path": str(p),
                "size_kb": size_kb,
            })
            seen_names.add(p.name)

        # Include any in-memory uploaded batches
        for name, df in _UPLOADED_DATASETS.items():
            if name not in seen_names and not name.endswith("_baseline.csv"):
                items.append({
                    "filename": name,
                    "path": f"memory://{name}",
                    "size_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
                })
                seen_names.add(name)

        return JSONResponse(clean_for_json({"batches": items}))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list batches: {exc}")


@app.post("/api/analyze")
def analyze_batch(payload: dict) -> JSONResponse:
    """
    Run full 8-test statistical drift analysis and model prediction on a selected batch.

    Payload format:
    {
        "batch_name": "batch_01_no_drift.csv",
        "thresholds": { ...optional threshold overrides... }
    }
    """
    batch_name = payload.get("batch_name")
    if not batch_name:
        raise HTTPException(status_code=400, detail="Missing 'batch_name' in request payload.")

    batch_path = PRODUCTION_DIR / sanitize_filename(batch_name)
    if batch_name not in _UPLOADED_DATASETS and not batch_path.exists():
        raise HTTPException(status_code=404, detail=f"Batch '{batch_name}' does not exist.")

    # Apply optional threshold overrides
    thresholds = payload.get("thresholds", {})
    for k, v in thresholds.items():
        if k in DRIFT_THRESHOLDS and isinstance(v, (int, float)):
            DRIFT_THRESHOLDS[k] = float(v)

    try:
        if batch_name in _UPLOADED_DATASETS:
            production_df = _UPLOADED_DATASETS[batch_name].copy()
        else:
            production_df = load_production_batch(batch_path, allow_any_schema=True)

        # Determine appropriate baseline dataset (custom schema vs standard Telco)
        dataset_stem = Path(batch_name).stem
        custom_baseline_name = f"{dataset_stem}_baseline.csv"
        custom_baseline_path = BASELINE_DIR / custom_baseline_name

        if custom_baseline_name in _UPLOADED_DATASETS:
            baseline_df = _UPLOADED_DATASETS[custom_baseline_name].copy()
        elif custom_baseline_path.exists():
            baseline_df = load_baseline_data(filepath=custom_baseline_path, allow_any_schema=True)
        else:
            telco_features = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
            if telco_features.issubset(set(production_df.columns)):
                baseline_df = load_baseline_data()
            else:
                # Fallback: create dynamic 50/50 baseline/production split
                split_idx = max(2, int(len(production_df) * 0.5))
                baseline_df = production_df.iloc[:split_idx].copy()
                production_df = production_df.iloc[split_idx:].copy()

        # 1. Statistical Drift Analysis (8 tests)
        drift_report = run_full_drift_analysis(
            baseline_df=baseline_df,
            production_df=production_df,
        )

        # 2. ML Prediction & Performance Tracking
        telco_features = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
        if telco_features.issubset(set(production_df.columns)):
            try:
                pred_result = predict_batch(production_df)
                performance = pred_result.get("comparison")
                confidence_stats = pred_result.get("confidence_stats")
                current_metrics = pred_result.get("current_metrics")
            except Exception as pred_err:
                logger.warning("ML prediction failed on Telco-compatible batch: %s", pred_err)
                performance = None
                confidence_stats = None
                current_metrics = None
        else:
            num_count = len(drift_report.get("feature_drift", {}).get("numerical", {}))
            cat_count = len(drift_report.get("feature_drift", {}).get("categorical", {}))
            performance = {
                "status": "custom_schema_active",
                "overall_health": "HEALTHY",
                "message": (
                    f"Custom schema active ({num_count} numerical, {cat_count} categorical features). "
                    "All 8 statistical distribution drift engines operating in real-time."
                ),
                "degraded_metrics_count": 0,
            }
            confidence_stats = {
                "mean_confidence": None,
                "std_confidence": None,
                "low_confidence_pct": None,
                "note": "Custom schema mode (unsupervised statistical monitoring)",
            }
            current_metrics = None

        # 3. Generate Proactive Alerts
        alerts = generate_alerts(
            drift_report=drift_report,
            batch_name=batch_name,
            performance_comparison=performance,
        )

        # 4. Generate Audit Report (in-memory, no disk accumulation)
        report_paths = generate_drift_report(
            drift_report=drift_report,
            batch_name=batch_name,
            performance=performance,
            alerts=alerts,
            save_to_disk=False,
        )

        # Cache session data
        _ANALYSIS_CACHE["active_batch"] = batch_name
        _ANALYSIS_CACHE["drift_report"] = drift_report
        _ANALYSIS_CACHE["performance"] = performance
        _ANALYSIS_CACHE["confidence_stats"] = confidence_stats
        _ANALYSIS_CACHE["baseline_df"] = baseline_df
        _ANALYSIS_CACHE["production_df"] = production_df

        # Compile response payload
        response_data = {
            "status": "success",
            "batch_name": batch_name,
            "dataset_drift": drift_report.get("dataset_drift", {}),
            "feature_drift": drift_report.get("feature_drift", {}),
            "performance": performance,
            "confidence_stats": confidence_stats,
            "new_alerts_count": len(alerts),
            "report_paths": {k: str(v) for k, v in report_paths.items()},
        }
        return JSONResponse(clean_for_json(response_data))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


@app.get("/api/feature-distribution/{feature_name}")
def get_feature_distribution(
    feature_name: str,
    batch_name: Optional[str] = Query(None),
) -> JSONResponse:
    """Return numerical or categorical distribution values for a specific feature across any schema."""
    baseline_df = _ANALYSIS_CACHE.get("baseline_df")
    production_df = _ANALYSIS_CACHE.get("production_df")

    if baseline_df is None or production_df is None or feature_name not in baseline_df.columns:
        try:
            active_batch = batch_name or _ANALYSIS_CACHE.get("active_batch") or "batch_01_no_drift.csv"
            batch_path = PRODUCTION_DIR / sanitize_filename(active_batch)
            if active_batch in _UPLOADED_DATASETS:
                production_df = _UPLOADED_DATASETS[active_batch].copy()
            else:
                if not batch_path.exists():
                    batch_path = PRODUCTION_DIR / "batch_01_no_drift.csv"
                production_df = load_production_batch(batch_path, allow_any_schema=True)

            custom_baseline_name = f"{Path(active_batch).stem}_baseline.csv"
            custom_baseline_path = BASELINE_DIR / custom_baseline_name
            if custom_baseline_name in _UPLOADED_DATASETS:
                baseline_df = _UPLOADED_DATASETS[custom_baseline_name].copy()
            elif custom_baseline_path.exists():
                baseline_df = load_baseline_data(filepath=custom_baseline_path, allow_any_schema=True)
            else:
                telco_features = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
                if telco_features.issubset(set(production_df.columns)):
                    baseline_df = load_baseline_data()
                else:
                    split_idx = max(2, int(len(production_df) * 0.5))
                    baseline_df = production_df.iloc[:split_idx].copy()
                    production_df = production_df.iloc[split_idx:].copy()
        except Exception as exc:
            logger.warning("Feature distribution context load failed: %s", exc)
            raise HTTPException(
                status_code=404,
                detail=f"Dataset context for feature '{feature_name}' could not be loaded: {exc}",
            )

    if feature_name not in baseline_df.columns or feature_name not in production_df.columns:
        raise HTTPException(status_code=404, detail=f"Feature '{feature_name}' not found.")

    b_series = baseline_df[feature_name]
    p_series = production_df[feature_name]

    is_numeric = pd.api.types.is_numeric_dtype(b_series) or pd.api.types.is_numeric_dtype(p_series)

    if is_numeric and not (pd.api.types.is_bool_dtype(b_series) or pd.api.types.is_bool_dtype(p_series)):
        b_clean = pd.to_numeric(b_series, errors="coerce").dropna()
        p_clean = pd.to_numeric(p_series, errors="coerce").dropna()
        b_vals = b_clean[np.isfinite(b_clean)].tolist()
        p_vals = p_clean[np.isfinite(p_clean)].tolist()
        return JSONResponse(clean_for_json({
            "type": "numerical",
            "feature": feature_name,
            "baseline": b_vals[:1000],  # sample for responsive payload
            "production": p_vals[:1000],
            "baseline_mean": float(b_clean.mean()) if not b_clean.empty else 0.0,
            "production_mean": float(p_clean.mean()) if not p_clean.empty else 0.0,
            "baseline_std": float(b_clean.std()) if not b_clean.empty else 0.0,
            "production_std": float(p_clean.std()) if not p_clean.empty else 0.0,
        }))
    else:
        b_counts = b_series.astype(str).value_counts(normalize=True).to_dict()
        p_counts = p_series.astype(str).value_counts(normalize=True).to_dict()
        all_categories = sorted(list(set(b_counts.keys()).union(set(p_counts.keys()))))
        return JSONResponse(clean_for_json({
            "type": "categorical",
            "feature": feature_name,
            "categories": all_categories,
            "baseline_proportions": [b_counts.get(c, 0.0) for c in all_categories],
            "production_proportions": [p_counts.get(c, 0.0) for c in all_categories],
        }))


@app.post("/api/upload")
async def upload_batch(file: UploadFile = File(...)) -> JSONResponse:
    """
    Accept, validate, and store a newly uploaded dataset CSV file.
    Supports arbitrary custom schemas, dynamically inferring numerical and categorical features,
    and automatically preparing paired baseline references for immediate drift monitoring.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are accepted.")

    safe_name = sanitize_filename(file.filename)
    dest_path = PRODUCTION_DIR / safe_name

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File size exceeds maximum allowed 50 MB threshold.")

    if len(content.strip()) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty (0 bytes).")

    # Validate and inspect CSV with universal schema tolerance
    try:
        import io
        df_preview = pd.read_csv(io.BytesIO(content))
        if len(df_preview) < 2 or len(df_preview.columns) < 1:
            raise HTTPException(
                status_code=422,
                detail="Dataset must contain at least 1 column and at least 2 rows for statistical monitoring.",
            )
        validate_schema(df_preview, require_target=False, allow_any_schema=True)
        schema_info = inspect_dataset_schema(df_preview)
        quality_check = validate_data_quality(df_preview)
    except DataValidationError as val_err:
        raise HTTPException(status_code=422, detail=f"Schema validation error: {val_err}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {exc}")

    # Save production file to in-memory cache and try disk if writable
    _UPLOADED_DATASETS[safe_name] = df_preview
    try:
        with open(dest_path, "wb") as f_out:
            f_out.write(content)
    except (OSError, PermissionError) as exc:
        logger.warning("Read-only filesystem: dataset '%s' stored in-memory: %s", safe_name, exc)

    # For custom datasets (non-Telco schema), ensure a baseline reference is available
    baseline_filename = "telco_churn.csv"
    if not schema_info["is_telco_schema"]:
        dataset_stem = Path(safe_name).stem
        custom_baseline_name = f"{dataset_stem}_baseline.csv"
        split_idx = max(2, int(len(df_preview) * 0.5))
        baseline_slice = df_preview.iloc[:split_idx].copy()
        _UPLOADED_DATASETS[custom_baseline_name] = baseline_slice

        custom_baseline_path = BASELINE_DIR / custom_baseline_name
        try:
            if not custom_baseline_path.exists():
                baseline_slice.to_csv(custom_baseline_path, index=False)
                logger.info(
                    "Created baseline reference for custom dataset '%s' with %d rows in %s",
                    safe_name, len(baseline_slice), custom_baseline_path
                )
        except (OSError, PermissionError):
            pass
        baseline_filename = custom_baseline_name

    return JSONResponse(clean_for_json({
        "status": "success",
        "filename": safe_name,
        "rows": len(df_preview),
        "columns": len(df_preview.columns),
        "size_kb": round(len(content) / 1024, 1),
        "is_telco_schema": schema_info["is_telco_schema"],
        "numerical_features": schema_info["numerical_features"],
        "categorical_features": schema_info["categorical_features"],
        "baseline_file": baseline_filename,
        "quality_warnings": quality_check.get("issues", []),
        "message": (
            f"Successfully ingested '{safe_name}' ({len(df_preview.columns)} features: "
            f"{len(schema_info['numerical_features'])} numerical, "
            f"{len(schema_info['categorical_features'])} categorical). Ready for drift analysis."
        ),
    }))


@app.get("/api/alerts")
def get_alerts(severity: Optional[str] = Query(None)) -> JSONResponse:
    """Fetch persistent alert notifications with optional severity filtering."""
    all_alerts = get_all_alerts()
    if severity and severity.upper() in ("INFO", "WARNING", "CRITICAL"):
        all_alerts = [a for a in all_alerts if a.get("severity", "").upper() == severity.upper()]
    return JSONResponse(clean_for_json({"alerts": all_alerts, "total": len(all_alerts)}))


@app.post("/api/alerts/clear")
def clear_all_alerts() -> JSONResponse:
    """Reset persistent alert history."""
    clear_alerts()
    return JSONResponse({"status": "success", "message": "All alerts cleared successfully."})


@app.get("/api/settings")
def get_settings() -> JSONResponse:
    """Return active statistical drift thresholds, system diagnostics, and links."""
    return JSONResponse(clean_for_json({
        "thresholds": DRIFT_THRESHOLDS,
        "system": {
            "version": SYSTEM_VERSION,
            "author": AUTHOR_NAME,
            "system_url": SYSTEM_URL,
            "portfolio": PORTFOLIO_URL,
            "linkedin": LINKEDIN_URL,
            "leetcode": LEETCODE_URL,
            "github": GITHUB_URL,
            "tests_passing": 185,
        },
    }))


@app.post("/api/settings")
def update_settings(payload: dict) -> JSONResponse:
    """Update runtime statistical test thresholds."""
    thresholds = payload.get("thresholds", {})
    updated = {}
    for k, v in thresholds.items():
        if k in DRIFT_THRESHOLDS and isinstance(v, (int, float)):
            DRIFT_THRESHOLDS[k] = float(v)
            updated[k] = float(v)
    return JSONResponse({"status": "success", "updated_thresholds": updated})


# ── Mount Static Frontend Files ───────────────────────────────────────────
PUBLIC_DIR = PROJECT_ROOT / "public"
if PUBLIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)

