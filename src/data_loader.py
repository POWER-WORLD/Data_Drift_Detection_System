"""
Data loading module for the Data Drift Detection System.

Handles loading baseline datasets, reading production batches, and generating
simulated data drift scenarios (no drift, mild drift, severe drift, missingness,
and categorical expansion). Includes validation and sanitization on all data ingested.
Follows PEP 8 and Global Naming Standardization.
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from src.config.settings import (
    BASELINE_FILE,
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    PRODUCTION_DIR,
    TARGET_COLUMN,
)
from src.data_validator import sanitize_dataframe, validate_file, validate_schema
from src.utils import setup_logger

logger = setup_logger(__name__)


def load_baseline_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load the baseline training dataset with schema validation and sanitization.

    Args:
        filepath: Optional path to the CSV file. Defaults to BASELINE_FILE.

    Returns:
        Validated and sanitized baseline DataFrame.

    Raises:
        DataValidationError: If the file fails integrity or schema checks.
        ValueError: If the file cannot be parsed or is empty.
    """
    target_path = Path(filepath) if filepath is not None else BASELINE_FILE
    logger.info("Loading baseline data from %s", target_path)

    # Validate file existence and constraints
    validate_file(target_path)

    try:
        df = pd.read_csv(target_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Baseline file is empty: {target_path}")
    except pd.errors.ParserError as exc:
        raise ValueError(f"Failed to parse baseline CSV: {exc}")

    logger.info("Loaded baseline data with %d rows and %d columns", len(df), len(df.columns))

    # Validate schema (target column is mandatory in baseline)
    schema_warnings = validate_schema(df, require_target=True)
    for warning_msg in schema_warnings:
        logger.warning(warning_msg)

    # Clean and coerce TotalCharges (frequently contains blank string spaces in Telco data)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        if df["TotalCharges"].isna().any():
            median_val = df["TotalCharges"].median()
            fill_val = float(median_val) if pd.notna(median_val) else 0.0
            df["TotalCharges"] = df["TotalCharges"].fillna(fill_val)
            logger.info("Filled missing TotalCharges with baseline median=%.2f", fill_val)

    # Sanitize infinities and empty rows
    sanitized_df = sanitize_dataframe(df)
    return sanitized_df


def load_production_batch(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a production data batch with schema validation and sanitization.

    Args:
        filepath: Path to the production CSV batch file.

    Returns:
        Validated production DataFrame.

    Raises:
        DataValidationError: If the file fails integrity or schema checks.
        ValueError: If the file is empty or cannot be parsed.
    """
    target_path = Path(filepath)
    logger.info("Loading production batch from %s", target_path)

    validate_file(target_path)

    try:
        df = pd.read_csv(target_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Production batch is empty: {target_path}")
    except pd.errors.ParserError as exc:
        raise ValueError(f"Failed to parse production CSV: {exc}")

    # Validate schema (target column is optional in production)
    schema_warnings = validate_schema(df, require_target=False)
    for warning_msg in schema_warnings:
        logger.warning(warning_msg)

    # Clean TotalCharges if present
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    sanitized_df = sanitize_dataframe(df)
    logger.info("Loaded production batch with %d rows", len(sanitized_df))
    return sanitized_df


def list_production_batches() -> List[Path]:
    """
    List all available production batch files in PRODUCTION_DIR sorted by filename.

    Returns:
        Sorted list of Path objects for available CSV batch files.
    """
    batches = sorted(PRODUCTION_DIR.glob("*.csv"))
    logger.info("Found %d production batch file(s) in %s", len(batches), PRODUCTION_DIR)
    return batches


def generate_production_batches(baseline_df: pd.DataFrame, seed: int = 42) -> None:
    """
    Generate 5 simulated production batches demonstrating varied data drift scenarios.

    Scenarios:
        1. batch_01_no_drift.csv: Statistical replicate of baseline.
        2. batch_02_mild_drift.csv: Minor numerical shifts (MonthlyCharges +5-15%, tenure jitter).
        3. batch_03_severe_drift.csv: Large distribution shifts across charges, tenure, and contracts.
        4. batch_04_missing_data.csv: Synthetically injected missing values across multiple columns.
        5. batch_05_category_drift.csv: Unseen categorical values (e.g. 'Crypto wallet', '5G Home').

    Args:
        baseline_df: Source baseline training DataFrame.
        seed: Random seed for deterministic reproducibility.
    """
    rng = np.random.RandomState(seed)
    batch_size = 1000
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

    # ── Scenario 1: No Drift ─────────────────────────────────────────────
    batch1 = baseline_df.sample(n=batch_size, replace=True, random_state=rng).reset_index(drop=True)
    batch1.to_csv(PRODUCTION_DIR / "batch_01_no_drift.csv", index=False)
    logger.info("Generated batch_01_no_drift.csv")

    # ── Scenario 2: Mild Drift ───────────────────────────────────────────
    batch2 = baseline_df.sample(n=batch_size, replace=True, random_state=rng).reset_index(drop=True)
    batch2["MonthlyCharges"] = batch2["MonthlyCharges"] * rng.uniform(1.05, 1.15, size=batch_size)
    batch2["tenure"] = (batch2["tenure"] + rng.randint(0, 5, size=batch_size)).clip(0, 72)
    senior_flip = rng.random(batch_size) < 0.10
    batch2.loc[senior_flip, "SeniorCitizen"] = 1
    batch2.to_csv(PRODUCTION_DIR / "batch_02_mild_drift.csv", index=False)
    logger.info("Generated batch_02_mild_drift.csv")

    # ── Scenario 3: Severe Drift ─────────────────────────────────────────
    batch3 = baseline_df.sample(n=batch_size, replace=True, random_state=rng).reset_index(drop=True)
    batch3["MonthlyCharges"] = batch3["MonthlyCharges"] * rng.uniform(1.5, 2.0, size=batch_size)
    batch3["TotalCharges"] = batch3["TotalCharges"] * rng.uniform(1.3, 1.8, size=batch_size)
    batch3["tenure"] = rng.randint(0, 12, size=batch_size)
    batch3["Contract"] = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=batch_size,
        p=[0.85, 0.10, 0.05],
    )
    batch3["Churn"] = rng.choice(["Yes", "No"], size=batch_size, p=[0.55, 0.45])
    batch3.to_csv(PRODUCTION_DIR / "batch_03_severe_drift.csv", index=False)
    logger.info("Generated batch_03_severe_drift.csv")

    # ── Scenario 4: Missing Data Drift ───────────────────────────────────
    batch4 = baseline_df.sample(n=batch_size, replace=True, random_state=rng).reset_index(drop=True)
    batch4.loc[rng.random(batch_size) < 0.20, "MonthlyCharges"] = np.nan
    batch4.loc[rng.random(batch_size) < 0.15, "TotalCharges"] = np.nan
    batch4.loc[rng.random(batch_size) < 0.10, "InternetService"] = np.nan
    batch4.loc[rng.random(batch_size) < 0.12, "tenure"] = np.nan
    batch4.to_csv(PRODUCTION_DIR / "batch_04_missing_data.csv", index=False)
    logger.info("Generated batch_04_missing_data.csv")

    # ── Scenario 5: Category Drift ───────────────────────────────────────
    batch5 = baseline_df.sample(n=batch_size, replace=True, random_state=rng).reset_index(drop=True)
    batch5.loc[rng.random(batch_size) < 0.15, "PaymentMethod"] = "Crypto wallet"
    batch5.loc[rng.random(batch_size) < 0.12, "InternetService"] = "5G Home"
    batch5.loc[rng.random(batch_size) < 0.08, "Contract"] = "Quarterly"
    batch5.to_csv(PRODUCTION_DIR / "batch_05_category_drift.csv", index=False)
    logger.info("Generated batch_05_category_drift.csv")

    logger.info("Successfully generated all 5 simulated production batches.")
