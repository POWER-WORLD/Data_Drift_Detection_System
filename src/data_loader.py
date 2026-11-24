"""
Data loading module for the Data Drift Detection System.
Handles loading baseline data, production batches, and generating simulated drift scenarios.
All loading functions include validation and error handling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.config.settings import (
    BASELINE_FILE, PRODUCTION_DIR, TARGET_COLUMN, ID_COLUMN,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)
from src.data_validator import validate_file, validate_schema, sanitize_dataframe
from src.utils import setup_logger

logger = setup_logger(__name__)


def load_baseline_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the baseline (training) dataset with validation.

    Args:
        filepath: Path to CSV file. Defaults to BASELINE_FILE from settings.

    Returns:
        Validated and cleaned DataFrame.

    Raises:
        DataValidationError: If the file fails validation.
        FileNotFoundError: If the file does not exist.
    """
    filepath = filepath or BASELINE_FILE
    logger.info(f"Loading baseline data from {filepath}")

    # Validate file
    validate_file(filepath)

    try:
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Baseline file is empty: {filepath}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse baseline CSV: {e}")

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Validate schema
    warnings = validate_schema(df, require_target=True)
    for w in warnings:
        logger.warning(w)

    # Clean TotalCharges: convert to numeric (has some blank strings)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges with median
    if df["TotalCharges"].isna().any():
        median_val = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_val)
        logger.info(f"Filled missing TotalCharges with median={median_val:.2f}")

    # Sanitize (replace inf, drop all-NaN rows)
    df = sanitize_dataframe(df)

    return df


def load_production_batch(filepath: Path) -> pd.DataFrame:
    """
    Load a production data batch with validation.

    Args:
        filepath: Path to the production batch CSV file.

    Returns:
        Validated DataFrame.

    Raises:
        DataValidationError: If the file fails validation.
    """
    logger.info(f"Loading production batch from {filepath}")

    # Validate file
    validate_file(filepath)

    try:
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Production batch is empty: {filepath}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse production CSV: {e}")

    # Validate schema (target is optional in production)
    warnings = validate_schema(df, require_target=False)
    for w in warnings:
        logger.warning(w)

    # Same cleaning as baseline
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Sanitize
    df = sanitize_dataframe(df)

    logger.info(f"Loaded production batch: {len(df)} rows")
    return df


def list_production_batches() -> list:
    """
    List all available production batch files.

    Returns:
        Sorted list of Path objects for each batch CSV.
    """
    batches = sorted(PRODUCTION_DIR.glob("*.csv"))
    logger.info(f"Found {len(batches)} production batches")
    return batches


def generate_production_batches(baseline_df: pd.DataFrame, seed: int = 42) -> None:
    """
    Generate 5 simulated production data batches with different drift scenarios.
    Each batch contains ~1000 rows sampled/modified from the baseline data.

    Scenarios:
        1. No Drift — random sample from baseline
        2. Mild Drift — slight shifts in numerical features
        3. Severe Drift — large distribution changes
        4. Missing Data Drift — increased null values
        5. Category Drift — new unseen categorical values

    Args:
        baseline_df: The baseline training DataFrame.
        seed: Random seed for reproducibility.
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
    flip_mask = rng.random(batch_size) < 0.10
    batch2.loc[flip_mask, "SeniorCitizen"] = 1
    batch2.to_csv(PRODUCTION_DIR / "batch_02_mild_drift.csv", index=False)
    logger.info("Generated batch_02_mild_drift.csv")

    # ── Scenario 3: Severe Drift ─────────────────────────────────────────
    batch3 = baseline_df.sample(n=batch_size, replace=True, random_state=rng).reset_index(drop=True)
    batch3["MonthlyCharges"] = batch3["MonthlyCharges"] * rng.uniform(1.5, 2.0, size=batch_size)
    batch3["TotalCharges"] = batch3["TotalCharges"] * rng.uniform(1.3, 1.8, size=batch_size)
    batch3["tenure"] = rng.randint(0, 12, size=batch_size)
    batch3["Contract"] = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=batch_size, p=[0.85, 0.10, 0.05]
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

    logger.info("All 5 production batches generated successfully.")
