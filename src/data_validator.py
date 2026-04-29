"""
Data Validation Layer for the Data Drift Detection System.

Provides schema enforcement, data quality checks, and sanitization
to reject corrupt or unsafe data before it enters the pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.config.settings import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN, ID_COLUMN
)
from src.utils import setup_logger

logger = setup_logger(__name__)

# All columns expected in the baseline dataset (order-independent)
REQUIRED_COLUMNS = set(
    [ID_COLUMN, TARGET_COLUMN] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES
)

# Maximum file size for uploads (50 MB)
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

# Maximum acceptable missing-value percentage per column
MAX_MISSING_PCT = 90.0


class DataValidationError(Exception):
    """Raised when data fails validation checks."""
    pass


def validate_file(filepath: Path) -> None:
    """
    Validate that a file exists, is a CSV, and is within size limits.

    Raises:
        DataValidationError: On any validation failure.
    """
    if not filepath.exists():
        raise DataValidationError(f"File not found: {filepath}")

    if filepath.suffix.lower() != ".csv":
        raise DataValidationError(
            f"Invalid file type '{filepath.suffix}'. Only .csv files are accepted."
        )

    size = filepath.stat().st_size
    if size == 0:
        raise DataValidationError(f"File is empty: {filepath}")

    if size > MAX_FILE_SIZE_BYTES:
        size_mb = size / (1024 * 1024)
        raise DataValidationError(
            f"File too large ({size_mb:.1f} MB). Maximum allowed is "
            f"{MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f} MB."
        )


def validate_schema(df: pd.DataFrame, require_target: bool = False) -> List[str]:
    """
    Validate that the DataFrame contains expected columns.

    Args:
        df: DataFrame to validate.
        require_target: If True, target column must be present.

    Returns:
        List of warning messages (empty if no issues).

    Raises:
        DataValidationError: If critical columns are missing.
    """
    warnings = []
    present = set(df.columns)

    # Check for feature columns
    expected_features = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
    missing_features = expected_features - present
    if missing_features:
        raise DataValidationError(
            f"Missing required feature columns: {sorted(missing_features)}"
        )

    if require_target and TARGET_COLUMN not in present:
        raise DataValidationError(
            f"Missing required target column: '{TARGET_COLUMN}'"
        )

    # Warn about extra columns (not an error)
    known = REQUIRED_COLUMNS
    extra = present - known
    if extra:
        warnings.append(f"Extra columns found (will be ignored): {sorted(extra)}")

    return warnings


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run data quality checks and return a report.

    Checks:
        - Rows with all NaN
        - Columns with >90% missing
        - Infinite values
        - Zero-variance numerical columns
        - Duplicate rows

    Returns:
        Dictionary with quality metrics and lists of flagged items.
    """
    report: Dict[str, Any] = {"passed": True, "issues": [], "stats": {}}

    # Empty DataFrame
    if len(df) == 0:
        report["passed"] = False
        report["issues"].append("DataFrame is empty (0 rows)")
        return report

    # Very small batch
    if len(df) < 10:
        report["issues"].append(
            f"Very small batch ({len(df)} rows). Statistical tests may be unreliable."
        )

    report["stats"]["rows"] = len(df)
    report["stats"]["columns"] = len(df.columns)

    # Rows that are entirely NaN
    all_nan_rows = int(df.isna().all(axis=1).sum())
    if all_nan_rows > 0:
        report["issues"].append(f"{all_nan_rows} row(s) are entirely NaN")

    # Columns with excessive missing values
    high_missing = []
    for col in df.columns:
        miss_pct = df[col].isna().mean() * 100
        if miss_pct > MAX_MISSING_PCT:
            high_missing.append((col, round(miss_pct, 1)))
    if high_missing:
        report["issues"].append(
            f"Columns with >{MAX_MISSING_PCT}% missing: "
            f"{', '.join(f'{c} ({p}%)' for c, p in high_missing)}"
        )

    # Infinite values in numerical columns
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    inf_cols = []
    for col in num_cols:
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        n_inf = int(np.isinf(numeric_col.dropna()).sum())
        if n_inf > 0:
            inf_cols.append((col, n_inf))
    if inf_cols:
        report["issues"].append(
            f"Infinite values found: "
            f"{', '.join(f'{c} ({n})' for c, n in inf_cols)}"
        )

    # Zero-variance numerical columns
    zero_var = []
    for col in num_cols:
        numeric_col = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(numeric_col) > 0 and numeric_col.std() == 0:
            zero_var.append(col)
    if zero_var:
        report["issues"].append(f"Zero-variance columns: {zero_var}")

    # Duplicate rows
    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        dupe_pct = round(n_dupes / len(df) * 100, 1)
        report["issues"].append(
            f"{n_dupes} duplicate rows ({dupe_pct}%)"
        )

    report["stats"]["duplicate_rows"] = n_dupes
    report["stats"]["issue_count"] = len(report["issues"])

    if report["issues"]:
        for issue in report["issues"]:
            logger.warning(f"Data quality: {issue}")

    return report


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame by handling infinities and fully-NaN rows.

    This is non-destructive: it replaces inf with NaN (which are later
    handled by the preprocessor) and drops all-NaN rows.

    Args:
        df: Raw DataFrame.

    Returns:
        Sanitized DataFrame (copy).
    """
    df = df.copy()

    # Replace infinities with NaN in numerical columns
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            df[col] = numeric_col.replace([np.inf, -np.inf], np.nan)

    # Drop rows that are entirely NaN
    before = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} all-NaN rows during sanitization")

    return df


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and special characters.

    Args:
        filename: Raw filename from upload.

    Returns:
        Safe filename string.
    """
    # Remove path components
    safe = Path(filename).name

    # Remove potentially dangerous characters
    safe = "".join(c for c in safe if c.isalnum() or c in "._- ")

    # Ensure it ends with .csv
    if not safe.lower().endswith(".csv"):
        safe += ".csv"

    # Fallback if empty
    if not safe or safe == ".csv":
        safe = "uploaded_batch.csv"

    return safe
