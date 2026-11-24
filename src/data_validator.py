"""
Data Validation Layer for the Data Drift Detection System.

Provides schema enforcement, data quality checks, file integrity checks,
and non-destructive sanitization to reject corrupt or unsafe data.
Follows PEP 8 and Global Naming Standardization.
"""

from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.config.settings import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from src.utils import setup_logger

logger = setup_logger(__name__)

# All columns expected in the baseline dataset (order-independent)
REQUIRED_COLUMNS: Set[str] = set(
    [ID_COLUMN, TARGET_COLUMN] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES
)

# Maximum file size for uploads (50 MB)
MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024

# Maximum acceptable missing-value percentage per column before flagging
MAX_MISSING_PCT: float = 90.0


class DataValidationError(Exception):
    """Raised when incoming dataset fails schema or integrity validation."""
    pass


def validate_file(filepath: Union[str, Path]) -> None:
    """
    Validate that a file exists, is a CSV, and does not exceed size limits.

    Args:
        filepath: Path to the target file.

    Raises:
        DataValidationError: If any file validation check fails.
    """
    path = Path(filepath)
    if not path.exists():
        raise DataValidationError(f"File not found: {path}")

    if not path.is_file():
        raise DataValidationError(f"Path is not a regular file: {path}")

    if path.suffix.lower() != ".csv":
        raise DataValidationError(
            f"Invalid file type '{path.suffix}'. Only .csv files are accepted."
        )

    file_size = path.stat().st_size
    if file_size == 0:
        raise DataValidationError(f"File is empty (0 bytes): {path}")

    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise DataValidationError(
            f"File too large ({size_mb:.1f} MB). Maximum allowed size is {max_mb:.0f} MB."
        )


def validate_schema(df: pd.DataFrame, require_target: bool = False) -> List[str]:
    """
    Validate that the DataFrame contains all mandatory feature columns.

    Args:
        df: DataFrame to validate.
        require_target: If True, the target column must be present.

    Returns:
        List of warning messages (empty if no issues).

    Raises:
        DataValidationError: If mandatory feature or target columns are absent.
    """
    warnings: List[str] = []
    present_columns = set(df.columns)

    # Check for expected feature columns
    expected_features = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
    missing_features = expected_features - present_columns
    if missing_features:
        raise DataValidationError(
            f"Missing required feature columns: {sorted(missing_features)}"
        )

    if require_target and TARGET_COLUMN not in present_columns:
        raise DataValidationError(
            f"Missing required target column: '{TARGET_COLUMN}'"
        )

    # Warn about extra unmapped columns
    extra_columns = present_columns - REQUIRED_COLUMNS
    if extra_columns:
        warnings.append(
            f"Extra columns found (will be ignored): {sorted(extra_columns)}"
        )

    return warnings


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute data quality checks and return a structured assessment report.

    Checks include:
        - Empty DataFrame detection
        - Small batch size warning (< 10 rows)
        - Entirely NaN rows
        - Columns with > 90% missing values
        - Infinite values in numerical columns
        - Zero-variance numerical columns (handling infinities without warning)
        - Duplicate row counts

    Args:
        df: Input DataFrame to check.

    Returns:
        Dictionary containing quality indicators, issues list, and basic statistics.
    """
    report: Dict[str, Any] = {"passed": True, "issues": [], "stats": {}}

    # Empty DataFrame check
    if len(df) == 0:
        report["passed"] = False
        report["issues"].append("DataFrame is empty (0 rows)")
        return report

    # Small batch check
    if len(df) < 10:
        report["issues"].append(
            f"Very small batch ({len(df)} rows). Statistical tests may be unreliable."
        )

    report["stats"]["rows"] = int(len(df))
    report["stats"]["columns"] = int(len(df.columns))

    # Entirely NaN rows check
    all_nan_rows = int(df.isna().all(axis=1).sum())
    if all_nan_rows > 0:
        report["issues"].append(f"{all_nan_rows} row(s) are entirely NaN")

    # High missingness check
    high_missing: List[Tuple[str, float]] = []
    for col in df.columns:
        miss_pct = float(df[col].isna().mean() * 100.0)
        if miss_pct > MAX_MISSING_PCT:
            high_missing.append((col, round(miss_pct, 1)))

    if high_missing:
        report["issues"].append(
            f"Columns with >{MAX_MISSING_PCT:.0f}% missing: "
            f"{', '.join(f'{c} ({p}%)' for c, p in high_missing)}"
        )

    # Infinite values check in numerical features
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    inf_cols: List[Tuple[str, int]] = []
    for col in num_cols:
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        non_na = numeric_series.dropna()
        n_inf = int(np.isinf(non_na).sum())
        if n_inf > 0:
            inf_cols.append((col, n_inf))

    if inf_cols:
        report["issues"].append(
            f"Infinite values found: "
            f"{', '.join(f'{c} ({n})' for c, n in inf_cols)}"
        )

    # Zero-variance check in numerical features
    # Filter strictly to finite values first to avoid RuntimeWarning when infinities exist
    zero_var_cols: List[str] = []
    for col in num_cols:
        numeric_series = pd.to_numeric(df[col], errors="coerce").dropna()
        finite_vals = numeric_series[np.isfinite(numeric_series)]
        if len(finite_vals) > 1 and float(finite_vals.std()) == 0.0:
            zero_var_cols.append(col)

    if zero_var_cols:
        report["issues"].append(f"Zero-variance columns: {zero_var_cols}")

    # Duplicate rows check
    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        dupe_pct = round(n_dupes / len(df) * 100.0, 1)
        report["issues"].append(f"{n_dupes} duplicate rows ({dupe_pct}%)")

    report["stats"]["duplicate_rows"] = n_dupes
    report["stats"]["issue_count"] = len(report["issues"])

    if report["issues"]:
        for issue in report["issues"]:
            logger.warning("Data quality check flagged: %s", issue)

    return report


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame non-destructively by converting infinities to NaN and dropping all-NaN rows.

    Args:
        df: Raw DataFrame.

    Returns:
        Sanitized DataFrame copy.
    """
    sanitized_df = df.copy()

    # Replace infinities with NaN in numerical columns
    for col in NUMERICAL_FEATURES:
        if col in sanitized_df.columns:
            numeric_col = pd.to_numeric(sanitized_df[col], errors="coerce")
            sanitized_df[col] = numeric_col.replace([np.inf, -np.inf], np.nan)

    # Drop rows that are entirely NaN
    initial_rows = len(sanitized_df)
    sanitized_df = sanitized_df.dropna(how="all").reset_index(drop=True)
    dropped_count = initial_rows - len(sanitized_df)

    if dropped_count > 0:
        logger.info("Dropped %d all-NaN rows during sanitization", dropped_count)

    return sanitized_df


def sanitize_filename(filename: str) -> str:
    """
    Sanitize an uploaded filename to prevent directory traversal and special character exploits.

    Args:
        filename: Raw filename string from upload payload.

    Returns:
        Safe, normalized filename ending with .csv.
    """
    # Extract basename to eliminate path components
    safe_name = Path(filename).name

    # Allow alphanumeric characters and standard delimiters
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "._- ")
    safe_name = safe_name.strip()

    # Fallback if empty or whitespace-only
    if not safe_name or safe_name.lower() == ".csv":
        return "uploaded_batch.csv"

    # Enforce .csv suffix
    if not safe_name.lower().endswith(".csv"):
        safe_name += ".csv"

    return safe_name
