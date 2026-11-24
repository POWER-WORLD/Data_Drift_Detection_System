"""
Tests for the Data Validation Layer.
Covers schema validation, data quality checks, file validation, and sanitization.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch

from src.data_validator import (
    validate_file, validate_schema, validate_data_quality,
    sanitize_dataframe, sanitize_filename, DataValidationError,
)
from src.config.settings import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN, ID_COLUMN


def _make_valid_df(n=100):
    """Create a minimal valid DataFrame matching the Telco schema."""
    rng = np.random.RandomState(42)
    data = {
        ID_COLUMN: [f"id_{i}" for i in range(n)],
        TARGET_COLUMN: rng.choice(["Yes", "No"], n),
    }
    for col in NUMERICAL_FEATURES:
        data[col] = rng.normal(50, 10, n)
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.choice(["A", "B", "C"], n)
    return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════════════════════
# Schema Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemaValidation:
    """Tests for validate_schema()."""

    def test_valid_schema_passes(self):
        df = _make_valid_df()
        warnings = validate_schema(df, require_target=True)
        assert isinstance(warnings, list)

    def test_missing_feature_column_raises(self):
        df = _make_valid_df()
        df = df.drop(columns=["tenure"])
        with pytest.raises(DataValidationError, match="Missing required"):
            validate_schema(df)

    def test_missing_target_when_required_raises(self):
        df = _make_valid_df()
        df = df.drop(columns=[TARGET_COLUMN])
        with pytest.raises(DataValidationError, match="target column"):
            validate_schema(df, require_target=True)

    def test_missing_target_not_required_passes(self):
        df = _make_valid_df()
        df = df.drop(columns=[TARGET_COLUMN])
        warnings = validate_schema(df, require_target=False)
        assert isinstance(warnings, list)

    def test_extra_columns_warned(self):
        df = _make_valid_df()
        df["EXTRA_COL"] = 99
        warnings = validate_schema(df)
        assert any("Extra" in w for w in warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Quality
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataQuality:
    """Tests for validate_data_quality()."""

    def test_clean_data_passes(self):
        df = _make_valid_df()
        report = validate_data_quality(df)
        assert report["stats"]["rows"] == 100

    def test_empty_df_reported(self):
        df = pd.DataFrame()
        report = validate_data_quality(df)
        assert report["passed"] is False

    def test_small_batch_warning(self):
        df = _make_valid_df(n=5)
        report = validate_data_quality(df)
        assert any("small batch" in i.lower() for i in report["issues"])

    def test_all_nan_rows_flagged(self):
        df = _make_valid_df(n=50)
        # Set last 3 rows to NaN
        df.iloc[-3:] = np.nan
        report = validate_data_quality(df)
        assert any("entirely NaN" in i for i in report["issues"])

    def test_infinite_values_flagged(self):
        df = _make_valid_df(n=50)
        df.loc[0, "tenure"] = np.inf
        report = validate_data_quality(df)
        assert any("Infinite" in i for i in report["issues"])

    def test_duplicate_rows_counted(self):
        df = _make_valid_df(n=20)
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)
        report = validate_data_quality(df)
        assert report["stats"]["duplicate_rows"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Sanitization
# ═══════════════════════════════════════════════════════════════════════════════

class TestSanitization:
    """Tests for sanitize_dataframe() and sanitize_filename()."""

    def test_inf_replaced_with_nan(self):
        df = _make_valid_df(n=10)
        df.loc[0, "tenure"] = np.inf
        df.loc[1, "MonthlyCharges"] = -np.inf
        result = sanitize_dataframe(df)
        assert not np.isinf(result["tenure"]).any()
        assert not np.isinf(result["MonthlyCharges"]).any()

    def test_all_nan_rows_dropped(self):
        df = _make_valid_df(n=10)
        df.iloc[9] = np.nan
        result = sanitize_dataframe(df)
        assert len(result) == 9

    def test_filename_sanitization(self):
        # Path traversal is stripped — only the filename remains
        result = sanitize_filename("../../../etc/passwd")
        assert ".." not in result  # No traversal
        assert result.endswith(".csv")
        assert sanitize_filename("batch data.csv") == "batch data.csv"
        assert sanitize_filename("test.csv") == "test.csv"

    def test_filename_empty_fallback(self):
        assert sanitize_filename("") == "uploaded_batch.csv"

    def test_filename_non_csv_gets_extension(self):
        assert sanitize_filename("data.xlsx").endswith(".csv")


# ═══════════════════════════════════════════════════════════════════════════════
# File Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestFileValidation:
    """Tests for validate_file()."""

    def test_nonexistent_file_raises(self, tmp_path):
        fake = tmp_path / "nonexistent.csv"
        with pytest.raises(DataValidationError, match="not found"):
            validate_file(fake)

    def test_wrong_extension_raises(self, tmp_path):
        bad = tmp_path / "data.txt"
        bad.write_text("hello")
        with pytest.raises(DataValidationError, match="Invalid file type"):
            validate_file(bad)

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        with pytest.raises(DataValidationError, match="empty"):
            validate_file(empty)

    def test_valid_csv_passes(self, tmp_path):
        good = tmp_path / "data.csv"
        good.write_text("a,b\n1,2\n")
        validate_file(good)  # Should not raise
