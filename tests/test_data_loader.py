"""
Tests for data loading functionality.
"""

import pytest
import pandas as pd
from src.data_loader import load_baseline_data, list_production_batches
from src.config.settings import BASELINE_FILE, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN


class TestLoadBaselineData:
    """Tests for the baseline data loader."""

    def test_loads_successfully(self):
        df = load_baseline_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_required_columns(self):
        df = load_baseline_data()
        for col in NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]:
            assert col in df.columns, f"Missing column: {col}"

    def test_total_charges_is_numeric(self):
        df = load_baseline_data()
        assert pd.api.types.is_numeric_dtype(df["TotalCharges"])

    def test_no_missing_total_charges(self):
        df = load_baseline_data()
        assert df["TotalCharges"].isna().sum() == 0

    def test_target_column_has_two_classes(self):
        df = load_baseline_data()
        assert set(df[TARGET_COLUMN].unique()) == {"Yes", "No"}


class TestListProductionBatches:
    """Tests for production batch listing."""

    def test_returns_list(self):
        batches = list_production_batches()
        assert isinstance(batches, list)

    def test_has_generated_batches(self):
        batches = list_production_batches()
        assert len(batches) >= 5, "Expected at least 5 production batches"

    def test_all_are_csv_files(self):
        batches = list_production_batches()
        for b in batches:
            assert b.suffix == ".csv"
