"""
Tests for preprocessing functionality.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import load_baseline_data
from src.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Tests for the DataPreprocessor class."""

    @pytest.fixture
    def baseline_df(self):
        return load_baseline_data()

    @pytest.fixture
    def preprocessor(self, baseline_df):
        p = DataPreprocessor()
        p.fit_transform(baseline_df)
        return p

    def test_fit_transform_returns_correct_types(self, baseline_df):
        p = DataPreprocessor()
        X, y = p.fit_transform(baseline_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_fit_transform_no_nans(self, baseline_df):
        p = DataPreprocessor()
        X, y = p.fit_transform(baseline_df)
        assert X.isna().sum().sum() == 0, "Processed data contains NaN values"

    def test_target_is_binary(self, baseline_df):
        p = DataPreprocessor()
        _, y = p.fit_transform(baseline_df)
        assert set(y.unique()) == {0, 1}

    def test_is_fitted_after_fit_transform(self, baseline_df):
        p = DataPreprocessor()
        p.fit_transform(baseline_df)
        assert p.is_fitted is True

    def test_transform_production_data(self, preprocessor, baseline_df):
        # Use a subset of baseline as fake production data
        prod_df = baseline_df.head(100).copy()
        X, y = preprocessor.transform(prod_df)
        assert isinstance(X, pd.DataFrame)
        assert X.isna().sum().sum() == 0

    def test_transform_handles_missing_values(self, preprocessor, baseline_df):
        prod_df = baseline_df.head(100).copy()
        prod_df.loc[0:10, "MonthlyCharges"] = np.nan
        X, y = preprocessor.transform(prod_df)
        assert X.isna().sum().sum() == 0

    def test_save_and_load(self, preprocessor, tmp_path):
        save_path = tmp_path / "test_preprocessor.joblib"
        preprocessor.save(save_path)
        loaded = DataPreprocessor.load(save_path)
        assert loaded.is_fitted is True
        assert len(loaded.label_encoders) == len(preprocessor.label_encoders)
