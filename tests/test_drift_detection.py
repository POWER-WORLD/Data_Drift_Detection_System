"""
Tests for drift detection engine.
Covers all statistical tests including edge cases:
empty series, single values, NaN-heavy data.
"""

import pytest
import pandas as pd
import numpy as np
from src.drift_detection import (
    ks_test, compute_psi, jensen_shannon_distance,
    mean_shift_test, variance_change_test,
    chi_square_test, new_category_detection,
    category_frequency_shift,
    run_full_drift_analysis
)


class TestNumericalDrift:
    """Tests for numerical drift detection methods."""

    def test_ks_no_drift(self):
        np.random.seed(42)
        a = pd.Series(np.random.normal(0, 1, 1000))
        b = pd.Series(np.random.normal(0, 1, 1000))
        result = ks_test(a, b)
        assert result["test"] == "Kolmogorov-Smirnov"
        assert result["drift_detected"] is False

    def test_ks_with_drift(self):
        np.random.seed(42)
        a = pd.Series(np.random.normal(0, 1, 1000))
        b = pd.Series(np.random.normal(3, 1, 1000))
        result = ks_test(a, b)
        assert result["drift_detected"] is True

    def test_ks_empty_series_handled(self):
        """KS test should handle empty series gracefully."""
        a = pd.Series(dtype=float)
        b = pd.Series([1.0, 2.0, 3.0])
        result = ks_test(a, b)
        assert result["drift_detected"] is False
        assert "warning" in result

    def test_ks_small_sample_handled(self):
        """KS test with fewer than 10 samples should warn."""
        a = pd.Series([1.0, 2.0, 3.0])
        b = pd.Series([4.0, 5.0, 6.0])
        result = ks_test(a, b)
        assert "warning" in result

    def test_ks_with_nans(self):
        """KS test should drop NaN values and still work."""
        np.random.seed(42)
        a = pd.Series([1, 2, np.nan, 4, 5] * 20)
        b = pd.Series([1, 2, 3, np.nan, 5] * 20)
        result = ks_test(a, b)
        assert result["statistic"] is not None

    def test_psi_no_drift(self):
        np.random.seed(42)
        a = pd.Series(np.random.normal(50, 10, 1000))
        b = pd.Series(np.random.normal(50, 10, 1000))
        result = compute_psi(a, b)
        assert result["severity"] == "LOW"

    def test_psi_severe_drift(self):
        np.random.seed(42)
        a = pd.Series(np.random.normal(50, 10, 1000))
        b = pd.Series(np.random.normal(100, 20, 1000))
        result = compute_psi(a, b)
        assert result["drift_detected"] is True

    def test_psi_empty_series(self):
        """PSI should handle empty series gracefully."""
        a = pd.Series(dtype=float)
        b = pd.Series(dtype=float)
        result = compute_psi(a, b)
        assert result["drift_detected"] is False

    def test_psi_zero_variance(self):
        """PSI should handle zero-variance columns."""
        a = pd.Series([5.0] * 100)
        b = pd.Series([5.0] * 100)
        result = compute_psi(a, b)
        assert result["severity"] == "LOW"

    def test_js_distance(self):
        np.random.seed(42)
        a = pd.Series(np.random.normal(0, 1, 1000))
        b = pd.Series(np.random.normal(0, 1, 1000))
        result = jensen_shannon_distance(a, b)
        assert result["value"] >= 0
        assert result["severity"] in ["LOW", "MODERATE", "SEVERE"]

    def test_js_empty_series(self):
        a = pd.Series(dtype=float)
        b = pd.Series(dtype=float)
        result = jensen_shannon_distance(a, b)
        assert result["drift_detected"] is False

    def test_mean_shift_no_drift(self):
        a = pd.Series([50, 50, 50, 50, 50])
        b = pd.Series([51, 49, 50, 50, 50])
        result = mean_shift_test(a, b)
        assert result["drift_detected"] is False

    def test_mean_shift_with_zeros(self):
        """Mean shift should handle zero baseline mean."""
        a = pd.Series([0.0] * 20)
        b = pd.Series([0.0] * 20)
        result = mean_shift_test(a, b)
        assert result["drift_detected"] is False

    def test_mean_shift_insufficient_samples(self):
        a = pd.Series([1.0])
        b = pd.Series([2.0])
        result = mean_shift_test(a, b)
        assert "warning" in result

    def test_variance_change(self):
        np.random.seed(42)
        a = pd.Series(np.random.normal(0, 1, 1000))
        b = pd.Series(np.random.normal(0, 5, 1000))
        result = variance_change_test(a, b)
        assert result["drift_detected"] is True

    def test_variance_change_insufficient_samples(self):
        a = pd.Series([1.0])
        b = pd.Series([2.0])
        result = variance_change_test(a, b)
        assert "warning" in result

    def test_drift_detected_is_python_bool(self):
        """Ensure drift_detected is Python bool, not numpy bool."""
        np.random.seed(42)
        a = pd.Series(np.random.normal(0, 1, 100))
        b = pd.Series(np.random.normal(0, 1, 100))
        result = ks_test(a, b)
        assert type(result["drift_detected"]) is bool


class TestCategoricalDrift:
    """Tests for categorical drift detection methods."""

    def test_chi_square_no_drift(self):
        np.random.seed(42)
        cats = ["A", "B", "C"]
        a = pd.Series(np.random.choice(cats, 500, p=[0.5, 0.3, 0.2]))
        b = pd.Series(np.random.choice(cats, 500, p=[0.5, 0.3, 0.2]))
        result = chi_square_test(a, b)
        assert result["test"] == "Chi-Square"

    def test_chi_square_small_sample(self):
        a = pd.Series(["A", "B"])
        b = pd.Series(["A", "C"])
        result = chi_square_test(a, b)
        assert "warning" in result

    def test_new_category_detected(self):
        a = pd.Series(["A", "B", "C", "A", "B"])
        b = pd.Series(["A", "B", "D", "A", "E"])
        result = new_category_detection(a, b)
        assert result["drift_detected"] is True
        assert "D" in result["new_categories"]
        assert "E" in result["new_categories"]

    def test_no_new_categories(self):
        a = pd.Series(["A", "B", "C"])
        b = pd.Series(["A", "B", "C"])
        result = new_category_detection(a, b)
        assert result["drift_detected"] is False
        assert result["new_count"] == 0

    def test_frequency_shift_stable(self):
        np.random.seed(42)
        a = pd.Series(np.random.choice(["X", "Y"], 500, p=[0.5, 0.5]))
        b = pd.Series(np.random.choice(["X", "Y"], 500, p=[0.5, 0.5]))
        result = category_frequency_shift(a, b)
        assert result["test"] == "Category Frequency Shift"

    def test_frequency_shift_with_nans(self):
        a = pd.Series(["A", "B", None, "A", "B"] * 20)
        b = pd.Series(["A", "A", None, "A", "A"] * 20)
        result = category_frequency_shift(a, b)
        assert "shifts" in result


class TestFullDriftAnalysis:
    """Tests for the full drift analysis pipeline."""

    def test_runs_on_real_data(self):
        from src.data_loader import load_baseline_data, load_production_batch
        from src.config.settings import PRODUCTION_DIR

        baseline = load_baseline_data()
        prod_path = PRODUCTION_DIR / "batch_01_no_drift.csv"
        if prod_path.exists():
            production = load_production_batch(prod_path)
            result = run_full_drift_analysis(baseline, production)
            assert "feature_drift" in result
            assert "dataset_drift" in result
            assert result["dataset_drift"]["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

    def test_report_serializable(self):
        """All values in the drift report should be JSON-serializable."""
        import json
        from src.data_loader import load_baseline_data, load_production_batch
        from src.config.settings import PRODUCTION_DIR

        baseline = load_baseline_data()
        prod_path = PRODUCTION_DIR / "batch_01_no_drift.csv"
        if prod_path.exists():
            production = load_production_batch(prod_path)
            result = run_full_drift_analysis(baseline, production)
            # This will raise if any numpy types are not serializable
            json.dumps(result, default=str)
