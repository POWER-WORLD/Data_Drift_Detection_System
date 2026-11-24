"""
Comprehensive edge case tests for the drift detection statistical engine (src/drift_detection.py).

Covers edge conditions:
- Identical constant distributions
- Single distinct value series
- Disjoint categorical spaces
- Zero-variance distributions
- Inverted probability density distributions
- Extreme outliers and heavy-tailed distributions
- Symmetry invariants (Jensen-Shannon distance)
- Numerical boundary stability under zero denominators
"""

import numpy as np
import pandas as pd
import pytest

from src.drift_detection import (
    category_frequency_shift,
    chi_square_test,
    compute_psi,
    jensen_shannon_distance,
    ks_test,
    mean_shift_test,
    new_category_detection,
    variance_change_test,
)


class TestDriftNumericalEdgeCases:
    """Edge case tests for numerical drift detection methods."""

    def test_ks_identical_constant_distributions(self) -> None:
        """Verify KS test reports no drift when both series are identically constant."""
        a = pd.Series([42.0] * 100)
        b = pd.Series([42.0] * 100)

        result = ks_test(a, b)
        assert result["drift_detected"] is False
        assert result["statistic"] == 0.0
        assert result["p_value"] == 1.0

    def test_ks_distinct_constant_distributions(self) -> None:
        """Verify KS test detects complete separation between two distinct constant series."""
        a = pd.Series([10.0] * 100)
        b = pd.Series([20.0] * 100)

        result = ks_test(a, b)
        assert result["drift_detected"] is True
        assert result["statistic"] == 1.0
        assert result["p_value"] < 0.001

    def test_ks_extreme_outliers(self) -> None:
        """Verify KS test behaves stably in presence of extreme heavy-tailed outliers."""
        np.random.seed(42)
        a = pd.Series(np.random.normal(0, 1, 500))
        b = a.copy()
        # Add extreme outliers
        b.iloc[0] = 1e9
        b.iloc[1] = -1e9

        result = ks_test(a, b)
        assert isinstance(result["drift_detected"], bool)
        assert result["statistic"] is not None

    def test_psi_identical_distributions(self) -> None:
        """Verify PSI calculation yields 0.0 (or negligible) for identical series."""
        np.random.seed(42)
        values = pd.Series(np.random.normal(50, 15, 1000))

        result = compute_psi(values, values)
        assert result["value"] < 0.001
        assert result["severity"] == "LOW"
        assert result["drift_detected"] is False

    def test_psi_all_identical_constant_values(self) -> None:
        """Verify PSI handles series where all values in both are identical constants."""
        a = pd.Series([100.0] * 200)
        b = pd.Series([100.0] * 200)

        result = compute_psi(a, b)
        assert result["severity"] == "LOW"
        assert result["drift_detected"] is False

    def test_psi_high_zero_proportion(self) -> None:
        """Verify PSI behaves reliably when majority of data is zero."""
        a = pd.Series([0.0] * 900 + list(range(100)))
        b = pd.Series([0.0] * 900 + list(range(100)))

        result = compute_psi(a, b)
        assert result["severity"] == "LOW"
        assert result["drift_detected"] is False

    def test_js_distance_symmetry(self) -> None:
        """Verify Jensen-Shannon distance satisfies mathematical symmetry JS(P, Q) == JS(Q, P)."""
        np.random.seed(42)
        a = pd.Series(np.random.normal(10, 2, 500))
        b = pd.Series(np.random.normal(15, 3, 500))

        res_ab = jensen_shannon_distance(a, b)
        res_ba = jensen_shannon_distance(b, a)

        assert abs(res_ab["value"] - res_ba["value"]) < 1e-4

    def test_js_distance_disjoint_distributions(self) -> None:
        """Verify Jensen-Shannon distance approaches upper bound for non-overlapping distributions."""
        a = pd.Series([1.0, 2.0, 3.0] * 100)
        b = pd.Series([100.0, 200.0, 300.0] * 100)

        result = jensen_shannon_distance(a, b)
        assert result["drift_detected"] is True
        assert result["severity"] == "SEVERE"
        assert result["value"] > 0.8

    def test_mean_shift_both_zero_means(self) -> None:
        """Verify mean shift handles case where both distributions have zero mean."""
        a = pd.Series([-5.0, 0.0, 5.0] * 20)
        b = pd.Series([-5.0, 0.0, 5.0] * 20)

        result = mean_shift_test(a, b)
        assert result["drift_detected"] is False
        assert result["pct_change"] == 0.0

    def test_mean_shift_negative_means(self) -> None:
        """Verify mean shift accurately evaluates percent change for negative distributions."""
        a = pd.Series([-100.0] * 20)
        b = pd.Series([-150.0] * 20)

        result = mean_shift_test(a, b)
        assert result["drift_detected"] is True
        assert abs(result["pct_change"] - 50.0) < 0.1

    def test_variance_change_both_zero_variance(self) -> None:
        """Verify variance test handles zero-variance in both inputs without zero division."""
        a = pd.Series([10.0] * 30)
        b = pd.Series([10.0] * 30)

        result = variance_change_test(a, b)
        assert result["drift_detected"] is False
        assert result["pct_change"] == 0.0
        assert result["baseline_var"] == 0.0
        assert result["production_var"] == 0.0


class TestDriftCategoricalEdgeCases:
    """Edge case tests for categorical drift detection methods."""

    def test_chi_square_identical_frequencies(self) -> None:
        """Verify chi-square test on identical categorical frequency distributions."""
        a = pd.Series(["Alpha", "Beta", "Gamma"] * 100)
        b = pd.Series(["Alpha", "Beta", "Gamma"] * 100)

        result = chi_square_test(a, b)
        assert result["drift_detected"] is False
        assert result["statistic"] == 0.0
        assert result["p_value"] == 1.0

    def test_chi_square_completely_disjoint_categories(self) -> None:
        """Verify chi-square handles disjoint category domains without crashing."""
        a = pd.Series(["A", "B", "C"] * 50)
        b = pd.Series(["X", "Y", "Z"] * 50)

        result = chi_square_test(a, b)
        assert result["test"] == "Chi-Square"
        assert isinstance(result["drift_detected"], bool)

    def test_chi_square_shifted_distribution(self) -> None:
        """Verify chi-square detects drift when category distribution shifts heavily."""
        a = pd.Series(["A"] * 90 + ["B"] * 10)
        b = pd.Series(["A"] * 10 + ["B"] * 90)

        result = chi_square_test(a, b)
        assert result["drift_detected"] is True
        assert result["p_value"] is not None
        assert result["p_value"] < 0.05

    def test_new_category_multiple_unseen(self) -> None:
        """Verify new category detection accurately catalogs multiple new unseen labels."""
        a = pd.Series(["Standard", "Gold"] * 50)
        b = pd.Series(["Standard", "Platinum", "Diamond", "Gold"] * 50)

        result = new_category_detection(a, b)
        assert result["drift_detected"] is True
        assert result["new_count"] == 2
        assert set(result["new_categories"]) == {"Platinum", "Diamond"}

    def test_new_category_empty_production_series(self) -> None:
        """Verify new category detection handles empty series gracefully."""
        a = pd.Series(["A", "B", "C"])
        b = pd.Series([], dtype=str)

        result = new_category_detection(a, b)
        assert result["drift_detected"] is False
        assert result["new_count"] == 0

    def test_category_frequency_shift_inverted_proportions(self) -> None:
        """Verify category frequency shift flags massive percentage changes across classes."""
        a = pd.Series(["A"] * 900 + ["B"] * 100)
        b = pd.Series(["A"] * 100 + ["B"] * 900)

        result = category_frequency_shift(a, b)
        assert result["drift_detected"] is True
        assert len(result["shifts"]) == 2
        assert "A" in result["shifts"]
        assert result["shifts"]["A"]["baseline_pct"] == 90.0
        assert result["shifts"]["A"]["production_pct"] == 10.0
