"""
Preprocessing module for the Data Drift Detection System.

Handles data cleaning, numeric imputation, categorical mode imputation,
label encoding, standard scaling, and artifact serialization.
Maintains natural feature ordering consistent with trained model estimators.
Follows PEP 8 and Global Naming Standardization.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config.settings import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from src.utils import setup_logger

logger = setup_logger(__name__)

# Canonical baseline feature sequence matching models/best_model.joblib
CANONICAL_FEATURE_ORDER: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


class DataPreprocessor:
    """
    End-to-end data preprocessor designed for ML inference stability.

    Responsibilities:
    - ID column elimination
    - Infinite float sanitization (replacement with NaN)
    - Target label encoding
    - Missing value imputation using fitted baseline statistics (median/mode)
    - Robust label encoding with graceful fallback for unseen production categories
    - Standard scaling for continuous numerical features
    - Strict feature sequence preservation matching trained model estimators
    """

    def __init__(self) -> None:
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: StandardScaler = StandardScaler()
        self.target_encoder: LabelEncoder = LabelEncoder()
        self.is_fitted: bool = False
        self._numerical_features: List[str] = list(NUMERICAL_FEATURES)
        self._categorical_features: List[str] = list(CATEGORICAL_FEATURES)
        self._baseline_medians: Dict[str, float] = {}
        self._baseline_modes: Dict[str, str] = {}
        self.feature_names_: List[str] = list(CANONICAL_FEATURE_ORDER)

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit preprocessor parameters on baseline data and return transformed feature matrix and target.

        Args:
            df: Raw baseline training DataFrame.

        Returns:
            Tuple of (X_processed DataFrame, y_encoded Series).
        """
        logger.info("Fitting data preprocessor on baseline dataset...")
        processed_df = df.copy()

        # Capture feature order prior to column modifications
        natural_features = [
            c for c in processed_df.columns if c not in (ID_COLUMN, TARGET_COLUMN)
        ]
        if natural_features:
            self.feature_names_ = natural_features
        else:
            self.feature_names_ = list(CANONICAL_FEATURE_ORDER)

        # Remove primary key identifier if present
        if ID_COLUMN in processed_df.columns:
            processed_df = processed_df.drop(columns=[ID_COLUMN])

        # Enforce numeric conversion for TotalCharges
        if "TotalCharges" in processed_df.columns:
            processed_df["TotalCharges"] = pd.to_numeric(
                processed_df["TotalCharges"], errors="coerce"
            )

        # Replace infinities with NaN in numerical columns
        for col in self._numerical_features:
            if col in processed_df.columns:
                numeric_series = pd.to_numeric(processed_df[col], errors="coerce")
                processed_df[col] = numeric_series.replace([np.inf, -np.inf], np.nan)

        # Compute and persist baseline medians for numerical imputation
        for col in self._numerical_features:
            if col in processed_df.columns:
                col_median = processed_df[col].median()
                fill_val = float(col_median) if pd.notna(col_median) else 0.0
                self._baseline_medians[col] = fill_val
                if processed_df[col].isna().any():
                    processed_df[col] = processed_df[col].fillna(fill_val)
                    logger.info("Imputed %s missing values with median=%.2f", col, fill_val)

        # Compute and persist baseline modes for categorical imputation
        for col in self._categorical_features:
            if col in processed_df.columns:
                mode_series = processed_df[col].dropna().mode()
                mode_val = str(mode_series.iloc[0]) if len(mode_series) > 0 else "Unknown"
                self._baseline_modes[col] = mode_val
                if processed_df[col].isna().any():
                    processed_df[col] = processed_df[col].fillna(mode_val)
                    logger.info("Imputed %s missing values with mode=%s", col, mode_val)

        # Fit and encode target variable
        if TARGET_COLUMN not in processed_df.columns:
            raise KeyError(f"Mandatory target column '{TARGET_COLUMN}' missing from baseline data.")

        y_encoded = self.target_encoder.fit_transform(processed_df[TARGET_COLUMN].astype(str))
        processed_df = processed_df.drop(columns=[TARGET_COLUMN])

        # Fit and encode categorical features
        for col in self._categorical_features:
            if col in processed_df.columns:
                encoder = LabelEncoder()
                processed_df[col] = encoder.fit_transform(processed_df[col].astype(str))
                self.label_encoders[col] = encoder

        # Fit and transform numerical standard scaler
        scaled_values = self.scaler.fit_transform(processed_df[self._numerical_features])
        processed_df[self._numerical_features] = scaled_values

        # Ensure consistent column ordering matching self.feature_names_
        processed_df = processed_df[self.feature_names_]

        self.is_fitted = True
        logger.info(
            "Preprocessor fit complete: %d samples, %d features",
            processed_df.shape[0],
            processed_df.shape[1],
        )
        return processed_df, pd.Series(y_encoded, name=TARGET_COLUMN)

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Transform production inference data using fitted preprocessor artifacts.

        Gracefully handles unseen categories, missing features, and missing values.

        Args:
            df: Raw production DataFrame.

        Returns:
            Tuple of (X_processed DataFrame, y_encoded Series or None).

        Raises:
            RuntimeError: If transform is called prior to fit_transform().
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call fit_transform() first.")

        logger.info("Transforming production batch features...")
        processed_df = df.copy()

        # Remove ID column if present
        if ID_COLUMN in processed_df.columns:
            processed_df = processed_df.drop(columns=[ID_COLUMN])

        # Convert TotalCharges if present
        if "TotalCharges" in processed_df.columns:
            processed_df["TotalCharges"] = pd.to_numeric(
                processed_df["TotalCharges"], errors="coerce"
            )

        # Ensure all numerical features exist and infinities are replaced with NaN
        missing_imputations: Dict[str, int] = {}
        for col in self._numerical_features:
            if col not in processed_df.columns:
                fill_val = self._baseline_medians.get(col, 0.0)
                processed_df[col] = fill_val
                missing_imputations[col] = len(processed_df)
            else:
                numeric_series = pd.to_numeric(processed_df[col], errors="coerce")
                sanitized_series = numeric_series.replace([np.inf, -np.inf], np.nan)
                num_missing = int(sanitized_series.isna().sum())
                if num_missing > 0:
                    fill_val = self._baseline_medians.get(col, 0.0)
                    sanitized_series = sanitized_series.fillna(fill_val)
                    missing_imputations[col] = num_missing
                processed_df[col] = sanitized_series

        if missing_imputations:
            logger.warning(
                "Missing numerical values imputed with baseline medians: %s",
                missing_imputations,
            )

        # Ensure all categorical features exist and missing values are filled with mode
        for col in self._categorical_features:
            fallback_mode = self._baseline_modes.get(col, "Unknown")
            if col not in processed_df.columns:
                processed_df[col] = fallback_mode
            else:
                if processed_df[col].isna().any():
                    processed_df[col] = processed_df[col].fillna(fallback_mode)

        # Encode target if present in production batch
        y_encoded: Optional[pd.Series] = None
        if TARGET_COLUMN in processed_df.columns:
            target_series = processed_df[TARGET_COLUMN].dropna().astype(str)
            if len(target_series) > 0:
                known_classes = set(self.target_encoder.classes_)
                safe_target = processed_df[TARGET_COLUMN].astype(str).apply(
                    lambda val, kc=known_classes, default_cls=self.target_encoder.classes_[0]: (
                        val if val in kc else default_cls
                    )
                )
                y_encoded = pd.Series(
                    self.target_encoder.transform(safe_target),
                    name=TARGET_COLUMN,
                )
            processed_df = processed_df.drop(columns=[TARGET_COLUMN])

        # Transform categorical features with fallback for unseen classes
        for col in self._categorical_features:
            if col in self.label_encoders:
                encoder = self.label_encoders[col]
                known_classes = set(encoder.classes_)
                default_class = encoder.classes_[0]

                safe_cat_series = processed_df[col].astype(str).apply(
                    lambda val, kc=known_classes, d_cls=default_class: val if val in kc else d_cls
                )
                processed_df[col] = encoder.transform(safe_cat_series)

        # Scale numerical features
        processed_df[self._numerical_features] = self.scaler.transform(
            processed_df[self._numerical_features]
        )

        # Align column order exactly with canonical feature order / self.feature_names_
        feature_order = getattr(self, "feature_names_", None) or CANONICAL_FEATURE_ORDER
        processed_df = processed_df[feature_order]

        logger.info(
            "Production batch transformed successfully: %d samples, %d features",
            processed_df.shape[0],
            processed_df.shape[1],
        )
        return processed_df, y_encoded

    def save(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """
        Serialize and persist the fitted preprocessor instance to disk.

        Args:
            filepath: Destination file path. Defaults to MODELS_DIR / 'preprocessor.joblib'.
        """
        target_path = Path(filepath) if filepath is not None else (MODELS_DIR / "preprocessor.joblib")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target_path)
        logger.info("Preprocessor artifact successfully saved to %s", target_path)

    @classmethod
    def load(cls, filepath: Optional[Union[str, Path]] = None) -> "DataPreprocessor":
        """
        Load and deserialize a fitted preprocessor instance from disk.

        Args:
            filepath: Path to the serialized preprocessor artifact.

        Returns:
            Fitted DataPreprocessor instance.

        Raises:
            FileNotFoundError: If the specified artifact path does not exist.
        """
        target_path = Path(filepath) if filepath is not None else (MODELS_DIR / "preprocessor.joblib")
        if not target_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at: {target_path}")

        instance: "DataPreprocessor" = joblib.load(target_path)
        # Ensure feature_names_ exists on legacy instances
        if not hasattr(instance, "feature_names_") or not instance.feature_names_:
            instance.feature_names_ = list(CANONICAL_FEATURE_ORDER)
        logger.info("Loaded preprocessor artifact from %s", target_path)
        return instance
