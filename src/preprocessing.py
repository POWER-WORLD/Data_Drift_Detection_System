"""
Preprocessing module for the Data Drift Detection System.
Handles data cleaning, encoding, scaling, and feature engineering.
Stores baseline statistics (medians) for consistent production transforms.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config.settings import (
    TARGET_COLUMN, ID_COLUMN, NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES, MODELS_DIR
)
from src.utils import setup_logger

logger = setup_logger(__name__)


class DataPreprocessor:
    """
    End-to-end data preprocessor that handles:
    - Dropping ID columns
    - Replacing infinities with NaN
    - Encoding the target variable
    - Label-encoding categorical features
    - Scaling numerical features
    - Handling missing values (stores baseline medians for production use)

    The preprocessor can be fitted on baseline data and then applied
    consistently to production batches.
    """

    def __init__(self):
        self.label_encoders: dict = {}
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
        self.is_fitted: bool = False
        self._numerical_features = NUMERICAL_FEATURES.copy()
        self._categorical_features = CATEGORICAL_FEATURES.copy()
        self._baseline_medians: dict = {}   # Stored for production imputation
        self._baseline_modes: dict = {}

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit the preprocessor on baseline data and return transformed features + target.

        Args:
            df: Raw baseline DataFrame.

        Returns:
            Tuple of (X_processed, y_encoded).
        """
        logger.info("Fitting preprocessor on baseline data")
        df = df.copy()

        # Drop ID column
        if ID_COLUMN in df.columns:
            df = df.drop(columns=[ID_COLUMN])

        # Handle TotalCharges conversion
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Replace infinities with NaN
        for col in self._numerical_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # Fill missing numerical values with median and store for production
        for col in self._numerical_features:
            if col in df.columns:
                median_val = df[col].median()
                self._baseline_medians[col] = float(median_val) if pd.notna(median_val) else 0.0
                if df[col].isna().any():
                    df[col] = df[col].fillna(self._baseline_medians[col])
                    logger.info(f"Filled {col} missing values with median={self._baseline_medians[col]:.2f}")

        # Fill missing categorical values with mode and store
        for col in self._categorical_features:
            if col in df.columns:
                mode_vals = df[col].mode()
                mode_val = mode_vals[0] if len(mode_vals) > 0 else "Unknown"
                self._baseline_modes[col] = mode_val
                if df[col].isna().any():
                    df[col] = df[col].fillna(mode_val)
                    logger.info(f"Filled {col} missing values with mode={mode_val}")

        # Encode target
        y = self.target_encoder.fit_transform(df[TARGET_COLUMN])
        df = df.drop(columns=[TARGET_COLUMN])

        # Encode categorical features
        for col in self._categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # Scale numerical features
        df[self._numerical_features] = self.scaler.fit_transform(df[self._numerical_features])

        self.is_fitted = True
        logger.info(f"Preprocessing complete: {df.shape[0]} rows, {df.shape[1]} features")

        return df, pd.Series(y, name=TARGET_COLUMN)

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Transform new production data using the fitted preprocessor.
        Uses baseline medians for imputation, handles unseen categories gracefully.

        Args:
            df: Raw production DataFrame.

        Returns:
            Tuple of (X_processed, y_encoded or None).
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform() first.")

        logger.info("Transforming production data")
        df = df.copy()

        # Drop ID column
        if ID_COLUMN in df.columns:
            df = df.drop(columns=[ID_COLUMN])

        # Handle TotalCharges conversion
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Replace infinities with NaN
        for col in self._numerical_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # Fill missing numerical values with baseline median (not 0)
        missing_counts = {}
        for col in self._numerical_features:
            if col in df.columns:
                n_missing = int(df[col].isna().sum())
                if n_missing > 0:
                    missing_counts[col] = n_missing
                    fill_val = self._baseline_medians.get(col, 0.0)
                    df[col] = df[col].fillna(fill_val)

        if missing_counts:
            logger.warning(f"Missing values in production data (filled with baseline medians): {missing_counts}")

        # Fill missing categorical values with baseline mode
        for col in self._categorical_features:
            if col in df.columns and df[col].isna().any():
                fill_val = self._baseline_modes.get(col, "Unknown")
                df[col] = df[col].fillna(fill_val)

        # Encode target if present
        y = None
        if TARGET_COLUMN in df.columns:
            known_classes = set(self.target_encoder.classes_)
            df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(
                lambda x: x if x in known_classes else self.target_encoder.classes_[0]
            )
            y = pd.Series(self.target_encoder.transform(df[TARGET_COLUMN]), name=TARGET_COLUMN)
            df = df.drop(columns=[TARGET_COLUMN])

        # Encode categorical features (handle unseen categories)
        for col in self._categorical_features:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                known_classes = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x, kc=known_classes, le_cls=le.classes_: x if x in kc else le_cls[0]
                )
                df[col] = le.transform(df[col])

        # Scale numerical features
        df[self._numerical_features] = self.scaler.transform(df[self._numerical_features])

        logger.info(f"Production data transformed: {df.shape[0]} rows, {df.shape[1]} features")
        return df, y

    def save(self, filepath: Optional[Path] = None) -> None:
        """Save the fitted preprocessor to disk."""
        filepath = filepath or (MODELS_DIR / "preprocessor.joblib")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "DataPreprocessor":
        """Load a fitted preprocessor from disk."""
        filepath = filepath or (MODELS_DIR / "preprocessor.joblib")
        preprocessor = joblib.load(filepath)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor
