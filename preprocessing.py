# preprocessing.py
# -------------------------------------------------
# Preprocessing for CICIDS2017
# (Everything is fitted only on the training set)
# -------------------------------------------------

import numpy as np
import pandas as pd
from typing import List


class Preprocessor:
    """
    This class prepares the numerical features for the model.

    Steps:
    1. Replace missing values with the median (from training data)
    2. Apply signed log transformation (helps with large feature ranges)
    3. Standardize features (zero mean, unit variance)
    """

    def __init__(self):
        self.feature_columns = None
        self.median = None
        self.mean = None
        self.std = None

    def fit(self, train_df: pd.DataFrame, feature_columns: List[str]):
        """
        Learn the required statistics from the training data only.
        """
        self.feature_columns = feature_columns
        data = train_df[feature_columns].copy()

        # Replace infinite values with NaN
        data = data.replace([np.inf, -np.inf], np.nan)

        # Calculate median for missing value imputation
        self.median = data.median()
        data = data.fillna(self.median)

        # Signed log transformation
        data = np.sign(data) * np.log1p(np.abs(data))

        # Calculate mean and std for standardization
        self.mean = data.mean()
        self.std = data.std()
        self.std = self.std.replace(0, 1.0)  # avoid division by zero

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply the same preprocessing steps to any dataset
        (train, validation, or test).
        """
        data = df[self.feature_columns].copy()
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(self.median)

        # Same log transformation
        data = np.sign(data) * np.log1p(np.abs(data))

        # Standardization
        data = (data - self.mean) / self.std

        return data.to_numpy(dtype=np.float32)
