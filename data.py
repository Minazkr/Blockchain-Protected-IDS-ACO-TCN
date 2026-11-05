# data.py
# -------------------------------------------------
# Data loading, cleaning, and chronological splitting
# for CICIDS2017 dataset
# -------------------------------------------------

import pandas as pd
import numpy as np
from typing import Tuple, Dict


def load_cicids(data_path: str) -> pd.DataFrame:
    """
    Load CICIDS2017 from a folder or zip file.
    - Reads all CSV files
    - Standardizes column names
    - Maps attack names to standard classes
    """
    
    raise NotImplementedError("کد بارگذاری CICIDS2017 اینجا قرار می‌گیرد")


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean the dataset:
    - Replace inf values with NaN
    - Remove exact duplicate rows
    - Return cleaning statistics
    """
    info = {
        "rows_before": len(df)
    }

    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Remove exact duplicates
    duplicate_mask = df.duplicated()
    info["duplicates_removed"] = int(duplicate_mask.sum())
    df = df.loc[~duplicate_mask].reset_index(drop=True)

    info["rows_after"] = len(df)
    return df, info


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    purge_seconds: float = 120.0
) -> pd.DataFrame:
    """
    Split CICIDS2017 chronologically per campaign.
    A purge gap is applied between splits to reduce leakage.
    """
   
    raise NotImplementedError("کد تقسیم زمانی اینجا قرار می‌گیرد")
