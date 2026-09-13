"""
===============================================================================
Data Splitting and Resampling (src/data_splitting.py)
===============================================================================
This module handles splitting the dataset into training and validation sets
and defining the cross-validation strategy for model evaluation.

No resampling happens here. The pipeline used to apply SMOTE to the whole
training set at this point; `resampling_study.py` measures what that is worth
on the representation this pipeline actually produces (-0.0004 ROC-AUC averaged
over the four base models, at 1.8-3.2x the training time), so it was removed
rather than kept as a default nobody had checked.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

def split_data(
    df: pd.DataFrame, 
    target_col: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separate features and target, then split into training and validation sets 
    using stratified sampling to maintain class distribution.
    """
    print(f"[INFO] Splitting data (test_size={test_size}, stratify=True)...")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"[INFO] Train shape: {X_train.shape}, Validation shape: {X_val.shape}")
    return X_train, X_val, y_train, y_val


def get_cv_strategy(
    n_splits: int = 5, 
    n_repeats: int = 2, 
    random_state: int = 42
) -> RepeatedStratifiedKFold:
    """
    Define and return the cross-validation strategy.
    RepeatedStratifiedKFold is ideal for imbalanced classification tasks.
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, 
        n_repeats=n_repeats, 
        random_state=random_state
    )
    
    print(f"[INFO] CV Strategy defined: {n_splits}-Fold Stratified CV, repeated {n_repeats} times.")
    return rskf


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = "임신 성공 여부"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, RepeatedStratifiedKFold]:
    """
    Execute the full data splitting pipeline.
    """
    # 1. Train/Validation Split
    X_train, X_val, y_train, y_val = split_data(df, target_col)

    # 2. Setup CV Strategy
    rskf = get_cv_strategy()

    return X_train, X_val, y_train, y_val, rskf