"""
===============================================================================
Data Splitting and Resampling (src/data_splitting.py)
===============================================================================
This module handles splitting the dataset into training and validation sets,
applying SMOTE to address class imbalance, and defining the cross-validation 
strategy for robust model evaluation.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE

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


def apply_smote(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to the training 
    data to synthesize samples for the minority class.
    
    Note: SMOTE should strictly be applied ONLY to the training set to prevent 
    data leakage into the validation set.
    """
    print("[INFO] Applying SMOTE to balance the training data...")
    
    smote = SMOTE(random_state=random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    
    class_counts = np.bincount(y_train_sm.astype(int))
    print(f"[INFO] After SMOTE, Train shape: {X_train_sm.shape}")
    print(f"[INFO] Class distribution after SMOTE: {class_counts}")
    
    return X_train_sm, y_train_sm


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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, RepeatedStratifiedKFold]:
    """
    Execute the full data splitting and resampling pipeline.
    """
    # 1. Train/Validation Split
    X_train, X_val, y_train, y_val = split_data(df, target_col)
    
    # 2. Apply SMOTE
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    
    # 3. Setup CV Strategy
    rskf = get_cv_strategy()
    
    return X_train, X_val, y_train, y_val, X_train_sm, y_train_sm, rskf