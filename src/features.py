"""
===============================================================================
Domain-Based Feature Engineering (src/features.py)
===============================================================================
This module contains functions for generating new features based on domain
knowledge and handling string-to-numeric conversions.

It no longer performs target encoding; see the note above run_feature_engineering.
"""

import pandas as pd
from typing import Tuple

def clean_count_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove non-numeric characters from count-related columns and convert them to numeric.
    """
    count_cols = [
        "총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수",
        "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
        "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수"
    ]
    
    for col in count_cols:
        if col in train_df.columns:
            # Extract digits only and convert to numeric
            train_df[col] = pd.to_numeric(
                train_df[col].astype(str).str.replace(r"[^\d]", "", regex=True), 
                errors='coerce'
            )
        if col in test_df.columns:
            test_df[col] = pd.to_numeric(
                test_df[col].astype(str).str.replace(r"[^\d]", "", regex=True), 
                errors='coerce'
            )
            
    return train_df, test_df

def copy_original_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preserve the original values of specific categorical features before encoding.
    """
    if "시술 유형" in train_df.columns:
        train_df["시술 유형_원본"] = train_df["시술 유형"].copy()
    if "시술 유형" in test_df.columns:
        test_df["시술 유형_원본"] = test_df["시술 유형"].copy()
        
    return train_df, test_df

def create_success_rate_feature(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate the pregnancy success rate per attempt.
    Uses Laplace smoothing (+1) to avoid division by zero errors.
    """
    if ("총 시술 횟수" in train_df.columns) and ("총 임신 횟수" in train_df.columns):
        train_df["임신 시도 대비 성공률"] = (train_df["총 임신 횟수"] + 1) / (train_df["총 시술 횟수"] + 1)
        test_df["임신 시도 대비 성공률"] = (test_df["총 임신 횟수"] + 1) / (test_df["총 시술 횟수"] + 1)
        
    return train_df, test_df

# ---------------------------------------------------------------------------
# Removed: create_age_success_rate
#
# This module used to target-encode the age bracket -- group the training set
# on "시술 당시 나이", take the mean of the label, and map it back -- producing
# the feature "연령대 평균 임신 성공률". Every training row's own label sat
# inside the group mean it received, and the mapping was fitted once on the
# whole training set, before any cross-validation split.
#
# target_encoding_study.py measures all of that. The leak is worth -0.0005
# ROC-AUC, which is to say nothing: the bracket takes 7 values and the smallest
# group holds 329 rows, so no single row moves its own group mean.
#
# What the study also showed is that the feature was load-bearing for a reason
# that had nothing to do with the labels. "시술 당시 나이" is an object column,
# so encoding.py dropped it, and this feature was the only path by which age
# reached the model at all -- removing it costs 0.0148 AUC. Replacing it with a
# plain one-hot of the same 7 brackets scores within 0.0004 of it.
#
# So the bracket is now one-hot encoded in encoding.py and the target encoding
# is gone. Same accuracy, and a whole category of error no longer has to be
# argued about.
# ---------------------------------------------------------------------------

def run_feature_engineering(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the full feature engineering pipeline.

    The target column is no longer a parameter: with the target encoding gone,
    nothing in this module reads the label, which is the point.
    """
    print("[INFO] Starting feature engineering pipeline...")

    train_df, test_df = clean_count_columns(train_df, test_df)
    train_df, test_df = copy_original_features(train_df, test_df)
    train_df, test_df = create_success_rate_feature(train_df, test_df)

    print(f"[INFO] After Feature Engineering, Train shape: {train_df.shape}")
    print(f"[INFO] After Feature Engineering, Test shape : {test_df.shape}")
    
    return train_df, test_df