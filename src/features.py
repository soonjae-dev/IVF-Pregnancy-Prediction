"""
===============================================================================
Domain-Based Feature Engineering (src/features.py)
===============================================================================
This module contains functions for generating new features based on domain 
knowledge, handling string-to-numeric conversions, and performing target encoding.
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

def create_age_success_rate(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform target encoding: Map the average pregnancy success rate grouped by age.
    The mapping is strictly derived from the training set to prevent data leakage.
    """
    if ("시술 당시 나이" in train_df.columns) and (target_col in train_df.columns):
        # Calculate mean success rate grouped by age from the training set
        age_success_rate = train_df.groupby("시술 당시 나이")[target_col].mean()
        
        # Apply the derived mapping to both train and test sets
        train_df["연령대 평균 임신 성공률"] = train_df["시술 당시 나이"].map(age_success_rate)
        if "시술 당시 나이" in test_df.columns:
            test_df["연령대 평균 임신 성공률"] = test_df["시술 당시 나이"].map(age_success_rate)
            
    return train_df, test_df

def run_feature_engineering(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "임신 성공 여부") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the full feature engineering pipeline.
    """
    print("[INFO] Starting feature engineering pipeline...")
    
    train_df, test_df = clean_count_columns(train_df, test_df)
    train_df, test_df = copy_original_features(train_df, test_df)
    train_df, test_df = create_success_rate_feature(train_df, test_df)
    train_df, test_df = create_age_success_rate(train_df, test_df, target_col)
    
    print(f"[INFO] After Feature Engineering, Train shape: {train_df.shape}")
    print(f"[INFO] After Feature Engineering, Test shape : {test_df.shape}")
    
    return train_df, test_df