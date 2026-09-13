"""
===============================================================================
Categorical Encoding and Column Synchronization (src/encoding.py)
===============================================================================
This module handles the encoding of categorical variables (One-Hot Encoding)
and ensures that the feature space is perfectly synchronized between the 
training and testing datasets.
"""

import pandas as pd
from typing import Tuple, List

def drop_unused_object_columns(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    columns_to_keep: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify all object (string) columns and drop those not specified in columns_to_keep.
    """
    obj_cols_train = train_df.select_dtypes(include=['object']).columns.tolist()
    obj_cols_test  = test_df.select_dtypes(include=['object']).columns.tolist()
    
    to_drop_train = [c for c in obj_cols_train if c not in columns_to_keep]
    to_drop_test  = [c for c in obj_cols_test if c not in columns_to_keep]
    
    print(f"[INFO] Dropping unused object columns in Train: {to_drop_train}")
    print(f"[INFO] Dropping unused object columns in Test: {to_drop_test}")
    
    train_df.drop(columns=to_drop_train, inplace=True, errors='ignore')
    test_df.drop(columns=to_drop_test, inplace=True, errors='ignore')
    
    return train_df, test_df

def apply_one_hot_encoding(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    columns_to_encode: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply One-Hot Encoding to the specified categorical columns.
    """
    encode_train = [col for col in columns_to_encode if col in train_df.columns]
    encode_test = [col for col in columns_to_encode if col in test_df.columns]
    
    if encode_train:
        train_df = pd.get_dummies(train_df, columns=encode_train, drop_first=False)
    if encode_test:
        test_df = pd.get_dummies(test_df, columns=encode_test, drop_first=False)
        
    return train_df, test_df

def synchronize_columns(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    target_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synchronize columns between train and test sets to ensure they have the 
    exact same features after encoding, aligning their dimensions for modeling.
    """
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    common_cols = list(train_cols.intersection(test_cols))

    if target_col in common_cols:
        common_cols.remove(target_col)

    # Taking the intersection is right -- a feature the model cannot see at
    # prediction time is useless -- but doing it silently is not. A one-hot
    # category present in only one of the two frames disappears from both here,
    # and that is worth seeing rather than discovering later as a shape mismatch.
    train_only = sorted(train_cols - test_cols - {target_col})
    test_only = sorted(test_cols - train_cols)
    if train_only:
        print(f"[WARN] Dropping {len(train_only)} train-only column(s): {train_only}")
    if test_only:
        print(f"[WARN] Dropping {len(test_only)} test-only column(s): {test_only}")

    # Sort for consistent column ordering
    sorted_features = sorted(common_cols)
    
    # Target must strictly remain in train_df
    train_df = train_df[[target_col] + sorted_features]
    test_df = test_df[sorted_features]
    
    return train_df, test_df

def run_encoding_pipeline(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    target_col: str = "임신 성공 여부",
    columns_to_encode: List[str] = ["시술 유형_원본", "시술 당시 나이"]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the complete encoding and synchronization pipeline.

    Note that columns_to_encode doubles as the keep-list: every other object
    column is dropped in step 1. "시술 당시 나이" is on it because age is the
    single most predictive variable in the dataset -- success rate runs 0.323
    to 0.118 across its 7 brackets -- and this is the only place it enters the
    feature space. It used to reach the model as a target encoding built in
    features.py, which was replaced by this one-hot; see
    target_encoding_study.py.
    """
    print("[INFO] Starting Categorical Encoding Pipeline...")
    
    # 1. Drop unused text columns
    train_df, test_df = drop_unused_object_columns(
        train_df, test_df, columns_to_keep=columns_to_encode
    )
    
    # 2. Apply One-Hot Encoding
    train_df, test_df = apply_one_hot_encoding(
        train_df, test_df, columns_to_encode=columns_to_encode
    )
    
    # 3. Synchronize features
    train_df, test_df = synchronize_columns(
        train_df, test_df, target_col=target_col
    )
    
    print(f"[INFO] After Encoding & Sync, Train shape: {train_df.shape}")
    print(f"[INFO] After Encoding & Sync, Test shape : {test_df.shape}")
    
    # 4. Validation: Check for remaining object columns
    rem_obj_train = train_df.select_dtypes(include=['object']).columns.tolist()
    rem_obj_test = test_df.select_dtypes(include=['object']).columns.tolist()
    
    if rem_obj_train or rem_obj_test:
        print(f"[WARN] Object columns remaining! Train: {rem_obj_train}, Test: {rem_obj_test}")
    else:
        print("[INFO] Success: All features are successfully encoded to numeric.")
        
    return train_df, test_df