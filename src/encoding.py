"""
===============================================================================
Categorical Encoding and Column Synchronization (src/encoding.py)
===============================================================================
This module handles the encoding of categorical variables (One-Hot Encoding)
and ensures that the feature space is perfectly synchronized between the 
training and testing datasets.
"""

import re

import pandas as pd
from typing import Tuple, List


def sanitize_column_names(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Make one-hot column names safe for LightGBM.

    LightGBM refuses feature names containing []{}":, with "Do not support
    special JSON characters in feature name" -- and one-hot names inherit
    whatever was in the category, so "특정 시술 유형" alone produces eleven
    offenders (ICSI:IVF, IVF / AH:ICSI / AH, ...) and "배아 생성 주요 이유"
    nine more, from comma-separated multi-labels.

    This bit only shows up where a DataFrame reaches the model with its names
    attached: main.py's StackingClassifier does, while anything that goes
    through a Pipeline with a scaler, or through .values, hands over a bare
    array and never trips it. So it stayed invisible until the discarded
    columns were added back.

    Offending characters become underscores, and any collisions that creates
    are de-duplicated. Both frames are renamed with the same rule, which keeps
    them aligned.
    """
    def clean(names):
        cleaned = [re.sub(r'[\[\]{}":,]', "_", str(name)) for name in names]
        return list(pd.io.common.dedup_names(cleaned, is_potential_multiindex=False))

    train_df.columns = clean(train_df.columns)
    test_df.columns = clean(test_df.columns)
    return train_df, test_df

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
    columns_to_encode: List[str] = [
        "시술 유형_원본",           # procedure type (the features.py copy)
        "시술 당시 나이",           # patient age band
        "시술 시기 코드",           # procedure period code
        "특정 시술 유형",           # specific procedure type
        "배란 유도 유형",           # ovulation induction type
        "배아 생성 주요 이유",       # main reason for embryo creation
        "클리닉 내 총 시술 횟수",    # total procedures at this clinic
        "난자 출처",                # egg source
        "정자 출처",                # sperm source
        "난자 기증자 나이",          # egg donor age band
        "정자 기증자 나이",          # sperm donor age band
    ]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the complete encoding and synchronization pipeline.

    columns_to_encode doubles as the keep-list: every object column not on it is
    dropped in step 1. For most of this project that list held one name, and
    everything else -- age, procedure type, donor ages, egg and sperm source --
    was silently thrown away.

    That was not a judgement anyone made. It surfaced when the age bracket was
    found to be worth 0.0148 ROC-AUC while sitting on the wrong side of this
    line, so column_study.py asked the same question of the rest: encode all
    nine and the matrix goes from 39 features to 104, worth +0.0030 AUC to
    LightGBM and +0.0069 to XGBoost.

    "시술 유형" is deliberately absent: features.py copies it to
    "시술 유형_원본" beforehand, so adding it here would duplicate those
    indicators. The copy is vestigial and the two could be collapsed, but that
    is a rename, not a measurement, so it is left alone.
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
    
    # 3. Make the generated names safe for the downstream models
    train_df, test_df = sanitize_column_names(train_df, test_df)

    # 4. Synchronize features
    train_df, test_df = synchronize_columns(
        train_df, test_df, target_col=target_col
    )

    print(f"[INFO] After Encoding & Sync, Train shape: {train_df.shape}")
    print(f"[INFO] After Encoding & Sync, Test shape : {test_df.shape}")
    
    # 5. Validation: Check for remaining object columns
    rem_obj_train = train_df.select_dtypes(include=['object']).columns.tolist()
    rem_obj_test = test_df.select_dtypes(include=['object']).columns.tolist()
    
    if rem_obj_train or rem_obj_test:
        print(f"[WARN] Object columns remaining! Train: {rem_obj_train}, Test: {rem_obj_test}")
    else:
        print("[INFO] Success: All features are successfully encoded to numeric.")
        
    return train_df, test_df