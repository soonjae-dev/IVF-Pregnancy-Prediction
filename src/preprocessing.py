"""
===============================================================================
Data Loading and Basic Preprocessing (src/preprocessing.py)
===============================================================================
This module handles loading raw datasets, removing unnecessary identifiers, 
and performing basic missing value imputation.
"""

import pandas as pd
from typing import Tuple, Optional

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and testing datasets from CSV files.
    
    Args:
        train_path (str): File path for the training data.
        test_path (str): File path for the testing data.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Loaded train and test dataframes.
    """
    print(f"[INFO] Loading data from {train_path} and {test_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def process_ids(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
    """
    Extract and save test IDs for the final submission, then drop the 'ID' 
    column from both dataframes to prevent data leakage.
    """
    # Save test IDs for submission
    test_ids = test_df['ID'].copy() if 'ID' in test_df.columns else None
        
    # Drop ID columns
    if 'ID' in train_df.columns:
        train_df.drop(columns=['ID'], inplace=True)
    if 'ID' in test_df.columns:
        test_df.drop(columns=['ID'], inplace=True)
        
    return train_df, test_df, test_ids

def handle_missing_values(df: pd.DataFrame, fill_value: int = -1) -> pd.DataFrame:
    """
    Impute missing values with a designated constant.
    
    Args:
        df (pd.DataFrame): The dataframe to process.
        fill_value (int): The value used to replace NaNs. Default is -1.
        
    Returns:
        pd.DataFrame: Dataframe with imputed missing values.
    """
    return df.fillna(fill_value)

def run_basic_preprocessing(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Execute the full basic preprocessing pipeline.
    """
    # 1. Load data
    train_df, test_df = load_data(train_path, test_path)
    
    # 2. Process IDs
    train_df, test_df, test_ids = process_ids(train_df, test_df)
    
    print(f"[INFO] Train shape after ID drop: {train_df.shape}")
    print(f"[INFO] Test shape after ID drop : {test_df.shape}")
    
    # 3. Impute missing values
    train_df = handle_missing_values(train_df, fill_value=-1)
    test_df = handle_missing_values(test_df, fill_value=-1)
    
    # Print summary of imputed values
    train_missing_count = (train_df == -1).sum().sum()
    test_missing_count = (test_df == -1).sum().sum()
    print(f"[INFO] Total -1 values in Train: {train_missing_count}")
    print(f"[INFO] Total -1 values in Test : {test_missing_count}")
    
    return train_df, test_df, test_ids