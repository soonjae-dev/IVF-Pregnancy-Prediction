"""
===============================================================================
Feature Selection and Correlation Analysis (src/feature_selection.py)
===============================================================================
This module analyzes the linear correlation between features and the target 
variable. It provides functions to visualize the correlation matrix and to 
drop features that have a minimal impact on the target to reduce noise.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Tuple, List

def setup_korean_font() -> None:
    """
    Configure Matplotlib to correctly display Korean characters.
    Optimized for macOS using 'AppleGothic'.
    """
    mpl.rc('font', family='AppleGothic')
    mpl.rc('axes', unicode_minus=False)

def plot_correlation_heatmap(train_df: pd.DataFrame, target_col: str = "임신 성공 여부") -> None:
    """
    Calculate and plot a heatmap of the correlation matrix for numeric columns.
    
    Args:
        train_df (pd.DataFrame): The training dataframe.
        target_col (str): The name of the target column.
    """
    setup_korean_font()
    
    # Ensure target column is numeric for correlation calculation
    if target_col in train_df.columns:
        train_df[target_col] = pd.to_numeric(train_df[target_col], errors='coerce')
        
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = train_df[num_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="RdBu_r", center=0)
    plt.title("Correlation Matrix (Train Data, Numeric Columns)")
    plt.show()

def drop_low_correlation_features(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    target_col: str = "임신 성공 여부", 
    threshold: float = 0.02
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify and remove features that have an absolute correlation with the 
    target variable below a specified threshold.
    
    Args:
        train_df (pd.DataFrame): The training dataframe.
        test_df (pd.DataFrame): The testing dataframe.
        target_col (str): The name of the target column.
        threshold (float): The minimum absolute correlation required to keep a feature.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Dataframes with low-correlation features removed.
    """
    print(f"[INFO] Evaluating feature correlations with threshold |corr| < {threshold}...")
    
    # Ensure target column is numeric
    if target_col in train_df.columns:
        train_df[target_col] = pd.to_numeric(train_df[target_col], errors='coerce')
        
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = train_df[num_cols].corr()
    
    if target_col not in corr_matrix.columns:
        print("[WARN] Target column not found in correlation matrix. Skipping feature removal.")
        return train_df, test_df
        
    # Calculate absolute correlation with target
    corr_with_target = corr_matrix[target_col].abs().sort_values(ascending=False)
    
    # Identify features below the threshold
    low_corr_vars = corr_with_target[corr_with_target < threshold].index.tolist()
    drop_cols = [c for c in low_corr_vars if c != target_col]
    
    print(f"[INFO] Removing {len(drop_cols)} low-correlation features: {drop_cols}")
    
    # Drop identified features from both train and test sets
    train_df.drop(columns=drop_cols, inplace=True, errors='ignore')
    test_df.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    return train_df, test_df

def run_feature_selection(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "임신 성공 여부") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the feature selection pipeline.
    """
    # plot_correlation_heatmap(train_df, target_col) # Uncomment to visualize during EDA
    
    train_df, test_df = drop_low_correlation_features(train_df, test_df, target_col=target_col, threshold=0.02)
    
    print(f"[INFO] After feature selection, Train shape: {train_df.shape}")
    print(f"[INFO] After feature selection, Test shape : {test_df.shape}")
    
    return train_df, test_df