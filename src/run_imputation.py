"""
===============================================================================
GAIN Imputation Execution (src/run_imputation.py)
===============================================================================
This module provides wrapper functions to execute GAIN imputation on 
pandas DataFrames, seamlessly handling the conversion between DataFrames 
and NumPy arrays.
"""

import torch
import numpy as np
import pandas as pd
from typing import Tuple

# Assuming gain_impute is imported from the previously created module
# from src.gain_imputer import gain_impute

def run_gain_on_dataframes(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    device: torch.device,
    iterations: int = 3000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert DataFrames to NumPy arrays, apply GAIN imputation, 
    and reconstruct the DataFrames with imputed values.
    
    Args:
        train_df (pd.DataFrame): Training dataframe with missing values (-1).
        test_df (pd.DataFrame): Testing dataframe with missing values (-1).
        device (torch.device): Compute device to use for GAIN.
        iterations (int): Number of training iterations for the GAN. Default is 3000.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Imputed train and test dataframes.
    """
    print("[INFO] Converting DataFrames to float32 arrays for imputation...")
    
    # Ensure all data is numeric and cast to float32 for PyTorch compatibility
    train_arr = train_df.values.astype(np.float32)
    test_arr = test_df.values.astype(np.float32)
    
    print(f"[INFO] Applying GAIN Imputation on Train Data using {device}...")
    train_imputed_arr = gain_impute(
        data_x=train_arr,
        batch_size=256,
        hint_rate=0.9,
        alpha=100.0,
        iterations=iterations,
        hidden_dims=[256, 256],
        dropout_rate=0.2,
        learning_rate=1e-3,
        device=device
    )
    
    print(f"[INFO] Applying GAIN Imputation on Test Data using {device}...")
    test_imputed_arr = gain_impute(
        data_x=test_arr,
        batch_size=256,
        hint_rate=0.9,
        alpha=100.0,
        iterations=iterations,
        hidden_dims=[256, 256],
        dropout_rate=0.2,
        learning_rate=1e-3,
        device=device
    )
    
    print("[INFO] Reconstructing DataFrames...")
    train_df_imputed = pd.DataFrame(train_imputed_arr, columns=train_df.columns)
    test_df_imputed = pd.DataFrame(test_imputed_arr, columns=test_df.columns)
    
    print(f"[INFO] Train Imputed Shape: {train_df_imputed.shape}")
    print(f"[INFO] Test Imputed Shape : {test_df_imputed.shape}")
    
    return train_df_imputed, test_df_imputed