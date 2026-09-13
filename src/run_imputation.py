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

from src.gain_imputer import gain_impute

def run_gain_on_dataframes(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    device: torch.device,
    iterations: int = 3000,
    target_col: str = "임신 성공 여부",
    seed: int = 42
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

    # The target is held out of the imputation. GAIN learns the joint
    # distribution of every column it is given and uses it to fill gaps, so
    # leaving the label in would let each row's own outcome inform the values
    # imputed for that row -- target leakage through the imputer. It also kept
    # train and test on different column counts, and therefore on differently
    # trained imputers.
    target = None
    if target_col in train_df.columns:
        target = train_df[target_col].copy()
        train_df = train_df.drop(columns=[target_col])
        print(f"[INFO] Holding out '{target_col}' from imputation.")

    if list(train_df.columns) != list(test_df.columns):
        raise ValueError(
            "train and test columns differ before imputation: "
            f"{set(train_df.columns) ^ set(test_df.columns)}"
        )

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
        device=device,
        seed=seed
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
        device=device,
        seed=seed
    )
    
    print("[INFO] Reconstructing DataFrames...")
    train_df_imputed = pd.DataFrame(train_imputed_arr, columns=train_df.columns,
                                    index=train_df.index)
    test_df_imputed = pd.DataFrame(test_imputed_arr, columns=test_df.columns,
                                   index=test_df.index)

    # Put the untouched target back, as the first column, matching what the
    # encoding step produced.
    if target is not None:
        train_df_imputed.insert(0, target_col, target.values)
    
    print(f"[INFO] Train Imputed Shape: {train_df_imputed.shape}")
    print(f"[INFO] Test Imputed Shape : {test_df_imputed.shape}")
    
    return train_df_imputed, test_df_imputed