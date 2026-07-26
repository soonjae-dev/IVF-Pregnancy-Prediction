"""
===============================================================================
Setup and Environment Configuration
===============================================================================
This module imports required libraries, configures global settings, and 
checks for hardware acceleration (macOS MPS / CPU).
"""

import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# PyTorch Deep Learning Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Scikit-learn Ecosystem
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    cross_val_score,
    learning_curve,
    StratifiedKFold
)
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve
)
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Imbalanced-learn Libraries
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Gradient Boosting & PyTorch Integration
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from skorch import NeuralNetClassifier

# Hyperparameter Optimization
import optuna

# Ignore non-critical warning messages
warnings.filterwarnings("ignore")

# Global Plotting Configuration
sns.set(style="whitegrid")


def set_seed(seed: int = 42):
    """
    Set random seed across all frameworks for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Check and return the available compute device (MPS for Apple Silicon, CUDA, or CPU).
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] MPS device is available. Using Apple Silicon Acceleration (MPS).")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("[INFO] CUDA device is available. Using NVIDIA GPU Acceleration.")
    else:
        device = torch.device("cpu")
        print("[INFO] Accelerator unavailable. Falling back to CPU.")
    return device


if __name__ == "__main__":
    # Test environment setup
    set_seed(42)
    device = get_device()