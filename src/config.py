"""
===============================================================================
Environment Setup (src/config.py)
===============================================================================
Seeding and device selection.

This module used to re-export the whole stack — pandas, seaborn, matplotlib,
sklearn, the three boosting libraries, skorch, torch, optuna — in the style of
a notebook's first cell. Nothing imported any of it: every consumer takes
set_seed and get_device and nothing else.

That mattered more than tidiness. skorch pulls in torch, so `from src.config
import set_seed` loaded torch into whatever process asked, and on macOS a
process holding torch alongside LightGBM, XGBoost and CatBoost dies on a
duplicate OpenMP runtime. main.py went through this import.

torch is now imported inside get_device, which only build_cache.py calls, and
nowhere else.
"""

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Seed Python's `random` and NumPy.

    Deliberately not torch: importing it here would put torch in every process
    that seeds anything. The one place torch randomness matters is GAIN, and
    `src.gain_imputer.gain_impute` seeds it directly from its own `seed`
    argument, inside the process that actually runs it.
    """
    random.seed(seed)
    np.random.seed(seed)


def get_device():
    """
    Return the best available compute device: MPS, CUDA, or CPU.

    Imports torch on call rather than at module scope — see the module
    docstring. Only call this from a process that has no boosting library
    loaded.
    """
    import torch

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
