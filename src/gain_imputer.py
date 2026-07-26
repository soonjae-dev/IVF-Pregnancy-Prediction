"""
===============================================================================
Generative Adversarial Imputation Nets (src/gain_imputer.py)
===============================================================================
This module implements GAIN (Generative Adversarial Imputation Nets) using PyTorch.
It leverages a GAN-based architecture to robustly impute missing values (indicated 
by -1.0) by training a Generator to output realistic values and a Discriminator 
to distinguish between observed and imputed components.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional

class Generator(nn.Module):
    """
    Generator network for GAIN.
    Learns to impute missing values to fool the Discriminator.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 256], dropout_rate: float = 0.2):
        super(Generator, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, input_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Discriminator(nn.Module):
    """
    Discriminator network for GAIN.
    Attempts to distinguish whether each feature is originally observed or imputed,
    using the input data and a hint matrix.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 256], dropout_rate: float = 0.2):
        super(Discriminator, self).__init__()
        combined_dim = input_dim * 2 # Takes concatenated (X, Hint)
        layers = []
        prev_dim = combined_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, input_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, h], dim=1)
        return self.net(inp)


def gain_impute(
    data_x: np.ndarray,
    batch_size: int = 256,
    hint_rate: float = 0.9,
    alpha: float = 100.0,
    iterations: int = 5000,
    hidden_dims: List[int] = [256, 256],
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Impute missing values using the GAIN algorithm.
    
    Args:
        data_x (np.ndarray): The dataset containing missing values (indicated by -1.0).
        batch_size (int): Mini-batch size for training.
        hint_rate (float): Probability of revealing the true mask in the hint matrix.
        alpha (float): Hyperparameter for the reconstruction loss.
        iterations (int): Number of training iterations.
        hidden_dims (List[int]): List of hidden layer dimensions.
        dropout_rate (float): Dropout probability.
        learning_rate (float): Learning rate for Adam optimizers.
        device (torch.device): Compute device (CPU, CUDA, or MPS).
        
    Returns:
        np.ndarray: The fully imputed dataset.
    """
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
    X = data_x.copy()
    n, d = X.shape
    
    # Identify missing values (-1.0) and create mask M (1 if observed, 0 if missing)
    M = (~np.isclose(X, -1.0)).astype(float)
    
    # Fill missing values with small random noise initially
    X_filled = np.where(np.isclose(X, -1.0), np.random.uniform(0, 0.01, size=X.shape), X)

    X_tensor = torch.tensor(X_filled, dtype=torch.float32, device=device)
    M_tensor = torch.tensor(M, dtype=torch.float32, device=device)

    # Initialize models
    G = Generator(d, hidden_dims, dropout_rate).to(device)
    D = Discriminator(d, hidden_dims, dropout_rate).to(device)

    G_opt = optim.Adam(G.parameters(), lr=learning_rate)
    D_opt = optim.Adam(D.parameters(), lr=learning_rate)

    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    print(f"[INFO] Starting GAIN Imputation for {iterations} iterations on {device}...")

    # Training Loop
    for it in range(iterations):
        # Sample mini-batch
        batch_idx = np.random.choice(n, batch_size, replace=True)
        X_mb = X_tensor[batch_idx]
        M_mb = M_tensor[batch_idx]
        Z_mb = torch.rand_like(X_mb)
        one_minus_M_mb = 1.0 - M_mb

        # Forward Generator
        X_mb_input = M_mb * X_mb + one_minus_M_mb * Z_mb
        G_sample = G(X_mb_input)
        X_hat = M_mb * X_mb + one_minus_M_mb * G_sample

        # Sample Hint Vector
        H_mb = (torch.rand(M_mb.shape, device=device) < hint_rate).float() * M_mb

        # Train Discriminator
        D_prob = D(X_hat, H_mb)
        D_loss = bce_loss(D_prob, M_mb)
        
        D_opt.zero_grad()
        D_loss.backward(retain_graph=True)
        D_opt.step()

        # Train Generator
        D_prob_for_G = D(X_hat, H_mb)
        G_loss_adv = bce_loss(D_prob_for_G, 1.0 - M_mb)
        G_loss_rec = mse_loss(M_mb * X_hat, M_mb * X_mb)
        G_loss = G_loss_adv + alpha * G_loss_rec

        G_opt.zero_grad()
        G_loss.backward()
        G_opt.step()

        if (it + 1) % 500 == 0:
            print(f"[GAIN] Iter {it + 1:04d}/{iterations} | D_Loss: {D_loss.item():.4f} | G_Loss: {G_loss.item():.4f}")

    # Final Imputation
    print("[INFO] GAIN training complete. Generating final imputed dataset...")
    Z_full = torch.rand_like(X_tensor)
    one_minus_M = 1.0 - M_tensor
    X_input = M_tensor * X_tensor + one_minus_M * Z_full
    
    with torch.no_grad():
        G_output = G(X_input)
        
    X_imputed = M_tensor * X_tensor + one_minus_M * G_output
    
    return X_imputed.cpu().numpy()