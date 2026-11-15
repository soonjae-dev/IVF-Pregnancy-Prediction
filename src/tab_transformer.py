"""
===============================================================================
TabTransformer (src/tab_transformer.py)
===============================================================================
A transformer over tabular features, implemented in PyTorch and tuned through
skorch. It is not part of the pipeline: main.py never calls it, and
configs/tab_params.json records a search run under the old leaky protocol.

It lives in its own module because it is the only thing in the optimization
code that needs torch. On macOS, importing torch into a process that already
holds LightGBM, XGBoost and CatBoost -- or the reverse -- takes the process
down with a duplicate-OpenMP segmentation fault. Keeping it here lets tune.py
and every study script import src.optimization without dragging torch in
behind it.
"""

from typing import Any, Dict, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_device() -> torch.device:
    """
    Check and return the available compute device (MPS for Apple Silicon or CPU).
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] MPS device is available. Using MPS!")
    else:
        device = torch.device("cpu")
        print("[INFO] MPS device is not available. Using CPU.")
    return device


# ---------------------------------------------------------------------------
# TabTransformer Architecture Classes
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        return x


class TabTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_hidden_dim: int = 64,
        dropout: float = 0.1,
        num_classes: int = 2,
        use_cls_token: bool = True
    ):
        super(TabTransformer, self).__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

        self.feature_embeds = nn.ModuleList(
            [nn.Linear(1, embed_dim) for _ in range(num_features)]
        )
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.num_tokens = num_features + 1
        else:
            self.cls_token = None
            self.num_tokens = num_features

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))

        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
             for _ in range(num_layers)]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.unsqueeze(-1)
        emb_list = []
        for i, fe in enumerate(self.feature_embeds):
            feat_i = x[:, i, :]
            emb_i = fe(feat_i)
            emb_list.append(emb_i)
        x_emb = torch.stack(emb_list, dim=1)
        if self.use_cls_token:
            cls_tok = self.cls_token.expand(batch_size, -1, -1)
            x_emb = torch.cat([cls_tok, x_emb], dim=1)

        x_emb = x_emb + self.pos_embed
        x_trans = x_emb.transpose(0, 1)
        for layer in self.transformer_layers:
            x_trans = layer(x_trans)
        x_trans = x_trans.transpose(0, 1)

        if self.use_cls_token:
            out = x_trans[:, 0, :]
        else:
            out = x_trans.mean(dim=1)
        logits = self.mlp_head(out)
        return logits


def optimize_tab_transformer(X_train: np.ndarray, y_train: np.ndarray, rskf, device: torch.device, n_trials: int = 10) -> Tuple[Dict[str, Any], float]:
    """
    Optimize hyperparameters for TabTransformer using Optuna and Skorch.
    """
    combos = [
        (16, 1), (16, 2), (16, 4), (16, 8),
        (24, 1), (24, 2), (24, 3), (24, 4), (24, 6), (24, 8),
        (32, 1), (32, 2), (32, 4), (32, 8),
        (40, 1), (40, 2), (40, 4), (40, 5), (40, 8),
        (48, 1), (48, 2), (48, 3), (48, 4), (48, 6), (48, 8),
        (56, 1), (56, 2), (56, 4), (56, 7), (56, 8),
        (64, 1), (64, 2), (64, 4), (64, 8),
    ]

    def objective_tab(trial):
        combo = trial.suggest_categorical('combo', combos)
        embed_dim, num_heads = combo

        ff_hidden_dim = trial.suggest_int('ff_hidden_dim', 32, 128, step=32)
        dropout = trial.suggest_float('dropout', 0.0, 0.3, step=0.1)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        max_epochs = trial.suggest_int('max_epochs', 10, 30, step=10)
        num_layers = trial.suggest_int('num_layers', 1, 3)

        net = NeuralNetClassifier(
            module=TabTransformer,
            module__num_features=X_train.shape[1],
            module__embed_dim=embed_dim,
            module__num_heads=num_heads,
            module__num_layers=num_layers,
            module__ff_hidden_dim=ff_hidden_dim,
            module__dropout=dropout,
            module__num_classes=2,
            module__use_cls_token=True,
            max_epochs=max_epochs,
            lr=lr,
            optimizer=optim.Adam,
            criterion=nn.CrossEntropyLoss,
            batch_size=64,
            iterator_train__shuffle=True,
            device=device,
            verbose=0
        )

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('tab', net)
        ])

        scores = cross_val_score(
            pipe,
            X_train, y_train,
            cv=rskf,
            scoring='roc_auc',
            n_jobs=-1
        )
        return scores.mean()

    print("[INFO] Starting Optuna tuning for TabTransformer...")
    study_tab = optuna.create_study(direction='maximize')
    study_tab.optimize(objective_tab, n_trials=n_trials, show_progress_bar=True)
    
    print("\n=== TabTransformer ===")
    print("Best params:", study_tab.best_params)
    print("Best value :", study_tab.best_value)
    return study_tab.best_params, study_tab.best_value
