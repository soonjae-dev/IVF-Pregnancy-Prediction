"""
===============================================================================
Hyperparameter Optimization using Optuna (src/optimization.py)
===============================================================================
This module defines Optuna objective functions and optimization studies for 
multiple machine learning models (XGBoost, LightGBM, RandomForest, CatBoost) 
and a custom deep learning model (TabTransformer via Skorch).
"""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from skorch import NeuralNetClassifier

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier

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


def optimize_xgboost(X_train_sm: np.ndarray, y_train_sm: np.ndarray, rskf, n_trials: int = 20) -> Tuple[Dict[str, Any], float]:
    """
    Optimize hyperparameters for XGBoost using Optuna.
    """
    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.1)
        }
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            **params
        )
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', xgb_model)
        ])
        scores = cross_val_score(pipe, X_train_sm, y_train_sm, cv=rskf, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    print("[INFO] Starting Optuna tuning for XGBoost...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=n_trials, show_progress_bar=True)
    
    print("\n=== XGBoost ===")
    print("Best params:", study_xgb.best_params)
    print("Best value :", study_xgb.best_value)
    return study_xgb.best_params, study_xgb.best_value


def optimize_lightgbm(X_train_sm: np.ndarray, y_train_sm: np.ndarray, rskf, n_trials: int = 20) -> Tuple[Dict[str, Any], float]:
    """
    Optimize hyperparameters for LightGBM using Optuna.
    """
    def objective_lgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 64),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.1)
        }
        lgb_model = lgb.LGBMClassifier(random_state=42, **params)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('lgb', lgb_model)
        ])
        scores = cross_val_score(pipe, X_train_sm, y_train_sm, cv=rskf, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    print("[INFO] Starting Optuna tuning for LightGBM...")
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=n_trials, show_progress_bar=True)
    
    print("\n=== LightGBM ===")
    print("Best params:", study_lgb.best_params)
    print("Best value :", study_lgb.best_value)
    return study_lgb.best_params, study_lgb.best_value


def optimize_random_forest(X_train_sm: np.ndarray, y_train_sm: np.ndarray, rskf, n_trials: int = 20) -> Tuple[Dict[str, Any], float]:
    """
    Optimize hyperparameters for RandomForest using Optuna.
    """
    def objective_rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, step=2),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
        rf_model = RandomForestClassifier(random_state=42, **params)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', rf_model)
        ])
        scores = cross_val_score(pipe, X_train_sm, y_train_sm, cv=rskf, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    print("[INFO] Starting Optuna tuning for RandomForest...")
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(objective_rf, n_trials=n_trials, show_progress_bar=True)
    
    print("\n=== RandomForest ===")
    print("Best params:", study_rf.best_params)
    print("Best value :", study_rf.best_value)
    return study_rf.best_params, study_rf.best_value


def optimize_catboost(X_train_sm: np.ndarray, y_train_sm: np.ndarray, rskf, n_trials: int = 20) -> Tuple[Dict[str, Any], float]:
    """
    Optimize hyperparameters for CatBoost using Optuna.
    """
    def objective_cat(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 600, step=200),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-4, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10)
        }
        cat_model = cb.CatBoostClassifier(random_state=42, verbose=0, **params)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('cat', cat_model)
        ])
        scores = cross_val_score(pipe, X_train_sm, y_train_sm, cv=rskf, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    print("[INFO] Starting Optuna tuning for CatBoost...")
    study_cat = optuna.create_study(direction='maximize')
    study_cat.optimize(objective_cat, n_trials=n_trials, show_progress_bar=True)
    
    print("\n=== CatBoost ===")
    print("Best params:", study_cat.best_params)
    print("Best value :", study_cat.best_value)
    return study_cat.best_params, study_cat.best_value


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


def optimize_tab_transformer(X_train_sm: np.ndarray, y_train_sm: np.ndarray, rskf, device: torch.device, n_trials: int = 10) -> Tuple[Dict[str, Any], float]:
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
            module__num_features=X_train_sm.shape[1],
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
            X_train_sm, y_train_sm,
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