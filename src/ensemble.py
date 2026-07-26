"""
===============================================================================
Ensemble Modeling: Stacking and Weighted Ensemble (src/ensemble.py)
===============================================================================
This module implements StackingClassifier with tuned base models and a meta-model,
as well as Optuna-based weight optimization for a Weighted Ensemble.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import optuna

import xgboost as xgb
import lightgbm as lgb
import catboost as cb


def build_stacking_classifier(
    best_params_xgb: Dict[str, Any],
    best_params_lgb: Dict[str, Any],
    best_params_rf: Dict[str, Any],
    best_params_cat: Dict[str, Any]
) -> StackingClassifier:
    """
    Build and return a StackingClassifier using tuned base models and LogisticRegression meta-model.
    """
    xgb_final = xgb.XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        **best_params_xgb
    )
    lgb_final = lgb.LGBMClassifier(
        random_state=42,
        **best_params_lgb
    )
    rf_final = RandomForestClassifier(
        random_state=42,
        **best_params_rf
    )
    cat_final = cb.CatBoostClassifier(
        random_state=42,
        verbose=0,
        **best_params_cat
    )

    base_estimators = [
        ('xgb', xgb_final),
        ('lgb', lgb_final),
        ('rf', rf_final),
        ('cat', cat_final)
    ]

    meta_model = LogisticRegression(solver='liblinear', random_state=42)

    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1,
        passthrough=False
    )
    return stacking_clf


def optimize_weighted_ensemble(
    X_train_sm: np.ndarray,
    y_train_sm: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_params_xgb: Dict[str, Any],
    best_params_lgb: Dict[str, Any],
    best_params_rf: Dict[str, Any],
    best_params_cat: Dict[str, Any],
    n_trials: int = 30
) -> Tuple[Dict[str, float], float, np.ndarray]:
    """
    Train base models fully, generate validation probabilities, and use Optuna 
    to find the optimal blending weights for a Weighted Ensemble.
    """
    xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **best_params_xgb)
    lgb_model = lgb.LGBMClassifier(random_state=42, **best_params_lgb)
    rf_model = RandomForestClassifier(random_state=42, **best_params_rf)
    cat_model = cb.CatBoostClassifier(random_state=42, verbose=0, **best_params_cat)

    print("[INFO] Training base models for Weighted Ensemble...")
    xgb_model.fit(X_train_sm, y_train_sm)
    lgb_model.fit(X_train_sm, y_train_sm)
    rf_model.fit(X_train_sm, y_train_sm)
    cat_model.fit(X_train_sm, y_train_sm)

    val_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
    val_pred_lgb = lgb_model.predict_proba(X_val)[:, 1]
    val_pred_rf = rf_model.predict_proba(X_val)[:, 1]
    val_pred_cat = cat_model.predict_proba(X_val)[:, 1]

    def objective(trial):
        w_xgb = trial.suggest_float('w_xgb', 0.0, 1.0)
        w_lgb = trial.suggest_float('w_lgb', 0.0, 1.0)
        w_rf = trial.suggest_float('w_rf', 0.0, 1.0)
        w_cat = trial.suggest_float('w_cat', 0.0, 1.0)
        
        sum_w = w_xgb + w_lgb + w_rf + w_cat
        if sum_w < 1e-9:
            return 0.0
        
        w_xgb /= sum_w
        w_lgb /= sum_w
        w_rf /= sum_w
        w_cat /= sum_w

        ensemble_proba = (
            w_xgb * val_pred_xgb +
            w_lgb * val_pred_lgb +
            w_rf * val_pred_rf +
            w_cat * val_pred_cat
        )
        return roc_auc_score(y_val, ensemble_proba)

    print("[INFO] Starting Optuna tuning for Weighted Ensemble weights...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_w = study.best_params
    sum_w = sum(best_w.values())
    if sum_w < 1e-9:
        sum_w = 1e-9

    normalized_weights = {k: v / sum_w for k, v in best_w.items()}
    
    best_ens_proba = (
        normalized_weights['w_xgb'] * val_pred_xgb +
        normalized_weights['w_lgb'] * val_pred_lgb +
        normalized_weights['w_rf'] * val_pred_rf +
        normalized_weights['w_cat'] * val_pred_cat
    )
    val_auc_ens = roc_auc_score(y_val, best_ens_proba)

    print(f"\n=== Weighted Ensemble ===")
    print("Best normalized weights:", normalized_weights)
    print(f"Validation AUC: {val_auc_ens:.4f}")

    return normalized_weights, val_auc_ens, best_ens_proba