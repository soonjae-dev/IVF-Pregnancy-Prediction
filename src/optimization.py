"""
===============================================================================
Hyperparameter Optimization using Optuna (src/optimization.py)
===============================================================================
This module defines Optuna objective functions and optimization studies for
multiple machine learning models (XGBoost, LightGBM, RandomForest, CatBoost)
and nothing else. The TabTransformer moved to src/tab_transformer.py, because
it was the only thing here that needed torch and importing torch alongside the
boosting libraries segfaults on macOS.

Every objective scores a scikit-learn Pipeline, not a bare estimator, so the
scaler is fitted on each fold's training part alone. An earlier version applied
SMOTE to the whole training set before `cross_val_score` and reported CV
ROC-AUC around 0.90; the fix was to move resampling inside the pipeline, and
`resampling_study.py` then showed the resampling itself was not doing anything,
so the step was removed entirely. See the README.
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier



# ---------------------------------------------------------------------------
# Search spaces
#
# One definition each, so main.py's optimize_* functions and tune.py cannot
# drift apart. Widening a range here widens it for both.
# ---------------------------------------------------------------------------

def suggest_params(trial, model: str) -> Dict[str, Any]:
    """Draw one hyperparameter set for `model` from its search space."""
    if model == "xgb":
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.1),
        }
    if model == "lgb":
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 64),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.1),
        }
    if model == "rf":
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, step=2),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        }
    if model == "cat":
        return {
            'iterations': trial.suggest_int('iterations', 200, 600, step=200),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-4, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
        }
    raise ValueError(f"unknown model: {model}")


# Keys that live in configs/*.json and tuning/best_params/*.json alongside the
# real hyperparameters: provenance, not settings. Anything starting with "_" is
# treated the same way. Passing one of these to a model constructor is a
# TypeError, so every consumer goes through model_params() rather than each
# remembering its own exclusion list.
METADATA_KEYS = {"best_value", "tuning_cv_auc", "referee_auc"}


def model_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Strip provenance fields, leaving only what a model constructor accepts."""
    return {k: v for k, v in params.items()
            if k not in METADATA_KEYS and not k.startswith("_")}


def load_params(model: str, config_dir: str = "configs") -> Dict[str, Any]:
    """Read configs/<model>_params.json, metadata already stripped."""
    import json
    import pathlib

    path = pathlib.Path(config_dir) / f"{model}_params.json"
    with open(path, encoding="utf-8") as handle:
        return model_params(json.load(handle))


def build_model(model: str, params: Dict[str, Any], seed: int = 42):
    """Instantiate one of the four base models from a parameter dict."""
    tuned = model_params(params)
    if model == "xgb":
        return xgb.XGBClassifier(random_state=seed, eval_metric='logloss',
                                 n_jobs=-1, tree_method='hist', **tuned)
    if model == "lgb":
        return lgb.LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **tuned)
    if model == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **tuned)
    if model == "cat":
        return cb.CatBoostClassifier(random_state=seed, verbose=0, thread_count=-1,
                                     allow_writing_files=False, **tuned)
    raise ValueError(f"unknown model: {model}")


def optimize_xgboost(X_train: np.ndarray, y_train: np.ndarray, rskf, n_trials: int = 20) -> Tuple[Dict[str, Any], float]:
    """
    Optimize hyperparameters for XGBoost using Optuna.
    """
    def objective_xgb(trial):
        params = suggest_params(trial, "xgb")
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
        scores = cross_val_score(pipe, X_train, y_train, cv=rskf, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    print("[INFO] Starting Optuna tuning for XGBoost...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=n_trials, show_progress_bar=True)
    
    print("\n=== XGBoost ===")
    print("Best params:", study_xgb.best_params)
    print("Best value :", study_xgb.best_value)
    return study_xgb.best_params, study_xgb.best_value


def optimize_lightgbm(X_train: np.ndarray, y_train: np.ndarray, rskf, n_trials: int = 20) -> Tuple[Dict[str, Any], float]:
    """
    Optimize hyperparameters for LightGBM using Optuna.
    """
    def objective_lgb(trial):
        params = suggest_params(trial, "lgb")
        lgb_model = lgb.LGBMClassifier(random_state=42, **params)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('lgb', lgb_model)
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=rskf, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    print("[INFO] Starting Optuna tuning for LightGBM...")
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=n_trials, show_progress_bar=True)
    
    print("\n=== LightGBM ===")
    print("Best params:", study_lgb.best_params)
    print("Best value :", study_lgb.best_value)
    return study_lgb.best_params, study_lgb.best_value


def optimize_random_forest(X_train: np.ndarray, y_train: np.ndarray, rskf, n_trials: int = 20) -> Tuple[Dict[str, Any], float]:
    """
    Optimize hyperparameters for RandomForest using Optuna.
    """
    def objective_rf(trial):
        params = suggest_params(trial, "rf")
        rf_model = RandomForestClassifier(random_state=42, **params)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', rf_model)
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=rskf, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    print("[INFO] Starting Optuna tuning for RandomForest...")
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(objective_rf, n_trials=n_trials, show_progress_bar=True)
    
    print("\n=== RandomForest ===")
    print("Best params:", study_rf.best_params)
    print("Best value :", study_rf.best_value)
    return study_rf.best_params, study_rf.best_value


def optimize_catboost(X_train: np.ndarray, y_train: np.ndarray, rskf, n_trials: int = 20) -> Tuple[Dict[str, Any], float]:
    """
    Optimize hyperparameters for CatBoost using Optuna.
    """
    def objective_cat(trial):
        params = suggest_params(trial, "cat")
        cat_model = cb.CatBoostClassifier(random_state=42, verbose=0, **params)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('cat', cat_model)
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=rskf, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    print("[INFO] Starting Optuna tuning for CatBoost...")
    study_cat = optuna.create_study(direction='maximize')
    study_cat.optimize(objective_cat, n_trials=n_trials, show_progress_bar=True)
    
    print("\n=== CatBoost ===")
    print("Best params:", study_cat.best_params)
    print("Best value :", study_cat.best_value)
    return study_cat.best_params, study_cat.best_value
