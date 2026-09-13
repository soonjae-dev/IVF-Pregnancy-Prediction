"""
===============================================================================
Main Pipeline Execution Script (main.py)
===============================================================================
Steps 1-6 -- preprocessing, feature engineering, selection, encoding and GAIN
imputation -- run in a separate process, through build_cache.py, and this
script picks up the matrix they produce.

That is not an optimisation, though it is one: the cache means GAIN runs once
across every script in the repository instead of once per script. It is a
requirement. GAIN needs PyTorch, this script needs LightGBM, XGBoost and
CatBoost, and on macOS a process holding torch alongside any of those dies on a
duplicate OpenMP runtime -- in either import order. See TUNING.md.
"""

import json

import numpy as np
import pandas as pd

from build_cache import ensure_cache
from src.logger import logger
from src.config import set_seed
from src.data_splitting import prepare_training_data
from src.optimization import (
    optimize_xgboost,
    optimize_lightgbm,
    optimize_random_forest,
    optimize_catboost
)
from src.ensemble import build_stacking_classifier, optimize_weighted_ensemble
from src.evaluation import optimize_threshold_and_evaluate, generate_submissions

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import catboost as cb

TARGET = "임신 성공 여부"
CACHE = "data/gain_cache.npz"


def load_config(config_path: str = "configs/config.json") -> dict:
    """
    Load pipeline configurations from a JSON file.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_prepared_data(iterations: int):
    """
    Steps 2-6, as build_cache.py produced them.

    Builds the cache first if it is missing; that build is a subprocess, so no
    torch is imported here. Returns the frames this script used to construct
    inline, so everything downstream is unchanged.
    """
    ensure_cache(CACHE, iterations=iterations)
    cached = np.load(CACHE, allow_pickle=True)

    columns = list(cached["columns"])
    train_df = pd.DataFrame(cached["X"], columns=columns)
    train_df.insert(0, TARGET, cached["y"])
    test_df = pd.DataFrame(cached["X_test"], columns=columns)
    test_ids = pd.Series(cached["test_ids"], name="ID")

    return train_df, test_df, test_ids

def main():
    logger.info("========== Pipeline Execution Started ==========")
    
    # 0. Load Configuration
    logger.info("[0] Loading configurations...")
    config = load_config()
    
    # 1. Environment Setup
    logger.info("[1] Setting up environment...")
    set_seed(config["pipeline"]["seed"])

    # 2-6. Preprocessing through GAIN imputation, in a separate process
    logger.info("[2-6] Loading the prepared matrix (built by build_cache.py)...")
    train_df, test_df, test_ids = load_prepared_data(
        config["imputation"]["gain_iterations"]
    )
    logger.info(f"      train {train_df.shape}, test {test_df.shape}")

    # 7. Data Splitting
    # No resampling. SMOTE used to be applied to the whole training set here;
    # resampling_study.py measures it as worth -0.0004 ROC-AUC on this
    # representation, at 1.8-3.2x the training time, so it was dropped.
    logger.info("[7] Splitting Data...")
    X_train, X_val, y_train, y_val, rskf = prepare_training_data(train_df)

    # 8. Hyperparameter Optimization
    logger.info("[8] Optimizing Hyperparameters...")
    n_trials = config["optimization"]["n_trials"]
    best_xgb, _ = optimize_xgboost(X_train, y_train, rskf, n_trials=n_trials)
    best_lgb, _ = optimize_lightgbm(X_train, y_train, rskf, n_trials=n_trials)
    best_rf, _ = optimize_random_forest(X_train, y_train, rskf, n_trials=n_trials)
    best_cat, _ = optimize_catboost(X_train, y_train, rskf, n_trials=n_trials)
    
    # 9. Ensemble Modeling
    logger.info("[9] Building Ensembles...")
    
    # 9.1 Stacking Ensemble
    stacking_clf = build_stacking_classifier(best_xgb, best_lgb, best_rf, best_cat)
    stacking_clf.fit(X_train, y_train)
    val_pred_stack = stacking_clf.predict_proba(X_val)[:, 1]
    best_thresh_s, stack_auc, _ = optimize_threshold_and_evaluate(y_val, val_pred_stack, "Stacking")
    
    # 9.2 Weighted Ensemble
    weights, val_auc_ens, val_ens_proba = optimize_weighted_ensemble(
        X_train, y_train, X_val, y_val,
        best_xgb, best_lgb, best_rf, best_cat, 
        n_trials=config["optimization"]["ensemble_trials"]
    )
    best_thresh_w, _, _ = optimize_threshold_and_evaluate(y_val, val_ens_proba, "Weighted Ensemble")
    
    # 10. Final Prediction & Submission
    logger.info("[10] Generating Submission Files...")
    xgb_model = xgb.XGBClassifier(
        random_state=config["pipeline"]["seed"], 
        use_label_encoder=False, 
        eval_metric='logloss', 
        **best_xgb
    ).fit(X_train, y_train)
    
    lgb_model = lgb.LGBMClassifier(
        random_state=config["pipeline"]["seed"], 
        **best_lgb
    ).fit(X_train, y_train)
    
    rf_model = RandomForestClassifier(
        random_state=config["pipeline"]["seed"], 
        **best_rf
    ).fit(X_train, y_train)
    
    cat_model = cb.CatBoostClassifier(
        random_state=config["pipeline"]["seed"], 
        verbose=0, 
        **best_cat
    ).fit(X_train, y_train)
    
    generate_submissions(
        stacking_clf, xgb_model, lgb_model, rf_model, cat_model,
        test_df, test_ids, best_thresh_s, best_thresh_w, weights
    )

    logger.info("========== Pipeline Executed Successfully! ==========")

if __name__ == "__main__":
    main()
