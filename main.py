"""
===============================================================================
Main Pipeline Execution Script (main.py)
===============================================================================
"""

import json
import pandas as pd
from src.logger import logger
from src.config import set_seed, get_device
from src.preprocessing import run_basic_preprocessing
from src.features import run_feature_engineering
from src.feature_selection import run_feature_selection
from src.encoding import run_encoding_pipeline
from src.run_imputation import run_gain_on_dataframes
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

def load_config(config_path: str = "configs/config.json") -> dict:
    """
    Load pipeline configurations from a JSON file.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)

def main():
    logger.info("========== Pipeline Execution Started ==========")
    
    # 0. Load Configuration
    logger.info("[0] Loading configurations...")
    config = load_config()
    
    # 1. Environment Setup
    logger.info("[1] Setting up environment...")
    set_seed(config["pipeline"]["seed"])
    device = get_device()
    
    # 2. Data Loading & Basic Preprocessing
    logger.info("[2] Loading data and basic preprocessing...")
    train_df, test_df, test_ids = run_basic_preprocessing(
        config["pipeline"]["train_data_path"], 
        config["pipeline"]["test_data_path"]
    )
    
    # 3. Feature Engineering
    logger.info("[3] Feature Engineering...")
    train_df, test_df = run_feature_engineering(train_df, test_df)
    
    # 4. Feature Selection
    logger.info("[4] Feature Selection...")
    train_df, test_df = run_feature_selection(train_df, test_df)
    
    # 5. Categorical Encoding
    logger.info("[5] Encoding Categorical Variables...")
    train_df, test_df = run_encoding_pipeline(train_df, test_df)
    
    # 6. Missing Value Imputation (GAIN)
    logger.info("[6] Imputing Missing Values with GAIN...")
    train_df, test_df = run_gain_on_dataframes(
        train_df, test_df, device, 
        iterations=config["imputation"]["gain_iterations"]
    )
    
    # 7. Data Splitting & Resampling (SMOTE)
    logger.info("[7] Splitting Data & Applying SMOTE...")
    X_train, X_val, y_train, y_val, X_train_sm, y_train_sm, rskf = prepare_training_data(train_df)
    
    # 8. Hyperparameter Optimization
    logger.info("[8] Optimizing Hyperparameters...")
    n_trials = config["optimization"]["n_trials"]
    best_xgb, _ = optimize_xgboost(X_train_sm, y_train_sm, rskf, n_trials=n_trials)
    best_lgb, _ = optimize_lightgbm(X_train_sm, y_train_sm, rskf, n_trials=n_trials)
    best_rf, _ = optimize_random_forest(X_train_sm, y_train_sm, rskf, n_trials=n_trials)
    best_cat, _ = optimize_catboost(X_train_sm, y_train_sm, rskf, n_trials=n_trials)
    
    # 9. Ensemble Modeling
    logger.info("[9] Building Ensembles...")
    
    # 9.1 Stacking Ensemble
    stacking_clf = build_stacking_classifier(best_xgb, best_lgb, best_rf, best_cat)
    stacking_clf.fit(X_train_sm, y_train_sm)
    val_pred_stack = stacking_clf.predict_proba(X_val)[:, 1]
    best_thresh_s, stack_auc, _ = optimize_threshold_and_evaluate(y_val, val_pred_stack, "Stacking")
    
    # 9.2 Weighted Ensemble
    weights, val_auc_ens, val_ens_proba = optimize_weighted_ensemble(
        X_train_sm, y_train_sm, X_val, y_val,
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
    ).fit(X_train_sm, y_train_sm)
    
    lgb_model = lgb.LGBMClassifier(
        random_state=config["pipeline"]["seed"], 
        **best_lgb
    ).fit(X_train_sm, y_train_sm)
    
    rf_model = RandomForestClassifier(
        random_state=config["pipeline"]["seed"], 
        **best_rf
    ).fit(X_train_sm, y_train_sm)
    
    cat_model = cb.CatBoostClassifier(
        random_state=config["pipeline"]["seed"], 
        verbose=0, 
        **best_cat
    ).fit(X_train_sm, y_train_sm)
    
    generate_submissions(
        stacking_clf, xgb_model, lgb_model, rf_model, cat_model,
        test_df, test_ids, best_thresh_s, best_thresh_w, weights
    )

    logger.info("========== Pipeline Executed Successfully! ==========")

if __name__ == "__main__":
    main()
