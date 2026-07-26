"""
===============================================================================
Evaluation, Learning Curve, and Inference Module (src/evaluation.py)
===============================================================================
This module handles optimal F1 threshold tuning, confusion matrix visualization, 
learning curve plotting, test set inference, and submission file generation 
for both Stacking and Weighted Ensemble models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple

from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import learning_curve


def optimize_threshold_and_evaluate(
    y_val: np.ndarray,
    val_pred_proba: np.ndarray,
    model_name: String = "Model"
) -> Tuple[float, float, np.ndarray]:
    """
    Find the optimal classification threshold that maximizes F1-score based on 
    precision-recall curve, print performance metrics, and plot confusion matrix.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, val_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = f1_scores.argmax()
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"[{model_name}] Best threshold for F1: {best_thresh:.4f}, F1={best_f1:.4f}")

    val_pred_label = (val_pred_proba >= best_thresh).astype(int)
    auc_score = roc_auc_score(y_val, val_pred_proba)
    f1_val = f1_score(y_val, val_pred_label)

    print(f"[{model_name}] AUC={auc_score:.4f}, F1={f1_val:.4f}")
    print(classification_report(y_val, val_pred_label))

    cm = confusion_matrix(y_val, val_pred_label)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix ({model_name} + best F1 threshold)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    return best_thresh, auc_score, val_pred_label


def plot_learning_curve(
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    scoring: str = 'roc_auc'
) -> None:
    """
    Compute and plot the learning curve for the given estimator.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=estimator,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        shuffle=True,
        random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Train AUC')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='Val AUC')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("ROC-AUC")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def generate_submissions(
    stacking_clf: Any,
    xgb_model: Any,
    lgb_model: Any,
    rf_model: Any,
    cat_model: Any,
    test_df_imputed: pd.DataFrame,
    test_ids: pd.Series,
    best_thresh_s: float,
    best_thresh_w: float,
    weights: Dict[str, float]
) -> None:
    """
    Generate test probabilities and submission CSV files for both Stacking 
    and Weighted Ensemble models.
    """
    # (A) Stacking Inference & Submission
    test_proba_stack = stacking_clf.predict_proba(test_df_imputed)[:, 1]
    
    submission_stack = pd.DataFrame({
        "ID": test_ids,
        "probability": test_proba_stack
    })
    submission_stack.to_csv("final_stacking_submission.csv", index=False)
    print("Stacking submission: 'final_stacking_submission.csv' created!")

    # (B) Weighted Ensemble Inference & Submission
    test_proba_xgb = xgb_model.predict_proba(test_df_imputed)[:, 1]
    test_proba_lgb = lgb_model.predict_proba(test_df_imputed)[:, 1]
    test_proba_rf = rf_model.predict_proba(test_df_imputed)[:, 1]
    test_proba_cat = cat_model.predict_proba(test_df_imputed)[:, 1]

    test_ens_proba = (
        weights['w_xgb'] * test_proba_xgb +
        weights['w_lgb'] * test_proba_lgb +
        weights['w_rf'] * test_proba_rf +
        weights['w_cat'] * test_proba_cat
    )

    submission_w = pd.DataFrame({
        "ID": test_ids,
        "probability": test_ens_proba
    })
    submission_w.to_csv("final_weighted_submission.csv", index=False)
    print("Weighted submission: 'final_weighted_submission.csv' created!")