#!/usr/bin/env python3
"""
evaluate_cv.py — leakage-free cross-validation for the IVF pregnancy model.

Runs two protocols side by side on the same data.

  [A] LEAKY    Reproduces the original main.py: SMOTE is fitted on the whole
               training set and the resampled data is handed to cross_val_score.
               This exists to show where the best_value figures in configs/*.json
               came from. It is not a number to report.

  [B] HONEST   StratifiedKFold(5). SMOTE is fitted on the training portion of
               each fold only; the validation fold keeps its original class
               distribution. ROC-AUC is computed on pooled out-of-fold
               probabilities. The F1 threshold is chosen on a calibration split
               carved out of the training portion and then applied to the
               validation fold, which is never used for any fitting or tuning
               decision.

Why [A] is wrong: SMOTE interpolates between neighbouring minority samples, so a
synthetic point derived from sample A can land in a training fold while A itself
sits in the validation fold. The model has effectively already seen the
neighbourhood of the answer.

Usage:
  python evaluate_cv.py --data <train_df_imputed.pkl | data/train.csv>
"""

import argparse
import json
import os
import time
import warnings
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)

import catboost as cb
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# The target keeps its Korean name, as it appears in the competition file.
TARGET = "임신 성공 여부"   # "pregnancy success"
SEED = 42


# ---------------------------------------------------------------- utilities

def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_dataset(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load either the preprocessed pickle or the raw competition CSV."""
    if path.endswith(".pkl"):
        df = pd.read_pickle(path)
    else:
        df = pd.read_csv(path)
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])
        df = df.fillna(-1)
        # Minimal encoding, as a fallback when the preprocessed pickle is absent.
        object_columns = df.select_dtypes(include=["object"]).columns.tolist()
        if object_columns:
            df = pd.get_dummies(df, columns=object_columns, dummy_na=False)

    if TARGET not in df.columns:
        raise SystemExit(
            f"target column {TARGET!r} not found. Columns: {list(df.columns)[:20]}"
        )

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(-1).astype(np.float32)
    return X, y


def build_model(name: str, params: Dict[str, Any]):
    tuned = model_params(params)
    if name == "xgb":
        return xgb.XGBClassifier(random_state=SEED, eval_metric="logloss",
                                 n_jobs=-1, tree_method="hist", **tuned)
    if name == "lgb":
        return lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **tuned)
    if name == "cat":
        return cb.CatBoostClassifier(random_state=SEED, verbose=0, thread_count=-1,
                                     allow_writing_files=False, **tuned)
    if name == "rf":
        return RandomForestClassifier(random_state=SEED, n_jobs=-1, **tuned)
    raise ValueError(name)


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, float]:
    """Threshold that maximises F1, and the F1 it reaches."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    index = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[index]), float(f1[index])


# ------------------------------------------------------- protocol A (leaky)

def run_leaky(X: pd.DataFrame, y: pd.Series, name: str, params: Dict[str, Any],
              n_splits: int = 5) -> Dict[str, Any]:
    """Reproduce the original pipeline: SMOTE applied before the CV split."""
    log(f"  [A/leaky] {name}: SMOTE on the whole training set, then CV")
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_resampled, y_resampled = SMOTE(random_state=SEED).fit_resample(X_train, y_train)
    log(f"            {X_resampled.shape[0]:,} rows after SMOTE "
        f"(from {X_train.shape[0]:,})")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    start = time.time()
    scores = cross_val_score(build_model(name, params), X_resampled, y_resampled,
                             cv=cv, scoring="roc_auc", n_jobs=1)
    log(f"            AUC {scores.mean():.4f} +/- {scores.std():.4f}  "
        f"({time.time() - start:.0f}s)")
    return {"auc_mean": float(scores.mean()), "auc_std": float(scores.std()),
            "folds": [float(s) for s in scores], "seconds": time.time() - start}


# ------------------------------------------------------ protocol B (honest)

def run_honest(X: pd.DataFrame, y: pd.Series, name: str, params: Dict[str, Any],
               n_splits: int = 5) -> Dict[str, Any]:
    """SMOTE inside the fold; threshold chosen on an inner calibration split."""
    log(f"  [B/honest] {name}: SMOTE on the training portion of each fold only")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    X_values, y_values = X.values, y.values

    out_of_fold = np.zeros(len(y), dtype=np.float64)
    fold_auc, fold_f1, fold_threshold = [], [], []
    start = time.time()

    for fold, (train_index, valid_index) in enumerate(cv.split(X_values, y_values), 1):
        fold_start = time.time()
        X_train, y_train = X_values[train_index], y_values[train_index]
        X_valid, y_valid = X_values[valid_index], y_values[valid_index]

        # Carve a calibration set out of the training portion. The validation
        # fold is never touched.
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train)

        # SMOTE only on the part the model is fitted to.
        X_fit_resampled, y_fit_resampled = SMOTE(random_state=SEED).fit_resample(
            X_fit, y_fit)

        model = build_model(name, params)
        model.fit(X_fit_resampled, y_fit_resampled)

        # Threshold from the calibration set, which keeps its original distribution.
        threshold, _ = best_f1_threshold(y_cal, model.predict_proba(X_cal)[:, 1])

        # Score on the fold the model has never seen.
        valid_proba = model.predict_proba(X_valid)[:, 1]
        out_of_fold[valid_index] = valid_proba
        auc = roc_auc_score(y_valid, valid_proba)
        f1 = f1_score(y_valid, (valid_proba >= threshold).astype(int))

        fold_auc.append(float(auc))
        fold_f1.append(float(f1))
        fold_threshold.append(float(threshold))
        log(f"            fold {fold}/{n_splits}  AUC {auc:.4f}  F1 {f1:.4f}  "
            f"thr {threshold:.3f}  ({time.time() - fold_start:.0f}s)")

    oof_auc = float(roc_auc_score(y_values, out_of_fold))
    log(f"            -> OOF AUC {oof_auc:.4f} | "
        f"AUC {np.mean(fold_auc):.4f} +/- {np.std(fold_auc):.4f} | "
        f"F1 {np.mean(fold_f1):.4f} +/- {np.std(fold_f1):.4f}  "
        f"({time.time() - start:.0f}s)")

    return {
        "oof_auc": oof_auc,
        "oof_average_precision": float(average_precision_score(y_values, out_of_fold)),
        "auc_mean": float(np.mean(fold_auc)), "auc_std": float(np.std(fold_auc)),
        "f1_mean": float(np.mean(fold_f1)), "f1_std": float(np.std(fold_f1)),
        "threshold_mean": float(np.mean(fold_threshold)),
        "fold_auc": fold_auc, "fold_f1": fold_f1, "fold_threshold": fold_threshold,
        "seconds": time.time() - start,
        "oof_proba": out_of_fold.tolist(),
    }


# ------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--params-dir", default="configs")
    parser.add_argument("--out", default="cv_results.json")
    parser.add_argument("--models", default="xgb,lgb,cat,rf")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--skip-leaky", action="store_true")
    parser.add_argument("--sample", type=int, default=0,
                        help="subsample rows, for timing runs")
    args = parser.parse_args()

    log(f"loading {args.data}")
    X, y = load_dataset(args.data)
    if args.sample:
        X, _, y, _ = train_test_split(X, y, train_size=args.sample,
                                      random_state=SEED, stratify=y)
        log(f"subsampled to {args.sample:,} rows")
    positives, total = int(y.sum()), len(y)
    log(f"shape {X.shape}, positives {positives:,}/{total:,} ({positives / total:.1%})")

    param_files = {"xgb": "xgb_params.json", "lgb": "lgb_params.json",
                   "cat": "cat_params.json", "rf": "rf_params.json"}
    results: Dict[str, Any] = {
        "dataset": {"rows": total, "features": int(X.shape[1]),
                    "positive_rate": positives / total,
                    "source": os.path.basename(args.data)},
        "protocol": {
            "honest": (f"StratifiedKFold({args.folds}), SMOTE inside the training "
                       "fold only, threshold tuned on an inner 20% calibration split"),
            "leaky": ("SMOTE on the full training set, then StratifiedKFold -- "
                      "reproduces the original configs/*.json"),
        },
        "models": {},
    }

    for name in (n.strip() for n in args.models.split(",")):
        params_path = os.path.join(args.params_dir, param_files[name])
        if not os.path.exists(params_path):
            log(f"  ! {name}: no parameter file at {params_path} -- skipping")
            continue
        params = json.load(open(params_path))
        tuned = model_params(params)
        log(f"-- {name} -- params={tuned}")

        entry: Dict[str, Any] = {"params": params,
                                 "recorded_best_value": params.get("best_value")}
        entry["honest"] = run_honest(X, y, name, params, args.folds)
        if not args.skip_leaky:
            entry["leaky"] = run_leaky(X, y, name, params, args.folds)
        results["models"][name] = entry

        # Checkpoint after each model: a full run takes a while.
        with open(args.out, "w") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)

    # Weighted ensemble over the out-of-fold probabilities. The folds are
    # identical across models, so the vectors line up row for row.
    oof = {name: np.array(entry["honest"]["oof_proba"])
           for name, entry in results["models"].items() if "honest" in entry}
    if len(oof) >= 2:
        log("-- weighted ensemble (blending out-of-fold probabilities) --")
        names = list(oof)
        rng = np.random.default_rng(SEED)
        best_auc, best_weights = -1.0, None
        for _ in range(2000):
            weights = rng.dirichlet(np.ones(len(names)))
            blended = sum(weights[i] * oof[n] for i, n in enumerate(names))
            auc = roc_auc_score(y.values, blended)
            if auc > best_auc:
                best_auc, best_weights = auc, weights
        blended = sum(best_weights[i] * oof[n] for i, n in enumerate(names))
        threshold, f1 = best_f1_threshold(y.values, blended)
        log(f"            weights={dict(zip(names, np.round(best_weights, 3)))}")
        log(f"            OOF AUC {best_auc:.4f}  F1 {f1:.4f} @ thr {threshold:.3f}")
        results["ensemble_weighted"] = {
            "weights": {n: float(best_weights[i]) for i, n in enumerate(names)},
            "oof_auc": float(best_auc), "oof_f1": float(f1),
            "threshold": float(threshold),
            "note": ("weights and threshold were both selected on the out-of-fold "
                     "probabilities, so this is optimistic relative to the "
                     "individual models"),
        }

    for entry in results["models"].values():
        entry["honest"].pop("oof_proba", None)
    with open(args.out, "w") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    log(f"written to {args.out}")


if __name__ == "__main__":
    main()
