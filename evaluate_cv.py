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
  python evaluate_cv.py                        # the shared cache
  python evaluate_cv.py --data data/train.csv  # a file instead
"""

import argparse
import json
import os
import pathlib
import time
import warnings
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)

import catboost as cb
import lightgbm as lgb
import xgboost as xgb

from build_cache import ensure_cache
from src.encoding import sanitize_column_names
from src.optimization import build_model, model_params

warnings.filterwarnings("ignore")

# The target keeps its Korean name, as it appears in the competition file.
TARGET = "임신 성공 여부"   # "pregnancy success"
CACHE = pathlib.Path("data/gain_cache.npz")
SEED = 42


# ---------------------------------------------------------------- utilities

def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_dataset(path: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    The matrix every other script uses, or a file if one is named.

    This used to build its own representation — read the CSV, fillna(-1),
    get_dummies every object column — which is how it ended up measuring
    something subtly different from the pipeline, and how it missed the column
    name sanitising that LightGBM needs. Defaulting to the shared cache means
    the leak comparison below runs on the representation that actually ships.

    --data is kept for the older artefacts (a pickle, or the raw CSV) so the
    historical figures stay reproducible.
    """
    if path is None:
        ensure_cache(CACHE)
        cached = np.load(CACHE, allow_pickle=True)
        X = pd.DataFrame(cached["X"], columns=list(cached["columns"]))
        return X, pd.Series(cached["y"].astype(int), name=TARGET)

    if path.endswith(".pkl"):
        df = pd.read_pickle(path)
    else:
        df = pd.read_csv(path)
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])
        df = df.fillna(-1)
        object_columns = df.select_dtypes(include=["object"]).columns.tolist()
        if object_columns:
            df = pd.get_dummies(df, columns=object_columns, dummy_na=False)
        # One-hot names inherit the category text, which can carry characters
        # LightGBM rejects. src/encoding.py does this for the real pipeline.
        df, _ = sanitize_column_names(df, df.copy())

    if TARGET not in df.columns:
        raise SystemExit(
            f"target column {TARGET!r} not found. Columns: {list(df.columns)[:20]}"
        )

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(-1).astype(np.float32)
    return X, y


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
    parser.add_argument("--data", default=None,
                        help="a .pkl or .csv; omit to use the shared cache")
    parser.add_argument("--params-dir", default="configs")
    parser.add_argument("--out", default=None,
                        help="default cv_results.json, or a _sample suffix when "
                             "--sample is given, so a quick run cannot clobber "
                             "the recorded figures")
    parser.add_argument("--models", default="xgb,lgb,cat,rf")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--skip-leaky", action="store_true")
    parser.add_argument("--sample", type=int, default=0,
                        help="subsample rows, for timing runs")
    args = parser.parse_args()

    # A subsampled run measures something else entirely; letting it write to
    # cv_results.json would silently replace the numbers the README cites.
    if args.out is None:
        args.out = (f"cv_results_sample{args.sample}.json" if args.sample
                    else "cv_results.json")

    log(f"loading {args.data or CACHE}")
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
                    "source": os.path.basename(args.data) if args.data else str(CACHE)},
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
