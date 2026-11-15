#!/usr/bin/env python3
"""
resampling_study.py — does SMOTE earn its place in this pipeline?

Runs the repository's own preprocessing (steps 1-6 of main.py, GAIN included),
then scores every base model under three imbalance strategies on identical
folds:

    none                no resampling at all
    smote               SMOTE fitted inside each training fold
    scale_pos_weight    cost reweighting, no synthetic data

ROC-AUC comes from pooled out-of-fold probabilities. F1 uses a threshold chosen
on a calibration split carved out of each training fold, so the validation fold
is never used for any tuning decision.

The imputed matrix is cached to data/gain_cache.npz, because GAIN takes a few
minutes and the comparison does not depend on re-running it.

Usage:
  python resampling_study.py [--sample N] [--out resampling_study.json]
"""

import argparse
import json
import pathlib
import time
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import catboost as cb
import lightgbm as lgb
import xgboost as xgb

from build_cache import ensure_cache
from src.optimization import model_params

warnings.filterwarnings("ignore")

TARGET = "임신 성공 여부"   # "pregnancy success"
SEED = 42
CACHE = pathlib.Path("data/gain_cache.npz")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_dataset(config):
    """
    The imputed matrix, built once by build_cache.py and shared with the other
    studies. That build runs in its own process — see build_cache.py for why.
    """
    ensure_cache(CACHE, iterations=config["imputation"]["gain_iterations"])
    cached = np.load(CACHE, allow_pickle=True)
    return cached["X"], cached["y"], list(cached["columns"])


def make_model(name, params, scale_pos_weight=None):
    tuned = model_params(params)
    if name == "xgb":
        extra = {"scale_pos_weight": scale_pos_weight} if scale_pos_weight else {}
        return xgb.XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=-1,
                                 tree_method="hist", **tuned, **extra)
    if name == "lgb":
        extra = {"scale_pos_weight": scale_pos_weight} if scale_pos_weight else {}
        return lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **tuned, **extra)
    if name == "cat":
        extra = {"scale_pos_weight": scale_pos_weight} if scale_pos_weight else {}
        return cb.CatBoostClassifier(random_state=SEED, verbose=0, thread_count=-1,
                                     allow_writing_files=False, **tuned, **extra)
    if name == "rf":
        extra = {"class_weight": "balanced"} if scale_pos_weight else {}
        return RandomForestClassifier(random_state=SEED, n_jobs=-1, **tuned, **extra)
    raise ValueError(name)


def best_f1_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    index = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[index])


def evaluate(X, y, name, params, strategy, n_splits=5):
    """Out-of-fold AUC and F1 for one (model, strategy) pair."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    ratio = float((y == 0).sum() / (y == 1).sum())

    out_of_fold = np.zeros(len(y))
    fold_f1 = []
    started = time.time()

    for train_index, valid_index in cv.split(X, y):
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X[train_index], y[train_index], test_size=0.2,
            random_state=SEED, stratify=y[train_index])

        if strategy == "smote":
            X_fit, y_fit = SMOTE(random_state=SEED).fit_resample(X_fit, y_fit)

        weight = ratio if strategy == "scale_pos_weight" else None
        model = make_model(name, params, scale_pos_weight=weight).fit(X_fit, y_fit)

        threshold = best_f1_threshold(y_cal, model.predict_proba(X_cal)[:, 1])
        proba = model.predict_proba(X[valid_index])[:, 1]
        out_of_fold[valid_index] = proba
        fold_f1.append(f1_score(y[valid_index], (proba >= threshold).astype(int)))

    result = {
        "oof_auc": float(roc_auc_score(y, out_of_fold)),
        "f1_mean": float(np.mean(fold_f1)),
        "f1_std": float(np.std(fold_f1)),
        "seconds": time.time() - started,
    }
    log(f"  {name:4s} {strategy:17s} AUC {result['oof_auc']:.4f}   "
        f"F1 {result['f1_mean']:.4f} ± {result['f1_std']:.4f}   "
        f"({result['seconds']:.0f}s)")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--out", default="resampling_study.json")
    args = parser.parse_args()

    config = json.load(open("configs/config.json"))
    X, y, columns = build_dataset(config)

    if args.sample:
        X, _, y, _ = train_test_split(X, y, train_size=args.sample,
                                      random_state=SEED, stratify=y)
        log(f"subsampled to {args.sample:,} rows")

    positives = int(y.sum())
    log(f"{len(y):,} rows, {X.shape[1]} features, "
        f"positive rate {positives / len(y):.1%} "
        f"(ratio {(len(y) - positives) / positives:.2f}:1)")

    param_files = {"xgb": "xgb_params.json", "lgb": "lgb_params.json",
                   "cat": "cat_params.json", "rf": "rf_params.json"}
    strategies = ["none", "smote", "scale_pos_weight"]

    results = {"rows": len(y), "features": X.shape[1],
               "positive_rate": positives / len(y), "models": {}}

    for name, filename in param_files.items():
        path = pathlib.Path("configs") / filename
        if not path.exists():
            log(f"  ! {name}: {path} missing, skipping")
            continue
        params = json.load(open(path))
        log(f"-- {name} --")
        results["models"][name] = {
            strategy: evaluate(X, y, name, params, strategy) for strategy in strategies
        }
        with open(args.out, "w") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)

    print()
    log("summary — out-of-fold ROC-AUC")
    header = f"  {'model':6s}" + "".join(f"{s:>20s}" for s in strategies)
    print(header)
    deltas = []
    for name, entry in results["models"].items():
        row = f"  {name:6s}"
        for strategy in strategies:
            row += f"{entry[strategy]['oof_auc']:>20.4f}"
        print(row)
        deltas.append(entry["smote"]["oof_auc"] - entry["none"]["oof_auc"])
    if deltas:
        print(f"\n  SMOTE minus no-resampling, averaged over "
              f"{len(deltas)} models: {np.mean(deltas):+.4f}")
        results["mean_smote_delta_auc"] = float(np.mean(deltas))
        with open(args.out, "w") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
    log(f"written to {args.out}")


if __name__ == "__main__":
    main()
