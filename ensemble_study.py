#!/usr/bin/env python3
"""
ensemble_study.py — what do the ensembles add over the best single model?

The repository builds a stacking classifier and an Optuna-weighted blend on top
of four base models. This measures what each is worth, on the representation the
pipeline actually produces (GAIN with the target held out) and on folds none of
them were fitted on.

Protocol, per outer fold of a 5-fold StratifiedKFold:

    1. split the training portion into a fit part (80%) and a calibration
       part (20%)
    2. fit all four base models on the fit part
    3. take their probabilities on the calibration part and on the validation
       fold
    4. build both ensembles using ONLY the calibration probabilities:
         weighted  — blend weights searched to maximise calibration AUC
         stacked   — logistic regression on the calibration probabilities
    5. apply them to the validation fold, which nothing has touched

Every reported number is therefore out-of-fold for the base models, the meta
learners and the decision threshold alike.

This differs from main.py's StackingClassifier, which fits its meta model on an
internal cross-validation of the training data. Using the calibration split
instead costs one round of base fits rather than five, and removes the internal
resampling leak noted in the audit.

Usage:
  python ensemble_study.py [--out ensemble_study.json]
"""

import argparse
import contextlib
import io
import json
import pathlib
import time

import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import catboost as cb
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

TARGET = "임신 성공 여부"   # "pregnancy success"
SEED = 42
CACHE = pathlib.Path("data/gain_cache.npz")
BASE_MODELS = ["xgb", "lgb", "cat", "rf"]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def build_dataset():
    """
    Steps 1-6 of main.py, cached.

    The cache is shared with resampling_study.py and target_encoding_study.py,
    so it has to carry `columns` even though this script does not need them —
    otherwise whichever script runs first decides whether the others work.
    """
    if CACHE.exists():
        log(f"loading cached matrix from {CACHE}")
        cached = np.load(CACHE, allow_pickle=True)
        return cached["X"], cached["y"]

    import torch

    from src.config import set_seed
    from src.encoding import run_encoding_pipeline
    from src.feature_selection import run_feature_selection
    from src.features import run_feature_engineering
    from src.preprocessing import run_basic_preprocessing
    from src.run_imputation import run_gain_on_dataframes

    config = json.load(open("configs/config.json"))
    set_seed(SEED)

    log("preprocessing and feature pipeline")
    train_df, test_df, _ = quiet(
        run_basic_preprocessing,
        config["pipeline"]["train_data_path"], config["pipeline"]["test_data_path"])
    train_df, test_df = quiet(run_feature_engineering, train_df, test_df)
    train_df, test_df = quiet(run_feature_selection, train_df, test_df)
    train_df, test_df = quiet(run_encoding_pipeline, train_df, test_df)

    log("GAIN imputation (target held out)")
    train_df, _ = quiet(run_gain_on_dataframes, train_df, test_df,
                        torch.device("cpu"),
                        iterations=config["imputation"]["gain_iterations"])

    y = train_df[TARGET].astype(int).values
    features = train_df.drop(columns=[TARGET])
    X = features.values.astype(np.float32)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, X=X, y=y,
                        columns=np.array(features.columns, dtype=object))
    return X, y


def make_model(name, params):
    tuned = {k: v for k, v in params.items() if k != "best_value"}
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


def best_f1_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    index = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[index])


def search_weights(y_cal, cal_probs, n_draws=3000):
    """Dirichlet search for blend weights, scored on the calibration split."""
    rng = np.random.default_rng(SEED)
    names = list(cal_probs)
    best_auc, best = -1.0, None
    for _ in range(n_draws):
        weights = rng.dirichlet(np.ones(len(names)))
        blended = sum(weights[i] * cal_probs[n] for i, n in enumerate(names))
        auc = roc_auc_score(y_cal, blended)
        if auc > best_auc:
            best_auc, best = auc, weights
    return {n: float(best[i]) for i, n in enumerate(names)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ensemble_study.json")
    args = parser.parse_args()

    X, y = build_dataset()
    log(f"{len(y):,} rows, {X.shape[1]} features, positive rate {y.mean():.1%}")

    params = {}
    for name in BASE_MODELS:
        path = pathlib.Path("configs") / f"{name}_params.json"
        if not path.exists():
            log(f"  ! {path} missing")
            return
        params[name] = json.load(open(path))

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    candidates = BASE_MODELS + ["weighted", "stacked"]
    oof = {name: np.zeros(len(y)) for name in candidates}
    fold_f1 = {name: [] for name in candidates}
    weight_log = []

    for fold, (train_index, valid_index) in enumerate(cv.split(X, y), 1):
        started = time.time()
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X[train_index], y[train_index], test_size=0.2,
            random_state=SEED, stratify=y[train_index])

        cal_probs, valid_probs = {}, {}
        for name in BASE_MODELS:
            model = make_model(name, params[name]).fit(X_fit, y_fit)
            cal_probs[name] = model.predict_proba(X_cal)[:, 1]
            valid_probs[name] = model.predict_proba(X[valid_index])[:, 1]

        # weighted blend — weights from the calibration split only
        weights = search_weights(y_cal, cal_probs)
        weight_log.append(weights)
        cal_probs["weighted"] = sum(weights[n] * cal_probs[n] for n in BASE_MODELS)
        valid_probs["weighted"] = sum(weights[n] * valid_probs[n] for n in BASE_MODELS)

        # stacked — logistic regression on the calibration probabilities
        meta = LogisticRegression(solver="liblinear", random_state=SEED)
        meta.fit(np.column_stack([cal_probs[n] for n in BASE_MODELS]), y_cal)
        cal_probs["stacked"] = meta.predict_proba(
            np.column_stack([cal_probs[n] for n in BASE_MODELS]))[:, 1]
        valid_probs["stacked"] = meta.predict_proba(
            np.column_stack([valid_probs[n] for n in BASE_MODELS]))[:, 1]

        for name in candidates:
            threshold = best_f1_threshold(y_cal, cal_probs[name])
            probs = valid_probs[name]
            oof[name][valid_index] = probs
            fold_f1[name].append(f1_score(y[valid_index], (probs >= threshold).astype(int)))

        log(f"  fold {fold}/5 done ({time.time() - started:.0f}s)   "
            f"weights " + " ".join(f"{n}={weights[n]:.2f}" for n in BASE_MODELS))

    results = {"rows": int(len(y)), "features": int(X.shape[1]),
               "positive_rate": float(y.mean()),
               "fold_weights": weight_log, "models": {}}

    print()
    log("out-of-fold results")
    print(f"  {'':20s}{'ROC-AUC':>10s}{'F1':>10s}{'':>8s}")
    baseline_f1 = 2 * y.mean() / (1 + y.mean())
    for name in candidates:
        auc = float(roc_auc_score(y, oof[name]))
        f1_mean, f1_std = float(np.mean(fold_f1[name])), float(np.std(fold_f1[name]))
        results["models"][name] = {"oof_auc": auc, "f1_mean": f1_mean, "f1_std": f1_std}
        mark = "  <- base" if name in BASE_MODELS else "  <- ensemble"
        print(f"  {name:20s}{auc:10.4f}{f1_mean:10.4f}{mark}")

    best_base = max(BASE_MODELS, key=lambda n: results["models"][n]["oof_auc"])
    base_auc = results["models"][best_base]["oof_auc"]
    base_f1 = results["models"][best_base]["f1_mean"]
    print(f"\n  best single model: {best_base}  "
          f"AUC {base_auc:.4f}  F1 {base_f1:.4f}")
    for name in ("weighted", "stacked"):
        entry = results["models"][name]
        print(f"  {name:9s} vs {best_base}:  "
              f"AUC {entry['oof_auc'] - base_auc:+.4f}   F1 {entry['f1_mean'] - base_f1:+.4f}")
    print(f"\n  all-positive baseline F1 = {baseline_f1:.4f}")
    results["best_base_model"] = best_base
    results["all_positive_f1"] = float(baseline_f1)

    with open(args.out, "w") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    log(f"written to {args.out}")


if __name__ == "__main__":
    main()
