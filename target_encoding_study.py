#!/usr/bin/env python3
"""
target_encoding_study.py — is the age target encoding leaking, and does it help?

src/features.py::create_age_success_rate builds the feature
"연령대 평균 임신 성공률" by grouping the training set on the age bracket and
taking the mean of the label. Two things are wrong with that in principle:

    1. every training row's own label is inside the group mean it receives
    2. the mapping is fitted once on the whole training set, before any
       cross-validation split, so validation-fold labels reach the training
       rows and vice versa

That is the shape of a target-encoding leak. Whether it is a leak worth
anything is an empirical question, and the answer depends entirely on the
cardinality: the age bracket takes 7 values and the smallest group holds 329
rows, so a single row moves its own group mean by at most 1/329.

This scores four variants on identical folds:

    global    the feature as shipped, fitted on the whole training set
    oof       fitted inside each fold: the mapping comes from the outer
              training portion only, and within that portion each row's value
              comes from a nested 5-fold split, so no row ever sees its own
              label
    dropped   the column removed entirely
    onehot    the column replaced by 7 indicator columns for the age bracket,
              which carry the same grouping and no label information at all

`global` minus `oof` is the size of the leak. `oof` minus `dropped` is what
the feature is worth once it is computed honestly. `oof` minus `onehot` is
what the label-derived *values* add over the bare grouping — if that is zero,
the encoding is carrying no label information worth having and the question of
whether it leaks stops mattering.

The grouping matters here more than it looks: "시술 당시 나이" is an object
column, so encoding.py drops it, and this feature is the only path by which age
reaches the model at all. Success rate runs 0.323 to 0.118 across the brackets,
which is why dropping it costs as much as it does.

Usage:
  python target_encoding_study.py [--out target_encoding_study.json]
"""

import argparse
import json
import pathlib
import time
import warnings

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

SEED = 42
CACHE = pathlib.Path("data/gain_cache.npz")
FEATURE = "연령대 평균 임신 성공률"   # "average pregnancy success rate by age band"
MODELS = ["lgb", "xgb"]
VARIANTS = ["global", "oof", "dropped", "onehot"]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_cached():
    """The matrix built by resampling_study.py / ensemble_study.py."""
    if not CACHE.exists():
        raise SystemExit(
            f"{CACHE} not found. Run `python resampling_study.py` first — it "
            "builds and caches the imputed matrix.")
    cached = np.load(CACHE, allow_pickle=True)
    return cached["X"], cached["y"], list(cached["columns"])


def make_model(name, params):
    tuned = {k: v for k, v in params.items() if k != "best_value"}
    if name == "lgb":
        return lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **tuned)
    if name == "xgb":
        return xgb.XGBClassifier(random_state=SEED, eval_metric="logloss",
                                 n_jobs=-1, tree_method="hist", **tuned)
    raise ValueError(name)


def best_f1_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    index = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[index])


def out_of_fold_encoding(groups, y, index, prior, n_splits=5):
    """
    Target encoding for `index` that no row in `index` contributed to.

    The mapping is fitted on the rows in `index` only. Within those rows the
    value each one receives comes from a nested split it was held out of, so a
    row's own label never reaches its own feature. Groups unseen in a nested
    training part fall back to the prior.
    """
    encoded = np.full(len(index), prior, dtype=np.float64)
    inner = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    for fit_positions, apply_positions in inner.split(index, y[index]):
        fit_rows = index[fit_positions]
        means = {}
        for group in np.unique(groups[fit_rows]):
            mask = groups[fit_rows] == group
            means[group] = float(y[fit_rows][mask].mean())
        encoded[apply_positions] = [
            means.get(g, prior) for g in groups[index[apply_positions]]
        ]
    return encoded


def full_encoding(groups, y, fit_index, apply_index, prior):
    """Mapping fitted on fit_index, applied to apply_index. No nesting."""
    means = {}
    for group in np.unique(groups[fit_index]):
        mask = groups[fit_index] == group
        means[group] = float(y[fit_index][mask].mean())
    return np.array([means.get(g, prior) for g in groups[apply_index]])


def evaluate(X, y, groups, column, name, params, variant, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    out_of_fold = np.zeros(len(y))
    fold_f1 = []
    started = time.time()

    if variant == "onehot":
        indicators = np.stack([(groups == g).astype(np.float32)
                               for g in np.unique(groups)], axis=1)
        X_onehot = np.hstack([np.delete(X, column, axis=1), indicators])

    for train_index, valid_index in cv.split(X, y):
        if variant == "dropped":
            X_work = np.delete(X, column, axis=1)
        elif variant == "onehot":
            X_work = X_onehot
        else:
            X_work = X

        if variant == "oof":
            # rebuild the column from the training fold alone
            X_work = X_work.copy()
            prior = float(y[train_index].mean())
            X_work[train_index, column] = out_of_fold_encoding(
                groups, y, train_index, prior)
            X_work[valid_index, column] = full_encoding(
                groups, y, train_index, valid_index, prior)

        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_work[train_index], y[train_index], test_size=0.2,
            random_state=SEED, stratify=y[train_index])

        model = make_model(name, params).fit(X_fit, y_fit)
        threshold = best_f1_threshold(y_cal, model.predict_proba(X_cal)[:, 1])
        proba = model.predict_proba(X_work[valid_index])[:, 1]
        out_of_fold[valid_index] = proba
        fold_f1.append(f1_score(y[valid_index], (proba >= threshold).astype(int)))

    result = {
        "oof_auc": float(roc_auc_score(y, out_of_fold)),
        "f1_mean": float(np.mean(fold_f1)),
        "f1_std": float(np.std(fold_f1)),
        "seconds": time.time() - started,
    }
    log(f"  {name:4s} {variant:9s} AUC {result['oof_auc']:.4f}   "
        f"F1 {result['f1_mean']:.4f} ± {result['f1_std']:.4f}   "
        f"({result['seconds']:.0f}s)")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target_encoding_study.json")
    args = parser.parse_args()

    X, y, columns = load_cached()
    if FEATURE not in columns:
        raise SystemExit(f"'{FEATURE}' is not in the cached matrix. Columns: {columns}")
    column = columns.index(FEATURE)

    # The encoded value is a function of the age bracket and nothing else, so
    # its distinct values recover the grouping without re-reading the raw file.
    groups = X[:, column].astype(np.float64).round(6)
    sizes = {float(g): int((groups == g).sum()) for g in np.unique(groups)}
    log(f"{len(y):,} rows, {X.shape[1]} features, positive rate {y.mean():.1%}")
    log(f"'{FEATURE}' takes {len(sizes)} distinct values; "
        f"smallest group {min(sizes.values()):,} rows, "
        f"largest {max(sizes.values()):,}")

    results = {"rows": int(len(y)), "features": int(X.shape[1]),
               "feature": FEATURE, "group_sizes": sizes, "models": {}}

    for name in MODELS:
        path = pathlib.Path("configs") / f"{name}_params.json"
        if not path.exists():
            log(f"  ! {path} missing, skipping {name}")
            continue
        params = json.load(open(path))
        log(f"-- {name} --")
        results["models"][name] = {
            variant: evaluate(X, y, groups, column, name, params, variant)
            for variant in VARIANTS
        }
        with open(args.out, "w") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)

    print()
    log("summary — out-of-fold ROC-AUC")
    print(f"  {'model':6s}" + "".join(f"{v:>11s}" for v in VARIANTS)
          + f"{'leak':>11s}{'vs drop':>11s}{'vs 1hot':>11s}")
    leaks, worth, over_onehot = [], [], []
    for name, entry in results["models"].items():
        row = f"  {name:6s}" + "".join(f"{entry[v]['oof_auc']:>11.4f}" for v in VARIANTS)
        leak = entry["global"]["oof_auc"] - entry["oof"]["oof_auc"]
        value = entry["oof"]["oof_auc"] - entry["dropped"]["oof_auc"]
        extra = entry["oof"]["oof_auc"] - entry["onehot"]["oof_auc"]
        leaks.append(leak); worth.append(value); over_onehot.append(extra)
        print(row + f"{leak:>+11.4f}{value:>+11.4f}{extra:>+11.4f}")
    if leaks:
        print(f"\n  leak         (global - oof)      averaged: {np.mean(leaks):+.4f}")
        print(f"  feature      (oof - dropped)    averaged: {np.mean(worth):+.4f}")
        print(f"  label values (oof - onehot)     averaged: {np.mean(over_onehot):+.4f}")
        results["mean_leak_auc"] = float(np.mean(leaks))
        results["mean_feature_value_auc"] = float(np.mean(worth))
        results["mean_value_over_onehot_auc"] = float(np.mean(over_onehot))
        with open(args.out, "w") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
    log(f"written to {args.out}")


if __name__ == "__main__":
    main()
