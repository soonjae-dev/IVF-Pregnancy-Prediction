"""Missing values handled by type, compared on the baseline's folds.

The raw training file has five kinds of gaps (docs/experiments/MISSING_TYPES.md):

    1. DI not applicable   embryo/egg columns are empty for every DI row
    2. step not performed  an elapsed-day column is empty because the step it
                           times did not happen (no transfer, no frozen embryo, ...)
    3. truly missing       the step happened but its day was not recorded
    4. "no" left blank     yes/no columns whose only recorded value is 1
    5. unknown             no structural reason (years since last pregnancy)

The baseline fills every numeric gap with the training-fold median, which turns
type 4 into "yes" and mixes types 2 and 3. This script encodes each type
explicitly and compares the result with the baseline and with leaving gaps as
NaN, all on the same five folds. Rules are either row-wise (no fitting) or fitted
on the training part of each fold; no label is used.

    baseline        baseline_cv.py unchanged (median for every numeric gap)
    native_nan      every numeric gap left as NaN for LightGBM
    median_flags0   type 4 set to 0, then the baseline's median imputation
    typed_median    types 1, 2, 4 encoded; type 3 flagged; remaining gaps median
    typed_nan       types 1, 2, 4 encoded; type 3 flagged; remaining gaps NaN
    nan_tuned, nan_catboost, nan_blend        stronger models on native_nan input
    typed_tuned, typed_catboost, typed_blend  the same models on typed_nan input

Usage:
  python missing_types_cv.py                         # all variants (about 35 minutes)
  python missing_types_cv.py --sample 20000 --trials 3
"""

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

from baseline_cv import SEED, build_pipeline, collect_provenance, file_sha256, load_dataset
from improve_cv import (PROCEDURE, catboost_native, fit_di_zero, lgb_fixed, lgb_tuned,
                        one_hot, paired)

NOT_APPLICABLE = -1.0
# Elapsed-day column -> (column recording the step, value meaning "not performed").
DAY_RULES = {
    "배아 이식 경과일": ("이식된 배아 수", 0),
    "배아 해동 경과일": ("동결 배아 사용 여부", 0),
    "난자 채취 경과일": ("수집된 신선 난자 수", 0),
    "난자 혼합 경과일": ("혼합된 난자 수", 0),
    "난자 해동 경과일": ("해동 난자 수", 0),
}
VARIANTS = ["baseline", "native_nan", "median_flags0", "typed_median", "typed_nan",
            "nan_tuned", "nan_catboost", "nan_blend",
            "typed_tuned", "typed_catboost", "typed_blend"]
COMPARISONS = [("baseline", "native_nan"), ("baseline", "median_flags0"),
               ("median_flags0", "typed_median"), ("native_nan", "typed_nan"),
               ("typed_median", "typed_nan"), ("nan_tuned", "typed_tuned"),
               ("nan_catboost", "typed_catboost"), ("nan_blend", "typed_blend"),
               ("baseline", "typed_blend")]


def fit_flag_columns(X_train):
    """Type 4: numeric columns with gaps whose only recorded value is 1."""
    numeric = X_train.select_dtypes(include=np.number)
    return [c for c in numeric.columns
            if numeric[c].isna().any() and set(numeric[c].dropna().unique()) == {1.0}]


def fit_typed(X_train):
    """Fit the type rules on the training part; returns a transform for any rows."""
    flags = fit_flag_columns(X_train)
    _, di_columns = fit_di_zero(X_train)
    days = [c for c in DAY_RULES if c in X_train]

    def transform(X):
        X = X.copy()
        di = X[PROCEDURE] == "DI"
        for column in days:
            source, not_done_value = DAY_RULES[column]
            gap = X[column].isna()
            not_applicable = di | (X[source] == not_done_value)
            # Type 3: the step happened (or cannot be ruled out) but no day was recorded.
            X[column + "_truly_missing"] = (gap & ~not_applicable).astype(float)
            # Types 1 and 2: a day that cannot exist gets its own value.
            X.loc[gap & not_applicable, column] = NOT_APPLICABLE
        for column in flags:                       # type 4: blank means "no"
            X[column] = X[column].fillna(0.0)
        others = [c for c in di_columns if c not in days and c not in flags]
        X.loc[di, others] = X.loc[di, others].fillna(0.0)   # type 1: counts are 0
        return X

    return transform, {"flag_columns": flags, "di_zero_columns": len(di_columns),
                       "day_columns": days}


def fit_flags0(X_train):
    flags = fit_flag_columns(X_train)

    def transform(X):
        X = X.copy()
        X[flags] = X[flags].fillna(0.0)
        return X
    return transform, {"flag_columns": flags}


def one_hot_median(X_train, X_valid):
    """Like improve_cv.one_hot, but remaining numeric gaps get the training median."""
    categorical = X_train.select_dtypes(exclude=np.number).columns.tolist()
    numeric = X_train.select_dtypes(include=np.number).columns.tolist()
    encoder = ColumnTransformer(
        [("num", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
         ("cat", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), categorical)],
        sparse_threshold=1.0)
    return encoder.fit_transform(X_train), encoder.transform(X_valid)


def predict(variant, X_train, y_train, X_valid, args, oofs, valid_idx):
    if variant == "baseline":
        pipeline = build_pipeline(X_train, n_estimators=args.estimators)
        return pipeline.fit(X_train, y_train).predict_proba(X_valid)[:, 1], {}
    if variant.endswith("_blend"):
        prefix = variant.split("_")[0]
        parts = [oofs[prefix + "_tuned"][valid_idx], oofs[prefix + "_catboost"][valid_idx]]
        return np.mean([rankdata(p) / len(p) for p in parts], axis=0), {}

    info = {}
    if variant == "median_flags0":
        transform, info = fit_flags0(X_train)
        X_train, X_valid = transform(X_train), transform(X_valid)
        pipeline = build_pipeline(X_train, n_estimators=args.estimators)
        return pipeline.fit(X_train, y_train).predict_proba(X_valid)[:, 1], info
    if variant.startswith("typed"):
        transform, info = fit_typed(X_train)
        X_train, X_valid = transform(X_train), transform(X_valid)

    if variant.endswith("_catboost"):
        proba, extra = catboost_native(X_train, y_train, X_valid)
    elif variant == "typed_median":
        M_train, M_valid = one_hot_median(X_train, X_valid)
        proba, extra = lgb_fixed(M_train, y_train, M_valid, args.estimators)
    else:
        M_train, M_valid = one_hot(X_train, X_valid)
        if variant.endswith("_tuned"):
            proba, extra = lgb_tuned(M_train, y_train, M_valid, args.trials)
        else:
            proba, extra = lgb_fixed(M_train, y_train, M_valid, args.estimators)
    info.update(extra)
    return proba, info


def run_variant(variant, X, y, folds, args, oofs):
    oof = np.full(len(y), np.nan)
    fold_results, details = [], []
    started = perf_counter()
    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        proba, info = predict(variant, X.iloc[train_idx], y[train_idx], X.iloc[valid_idx],
                              args, oofs, valid_idx)
        if not np.isfinite(proba).all():
            raise ValueError(f"{variant}: non-finite predictions in fold {fold}")
        oof[valid_idx] = proba
        fold_results.append({"fold": fold,
                             "roc_auc": float(roc_auc_score(y[valid_idx], proba)),
                             "average_precision": float(average_precision_score(y[valid_idx], proba))})
        details.append(info)
        print(f"  {variant:14s} fold {fold}  AUC {fold_results[-1]['roc_auc']:.5f}", flush=True)
    if np.isnan(oof).any():
        raise ValueError(f"{variant}: some rows have no OOF prediction")
    return oof, {"oof_roc_auc": float(roc_auc_score(y, oof)),
                 "oof_average_precision": float(average_precision_score(y, oof)),
                 "fold_results": fold_results, "fold_details": details,
                 "elapsed_seconds": perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--estimators", type=int, default=250)
    args = parser.parse_args()
    args.folds = 5

    provenance = collect_provenance(args.data, args)
    provenance["code"]["missing_types_cv_sha256"] = file_sha256(__file__)
    provenance["code"]["improve_cv_sha256"] = file_sha256(Path(__file__).with_name("improve_cv.py"))
    provenance["environment"]["packages"].update(
        {name: version(name) for name in ["catboost", "optuna", "scipy"]})
    X, y = load_dataset(args.data)
    if args.sample is not None and args.sample < len(y):
        X, _, y, _ = train_test_split(X, y, train_size=args.sample, stratify=y,
                                      random_state=SEED)
    y_values = y.to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y_values))
    print(f"Rows {len(y):,}, input columns {X.shape[1]}", flush=True)

    output = args.output_dir / ("missing_types_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    results, oofs = {}, {}
    report = {"question": "Does encoding each type of missing value explicitly raise OOF ROC-AUC?",
              "settings": {"trials": args.trials, "sample": args.sample, "seed": SEED,
                           "not_applicable_day_value": NOT_APPLICABLE,
                           "day_rules": {k: list(v) for k, v in DAY_RULES.items()}},
              "provenance": provenance, "variants": results}

    for variant in VARIANTS:
        oofs[variant], results[variant] = run_variant(variant, X, y_values, folds, args, oofs)
        path = output / f"{variant}_oof.csv"
        pd.DataFrame({"source_row_index": y.index.to_numpy(), "y_true": y_values,
                      "oof_score": oofs[variant]}).to_csv(path, index=False)
        results[variant]["oof_artifact"] = {"filename": path.name, "sha256": file_sha256(path)}
        print(f"{variant:14s} OOF AUC {results[variant]['oof_roc_auc']:.5f}", flush=True)
        with (output / "comparison.partial.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)

    if file_sha256(args.data) != provenance["dataset"]["sha256"]:
        raise RuntimeError("Dataset changed during the experiment")
    report["paired"] = {f"{b} vs {a}": paired(results[a], results[b]) for a, b in COMPARISONS}
    report["interpretation"] = ("One seed and five folds that share training rows; deltas "
                                "are descriptive, not independent replicates or a "
                                "significance test.")
    with (output / "comparison.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    (output / "comparison.partial.json").unlink()
    print(f"Saved: {output / 'comparison.json'}")


if __name__ == "__main__":
    main()
