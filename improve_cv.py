"""Controlled improvement experiments on top of the fold-fitted baseline.

Every variant uses the same rows and the same five shuffled stratified folds
(seed 42) as baseline_cv.py, so each step can be compared with the previous one
fold by fold. Anything learned from data is learned inside the training part of
each outer fold:

    baseline     baseline_cv.py unchanged (median imputation, one-hot, LightGBM)
    native_nan   numeric gaps left as NaN for LightGBM's own missing-value split
    di_zero      gaps that mean "not applicable" (columns that are empty for every
                 DI row of the training fold) set to 0 on DI rows
    ordinal      count strings ("3회", "6회 이상") and age bands as ordered numbers
    features     row-wise domain ratios (no labels, nothing fitted)
    lgb_tuned    LightGBM tuned by Optuna on an inner split of each training fold,
                 tree count by early stopping on that split, then refitted
    catboost     CatBoost with native categorical handling on the same features,
                 tree count by early stopping on an inner split, then refitted
    blend        equal-weight rank average of lgb_tuned and catboost (no weights
                 are fitted, so nothing is selected on the validation fold)

The ladder is cumulative: each step keeps the changes above it, and the report
gives paired fold differences against the previous step and against baseline.
One seed and five folds that share training rows: differences are descriptive,
not a significance test.

Usage:
  python improve_cv.py                    # all variants (about an hour)
  python improve_cv.py --trials 10        # fewer Optuna trials per fold
  python improve_cv.py --sample 20000     # quick structural check
"""

import argparse
import json
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

from baseline_cv import (SEED, build_pipeline, collect_provenance, file_sha256,
                         load_dataset)

MISSING = "__MISSING__"
PROCEDURE = "시술 유형"
COUNT_PATTERN = re.compile(r"^(\d+)회")
AGE_BANDS = {
    "시술 당시 나이": ["만18-34세", "만35-37세", "만38-39세", "만40-42세", "만43-44세", "만45-50세"],
    "난자 기증자 나이": ["만20세 이하", "만21-25세", "만26-30세", "만31-35세"],
    "정자 기증자 나이": ["만20세 이하", "만21-25세", "만26-30세", "만31-35세", "만36-40세", "만41-45세"],
}
VARIANTS = ["baseline", "native_nan", "di_zero", "ordinal", "features",
            "lgb_tuned", "catboost", "blend"]
INNER_SPLIT = 0.2


# ------------------------------------------------------------ preprocessing
# Stateless steps act on one row at a time. Fitted steps take the training part
# of a fold and return a function that is applied to both parts.

def count_to_number(series):
    """'0회' .. '5회' -> 0..5, '6회 이상' -> 6, anything else -> NaN."""
    return series.map(lambda v: float(COUNT_PATTERN.match(v).group(1))
                      if isinstance(v, str) and COUNT_PATTERN.match(v) else np.nan)


def ordinal_columns(X):
    """Count strings and age bands as ordered numbers. Unknown values become NaN."""
    X = X.copy()
    for column in X.columns:
        if X[column].dtype == object and X[column].str.match(COUNT_PATTERN).any():
            X[column] = count_to_number(X[column])
    for column, bands in AGE_BANDS.items():
        if column in X:
            # Keep the band as a category too: "unknown" behaves unlike any band.
            X[column + "_ord"] = X[column].map({b: float(i) for i, b in enumerate(bands)})
    return X


def ratio(numerator, denominator):
    numerator, denominator = numerator.astype(float), denominator.astype(float)
    out = np.full(len(numerator), np.nan)
    valid = denominator.to_numpy() > 0
    out[valid] = numerator.to_numpy()[valid] / denominator.to_numpy()[valid]
    return out


def domain_features(X):
    """Row-wise ratios from the treatment record. Requires ordinal_columns first."""
    X = X.copy()
    eggs = X["수집된 신선 난자 수"].fillna(0) + X["해동 난자 수"].fillna(0)
    X["f_transfer_per_embryo"] = ratio(X["이식된 배아 수"], X["총 생성 배아 수"])
    X["f_stored_per_embryo"] = ratio(X["저장된 배아 수"], X["총 생성 배아 수"])
    X["f_embryo_per_egg"] = ratio(X["총 생성 배아 수"], eggs)
    X["f_icsi_fertilisation"] = ratio(X["미세주입에서 생성된 배아 수"], X["미세주입된 난자 수"])
    X["f_mixed_per_egg"] = ratio(X["혼합된 난자 수"], eggs)
    X["f_partner_sperm_share"] = ratio(X["파트너 정자와 혼합된 난자 수"], X["혼합된 난자 수"])
    X["f_prior_pregnancy_rate"] = ratio(X["총 임신 횟수"], X["총 시술 횟수"])
    X["f_prior_birth_per_pregnancy"] = ratio(X["총 출산 횟수"], X["총 임신 횟수"])
    X["f_clinic_share"] = ratio(X["클리닉 내 총 시술 횟수"], X["총 시술 횟수"])
    X["f_embryo_age_at_transfer"] = X["배아 이식 경과일"] - X["난자 혼합 경과일"]
    causes = [c for c in X.columns if c.startswith("불임 원인 - ")]
    X["f_cause_count"] = X[causes].sum(axis=1)
    return X


def fit_di_zero(X_train):
    """Columns empty for every DI row of the training part; set to 0 on DI rows."""
    di = X_train[PROCEDURE] == "DI"
    numeric = X_train.select_dtypes(include=np.number).columns
    columns = [c for c in numeric if di.any() and X_train.loc[di, c].isna().all()]

    def transform(X):
        X = X.copy()
        rows = X[PROCEDURE] == "DI"
        X.loc[rows, columns] = X.loc[rows, columns].fillna(0)
        return X
    return transform, columns


def stage(variant):
    """Which preprocessing steps a variant uses (the ladder is cumulative)."""
    order = VARIANTS.index(variant)
    return {"di_zero": order >= VARIANTS.index("di_zero"),
            "ordinal": order >= VARIANTS.index("ordinal"),
            "features": order >= VARIANTS.index("features")}


def prepare(X_train, X_valid, variant):
    """Apply a variant's preprocessing; fitted parts see the training rows only."""
    steps = stage(variant)
    info = {}
    if steps["di_zero"]:
        transform, columns = fit_di_zero(X_train)
        X_train, X_valid = transform(X_train), transform(X_valid)
        info["di_zero_columns"] = len(columns)
    if steps["ordinal"]:
        X_train, X_valid = ordinal_columns(X_train), ordinal_columns(X_valid)
    if steps["features"]:
        X_train, X_valid = domain_features(X_train), domain_features(X_valid)
    return X_train, X_valid, info


def one_hot(X_train, X_valid):
    """One-hot for string columns, numeric columns passed through with NaN."""
    categorical = X_train.select_dtypes(exclude=np.number).columns.tolist()
    numeric = X_train.select_dtypes(include=np.number).columns.tolist()
    encoder = ColumnTransformer(
        [("num", "passthrough", numeric),
         ("cat", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), categorical)],
        sparse_threshold=1.0)
    return encoder.fit_transform(X_train), encoder.transform(X_valid)


# ------------------------------------------------------------------- models

def lgb_fixed(X_fit, y_fit, X_valid, n_estimators):
    model = LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
                           random_state=SEED, n_jobs=-1, verbosity=-1)
    return model.fit(X_fit, y_fit).predict_proba(X_valid)[:, 1], {}


def lgb_tuned(X_fit, y_fit, X_valid, trials):
    """Optuna on an inner split of the training part; the outer fold is untouched."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    inner = train_test_split(np.arange(len(y_fit)), test_size=INNER_SPLIT,
                             stratify=y_fit, random_state=SEED)
    X_in, X_out = X_fit[inner[0]], X_fit[inner[1]]
    y_in, y_out = y_fit[inner[0]], y_fit[inner[1]]

    def build(params, n_estimators):
        return LGBMClassifier(n_estimators=n_estimators, learning_rate=0.03,
                              subsample_freq=1, random_state=SEED, n_jobs=-1,
                              verbosity=-1, **params)

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        }
        model = build(params, 4000).fit(
            X_in, y_in, eval_set=[(X_out, y_out)], eval_metric="auc",
            callbacks=[early_stopping(150, verbose=False)])
        trial.set_user_attr("best_iteration", int(model.best_iteration_))
        return roc_auc_score(y_out, model.predict_proba(X_out)[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    best = study.best_trial
    # Refit on the whole training part; scale trees to the larger sample.
    n_estimators = max(50, int(round(best.user_attrs["best_iteration"] / (1 - INNER_SPLIT))))
    model = build(best.params, n_estimators).fit(X_fit, y_fit)
    return model.predict_proba(X_valid)[:, 1], {
        "params": best.params, "n_estimators": n_estimators,
        "inner_auc": float(best.value)}


def catboost_native(X_fit, y_fit, X_valid):
    """CatBoost on raw string categories; tree count from an inner split."""
    from catboost import CatBoostClassifier
    categorical = X_fit.select_dtypes(exclude=np.number).columns.tolist()
    inner = train_test_split(np.arange(len(y_fit)), test_size=INNER_SPLIT,
                             stratify=y_fit, random_state=SEED)

    def build(iterations):
        return CatBoostClassifier(iterations=iterations, learning_rate=0.05, depth=6,
                                  eval_metric="AUC", random_seed=SEED, thread_count=-1,
                                  verbose=0, allow_writing_files=False,
                                  cat_features=categorical)

    probe = build(4000).fit(X_fit.iloc[inner[0]], y_fit[inner[0]],
                            eval_set=(X_fit.iloc[inner[1]], y_fit[inner[1]]),
                            early_stopping_rounds=200, use_best_model=True)
    iterations = max(50, int(round((probe.get_best_iteration() + 1) / (1 - INNER_SPLIT))))
    model = build(iterations).fit(X_fit, y_fit)
    return model.predict_proba(X_valid)[:, 1], {"iterations": iterations}


# --------------------------------------------------------------- experiment

def run_variant(variant, X, y, folds, args, previous_oof):
    """Out-of-fold probabilities and per-fold metrics for one variant."""
    oof = np.full(len(y), np.nan)
    fold_results, details = [], []
    started = perf_counter()
    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        y_train, y_valid = y[train_idx], y[valid_idx]
        if variant == "baseline":
            pipeline = build_pipeline(X.iloc[train_idx], n_estimators=args.estimators)
            proba = pipeline.fit(X.iloc[train_idx], y_train).predict_proba(X.iloc[valid_idx])[:, 1]
            info = {}
        elif variant == "blend":
            parts = [previous_oof["lgb_tuned"][valid_idx], previous_oof["catboost"][valid_idx]]
            proba = np.mean([rankdata(p) / len(p) for p in parts], axis=0)
            info = {}
        else:
            X_train, X_valid, info = prepare(X.iloc[train_idx], X.iloc[valid_idx], variant)
            if variant == "catboost":
                proba, extra = catboost_native(X_train, y_train, X_valid)
            else:
                M_train, M_valid = one_hot(X_train, X_valid)
                if variant == "lgb_tuned":
                    proba, extra = lgb_tuned(M_train, y_train, M_valid, args.trials)
                else:
                    proba, extra = lgb_fixed(M_train, y_train, M_valid, args.estimators)
            info.update(extra)
        if not np.isfinite(proba).all():
            raise ValueError(f"{variant}: non-finite predictions in fold {fold}")
        oof[valid_idx] = proba
        fold_results.append({"fold": fold, "roc_auc": float(roc_auc_score(y_valid, proba)),
                             "average_precision": float(average_precision_score(y_valid, proba))})
        details.append(info)
        print(f"  {variant:11s} fold {fold}  AUC {fold_results[-1]['roc_auc']:.5f}", flush=True)
    if np.isnan(oof).any():
        raise ValueError(f"{variant}: some rows have no OOF prediction")
    # Rank-averaged blend scores are not probabilities; AUC is unaffected and AP
    # depends only on order, so both remain comparable.
    return oof, {"oof_roc_auc": float(roc_auc_score(y, oof)),
                 "oof_average_precision": float(average_precision_score(y, oof)),
                 "fold_results": fold_results, "fold_details": details,
                 "elapsed_seconds": perf_counter() - started}


def paired(a, b):
    deltas = [y["roc_auc"] - x["roc_auc"] for x, y in zip(a["fold_results"], b["fold_results"])]
    return {"fold_auc_deltas": deltas, "mean_fold_delta": float(np.mean(deltas)),
            "sd_fold_delta": float(np.std(deltas, ddof=1)),
            "positive_folds": int(sum(d > 0 for d in deltas)),
            "pooled_auc_delta": b["oof_roc_auc"] - a["oof_roc_auc"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--estimators", type=int, default=250)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    args = parser.parse_args()
    args.folds = 5
    variants = [v.strip() for v in args.variants.split(",")]
    unknown = set(variants) - set(VARIANTS)
    if unknown or "baseline" not in variants:
        parser.error(f"variants must include baseline and come from {VARIANTS}")
    if "blend" in variants and not {"lgb_tuned", "catboost"} <= set(variants):
        parser.error("blend needs lgb_tuned and catboost")

    provenance = collect_provenance(args.data, args)
    provenance["code"]["improve_cv_sha256"] = file_sha256(__file__)
    provenance["environment"]["packages"].update(
        {name: version(name) for name in ["catboost", "optuna", "scipy"]})
    X, y = load_dataset(args.data)
    if args.sample is not None and args.sample < len(y):
        X, _, y, _ = train_test_split(X, y, train_size=args.sample, stratify=y,
                                      random_state=SEED)
    y_values = y.to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y_values))
    print(f"Rows {len(y):,}, input columns {X.shape[1]}", flush=True)

    output = args.output_dir / ("improve_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    results, oofs = {}, {}
    report = {"question": "Which changes to the fold-fitted baseline raise OOF ROC-AUC?",
              "ladder": variants, "settings": {"trials": args.trials, "sample": args.sample,
                                              "inner_split": INNER_SPLIT, "seed": SEED},
              "provenance": provenance, "variants": results}

    for variant in variants:
        oofs[variant], results[variant] = run_variant(variant, X, y_values, folds, args, oofs)
        path = output / f"{variant}_oof.csv"
        pd.DataFrame({"source_row_index": y.index.to_numpy(), "y_true": y_values,
                      "oof_score": oofs[variant]}).to_csv(path, index=False)
        results[variant]["oof_artifact"] = {"filename": path.name, "sha256": file_sha256(path)}
        print(f"{variant:11s} OOF AUC {results[variant]['oof_roc_auc']:.5f}", flush=True)
        # Checkpoint so a long run keeps what it has finished.
        with (output / "comparison.partial.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)

    if file_sha256(args.data) != provenance["dataset"]["sha256"]:
        raise RuntimeError("Dataset changed during the experiment")
    report["paired_vs_previous"] = {b: paired(results[a], results[b])
                                    for a, b in zip(variants, variants[1:])}
    report["paired_vs_baseline"] = {v: paired(results["baseline"], results[v])
                                    for v in variants[1:]}
    report["interpretation"] = ("One seed and five folds that share training rows; deltas "
                                "are descriptive, not independent replicates or a "
                                "significance test. Cross-validation AUC on the training "
                                "data is not comparable with leaderboard scores.")
    with (output / "comparison.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    (output / "comparison.partial.json").unlink()
    print(f"Saved: {output / 'comparison.json'}")


if __name__ == "__main__":
    main()
