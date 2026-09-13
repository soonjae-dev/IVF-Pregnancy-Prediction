#!/usr/bin/env python3
"""
column_study.py — what is in the columns the pipeline throws away?

`src/encoding.py::drop_unused_object_columns` discards every object column that
is not on its keep-list. That list was ["시술 유형_원본"] until the age bracket
was added to it, and the age bracket turned out to be worth 0.0148 ROC-AUC —
it had been discarded by accident, not by judgement. This asks the same
question of everything else on the wrong side of that line.

Nine columns are affected. `시술 유형` looks like a tenth but is not: features.py
copies it to `시술 유형_원본` before encoding, so it survives.

Protocol. For each candidate the FULL pipeline is re-run — feature engineering,
correlation-based selection, one-hot encoding and GAIN imputation — with that
column added to the encode list, because adding a column changes what GAIN sees
and therefore every imputed value in the matrix. Anything less would be
measuring a different pipeline from the one that ships. Each variant is then
scored by LightGBM on the same five folds, with the F1 threshold taken from a
calibration split carved out of each training fold.

The baseline is the pipeline exactly as it stands.

Usage:
  python column_study.py [--out column_study.json] [--iterations 1000]
"""

import argparse
import contextlib
import io
import json
import pathlib
import time
import warnings

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import lightgbm as lgb

warnings.filterwarnings("ignore")

TARGET = "임신 성공 여부"
SEED = 42
CACHE_DIR = pathlib.Path("data/column_cache")
BASE_ENCODE = ["시술 유형_원본", "시술 당시 나이"]

# Every object column drop_unused_object_columns currently discards, except
# 시술 유형, which survives as the 시술 유형_원본 copy.
CANDIDATES = [
    "특정 시술 유형",          # specific procedure type        (24 values)
    "배아 생성 주요 이유",      # main reason for embryo creation (13)
    "난자 출처",               # egg source                      (3)
    "정자 출처",               # sperm source                    (4)
    "난자 기증자 나이",         # egg donor age band              (5)
    "정자 기증자 나이",         # sperm donor age band            (7)
    "클리닉 내 총 시술 횟수",   # total procedures at the clinic  (7)
    "시술 시기 코드",           # procedure period code           (7)
    "배란 유도 유형",           # ovulation induction type        (4)
]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def build(encode_columns, tag, iterations):
    """Steps 1-6 of main.py with a given encode list. Cached per variant."""
    cache = CACHE_DIR / f"{tag}.npz"
    if cache.exists():
        cached = np.load(cache, allow_pickle=True)
        return cached["X"], cached["y"], list(cached["columns"])

    from src.config import get_device, set_seed
    from src.encoding import run_encoding_pipeline
    from src.feature_selection import run_feature_selection
    from src.features import run_feature_engineering
    from src.preprocessing import run_basic_preprocessing
    from src.run_imputation import run_gain_on_dataframes

    config = json.load(open("configs/config.json"))
    set_seed(SEED)
    device = get_device()

    train_df, test_df, _ = quiet(
        run_basic_preprocessing,
        config["pipeline"]["train_data_path"], config["pipeline"]["test_data_path"])
    train_df, test_df = quiet(run_feature_engineering, train_df, test_df)
    train_df, test_df = quiet(run_feature_selection, train_df, test_df)
    train_df, test_df = quiet(run_encoding_pipeline, train_df, test_df,
                              columns_to_encode=encode_columns)
    train_df, _ = quiet(run_gain_on_dataframes, train_df, test_df, device,
                        iterations=iterations)

    y = train_df[TARGET].astype(int).values
    features = train_df.drop(columns=[TARGET])
    X = features.values.astype(np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y,
                        columns=np.array(features.columns, dtype=object))
    return X, y, list(features.columns)


def best_f1_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    index = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[index])


def score(X, y, params, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    out_of_fold = np.zeros(len(y))
    fold_f1 = []
    for train_index, valid_index in cv.split(X, y):
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X[train_index], y[train_index], test_size=0.2,
            random_state=SEED, stratify=y[train_index])
        model = lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1,
                                   **params).fit(X_fit, y_fit)
        threshold = best_f1_threshold(y_cal, model.predict_proba(X_cal)[:, 1])
        proba = model.predict_proba(X[valid_index])[:, 1]
        out_of_fold[valid_index] = proba
        fold_f1.append(f1_score(y[valid_index], (proba >= threshold).astype(int)))
    return {"oof_auc": float(roc_auc_score(y, out_of_fold)),
            "f1_mean": float(np.mean(fold_f1)),
            "f1_std": float(np.std(fold_f1))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="column_study.json")
    parser.add_argument("--iterations", type=int, default=1000,
                        help="GAIN iterations; matches configs/config.json")
    args = parser.parse_args()

    params = {k: v for k, v in json.load(open("configs/lgb_params.json")).items()
              if k != "best_value"}

    log("baseline — the pipeline as it stands")
    X, y, columns = build(BASE_ENCODE, "baseline", args.iterations)
    baseline = score(X, y, params)
    baseline["features"] = len(columns)
    log(f"  {len(columns)} features   AUC {baseline['oof_auc']:.4f}   "
        f"F1 {baseline['f1_mean']:.4f}")

    results = {"baseline": baseline, "candidates": {}}

    for candidate in CANDIDATES:
        tag = f"plus_{CANDIDATES.index(candidate):02d}"
        started = time.time()
        X, y, columns = build(BASE_ENCODE + [candidate], tag, args.iterations)
        entry = score(X, y, params)
        entry["features"] = len(columns)
        entry["added_columns"] = len(columns) - baseline["features"]
        entry["delta_auc"] = entry["oof_auc"] - baseline["oof_auc"]
        entry["delta_f1"] = entry["f1_mean"] - baseline["f1_mean"]
        results["candidates"][candidate] = entry
        log(f"  +{candidate:18s} {entry['added_columns']:3d} cols   "
            f"AUC {entry['oof_auc']:.4f} ({entry['delta_auc']:+.4f})   "
            f"F1 {entry['f1_mean']:.4f} ({entry['delta_f1']:+.4f})   "
            f"({time.time() - started:.0f}s)")
        with open(args.out, "w") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)

    helpful = [c for c, e in results["candidates"].items() if e["delta_auc"] > 0]
    if helpful:
        log(f"together: {', '.join(helpful)}")
        X, y, columns = build(BASE_ENCODE + helpful, "combined", args.iterations)
        entry = score(X, y, params)
        entry["features"] = len(columns)
        entry["columns"] = helpful
        entry["delta_auc"] = entry["oof_auc"] - baseline["oof_auc"]
        entry["delta_f1"] = entry["f1_mean"] - baseline["f1_mean"]
        results["combined"] = entry
        log(f"  {len(columns)} features   AUC {entry['oof_auc']:.4f} "
            f"({entry['delta_auc']:+.4f})   F1 {entry['f1_mean']:.4f} "
            f"({entry['delta_f1']:+.4f})")

    print()
    log("summary — OOF ROC-AUC against the current pipeline")
    print(f"  {'column':24s}{'cols':>6s}{'AUC':>10s}{'Δ AUC':>10s}{'Δ F1':>10s}")
    print(f"  {'(baseline)':24s}{baseline['features']:>6d}"
          f"{baseline['oof_auc']:>10.4f}{'':>10s}{'':>10s}")
    for candidate, entry in sorted(results["candidates"].items(),
                                   key=lambda kv: -kv[1]["delta_auc"]):
        print(f"  {candidate:24s}{entry['added_columns']:>6d}"
              f"{entry['oof_auc']:>10.4f}{entry['delta_auc']:>+10.4f}"
              f"{entry['delta_f1']:>+10.4f}")
    if "combined" in results:
        entry = results["combined"]
        print(f"  {'(all that helped)':24s}{entry['features']:>6d}"
              f"{entry['oof_auc']:>10.4f}{entry['delta_auc']:>+10.4f}"
              f"{entry['delta_f1']:>+10.4f}")

    with open(args.out, "w") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    log(f"written to {args.out}")


if __name__ == "__main__":
    main()
