"""Fit the fixed baseline on all training rows and predict the competition test set.

The model, preprocessing and seed are exactly those of baseline_cv.py. Nothing
is tuned or selected using test data. The output follows the DACON submission
format: columns ID and probability, rows in test-file order.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_cv import (TARGET, build_pipeline, collect_provenance,
                         file_sha256, load_dataset)


def load_test_features(path, train_columns):
    """Apply the same stateless normalization as load_dataset, without a target."""
    df = pd.read_csv(path, low_memory=False)
    if TARGET in df.columns:
        raise ValueError("Test file must not contain the target column")
    if "ID" not in df.columns or df["ID"].isna().any() or df["ID"].duplicated().any():
        raise ValueError("Test file needs a unique, nonmissing ID column")

    missing = [c for c in train_columns if c not in df.columns]
    extra = [c for c in df.columns if c not in train_columns and c != "ID"]
    if missing or extra:
        raise ValueError(f"Test columns differ from training: missing={missing}, extra={extra}")

    ids = df["ID"].copy()
    X = df[list(train_columns)].copy()
    for col in X.select_dtypes(exclude=np.number).columns:
        X[col] = X[col].astype("string").fillna("__MISSING__").astype(str)
    numeric = X.select_dtypes(include=np.number).columns
    X[numeric] = X[numeric].replace([np.inf, -np.inf], np.nan)
    return ids, X


def fit_and_predict(X_train, y_train, X_test, n_estimators=250):
    numeric_train = set(X_train.select_dtypes(include=np.number).columns)
    numeric_test = set(X_test.select_dtypes(include=np.number).columns)
    if numeric_train != numeric_test:
        raise ValueError("Numeric/categorical column types differ between train and test: "
                         f"{sorted(numeric_train ^ numeric_test)}")
    pipeline = build_pipeline(X_train, n_estimators=n_estimators)
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid test probabilities")
    return pipeline, probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--estimators", type=int, default=250)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.estimators < 1:
        parser.error("--estimators must be positive")

    args.folds, args.sample = None, None
    provenance = collect_provenance(args.data, args)
    provenance["code"]["predict_test_sha256"] = file_sha256(__file__)
    provenance["test_dataset"] = {"filename": args.test.name,
                                  "sha256": file_sha256(args.test),
                                  "bytes": args.test.stat().st_size}

    X_train, y_train = load_dataset(args.data)
    ids, X_test = load_test_features(args.test, X_train.columns)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}", flush=True)

    pipeline, probabilities = fit_and_predict(X_train, y_train, X_test, args.estimators)

    if (file_sha256(args.data) != provenance["dataset"]["sha256"]
            or file_sha256(args.test) != provenance["test_dataset"]["sha256"]):
        raise RuntimeError("Input data changed during the run; nothing was saved")

    output_dir = args.output_dir / ("submission_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output_dir.mkdir(parents=True, exist_ok=False)
    submission_path = output_dir / "submission.csv"
    pd.DataFrame({"ID": ids, "probability": probabilities}).to_csv(submission_path, index=False)

    summary = {
        "train_rows": len(y_train), "test_rows": len(ids), "features": X_train.shape[1],
        "model": "LightGBM", "preprocessing": "SimpleImputer + OneHotEncoder fitted on all training rows",
        "model_parameters": pipeline.named_steps["model"].get_params(),
        "test_probability": {"min": float(probabilities.min()), "mean": float(probabilities.mean()),
                             "max": float(probabilities.max())},
        "submission_artifact": {"filename": submission_path.name, "sha256": file_sha256(submission_path)},
        "leaderboard": None,
        "provenance": provenance,
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Saved: {submission_path}")


if __name__ == "__main__":
    main()
