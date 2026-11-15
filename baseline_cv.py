
"""
Fold-safe IVF prediction baseline.

Preprocessing is fitted separately within each training fold.
No GAIN, SMOTE, target-based feature selection, or tuning.
"""

import argparse
import json
import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "임신 성공 여부"
SEED = 42


def load_dataset(path):
    df = pd.read_csv(path, low_memory=False)

    if TARGET not in df.columns:
        raise ValueError(f"Target not found: {TARGET}")

    # Validate before integer conversion so fractional labels cannot be truncated.
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Target must contain only nonmissing binary values 0 and 1")

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET, "ID"], errors="ignore").copy()

    # Stateless type normalization; no statistics are learned.
    categorical = X.select_dtypes(exclude=np.number).columns

    for col in categorical:
        X[col] = (
            X[col]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    numeric = X.select_dtypes(include=np.number).columns
    X[numeric] = X[numeric].replace(
        [np.inf, -np.inf], np.nan
    )

    return X, y


def build_pipeline(X, n_estimators=250, missing_indicators=False):
    numeric_cols = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    transformers = []

    if numeric_cols:
        transformers.append((
            "num",
            SimpleImputer(
                strategy="median",
                add_indicator=missing_indicators,
                keep_empty_features=True,
            ),
            numeric_cols,
        ))

    if categorical_cols:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        ])

        transformers.append((
            "cat",
            categorical_pipeline,
            categorical_cols,
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        sparse_threshold=1.0,
    )

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def run_cv(X, y, n_splits=5, n_estimators=250, missing_indicators=False):
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    oof = np.full(len(y), np.nan, dtype=np.float64)
    oof_folds = np.zeros(len(y), dtype=np.int32)
    assignment_counts = np.zeros(len(y), dtype=np.int32)
    fold_results = []

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, y), start=1
    ):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        # New preprocessing and model for every fold.
        pipeline = build_pipeline(
            X_train,
            n_estimators=n_estimators,
            missing_indicators=missing_indicators,
        )

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(
            X_valid
        )[:, 1]

        if (assignment_counts[valid_idx] != 0).any():
            raise ValueError("Validation rows assigned more than once")
        if not np.isfinite(probabilities).all() or not (
            (probabilities >= 0) & (probabilities <= 1)
        ).all():
            raise ValueError("Invalid OOF probabilities")
        assignment_counts[valid_idx] += 1
        oof[valid_idx] = probabilities
        oof_folds[valid_idx] = fold

        auc = roc_auc_score(y_valid, probabilities)
        ap = average_precision_score(
            y_valid, probabilities
        )
        f1 = f1_score(
            y_valid,
            (probabilities >= 0.5).astype(int),
            zero_division=0,
        )

        fold_results.append({
            "fold": fold,
            "roc_auc": float(auc),
            "average_precision": float(ap),
            "f1_at_0.5": float(f1),
        })

        print(
            f"Fold {fold}/{n_splits} | "
            f"AUC={auc:.5f} | "
            f"AP={ap:.5f} | "
            f"F1@0.5={f1:.5f}",
            flush=True,
        )

    if not (assignment_counts == 1).all() or not np.isfinite(oof).all():
        raise ValueError("Every row must receive exactly one OOF prediction")

    summary = {
        "rows": len(y),
        "features": X.shape[1],
        "folds": n_splits,
        "seed": SEED,
        "model": "LightGBM",
        "numeric_missing_indicators": missing_indicators,
        "preprocessing": "Fold-fitted SimpleImputer + OneHotEncoder",
        "oof_roc_auc": float(roc_auc_score(y, oof)),
        "oof_average_precision": float(
            average_precision_score(y, oof)
        ),
        "oof_f1_at_0.5": float(
            f1_score(
                y,
                (oof >= 0.5).astype(int),
                zero_division=0,
            )
        ),
        "fold_results": fold_results,
        "model_parameters": pipeline.named_steps["model"].get_params(),
    }

    # Verify every row received exactly one OOF prediction.
    # Assignment counts above also detect repeated validation rows.

    oof_df = pd.DataFrame({
        "row_index": np.arange(len(y)),
        "source_row_index": y.index.to_numpy(),
        "fold": oof_folds,
        "y_true": y.to_numpy(),
        "oof_probability": oof,
    })

    return summary, oof_df


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_provenance(data_path, args):
    root = Path(__file__).resolve().parent
    def git(*arguments):
        try:
            return subprocess.check_output(
                ["git", *arguments], cwd=root, text=True,
                stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain")
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"filename": Path(data_path).name,
                    "sha256": file_sha256(data_path),
                    "bytes": Path(data_path).stat().st_size},
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform(),
                        "packages": {name: version(name) for name in
                                     ["numpy", "pandas", "scikit-learn", "lightgbm"]}},
        "code": {"commit": git("rev-parse", "HEAD"),
                 "dirty": None if status is None else bool(status),
                 "baseline_cv_sha256": file_sha256(__file__)},
        "configuration": {"folds": args.folds, "sample": args.sample,
                          "estimators": args.estimators, "seed": SEED},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--estimators", type=int, default=250)
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.estimators < 1 or args.folds < 2:
        parser.error("--estimators must be positive and --folds must be at least 2")

    provenance = collect_provenance(args.data, args)
    X, y = load_dataset(args.data)

    dataset_rows = len(y)

    if args.sample is not None:
        if not 2 <= args.sample <= len(y):
            parser.error("--sample must be between 2 and dataset size")

        if args.sample < len(y):
            X, _, y, _ = train_test_split(
                X,
                y,
                train_size=args.sample,
                stratify=y,
                random_state=SEED,
            )

            # Preserve original CSV row positions through stratified sampling.

    if y.nunique() != 2 or y.value_counts().min() < args.folds:
        parser.error("Both classes need at least --folds rows")

    print(f"Dataset: {X.shape}", flush=True)
    print(f"Positive rate: {y.mean():.4f}", flush=True)

    summary, oof_df = run_cv(
        X,
        y,
        n_splits=args.folds,
        n_estimators=args.estimators,
    )

    if file_sha256(args.data) != provenance["dataset"]["sha256"]:
        raise RuntimeError("Dataset changed during the run; results were not saved")
    summary["provenance"] = provenance

    print("\n=== OOF Results ===")
    print(f"ROC-AUC: {summary['oof_roc_auc']:.5f}")
    print(f"AP:      {summary['oof_average_precision']:.5f}")
    print(f"F1@0.5:  {summary['oof_f1_at_0.5']:.5f}")

    # Every run gets a new directory; exist_ok=False protects previous runs.
    run_name = "baseline_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    oof_path = output_dir / "predictions_oof.csv"
    oof_df.to_csv(oof_path, index=False)
    summary["oof_artifact"] = {"filename": oof_path.name,
                               "sha256": file_sha256(oof_path)}
    output_path = output_dir / "summary.json"
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Saved: {output_path}")
    print(f"Saved OOF predictions: {oof_path}")


if __name__ == "__main__":
    main()
