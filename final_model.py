#!/usr/bin/env python3
"""
final_model.py — train, save, load and use the best model from the experiments.

The model is the one that won the comparisons (docs/experiments/SEEDS.md):
numeric gaps left as NaN, no hand-made features, a LightGBM tuned with Optuna
and a CatBoost on raw categories, combined into one score.

Training follows the procedure each cross-validation fold used, with all
training rows in place of a training fold:

  1. One-hot encode string columns for LightGBM (numeric columns pass through
     with NaN). The encoder is fitted on the training rows only.
  2. LightGBM: 25 Optuna trials on an inner 80/20 split, early stopping on the
     inner 20%, then a refit on all rows with the tree count scaled by 1/0.8.
  3. CatBoost: tree count found by early stopping on the same kind of inner
     split, then a refit on all rows.
  4. Combine: the mean of the two predicted probabilities.

Why a probability mean and not the rank mean used in the experiments: a rank
depends on the other rows in the batch, so one patient's score would change
with whoever else is scored with them, and a single row has no rank at all. On
the saved out-of-fold predictions of all five split seeds the probability mean
scores within 0.00005 AUC of the rank mean and its average equals the
positive rate (0.258); on split seed 42, predicted and observed rates match
within 0.003 in every decile. That calibration was measured on
cross-validation predictions; it has not been checked on new data.

What the score is: the model's estimate of the chance of pregnancy for a
treatment cycle, given the full record of that cycle, including inputs only
known after treatment has progressed (see docs/experiments/ERROR_ANALYSIS.md).
It is not a pre-treatment prediction.

Commands:
  python final_model.py train                       # about 6 minutes on 15 cores
  python final_model.py predict --model models/final_<time> --input data/test.csv
  python final_model.py train --sample 5000 --trials 2   # quick check

The saved bundle is a joblib pickle. Loading a pickle runs code, so `predict`
first checks the file's SHA-256 against the manifest written next to it. Load
only bundles you built yourself.
"""
import argparse
import json
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from baseline_cv import SEED, TARGET, collect_provenance, file_sha256, load_dataset
from improve_cv import INNER_SPLIT

MISSING_CATEGORY = "__MISSING__"
BUNDLE_NAME = "model.joblib"
MANIFEST_NAME = "manifest.json"
PACKAGES = ["numpy", "pandas", "scikit-learn", "lightgbm", "catboost", "optuna", "joblib"]


# ------------------------------------------------------------------ fitting
# Same settings and search space as improve_cv.lgb_tuned and
# improve_cv.catboost_native; tests/test_final_model.py checks that they give
# identical predictions, so the two cannot drift apart unnoticed.

def fit_lgb_tuned(M, y, trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    inner = train_test_split(np.arange(len(y)), test_size=INNER_SPLIT,
                             stratify=y, random_state=SEED)
    M_in, M_out, y_in, y_out = M[inner[0]], M[inner[1]], y[inner[0]], y[inner[1]]

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
            M_in, y_in, eval_set=[(M_out, y_out)], eval_metric="auc",
            callbacks=[early_stopping(150, verbose=False)])
        trial.set_user_attr("best_iteration", int(model.best_iteration_))
        return roc_auc_score(y_out, model.predict_proba(M_out)[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    best = study.best_trial
    n_estimators = max(50, int(round(best.user_attrs["best_iteration"] / (1 - INNER_SPLIT))))
    model = build(best.params, n_estimators).fit(M, y)
    return model, {"params": best.params, "n_estimators": n_estimators,
                   "inner_auc": float(best.value), "trials": trials}


def fit_catboost(X, y):
    from catboost import CatBoostClassifier
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    inner = train_test_split(np.arange(len(y)), test_size=INNER_SPLIT,
                             stratify=y, random_state=SEED)

    def build(iterations):
        return CatBoostClassifier(iterations=iterations, learning_rate=0.05, depth=6,
                                  eval_metric="AUC", random_seed=SEED, thread_count=-1,
                                  verbose=0, allow_writing_files=False,
                                  cat_features=categorical)

    probe = build(4000).fit(X.iloc[inner[0]], y[inner[0]],
                            eval_set=(X.iloc[inner[1]], y[inner[1]]),
                            early_stopping_rounds=200, use_best_model=True)
    iterations = max(50, int(round((probe.get_best_iteration() + 1) / (1 - INNER_SPLIT))))
    model = build(iterations).fit(X, y)
    return model, {"iterations": iterations,
                   "inner_auc": float(probe.get_best_score()["validation"]["AUC"])}


def fit_encoder(X):
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    encoder = ColumnTransformer(
        [("num", "passthrough", numeric),
         ("cat", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), categorical)],
        sparse_threshold=1.0)
    return encoder.fit(X)


# ------------------------------------------------------------------- model

class InputError(ValueError):
    """The input cannot be scored as given."""


class FinalModel:
    """Schema, encoder, both models and the combining rule, saved as one object."""

    def __init__(self, columns, categorical, categories, encoder, lgb, catboost, details):
        self.columns = list(columns)
        self.categorical = list(categorical)
        self.categories = {c: sorted(v) for c, v in categories.items()}
        self.encoder, self.lgb, self.catboost = encoder, lgb, catboost
        self.details = details
        self.combine = "mean of LightGBM and CatBoost probabilities"

    @classmethod
    def fit(cls, X, y, trials=25):
        y = np.asarray(y)
        categorical = X.select_dtypes(exclude=np.number).columns.tolist()
        model = cls(X.columns, categorical, {c: [] for c in categorical},
                    None, None, None, {})
        # Training rows go through the same normalisation as any later input.
        X, _ = model.prepare(X)
        model.categories = {c: sorted(X[c].unique()) for c in categorical}
        model.encoder = fit_encoder(X)
        started = perf_counter()
        model.lgb, lgb_info = fit_lgb_tuned(model.encoder.transform(X), y, trials)
        lgb_info["seconds"] = perf_counter() - started
        started = perf_counter()
        model.catboost, cat_info = fit_catboost(X, y)
        cat_info["seconds"] = perf_counter() - started
        model.details = {"lightgbm": lgb_info, "catboost": cat_info,
                         "train_rows": int(len(y)), "positive_rate": float(y.mean())}
        return model

    def prepare(self, frame):
        """
        Check an input frame against the training schema and normalise it.

        Rules:
          - a training column that is missing: error
          - a column the model does not use (for example ID): dropped and reported
          - a numeric column with a value that is not a number: error
          - a string column: blanks become the missing category, as in training
          - a category never seen in training: allowed, counted and reported;
            LightGBM sees no category column switched on, CatBoost treats it as new
        """
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise InputError(f"missing columns: {missing}")
        report = {"rows": int(len(frame)),
                  "ignored_columns": [c for c in frame.columns if c not in self.columns],
                  "unseen_categories": {}}
        X = frame[self.columns].copy()
        for column in self.columns:
            if column in self.categorical:
                X[column] = X[column].astype("string").fillna(MISSING_CATEGORY).astype(str)
                unseen = ~X[column].isin(set(self.categories[column]))
                if unseen.any():
                    report["unseen_categories"][column] = int(unseen.sum())
            else:
                raw = X[column]
                if raw.dtype == object:          # numbers read as text; blank = missing
                    raw = raw.astype("string").str.strip().replace("", pd.NA)
                values = pd.to_numeric(raw, errors="coerce")
                bad = values.isna() & raw.notna()
                if bad.any():
                    raise InputError(f"column {column!r} has {int(bad.sum())} non-numeric values")
                X[column] = values.replace([np.inf, -np.inf], np.nan).astype(float)
        return X, report

    def predict_parts(self, frame):
        X, report = self.prepare(frame)
        p_lgb = self.lgb.predict_proba(self.encoder.transform(X))[:, 1]
        p_cat = self.catboost.predict_proba(X)[:, 1]
        return p_lgb, p_cat, report

    def predict_proba(self, frame):
        """One score per row; a row's score does not depend on the other rows."""
        p_lgb, p_cat, report = self.predict_parts(frame)
        score = (p_lgb + p_cat) / 2
        if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
            raise RuntimeError("invalid predictions")
        return score, report


# ----------------------------------------------------------- save and load

def save(model, directory, extra):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    bundle = directory / BUNDLE_NAME
    joblib.dump(model, bundle)
    manifest = {"bundle": BUNDLE_NAME, "bundle_sha256": file_sha256(bundle),
                "packages": {name: version(name) for name in PACKAGES},
                "columns": model.columns, "categorical": model.categorical,
                "combine": model.combine, "details": model.details, **extra}
    with (directory / MANIFEST_NAME).open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return manifest


def load(directory):
    """Load a saved bundle after checking its hash and package versions."""
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    bundle = directory / manifest["bundle"]
    if file_sha256(bundle) != manifest["bundle_sha256"]:
        raise RuntimeError(f"{bundle} does not match its manifest; refusing to load it")
    installed = {name: version(name) for name in PACKAGES}
    differ = {n: (manifest["packages"][n], installed[n]) for n in PACKAGES
              if manifest["packages"].get(n) != installed[n]}
    if differ:
        warnings.warn(f"package versions differ from training (saved, installed): {differ}")
    return joblib.load(bundle), manifest


# ---------------------------------------------------------------- commands

def read_input(path):
    frame = pd.read_csv(path, low_memory=False)
    if TARGET in frame.columns:
        raise InputError("input must not contain the target column")
    if "ID" in frame.columns and (frame["ID"].isna().any() or frame["ID"].duplicated().any()):
        raise InputError("ID must be unique and nonmissing")
    return frame


def write_scores(frame, scores, path):
    ids = frame["ID"] if "ID" in frame.columns else pd.RangeIndex(len(frame))
    pd.DataFrame({"ID": ids, "probability": scores}).to_csv(path, index=False)


def train(args):
    provenance = collect_provenance(args.data, args)
    provenance["code"]["final_model_sha256"] = file_sha256(__file__)
    X, y = load_dataset(args.data)
    if args.sample is not None and args.sample < len(y):
        X, _, y, _ = train_test_split(X, y, train_size=args.sample, stratify=y,
                                      random_state=SEED)
    print(f"Training on {len(y):,} rows, {X.shape[1]} columns", flush=True)
    started = perf_counter()
    model = FinalModel.fit(X, y.to_numpy(), trials=args.trials)
    print(f"Trained in {perf_counter() - started:.0f}s", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = args.model_dir / f"final_{stamp}"
    record = {"provenance": provenance,
              "settings": {"trials": args.trials, "sample": args.sample, "seed": SEED,
                           "inner_split": INNER_SPLIT}}

    # Score the test file before saving and again after loading the saved copy.
    check = None
    if args.test is not None:
        test = read_input(args.test)
        before, report = model.predict_proba(test)
        record["test_dataset"] = {"filename": args.test.name, "sha256": file_sha256(args.test)}
        record["test_input_report"] = report
    manifest = save(model, directory, record)
    if args.test is not None:
        reloaded, _ = load(directory)
        after, _ = reloaded.predict_proba(test)
        check = {"rows": int(len(after)), "max_abs_difference": float(np.max(np.abs(after - before))),
                 "identical": bool(np.array_equal(after, before))}
        if not check["identical"]:
            raise RuntimeError(f"reloaded model predicts differently: {check}")
        write_scores(test, after, directory / "test_predictions.csv")
        check["test_probability"] = {"min": float(after.min()), "mean": float(after.mean()),
                                     "max": float(after.max())}
        check["predictions_sha256"] = file_sha256(directory / "test_predictions.csv")

    if file_sha256(args.data) != provenance["dataset"]["sha256"]:
        raise RuntimeError("training data changed during the run")
    summary = {"model_directory": str(directory), "bundle_sha256": manifest["bundle_sha256"],
               "packages": manifest["packages"], "combine": model.combine,
               "details": model.details, "save_load_check": check, **record}
    out = args.output_dir / f"final_model_{stamp}.json"
    with out.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Saved model to {directory}; record {out}")
    if check:
        print(f"Reload check: {check['rows']:,} rows, identical={check['identical']}")


def predict(args):
    model, _ = load(args.model)
    frame = read_input(args.input)
    scores, report = model.predict_proba(frame)
    out = args.out or Path(args.model) / f"predictions_{Path(args.input).stem}.csv"
    write_scores(frame, scores, out)
    print(json.dumps(report, ensure_ascii=False))
    print(f"Wrote {len(scores):,} scores to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--data", type=Path, default=Path("data/train.csv"))
    fit.add_argument("--test", type=Path, default=Path("data/test.csv"),
                     help="scored before saving and after loading, to check the round trip")
    fit.add_argument("--model-dir", type=Path, default=Path("models"))
    fit.add_argument("--output-dir", type=Path, default=Path("results"))
    fit.add_argument("--sample", type=int, default=None)
    fit.add_argument("--trials", type=int, default=25)
    use = commands.add_parser("predict")
    use.add_argument("--model", type=Path, required=True)
    use.add_argument("--input", type=Path, required=True)
    use.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "train":
        # collect_provenance records these; the final model has no folds and
        # its tree counts come from early stopping.
        args.folds, args.estimators = None, None
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
