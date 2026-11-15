#!/usr/bin/env python3
"""
error_analysis.py — where the best model gains and where it fails, by subgroup.

Reads the out-of-fold predictions an earlier run already saved (no retraining)
and reports, for each subgroup of the training data:

  rows, positives, positive rate, ROC-AUC and AP of the baseline and of the
  chosen model, the AUC difference on the same rows, and a paired bootstrap
  95% interval for that difference.

Only aggregates are written. No input rows or per-row predictions leave this
script.

Reading the numbers:
- AUC is not computed for a group with a single class.
- Groups with fewer than SMALL_ROWS rows or SMALL_POSITIVES positives are
  flagged as small; their numbers move a lot between resamples.
- AP depends on the positive rate, so AP is compared between models within a
  group, never between groups.
- The bootstrap resamples rows of one fixed split. It shows how stable the
  difference is on these rows, not how it would vary on new data or another
  split.
- A blend score is a rank within the whole training set. Ranking is preserved
  inside any subgroup, so AUC and AP inside a group are still well defined.

Usage:
  python error_analysis.py                                  # latest missing_types run, nan_blend vs baseline
  python error_analysis.py --run results/improve_... --model blend
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from baseline_cv import SEED, TARGET, file_sha256

SMALL_ROWS = 1000
SMALL_POSITIVES = 100

TRANSFER = "이식된 배아 수"


def transfer_bucket(values):
    """Number of embryos transferred, in buckets; blank means not recorded."""
    out = pd.Series("기록 없음", index=values.index, dtype=object)
    out[values == 0] = "0개"
    out[values == 1] = "1개"
    out[values == 2] = "2개"
    out[values >= 3] = "3개 이상"
    return out


def flag_label(values):
    out = pd.Series("기록 없음", index=values.index, dtype=object)
    out[values == 0] = "0"
    out[values == 1] = "1"
    return out


# Each axis: name -> function(frame) -> Series of group labels.
AXES = {
    "시술 유형": lambda df: df["시술 유형"].astype(str),
    "시술 당시 나이": lambda df: df["시술 당시 나이"].astype(str),
    "난자 출처": lambda df: df["난자 출처"].astype(str),
    "동결 배아 사용 여부": lambda df: flag_label(df["동결 배아 사용 여부"]),
    "이식된 배아 수": lambda df: transfer_bucket(df[TRANSFER]),
}


# Row subsets reported like the overall row. Rows with no embryo transferred
# almost never become pregnant, and that is known only after the procedure; the
# subset shows how the models rank the rows where pregnancy was possible.
SUBSETS = {
    "배아를 이식했거나 DI 시술인 행":
        lambda df: (df[TRANSFER] >= 1) | (df["시술 유형"] == "DI"),
}


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def paired_bootstrap(y, a, b, rounds, rng):
    """95% interval of AUC(b) - AUC(a), resampling positives and negatives separately."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return None
    deltas = np.empty(rounds)
    for i in range(rounds):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        deltas[i] = roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx])
    low, high = np.percentile(deltas, [2.5, 97.5])
    return [float(low), float(high)]


def summarize(y, base, model, rounds):
    n, positives = len(y), int(y.sum())
    auc_base, auc_model = safe_auc(y, base), safe_auc(y, model)
    row = {
        "rows": n, "positives": positives,
        "positive_rate": positives / n if n else None,
        "small": n < SMALL_ROWS or positives < SMALL_POSITIVES,
        "auc_baseline": auc_base, "auc_model": auc_model,
        "auc_delta": None if auc_base is None else auc_model - auc_base,
        "auc_delta_ci95": None,
        "ap_baseline": float(average_precision_score(y, base)) if positives else None,
        "ap_model": float(average_precision_score(y, model)) if positives else None,
    }
    if auc_base is not None and rounds:
        # A fresh generator per group: identical rows give identical intervals
        # whatever order the groups are visited in.
        row["auc_delta_ci95"] = paired_bootstrap(y, base, model, rounds,
                                                  np.random.default_rng(SEED))
    return row


def group_rate_auc(y, labels, seed=SEED):
    """
    ROC-AUC of a score that is only the positive rate of the row's group,
    with rates learned on the other folds. It measures how much ranking one
    variable gives on its own.
    """
    score = np.zeros(len(y))
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    for train, valid in folds.split(np.zeros(len(y)), y):
        rates = pd.Series(y[train]).groupby(labels[train]).mean()
        score[valid] = pd.Series(labels[valid]).map(rates).fillna(y[train].mean()).values
    return float(roc_auc_score(y, score))


def load_oof(run, variant, recorded):
    path = run / f"{variant}_oof.csv"
    expected = recorded["variants"][variant]["oof_artifact"]["sha256"]
    if file_sha256(path) != expected:
        raise ValueError(f"{path} does not match the hash recorded in comparison.json")
    frame = pd.read_csv(path)
    return frame.sort_values("source_row_index").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--run", default=None,
                        help="results folder with comparison.json and OOF files "
                             "(default: latest results/missing_types_*)")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--model", default="nan_blend")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run = Path(args.run) if args.run else sorted(Path("results").glob("missing_types_*"))[-1]
    recorded = json.loads((run / "comparison.json").read_text(encoding="utf-8"))
    data_sha = file_sha256(args.data)
    if data_sha != recorded["provenance"]["dataset"]["sha256"]:
        raise ValueError("training data differs from the data the predictions were made on")

    df = pd.read_csv(args.data, low_memory=False)
    base = load_oof(run, args.baseline, recorded)
    model = load_oof(run, args.model, recorded)
    for frame in (base, model):
        if not np.array_equal(frame["source_row_index"].values, np.arange(len(df))):
            raise ValueError("OOF rows do not cover the training rows one to one")
        if not np.array_equal(frame["y_true"].values, df[TARGET].astype(int).values):
            raise ValueError("OOF labels do not match the training target")

    y = df[TARGET].astype(int).values
    a, b = base["oof_score"].values, model["oof_score"].values

    result = {
        "question": "Where does the best model gain over the baseline, and where do both fail?",
        "settings": {"run": str(run), "baseline": args.baseline, "model": args.model,
                     "bootstrap_rounds": args.bootstrap, "seed": SEED,
                     "small_rows": SMALL_ROWS, "small_positives": SMALL_POSITIVES},
        "provenance": {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": data_sha,
            "oof_sha256": {v: recorded["variants"][v]["oof_artifact"]["sha256"]
                           for v in (args.baseline, args.model)},
            "error_analysis_sha256": file_sha256(__file__),
        },
        "overall": summarize(y, a, b, args.bootstrap),
        "subsets": {},
        "axes": {},
        "single_variable_auc": {},
    }
    for name, select in SUBSETS.items():
        mask = select(df).values
        result["subsets"][name] = summarize(y[mask], a[mask], b[mask], args.bootstrap)
    for axis, labeler in AXES.items():
        labels = labeler(df).values
        groups = {}
        for label in sorted(pd.unique(labels), key=str):
            mask = labels == label
            groups[str(label)] = summarize(y[mask], a[mask], b[mask], args.bootstrap)
            print(f"{axis:>12} | {label:<10} n={mask.sum():>7,} "
                  f"auc {groups[str(label)]['auc_baseline']} -> {groups[str(label)]['auc_model']}")
        result["axes"][axis] = groups
        result["single_variable_auc"][axis] = group_rate_auc(y, labels)

    out = Path(args.out) if args.out else Path("results") / (
        "error_analysis_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
