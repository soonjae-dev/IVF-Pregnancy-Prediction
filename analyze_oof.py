"""Validate saved OOF predictions and explore F1 thresholds without retraining."""
import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score)


def validate_oof(frame):
    required = ["row_index", "fold", "y_true", "oof_probability"]
    if not set(required).issubset(frame.columns) or frame.empty:
        raise ValueError("Expected nonempty OOF data with row_index, fold, y_true, oof_probability")
    if frame[required].isna().any().any():
        raise ValueError("OOF data contains missing values")
    if not np.array_equal(np.sort(frame.row_index.to_numpy()), np.arange(len(frame))):
        raise ValueError("row_index must contain each position exactly once")
    if not frame.y_true.isin([0, 1]).all() or frame.y_true.nunique() != 2:
        raise ValueError("Both binary target classes are required")
    if not frame.oof_probability.between(0, 1).all():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    if not ((frame.fold >= 1) & (frame.fold % 1 == 0)).all():
        raise ValueError("Fold IDs must be positive integers")

    if "source_row_index" in frame:
        source = frame.source_row_index
        if source.isna().any() or source.duplicated().any() or not (
            (source >= 0) & (source % 1 == 0)
        ).all():
            raise ValueError("Source row positions must be unique nonnegative integers")


def best_threshold(y, probabilities):
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    denominator = precision[:-1] + recall[:-1]
    scores = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                       out=np.zeros_like(denominator), where=denominator > 0)
    # Stable tie rule: lowest threshold among equally good F1 values.
    index = int(np.argmax(scores))
    return {"threshold": float(thresholds[index]), "f1": float(scores[index]),
            "precision": float(precision[index]), "recall": float(recall[index])}


def analyze(frame, summary=None):
    validate_oof(frame)
    y, probability = frame.y_true, frame.oof_probability
    metrics = {"oof_roc_auc": float(roc_auc_score(y, probability)),
               "oof_average_precision": float(average_precision_score(y, probability)),
               "oof_f1_at_0.5": float(f1_score(y, probability >= 0.5))}
    if summary is not None:
        if summary["rows"] != len(frame) or summary["folds"] != frame.fold.nunique():
            raise ValueError("Summary row/fold counts disagree with OOF data")
        if not np.array_equal(np.sort(frame.fold.unique()), np.arange(1, summary["folds"] + 1)):
            raise ValueError("Fold IDs disagree with summary")
        for key, value in metrics.items():
            if not np.isclose(summary[key], value, rtol=0, atol=1e-12):
                raise ValueError(f"Summary metric disagrees: {key}")
    return {"rows": len(frame), "metrics": metrics,
            "exploratory_best": best_threshold(y, probability),
            "interpretation": "Threshold selected and scored on the same OOF labels; exploratory F1 is optimistic. Confirm a frozen threshold on independent held-out data or nested validation.",
            "per_fold_exploratory": [dict(fold=int(fold), **best_threshold(group.y_true, group.oof_probability))
                                     for fold, group in frame.groupby("fold")]}


def analyze_files(oof_path, summary_path=None):
    summary = json.loads(Path(summary_path).read_text()) if summary_path else None
    if summary and "oof_artifact" in summary:
        digest = hashlib.sha256()
        with Path(oof_path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != summary["oof_artifact"]["sha256"]:
            raise ValueError("OOF file SHA-256 does not match the recorded artifact")
    return analyze(pd.read_csv(oof_path), summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oof", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path, help="New report path; existing files are never overwritten")
    args = parser.parse_args()
    report = analyze_files(args.oof, args.summary)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
