#!/usr/bin/env python3
"""
validate_tuning.py — do the newly tuned parameters actually beat the incumbents?

tune.py searches on 75% of the data. This scores its winners against the
parameters currently in configs/ on the other 25%, which the search never saw,
and only then offers to write configs/.

Why a separate split rather than just comparing search scores: the search score
is the maximum over every set Optuna tried, on the folds it tried them on. That
maximum is biased upward by however many sets were tried, so comparing it to
anything is comparing a best-of-N to a single draw. The referee split has been
used to choose nothing, so a win there is a win.

    fit on the 75% tuning split, predict the 25% referee split
    both parameter sets, same fit data, same evaluation rows

What this does not do is produce the README's numbers. Those come from
ensemble_study.py over the full five folds. Re-run it after adopting anything.

Usage:
  python validate_tuning.py                 # compare, write nothing
  python validate_tuning.py --write         # compare, then update configs/
  python validate_tuning.py --model lgb     # just one
"""

import argparse
import json
import pathlib
import shutil
import time
import warnings

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from build_cache import ensure_cache

warnings.filterwarnings("ignore")

SEED = 42
SPLIT_SEED = 20250913     # must match tune.py, or the referee has seen the data
CACHE = pathlib.Path("data/gain_cache.npz")
TUNED = pathlib.Path("tuning/best_params")
CONFIGS = pathlib.Path("configs")
MODELS = ["xgb", "lgb", "rf", "cat"]
NAMES = {"xgb": "XGBoost", "lgb": "LightGBM", "rf": "RandomForest", "cat": "CatBoost"}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def splits():
    ensure_cache(CACHE)
    cached = np.load(CACHE, allow_pickle=True)
    X, y = cached["X"], cached["y"]
    return train_test_split(X, y, test_size=0.25,
                            random_state=SPLIT_SEED, stratify=y)


def evaluate(model, params, X_fit, y_fit, X_ref, y_ref):
    """Fit on the tuning split, score on the referee split."""
    from src.optimization import build_model

    estimator = build_model(model, params, seed=SEED)
    estimator.fit(X_fit, y_fit)
    proba = estimator.predict_proba(X_ref)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_ref, proba)
    f1_curve = 2 * precision * recall / (precision + recall + 1e-12)
    # The threshold is read off the referee split itself, so this F1 is
    # optimistic in absolute terms. Both sides get the same treatment, which is
    # all the comparison needs.
    threshold = float(thresholds[int(np.nanargmax(f1_curve[:-1]))])
    return {
        "auc": float(roc_auc_score(y_ref, proba)),
        "f1": float(f1_score(y_ref, (proba >= threshold).astype(int))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, help="just one model")
    parser.add_argument("--write", action="store_true",
                        help="copy winners into configs/ (originals kept as .leaky.json)")
    parser.add_argument("--out", default="tuning/referee.json")
    args = parser.parse_args()

    X_fit, X_ref, y_fit, y_ref = splits()
    log(f"fit on {len(y_fit):,} rows, score on {len(y_ref):,} held-out rows "
        f"({X_fit.shape[1]} features)")

    targets = [args.model] if args.model else MODELS
    results = {}

    for model in targets:
        tuned_path = TUNED / f"{model}_params.json"
        current_path = CONFIGS / f"{model}_params.json"
        if not tuned_path.exists():
            log(f"  {model}: no tuned parameters yet — run `python tune.py --model {model}`")
            continue
        if not current_path.exists():
            log(f"  {model}: {current_path} missing")
            continue

        from src.optimization import model_params
        tuned = model_params(json.load(open(tuned_path)))
        current = model_params(json.load(open(current_path)))

        log(f"  {model}: fitting incumbent...")
        before = evaluate(model, current, X_fit, y_fit, X_ref, y_ref)
        log(f"  {model}: fitting tuned...")
        after = evaluate(model, tuned, X_fit, y_fit, X_ref, y_ref)

        entry = {
            "incumbent": before, "tuned": after,
            "delta_auc": after["auc"] - before["auc"],
            "delta_f1": after["f1"] - before["f1"],
            "tuned_params": tuned, "incumbent_params": current,
        }
        entry["adopt"] = entry["delta_auc"] > 0
        results[model] = entry
        log(f"  {model}: {before['auc']:.4f} -> {after['auc']:.4f} "
            f"({entry['delta_auc']:+.4f} AUC)   "
            f"{'adopt' if entry['adopt'] else 'keep incumbent'}")

    if not results:
        return

    print()
    log("referee split — held out of the search entirely")
    print(f"  {'model':14s}{'incumbent':>12s}{'tuned':>10s}{'Δ AUC':>10s}{'Δ F1':>10s}   verdict")
    for model, entry in results.items():
        print(f"  {NAMES[model]:14s}{entry['incumbent']['auc']:>12.4f}"
              f"{entry['tuned']['auc']:>10.4f}{entry['delta_auc']:>+10.4f}"
              f"{entry['delta_f1']:>+10.4f}   "
              f"{'adopt' if entry['adopt'] else 'keep incumbent'}")

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    log(f"written to {args.out}")

    adopt = [m for m, e in results.items() if e["adopt"]]
    if not args.write:
        print()
        if adopt:
            log(f"would adopt: {', '.join(adopt)} — rerun with --write to update configs/")
        else:
            log("nothing beats the incumbents; configs/ unchanged")
        return

    if not adopt:
        log("nothing to write")
        return

    for model in adopt:
        current_path = CONFIGS / f"{model}_params.json"
        backup = CONFIGS / f"{model}_params.leaky.json"
        if not backup.exists():
            shutil.copy(current_path, backup)
        payload = dict(results[model]["tuned_params"])
        payload["referee_auc"] = results[model]["tuned"]["auc"]
        payload["_note"] = (
            "Selected by tune.py on a 75% tuning split under a leakage-free "
            "protocol, then beaten against the previous parameters on the 25% "
            "referee split. referee_auc is that held-out score, not the "
            "figure the README reports.")
        with open(current_path, "w") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        log(f"  wrote {current_path} (previous kept as {backup.name})")

    print()
    log("configs/ updated. The published figures are now stale — re-run:")
    log("  python ensemble_study.py")


if __name__ == "__main__":
    main()
