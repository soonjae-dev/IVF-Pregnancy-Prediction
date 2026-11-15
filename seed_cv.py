#!/usr/bin/env python3
"""
seed_cv.py — does the best model's gain survive a different split?

The best result so far (nan_blend, OOF ROC-AUC 0.74052, +0.0013 over the
baseline) comes from one split: StratifiedKFold(5, shuffle, seed 42). This
script repeats the four models that matter on several split seeds:

    baseline       baseline_cv.py unchanged
    nan_tuned      LightGBM, Optuna on an inner split of each training fold
    nan_catboost   CatBoost on raw categories, trees from an inner split
    nan_blend      rank average of the two

Within one split seed all four models see the same rows and folds, so their
differences are paired. Only the split seed changes between runs. Model and
tuning randomness (LightGBM, CatBoost, Optuna, inner splits) stays at seed 42,
so the spread across split seeds is sensitivity to the split alone, not to
model randomness.

Split seed 42 reproduces results/missing_types_*: the script checks this when
the earlier run's comparison.json is present.

Reading the results: the split seeds reuse the same 256,351 rows. They show
whether the gain holds when the folds change; they are not independent samples
and do not replace an independent test set.

Usage:
  python seed_cv.py                                   # seeds 42,1,2,3,4 (about 1h45 on 15 cores)
  python seed_cv.py --seeds 42,1 --sample 5000 --trials 2
"""
import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from baseline_cv import SEED, collect_provenance, file_sha256, load_dataset
from improve_cv import paired
from missing_types_cv import run_variant

VARIANTS = ["baseline", "nan_tuned", "nan_catboost", "nan_blend"]
DEFAULT_SEEDS = "42,1,2,3,4"


def parse_seeds(text):
    seeds = [int(s) for s in text.split(",") if s.strip()]
    if not seeds:
        raise ValueError("give at least one split seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("each split seed may be listed once")
    return seeds


def spread(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "min": float(values.min()), "max": float(values.max())}


def summarize(per_seed):
    """Across split seeds: AUC, AP and the paired gain over the baseline."""
    out = {}
    for variant in VARIANTS:
        rows = [s["variants"][variant] for s in per_seed.values()]
        out[variant] = {"oof_roc_auc": spread([r["oof_roc_auc"] for r in rows]),
                        "oof_average_precision": spread([r["oof_average_precision"] for r in rows]),
                        "elapsed_seconds": spread([r["elapsed_seconds"] for r in rows])}
        if variant != "baseline":
            gains = [s["vs_baseline"][variant]["pooled_auc_delta"] for s in per_seed.values()]
            folds_up = [s["vs_baseline"][variant]["folds_improved"] for s in per_seed.values()]
            out[variant]["auc_gain_over_baseline"] = spread(gains)
            out[variant]["seeds_with_gain"] = int(sum(g > 0 for g in gains))
            out[variant]["folds_improved"] = f"{sum(folds_up)}/{5 * len(gains)}"
    return out


def check_seed42(per_seed, sample):
    """Compare split seed 42 with the recorded missing_types run, when it applies."""
    if 42 not in per_seed or sample is not None:
        return None
    runs = sorted(Path("results").glob("missing_types_*/comparison.json"))
    if not runs:
        return None
    recorded = json.loads(runs[-1].read_text(encoding="utf-8"))["variants"]
    return {"run": str(runs[-1].parent),
            "auc_abs_difference": {v: abs(per_seed[42]["variants"][v]["oof_roc_auc"]
                                         - recorded[v]["oof_roc_auc"]) for v in VARIANTS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--estimators", type=int, default=250)
    args = parser.parse_args()
    args.folds = 5
    seeds = parse_seeds(args.seeds)

    provenance = collect_provenance(args.data, args)
    for name in ["seed_cv.py", "missing_types_cv.py", "improve_cv.py"]:
        provenance["code"][name.replace(".py", "_sha256")] = file_sha256(
            Path(__file__).with_name(name))
    provenance["environment"]["packages"].update(
        {name: version(name) for name in ["catboost", "optuna", "scipy"]})
    X, y = load_dataset(args.data)
    if args.sample is not None and args.sample < len(y):
        X, _, y, _ = train_test_split(X, y, train_size=args.sample, stratify=y,
                                      random_state=SEED)
    y_values = y.to_numpy()
    print(f"Rows {len(y):,}, input columns {X.shape[1]}, split seeds {seeds}", flush=True)

    output = args.output_dir / ("seeds_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    per_seed = {}
    report = {
        "question": "Does the gain of the tuned LightGBM-CatBoost blend over the baseline "
                    "hold on other split seeds?",
        "settings": {"split_seeds": seeds, "model_and_tuning_seed": SEED,
                     "folds": 5, "trials": args.trials, "estimators": args.estimators,
                     "sample": args.sample, "variants": VARIANTS,
                     "input": "native_nan: numeric gaps left as NaN, no hand-made features"},
        "provenance": provenance, "per_seed": per_seed,
    }

    def save(name):
        with (output / name).open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)

    for split_seed in seeds:
        print(f"== split seed {split_seed}", flush=True)
        started = perf_counter()
        folds = list(StratifiedKFold(5, shuffle=True, random_state=split_seed)
                     .split(X, y_values))
        results, oofs = {}, {}
        for variant in VARIANTS:
            oofs[variant], results[variant] = run_variant(variant, X, y_values, folds, args, oofs)
            path = output / f"seed{split_seed}_{variant}_oof.csv"
            pd.DataFrame({"source_row_index": y.index.to_numpy(), "y_true": y_values,
                          "oof_score": oofs[variant]}).to_csv(path, index=False)
            results[variant]["oof_artifact"] = {"filename": path.name, "sha256": file_sha256(path)}
            print(f"seed {split_seed} {variant:13s} OOF AUC "
                  f"{results[variant]['oof_roc_auc']:.5f}", flush=True)
        vs_baseline = {}
        for variant in VARIANTS[1:]:
            fold_view = paired(results["baseline"], results[variant])
            vs_baseline[variant] = {
                "pooled_auc_delta": results[variant]["oof_roc_auc"] - results["baseline"]["oof_roc_auc"],
                "pooled_ap_delta": (results[variant]["oof_average_precision"]
                                    - results["baseline"]["oof_average_precision"]),
                "folds_improved": int(sum(d > 0 for d in fold_view["fold_auc_deltas"])),
                **fold_view}
        per_seed[split_seed] = {"variants": results, "vs_baseline": vs_baseline,
                                "elapsed_seconds": perf_counter() - started}
        save("comparison.partial.json")

    if file_sha256(args.data) != provenance["dataset"]["sha256"]:
        raise RuntimeError("Dataset changed during the experiment")
    report["summary"] = summarize(per_seed)
    report["seed42_reproduces_missing_types"] = check_seed42(per_seed, args.sample)
    report["interpretation"] = (
        "Split seeds reuse the same rows. Agreement across seeds shows the gain does not "
        "depend on one particular split; the seeds are not independent samples, and model "
        "and tuning randomness was held at seed 42.")
    with (output / "comparison.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    (output / "comparison.partial.json").unlink()
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved: {output / 'comparison.json'}")


if __name__ == "__main__":
    main()
