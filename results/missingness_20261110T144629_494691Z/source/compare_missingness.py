"""Paired fixed-configuration experiment: numeric missingness indicators."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from baseline_cv import collect_provenance, file_sha256, load_dataset, run_cv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.folds, args.sample, args.estimators = 5, None, 250
    provenance = collect_provenance(args.data, args)
    provenance["code"]["comparison_script_sha256"] = file_sha256(__file__)
    X, y = load_dataset(args.data)
    output = args.output_dir / ("missingness_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    results, frames = {}, {}
    for name, enabled in [("baseline", False), ("missing_indicators", True)]:
        start = perf_counter()
        summary, frame = run_cv(X, y, n_splits=5, n_estimators=250,
                                missing_indicators=enabled)
        summary["elapsed_seconds"] = perf_counter() - start
        path = output / (name + "_oof.csv")
        frame.to_csv(path, index=False)
        summary["oof_artifact"] = {"filename": path.name, "sha256": file_sha256(path)}
        results[name], frames[name] = summary, frame
    for column in ["source_row_index", "fold", "y_true"]:
        if not np.array_equal(frames["baseline"][column], frames["missing_indicators"][column]):
            raise RuntimeError("Paired comparison alignment failed: " + column)
    if file_sha256(args.data) != provenance["dataset"]["sha256"]:
        raise RuntimeError("Dataset changed during experiment")
    differences = [b["roc_auc"] - a["roc_auc"] for a, b in zip(
        results["baseline"]["fold_results"], results["missing_indicators"]["fold_results"])]
    report = {"question": "Does adding numeric missingness indicators change ROC-AUC at fixed model settings?",
              "provenance": provenance, "variants": results,
              "paired_fold_auc_deltas": differences,
              "mean_fold_delta": float(np.mean(differences)),
              "sd_fold_delta": float(np.std(differences, ddof=1)),
              "pooled_oof_auc_delta": results["missing_indicators"]["oof_roc_auc"] - results["baseline"]["oof_roc_auc"],
              "interpretation": "One fixed seed and five overlapping-training folds; deltas are descriptive, not independent replicates or a significance test. Timing is sequential wall time, not a controlled benchmark."}
    with (output / "comparison.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"output": str(output), "fold_deltas": differences,
                      "pooled_delta": report["pooled_oof_auc_delta"]}, indent=2))


if __name__ == "__main__":
    main()
