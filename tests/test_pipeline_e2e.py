"""
End-to-end run of the experiment scripts on synthetic data.

Each script runs as a command, as a user would run it, on a small synthetic
file with the competition's schema (tests/synthetic_ivf.py): inner tuning,
LightGBM and CatBoost, out-of-fold predictions, the blend, the saved files and
the summary. The point is to catch broken connections between the pieces, not
to reproduce any score. The competition data is never used.
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from synthetic_ivf import TARGET, write_competition_files
except ImportError:                       # run from the repository root
    from tests.synthetic_ivf import TARGET, write_competition_files

ROOT = Path(__file__).resolve().parents[1]
HAVE_EXTRAS = all(importlib.util.find_spec(m) for m in ["catboost", "optuna"])
FAST = ["--trials", "2", "--estimators", "20"]


def sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@unittest.skipUnless(HAVE_EXTRAS, "needs catboost and optuna (requirements-improve.txt)")
class TestPipelineEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.train, cls.test = write_competition_files(cls.dir)
        cls.frame = pd.read_csv(cls.train)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_script(self, *arguments):
        completed = subprocess.run([sys.executable, *arguments], cwd=ROOT,
                                   capture_output=True, text=True, timeout=900)
        self.assertEqual(completed.returncode, 0,
                         f"{arguments[0]} failed:\n{completed.stderr[-3000:]}")
        return completed.stdout

    def only_run(self, pattern):
        runs = sorted(self.dir.glob(pattern))
        self.assertEqual(len(runs), 1, f"expected one {pattern} folder, found {runs}")
        return runs[0]

    def check_oof(self, folder, filename, recorded):
        """Rows aligned with the input, labels intact, scores finite, metrics consistent."""
        path = folder / filename
        self.assertEqual(sha256(path), recorded["oof_artifact"]["sha256"])
        oof = pd.read_csv(path, float_precision="round_trip")
        self.assertEqual(len(oof), len(self.frame))
        self.assertEqual(sorted(oof["source_row_index"]), list(range(len(self.frame))))
        labels = self.frame[TARGET].to_numpy()[oof["source_row_index"]]
        np.testing.assert_array_equal(oof["y_true"].to_numpy(), labels)
        scores = oof["oof_score"].to_numpy()
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())
        self.assertAlmostEqual(roc_auc_score(labels, scores), recorded["oof_roc_auc"], places=12)
        self.assertAlmostEqual(average_precision_score(labels, scores),
                               recorded["oof_average_precision"], places=12)
        self.assertEqual(len(recorded["fold_results"]), 5)
        return scores

    def check_report(self, folder):
        self.assertFalse((folder / "comparison.partial.json").exists())
        report = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(report["provenance"]["dataset"]["sha256"], sha256(self.train))
        return report

    def test_1_improvement_ladder(self):
        self.run_script("improve_cv.py", "--data", str(self.train),
                        "--output-dir", str(self.dir), *FAST)
        folder = self.only_run("improve_*")
        report = self.check_report(folder)
        self.assertEqual(list(report["variants"]), report["ladder"])
        for variant, recorded in report["variants"].items():
            self.check_oof(folder, f"{variant}_oof.csv", recorded)
        self.assertEqual(set(report["paired_vs_baseline"]), set(report["ladder"][1:]))
        tuned = report["variants"]["lgb_tuned"]["fold_details"][0]
        self.assertTrue({"params", "n_estimators", "inner_auc"} <= set(tuned))

    def test_2_missing_types(self):
        self.run_script("missing_types_cv.py", "--data", str(self.train),
                        "--output-dir", str(self.dir), *FAST)
        folder = self.only_run("missing_types_*")
        report = self.check_report(folder)
        scores = {v: self.check_oof(folder, f"{v}_oof.csv", r)
                  for v, r in report["variants"].items()}
        # The blend is the rank average of its two parts, fold by fold.
        self.assertTrue(np.isfinite(scores["nan_blend"]).all())
        typed = report["variants"]["typed_nan"]["fold_details"][0]
        self.assertIn("flag_columns", typed)
        self.assertEqual(set(typed["flag_columns"]),
                         {"착상 전 유전 검사 사용 여부", "PGD 시술 여부", "PGS 시술 여부"})

    def test_3_split_seeds_and_error_analysis(self):
        self.run_script("seed_cv.py", "--data", str(self.train), "--output-dir", str(self.dir),
                        "--seeds", "42,1", *FAST)
        folder = self.only_run("seeds_*")
        report = self.check_report(folder)
        for seed, result in report["per_seed"].items():
            for variant, recorded in result["variants"].items():
                self.check_oof(folder, f"seed{seed}_{variant}_oof.csv", recorded)
        self.assertEqual(report["summary"]["nan_blend"]["folds_improved"].split("/")[1], "10")

        # Split seed 42 must reproduce missing_types_cv.py on the same data.
        missing = self.dir.glob("missing_types_*/comparison.json")
        if any(missing):
            check = report["seed42_reproduces_missing_types"]
            self.assertIsNotNone(check)
            self.assertEqual(max(check["auc_abs_difference"].values()), 0.0)

            run = self.only_run("missing_types_*")
            out = self.dir / "error_analysis"
            self.run_script("error_analysis.py", "--data", str(self.train), "--run", str(run),
                            "--bootstrap", "20", "--out", str(out))
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            recorded = json.loads((run / "comparison.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(summary["overall"]["auc_model"],
                                   recorded["variants"]["nan_blend"]["oof_roc_auc"], places=12)
            self.assertEqual(set(summary["axes"]),
                             {"시술 유형", "시술 당시 나이", "난자 출처", "동결 배아 사용 여부",
                              "이식된 배아 수"})

    def test_4_final_model(self):
        models = self.dir / "models"
        self.run_script("final_model.py", "train", "--data", str(self.train),
                        "--test", str(self.test), "--model-dir", str(models),
                        "--output-dir", str(self.dir), "--trials", "2")
        model_dir = self.only_run("models/final_*")
        record = json.loads(self.only_run("final_model_*.json").read_text(encoding="utf-8"))
        self.assertTrue(record["save_load_check"]["identical"])
        self.assertEqual(record["bundle_sha256"], sha256(model_dir / "model.joblib"))

        out = self.dir / "scores.csv"
        self.run_script("final_model.py", "predict", "--model", str(model_dir),
                        "--input", str(self.test), "--out", str(out))
        scores = pd.read_csv(out)
        test = pd.read_csv(self.test)
        self.assertEqual(list(scores.columns), ["ID", "probability"])
        self.assertEqual(list(scores["ID"]), list(test["ID"]))
        self.assertTrue(scores["probability"].between(0, 1).all())
        self.assertEqual(sha256(out), record["save_load_check"]["predictions_sha256"])


if __name__ == "__main__":
    unittest.main()
