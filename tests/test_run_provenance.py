"""Exercise the real CLI on synthetic data without touching existing results."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from analyze_oof import analyze, analyze_files


class TestRunProvenance(unittest.TestCase):
    def test_repeated_runs_preserve_artifacts_and_original_rows(self):
        script = Path(__file__).resolve().parents[1] / "baseline_cv.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "synthetic.csv"
            original = pd.DataFrame({"numeric": range(80),
                                     "category": ["a", "b"] * 40,
                                     "임신 성공 여부": [0, 1] * 40})
            original.to_csv(data, index=False)
            command = [sys.executable, str(script), "--data", str(data),
                       "--output-dir", str(root), "--sample", "40",
                       "--folds", "2", "--estimators", "2"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            first = next(root.glob("baseline_*/summary.json"))
            first_bytes = first.read_bytes()
            subprocess.run(command, check=True, capture_output=True, text=True)
            reports = list(root.glob("baseline_*/summary.json"))
            self.assertEqual(len(reports), 2)
            self.assertEqual(first.read_bytes(), first_bytes)
            for report in reports:
                summary = json.loads(report.read_text())
                prediction = report.parent / summary["oof_artifact"]["filename"]
                frame = pd.read_csv(prediction)
                self.assertEqual(summary["provenance"]["dataset"]["sha256"],
                                 hashlib.sha256(data.read_bytes()).hexdigest())
                self.assertEqual(summary["oof_artifact"]["sha256"],
                                 hashlib.sha256(prediction.read_bytes()).hexdigest())
                self.assertEqual(summary["provenance"]["code"]["baseline_cv_sha256"],
                                 hashlib.sha256(script.read_bytes()).hexdigest())
                self.assertEqual(summary["model_parameters"]["n_estimators"], 2)
                self.assertIn("lightgbm", summary["provenance"]["environment"]["packages"])
                self.assertEqual(frame.source_row_index.nunique(), 40)
                self.assertGreater(frame.source_row_index.max(), 39)
                np.testing.assert_array_equal(frame.y_true,
                    original.iloc[frame.source_row_index]["임신 성공 여부"].to_numpy())
                analyze(frame, summary)
                analyze_files(prediction, report)
                # A numerically equivalent CSV still has a different artifact identity.
                prediction.write_bytes(prediction.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    analyze_files(prediction, report)


if __name__ == "__main__":
    unittest.main()
