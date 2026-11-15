import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from analyze_oof import analyze, best_threshold, validate_oof
from baseline_cv import run_cv


class TestOOF(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"row_index": range(6), "fold": [1, 1, 1, 2, 2, 2],
                                   "y_true": [0, 1, 1, 0, 1, 0],
                                   "oof_probability": [0.1, 0.2, 0.2, 0.2, 0.8, 0.9]})

    def test_threshold_with_ties_matches_brute_force(self):
        result = best_threshold(self.frame.y_true, self.frame.oof_probability)
        expected = max(f1_score(self.frame.y_true, self.frame.oof_probability >= t)
                       for t in self.frame.oof_probability.unique())
        self.assertAlmostEqual(result["f1"], expected)
        self.assertAlmostEqual(result["f1"], f1_score(
            self.frame.y_true, self.frame.oof_probability >= result["threshold"]))

    def test_corrupt_saved_data_rejected(self):
        for column, value in [("row_index", 1), ("oof_probability", np.inf),
                              ("y_true", 2), ("fold", 0), ("fold", 1.5)]:
            with self.subTest(column=column, value=value):
                frame = self.frame.copy()
                frame[column] = frame[column].astype(float)
                frame.loc[0, column] = value
                with self.assertRaises(ValueError):
                    validate_oof(frame)

    def test_summary_mismatch_rejected(self):
        summary = dict(rows=6, folds=2, **analyze(self.frame)["metrics"])
        analyze(self.frame, summary)
        summary["oof_f1_at_0.5"] += 0.01
        with self.assertRaises(ValueError):
            analyze(self.frame, summary)

    def test_cv_returns_one_prediction_per_position(self):
        X = pd.DataFrame({"number": np.arange(30)}, index=np.arange(100, 130))
        y = pd.Series([0, 1] * 15, index=X.index)
        summary, frame = run_cv(X, y, n_splits=3, n_estimators=2)
        validate_oof(frame)
        np.testing.assert_array_equal(frame.y_true, y.to_numpy())
        np.testing.assert_array_equal(frame.source_row_index, X.index)
        analyze(frame, summary)

    def test_duplicate_source_rows_rejected(self):
        frame = self.frame.copy()
        frame["source_row_index"] = [0, 1, 2, 3, 4, 4]
        with self.assertRaisesRegex(ValueError, "Source row"):
            validate_oof(frame)


if __name__ == "__main__":
    unittest.main()
