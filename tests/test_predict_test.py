import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_cv import TARGET, load_dataset
from predict_test import fit_and_predict, load_test_features


def synthetic_frames(rows=60, seed=0):
    rng = np.random.default_rng(seed)
    numeric = rng.normal(size=rows)
    numeric[::7] = np.nan
    category = rng.choice(["A", "B", None], size=rows)
    target = (rng.random(rows) < 0.3).astype(int)
    train = pd.DataFrame({"ID": [f"TRAIN_{i}" for i in range(rows)],
                          "numeric": numeric, "category": category, TARGET: target})
    test = pd.DataFrame({"ID": [f"TEST_{i}" for i in range(5)],
                         "category": ["A", "Z", None, "B", "A"],
                         "numeric": [0.1, np.nan, np.inf, -2.0, 5.0]})
    return train, test


class TestPredictTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        train, test = synthetic_frames()
        self.train_path, self.test_path = self.dir / "train.csv", self.dir / "test.csv"
        train.to_csv(self.train_path, index=False)
        test.to_csv(self.test_path, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_matches_training_normalization_and_order(self):
        X_train, y_train = load_dataset(self.train_path)
        ids, X_test = load_test_features(self.test_path, X_train.columns)
        self.assertEqual(list(X_test.columns), list(X_train.columns))
        self.assertEqual(ids.tolist(), [f"TEST_{i}" for i in range(5)])
        self.assertEqual(X_test["category"].iloc[2], "__MISSING__")
        self.assertTrue(np.isnan(X_test["numeric"].iloc[2]))

        _, probabilities = fit_and_predict(X_train, y_train, X_test, n_estimators=5)
        self.assertEqual(len(probabilities), 5)
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_rejects_column_mismatch(self):
        X_train, _ = load_dataset(self.train_path)
        pd.DataFrame({"ID": ["T0"], "numeric": [1.0]}).to_csv(self.test_path, index=False)
        with self.assertRaises(ValueError):
            load_test_features(self.test_path, X_train.columns)

    def test_rejects_target_in_test(self):
        X_train, _ = load_dataset(self.train_path)
        frame = pd.read_csv(self.test_path)
        frame[TARGET] = 0
        frame.to_csv(self.test_path, index=False)
        with self.assertRaises(ValueError):
            load_test_features(self.test_path, X_train.columns)


if __name__ == "__main__":
    unittest.main()
