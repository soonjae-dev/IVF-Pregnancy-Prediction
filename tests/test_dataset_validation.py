"""Validate the raw CSV target before labels can influence training or scoring."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from baseline_cv import TARGET, load_dataset


class TestDatasetValidation(unittest.TestCase):
    def test_binary_labels_and_feature_columns_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            pd.DataFrame({"ID": ["a", "b"], "numeric": [1, 2],
                          TARGET: [0.0, 1.0]}).to_csv(path, index=False)
            features, labels = load_dataset(path)
        self.assertEqual(features.columns.tolist(), ["numeric"])
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertEqual(labels.dtype.kind, "i")

    def test_invalid_labels_are_rejected_before_conversion(self):
        cases = {
            "fractional": [0.2, 1.8],
            "other_binary_labels": [1, 2],
            "negative": [-1, 1],
            "missing": [0, None],
            "infinite": [0, float("inf")],
            "text": ["no", "yes"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            for name, labels in cases.items():
                with self.subTest(name=name):
                    pd.DataFrame({"numeric": [1, 2], TARGET: labels}).to_csv(
                        path, index=False)
                    with self.assertRaisesRegex(ValueError, "nonmissing binary values 0 and 1"):
                        load_dataset(path)

    def test_missing_target_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            pd.DataFrame({"numeric": [1, 2]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Target not found"):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
