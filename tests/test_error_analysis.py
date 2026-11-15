"""Tests for the subgroup error analysis on synthetic data."""
import unittest

import numpy as np
import pandas as pd

from error_analysis import (group_rate_auc, paired_bootstrap, summarize, transfer_bucket)


class TestErrorAnalysis(unittest.TestCase):

    def test_transfer_buckets(self):
        values = pd.Series([0, 1, 2, 3, 5, np.nan])
        self.assertEqual(list(transfer_bucket(values)),
                         ["0개", "1개", "2개", "3개 이상", "3개 이상", "기록 없음"])

    def test_single_class_group_has_no_auc(self):
        y = np.zeros(20, dtype=int)
        row = summarize(y, np.linspace(0, 1, 20), np.linspace(0, 1, 20), rounds=10)
        self.assertIsNone(row["auc_baseline"])
        self.assertIsNone(row["auc_delta_ci95"])
        self.assertTrue(row["small"])

    def test_better_model_has_positive_interval(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 2000)
        weak = y + rng.normal(0, 2.0, len(y))
        strong = y + rng.normal(0, 0.5, len(y))
        row = summarize(y, weak, strong, rounds=50)
        self.assertGreater(row["auc_delta"], 0)
        self.assertGreater(row["auc_delta_ci95"][0], 0)

    def test_bootstrap_is_repeatable_for_identical_rows(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 500)
        a, b = rng.random(500), rng.random(500)
        first = summarize(y, a, b, rounds=20)["auc_delta_ci95"]
        second = summarize(y, a, b, rounds=20)["auc_delta_ci95"]
        self.assertEqual(first, second)
        self.assertIsNone(paired_bootstrap(np.ones(5, dtype=int), a[:5], b[:5], 5,
                                           np.random.default_rng(0)))

    def test_group_rate_auc(self):
        labels = np.array(["a"] * 500 + ["b"] * 500, dtype=object)
        y = np.array([0] * 500 + [1] * 500)
        self.assertAlmostEqual(group_rate_auc(y, labels), 1.0)
        rng = np.random.default_rng(2)
        noise = rng.integers(0, 2, 1000)
        self.assertLess(abs(group_rate_auc(noise, labels) - 0.5), 0.1)


if __name__ == "__main__":
    unittest.main()
