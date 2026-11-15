"""Tests for the split-seed comparison helpers (no training)."""
import unittest

from seed_cv import VARIANTS, parse_seeds, summarize


def fake_seed(base_auc, gain, folds_up):
    variants = {v: {"oof_roc_auc": base_auc + (0 if v == "baseline" else gain),
                    "oof_average_precision": 0.4, "elapsed_seconds": 1.0} for v in VARIANTS}
    vs = {v: {"pooled_auc_delta": gain, "folds_improved": folds_up} for v in VARIANTS[1:]}
    return {"variants": variants, "vs_baseline": vs}


class TestSeedCv(unittest.TestCase):

    def test_parse_seeds(self):
        self.assertEqual(parse_seeds("42, 1,2"), [42, 1, 2])
        for text in ["", "1,1"]:
            with self.assertRaises(ValueError, msg=text):
                parse_seeds(text)

    def test_summary_counts_seeds_and_folds(self):
        per_seed = {42: fake_seed(0.74, 0.0013, 5), 1: fake_seed(0.73, -0.0002, 2)}
        blend = summarize(per_seed)["nan_blend"]
        self.assertEqual(blend["seeds_with_gain"], 1)
        self.assertEqual(blend["folds_improved"], "7/10")
        self.assertAlmostEqual(blend["auc_gain_over_baseline"]["mean"], 0.00055)
        self.assertAlmostEqual(blend["auc_gain_over_baseline"]["min"], -0.0002)
        self.assertNotIn("auc_gain_over_baseline", summarize(per_seed)["baseline"])

    def test_single_seed_has_no_sd(self):
        summary = summarize({42: fake_seed(0.74, 0.001, 5)})
        self.assertIsNone(summary["baseline"]["oof_roc_auc"]["sd"])


if __name__ == "__main__":
    unittest.main()
