"""Save, load and predict with the final model on synthetic data."""
import importlib.util
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HAVE_EXTRAS = all(importlib.util.find_spec(m) for m in ["catboost", "optuna"])


def synthetic(rows=600, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.choice(["young", "middle", "old"], rows)
    count = rng.integers(0, 4, rows).astype(float)
    count[rng.random(rows) < 0.1] = np.nan
    days = rng.normal(5, 1, rows)
    logit = (age == "young") * 1.0 - (age == "old") * 1.0 + np.nan_to_num(count) * 0.4 - 1
    y = (rng.random(rows) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"age": age, "count": count, "days": days})
    return X, y


@unittest.skipUnless(HAVE_EXTRAS, "needs catboost and optuna (requirements-improve.txt)")
class TestFinalModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from final_model import FinalModel
        cls.X, cls.y = synthetic()
        cls.model = FinalModel.fit(cls.X, cls.y, trials=2)

    def test_matches_experiment_code(self):
        """The final fit uses exactly the procedure the experiments evaluated."""
        from final_model import fit_catboost, fit_encoder, fit_lgb_tuned
        from improve_cv import catboost_native, lgb_tuned, one_hot
        X_fit, X_valid = self.X.iloc[:450], self.X.iloc[450:]
        y_fit = self.y[:450]
        M_fit, M_valid = one_hot(X_fit, X_valid)
        expected_lgb, _ = lgb_tuned(M_fit, y_fit, M_valid, 2)
        encoder = fit_encoder(X_fit)
        lgb, _ = fit_lgb_tuned(encoder.transform(X_fit), y_fit, 2)
        np.testing.assert_array_equal(lgb.predict_proba(encoder.transform(X_valid))[:, 1],
                                      expected_lgb)
        expected_cat, _ = catboost_native(X_fit, y_fit, X_valid)
        cat, _ = fit_catboost(X_fit, y_fit)
        np.testing.assert_array_equal(cat.predict_proba(X_valid)[:, 1], expected_cat)

    def test_save_load_round_trip(self):
        from final_model import load, save
        before, _ = self.model.predict_proba(self.X)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "m"
            manifest = save(self.model, directory, {})
            self.assertEqual(manifest["columns"], ["age", "count", "days"])
            reloaded, _ = load(directory)
            after, _ = reloaded.predict_proba(self.X)
        np.testing.assert_array_equal(before, after)

    def test_tampered_bundle_is_refused(self):
        from final_model import save, load
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "m"
            save(self.model, directory, {})
            with (directory / "model.joblib").open("ab") as handle:
                handle.write(b"x")
            with self.assertRaises(RuntimeError):
                load(directory)

    def test_package_mismatch_warns(self):
        from final_model import load, save
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "m"
            save(self.model, directory, {})
            path = directory / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["packages"]["lightgbm"] = "0.0"
            path.write_text(json.dumps(manifest))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                load(directory)
            self.assertTrue(any("lightgbm" in str(w.message) for w in caught))

    def test_scores_are_probabilities_and_row_independent(self):
        scores, _ = self.model.predict_proba(self.X)
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())
        single, _ = self.model.predict_proba(self.X.iloc[[7]])
        self.assertAlmostEqual(float(single[0]), float(scores[7]), places=12)
        shuffled, _ = self.model.predict_proba(self.X.iloc[::-1])
        np.testing.assert_allclose(shuffled[::-1], scores, rtol=0, atol=1e-12)

    def test_mean_of_the_two_models(self):
        p_lgb, p_cat, _ = self.model.predict_parts(self.X)
        scores, _ = self.model.predict_proba(self.X)
        np.testing.assert_allclose(scores, (p_lgb + p_cat) / 2)

    def test_input_rules(self):
        from final_model import InputError
        with self.assertRaises(InputError):
            self.model.predict_proba(self.X.drop(columns=["days"]))
        bad = self.X.astype({"count": object}).copy()
        bad.loc[0, "count"] = "three"
        with self.assertRaises(InputError):
            self.model.predict_proba(bad)

        extra = self.X.head(5).assign(ID=[f"r{i}" for i in range(5)], note="x")
        _, report = self.model.predict_proba(extra)
        self.assertEqual(report["ignored_columns"], ["ID", "note"])

        unseen = self.X.head(5).copy()
        unseen.loc[unseen.index[:2], "age"] = "unknown-group"
        unseen.loc[unseen.index[2], "age"] = np.nan
        scores, report = self.model.predict_proba(unseen)
        self.assertEqual(report["unseen_categories"], {"age": 3})
        self.assertTrue(np.isfinite(scores).all())

    def test_numbers_read_as_text_are_accepted(self):
        as_text = self.X.head(5).astype({"count": str, "days": str})
        as_text["count"] = as_text["count"].replace("nan", "")
        expected, _ = self.model.predict_proba(self.X.head(5))
        scores, _ = self.model.predict_proba(as_text)
        np.testing.assert_allclose(scores, expected)


if __name__ == "__main__":
    unittest.main()
