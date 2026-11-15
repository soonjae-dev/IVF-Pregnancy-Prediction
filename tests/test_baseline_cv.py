
import unittest

import numpy as np
import pandas as pd

from baseline_cv import build_pipeline


class TestBaselinePreprocessing(unittest.TestCase):

    def test_training_only_statistics(self):
        train = pd.DataFrame({
            "numeric": [1.0, 3.0, np.nan],
            "category": ["A", "A", "B"],
        })

        validation = pd.DataFrame({
            "numeric": [np.nan, 1000.0],
            "category": ["C", "A"],
        })

        pipeline = build_pipeline(train, n_estimators=5)
        preprocessor = pipeline.named_steps["preprocessor"]

        preprocessor.fit(train)
        transformed = preprocessor.transform(validation)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        # Training median = 2, not affected by validation.
        self.assertAlmostEqual(
            preprocessor.named_transformers_["num"]
            .statistics_[0],
            2.0,
        )

        self.assertAlmostEqual(
            transformed[0, 0],
            2.0,
        )

        # Unseen category C has no fitted one-hot column.
        np.testing.assert_array_equal(
            transformed[0, 1:],
            np.zeros(2),
        )

    def test_missing_indicators_are_fitted_on_training_only(self):
        train = pd.DataFrame({"missing_in_train": [1., np.nan, 3.],
                              "complete_in_train": [2., 4., 6.]})
        valid = pd.DataFrame({"missing_in_train": [np.nan],
                              "complete_in_train": [np.nan]})
        preprocessor = build_pipeline(train, missing_indicators=True).named_steps["preprocessor"]
        preprocessor.fit(train)
        transformed = preprocessor.transform(valid)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        np.testing.assert_allclose(transformed, [[2., 4., 1.]])
        np.testing.assert_array_equal(
            preprocessor.named_transformers_["num"].indicator_.features_, [0])


if __name__ == "__main__":
    unittest.main()
