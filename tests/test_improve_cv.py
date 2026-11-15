import unittest

import numpy as np
import pandas as pd

from improve_cv import (count_to_number, domain_features, fit_di_zero,
                        ordinal_columns, prepare, ratio)


def frame(procedure, embryos):
    return pd.DataFrame({"시술 유형": procedure, "총 생성 배아 수": embryos,
                         "총 시술 횟수": ["0회"] * len(procedure)})


class TestImprovePreprocessing(unittest.TestCase):

    def test_count_strings_become_ordered_numbers(self):
        values = pd.Series(["0회", "3회", "6회 이상", "__MISSING__"])
        result = count_to_number(values).tolist()
        self.assertEqual(result[:3], [0.0, 3.0, 6.0])
        self.assertTrue(np.isnan(result[3]))

    def test_age_band_keeps_category_and_adds_order(self):
        X = pd.DataFrame({"시술 당시 나이": ["만18-34세", "만45-50세", "알 수 없음"]})
        result = ordinal_columns(X)
        self.assertEqual(result["시술 당시 나이"].tolist(), X["시술 당시 나이"].tolist())
        self.assertEqual(result["시술 당시 나이_ord"].tolist()[:2], [0.0, 5.0])
        self.assertTrue(np.isnan(result["시술 당시 나이_ord"].iloc[2]))

    def test_di_zero_columns_come_from_training_rows_only(self):
        train = frame(["DI", "IVF"], [np.nan, 4.0])
        # In validation a DI row has a value; it must not change the fitted list.
        valid = frame(["DI", "DI", "IVF"], [np.nan, 2.0, np.nan])
        transform, columns = fit_di_zero(train)
        self.assertEqual(columns, ["총 생성 배아 수"])
        result = transform(valid)["총 생성 배아 수"].tolist()
        self.assertEqual(result[:2], [0.0, 2.0])
        self.assertTrue(np.isnan(result[2]), "IVF gaps stay missing")

    def test_di_zero_skips_columns_observed_for_training_di_rows(self):
        train = frame(["DI", "DI", "IVF"], [np.nan, 1.0, 4.0])
        _, columns = fit_di_zero(train)
        self.assertEqual(columns, [])

    def test_ratio_is_missing_when_denominator_is_not_positive(self):
        result = ratio(pd.Series([1.0, 2.0, 3.0]), pd.Series([2.0, 0.0, np.nan]))
        self.assertEqual(result[0], 0.5)
        self.assertTrue(np.isnan(result[1:]).all())

    def test_domain_features_use_no_label(self):
        columns = ["수집된 신선 난자 수", "해동 난자 수", "이식된 배아 수", "총 생성 배아 수",
                   "저장된 배아 수", "미세주입에서 생성된 배아 수", "미세주입된 난자 수",
                   "혼합된 난자 수", "파트너 정자와 혼합된 난자 수", "총 임신 횟수",
                   "총 시술 횟수", "총 출산 횟수", "클리닉 내 총 시술 횟수",
                   "배아 이식 경과일", "난자 혼합 경과일", "불임 원인 - 난관 질환"]
        X = pd.DataFrame({c: [1.0, 2.0] for c in columns})
        first = domain_features(X)
        # Changing one row must not change another row's features.
        X.iloc[1] = 9.0
        second = domain_features(X)
        new = [c for c in first.columns if c.startswith("f_")]
        pd.testing.assert_series_equal(first[new].iloc[0], second[new].iloc[0])

    def test_baseline_variant_is_unchanged_by_prepare(self):
        train = frame(["DI", "IVF"], [np.nan, 4.0])
        X_train, X_valid, info = prepare(train, train, "native_nan")
        pd.testing.assert_frame_equal(X_train, train)
        self.assertEqual(info, {})


if __name__ == "__main__":
    unittest.main()
