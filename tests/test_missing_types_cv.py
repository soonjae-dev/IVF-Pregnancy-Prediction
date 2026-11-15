import unittest

import numpy as np
import pandas as pd

from missing_types_cv import NOT_APPLICABLE, fit_flag_columns, fit_flags0, fit_typed


def records(**columns):
    return pd.DataFrame(columns)


class TestMissingTypes(unittest.TestCase):

    def setUp(self):
        # Two IVF rows (one transfer, one without) and one DI row.
        self.train = records(**{
            "시술 유형": ["IVF", "IVF", "DI"],
            "이식된 배아 수": [2.0, 0.0, np.nan],
            "배아 이식 경과일": [5.0, np.nan, np.nan],
            "총 생성 배아 수": [4.0, 3.0, np.nan],
            "PGS 시술 여부": [1.0, np.nan, np.nan],
        })

    def test_flag_columns_are_those_recording_only_one(self):
        self.assertEqual(fit_flag_columns(self.train), ["PGS 시술 여부"])
        both = self.train.assign(**{"PGS 시술 여부": [1.0, 0.0, np.nan]})
        self.assertEqual(fit_flag_columns(both), [])

    def test_flags0_turns_blank_into_no(self):
        transform, _ = fit_flags0(self.train)
        self.assertEqual(transform(self.train)["PGS 시술 여부"].tolist(), [1.0, 0.0, 0.0])

    def test_typed_separates_not_applicable_from_truly_missing(self):
        transform, info = fit_typed(self.train)
        valid = records(**{
            "시술 유형": ["IVF", "IVF", "DI"],
            # Row 0: embryos transferred but no day recorded -> truly missing.
            "이식된 배아 수": [1.0, 0.0, np.nan],
            "배아 이식 경과일": [np.nan, np.nan, np.nan],
            "총 생성 배아 수": [np.nan, 2.0, np.nan],
            "PGS 시술 여부": [np.nan, 1.0, np.nan],
        })
        result = transform(valid)
        day = result["배아 이식 경과일"].tolist()
        self.assertTrue(np.isnan(day[0]))
        self.assertEqual(day[1:], [NOT_APPLICABLE, NOT_APPLICABLE])
        self.assertEqual(result["배아 이식 경과일_truly_missing"].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(result["PGS 시술 여부"].tolist(), [0.0, 1.0, 0.0])
        counts = result["총 생성 배아 수"].tolist()
        self.assertTrue(np.isnan(counts[0]), "an IVF gap in a count is not assumed to be 0")
        self.assertEqual(counts[2], 0.0, "DI counts are not applicable, so 0")
        self.assertEqual(info["flag_columns"], ["PGS 시술 여부"])

    def test_rules_come_from_training_rows_only(self):
        transform, _ = fit_typed(self.train)
        # A validation DI row with a recorded embryo count keeps it.
        valid = self.train.assign(**{"총 생성 배아 수": [np.nan, np.nan, 7.0]})
        self.assertEqual(transform(valid)["총 생성 배아 수"].iloc[2], 7.0)


if __name__ == "__main__":
    unittest.main()
