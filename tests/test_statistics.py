import unittest

import numpy as np
import pandas as pd

from analytics.descriptive import describe_series, iqr_details, standardized_team_means


class StatisticsTests(unittest.TestCase):
    def test_sample_statistics_and_linear_quartiles(self):
        s = describe_series(pd.Series([1, 2, 3, 4, 100]))
        self.assertAlmostEqual(s["mean"], 22)
        self.assertEqual(s["median"], 3)
        self.assertEqual((s["q1"], s["q3"], s["iqr"]), (2, 4, 2))
        self.assertAlmostEqual(s["variance"], 1902.5)
        self.assertAlmostEqual(s["std"], np.sqrt(1902.5))
        self.assertEqual(s["outliers"].tolist(), [100])
        self.assertEqual((s["lower"], s["upper"]), (-1, 7))
        self.assertEqual((s["whisker_low"], s["whisker_high"]), (1, 4))
        self.assertEqual(s["modes"], [])
        q = iqr_details(pd.Series([1, 2, 3, 4]))
        self.assertEqual((q["q1"], q["q3"]), (1.75, 3.25))

    def test_multiple_modes_and_valid_zero(self):
        s = describe_series(pd.Series([0, 0, 1, 1, 2, np.nan]))
        self.assertEqual(s["modes"], [0, 1])
        self.assertEqual(s["n"], 5)
        self.assertEqual(s["missing"], 1)
        self.assertEqual(s["min"], 0)

    def test_coefficient_of_variation(self):
        s = describe_series(pd.Series([1, 2, 3]))
        self.assertEqual(s["cv"], 50)
        self.assertTrue(np.isnan(describe_series(pd.Series([0, 0, 0]))["cv"]))

    def test_skew_and_insufficient_samples(self):
        self.assertAlmostEqual(describe_series(pd.Series([1, 2, 3, 4, 5]))["skew"], 0)
        self.assertGreater(describe_series(pd.Series([1, 1, 1, 2, 20]))["skew"], 1)
        self.assertTrue(np.isnan(describe_series(pd.Series([1, 1, 1]))["skew"]))
        self.assertTrue(np.isnan(describe_series(pd.Series([1, 2]))["skew"]))
        self.assertEqual(describe_series(pd.Series([], dtype=float))["n"], 0)

    def test_standardization_uses_joint_reference(self):
        df = pd.DataFrame({"time_id": ["a", "a", "b", "b"], "points": [1, 3, 5, 7]})
        means, reference = standardized_team_means(df, "points", ["a", "b"])
        visible, same_reference = standardized_team_means(df, "points", ["a"])
        self.assertEqual(reference["mean"], 4)
        self.assertEqual(reference, same_reference)
        self.assertEqual(visible["a"], means["a"])
        self.assertLess(means["a"], 0)
        self.assertGreater(means["b"], 0)


if __name__ == "__main__":
    unittest.main()

