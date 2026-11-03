import unittest

import numpy as np
import pandas as pd

from analytics.outliers import METHODS, detect_outliers


class OutlierTests(unittest.TestCase):
    def test_iqr_identifies_source_rows(self):
        series = pd.Series([1, 2, 3, 4, 100], index=[10, 20, 30, 40, 90])
        result = detect_outliers(series, "iqr")
        self.assertEqual(result["rows"].index[result["rows"]["sinalizado"]].tolist(), [90])
        self.assertEqual(result["threshold"], (-1, 7))

    def test_grubbs_checks_one_extreme(self):
        series = pd.Series([199.31, 199.53, 200.19, 200.82, 201.92, 201.95, 203.42, 245.57])
        result = detect_outliers(series, "grubbs")
        self.assertEqual(result["count"], 1)
        self.assertTrue(result["rows"].loc[7, "sinalizado"])
        self.assertAlmostEqual(result["threshold"], 2.126645, places=5)

    def test_knn_excludes_self_even_with_duplicates(self):
        series = pd.Series([0.0, 0.0, 2.0, 10.0])
        result = detect_outliers(series, "knn", {"neighbors": 1})
        scale = np.std(series, ddof=0)
        np.testing.assert_allclose(result["rows"]["escore"], [0, 0, 2 / scale, 8 / scale])

    def test_all_methods_preserve_indices_and_use_common_values(self):
        series = pd.Series(list(range(30)) + [200, np.nan], index=range(100, 132))
        for method in METHODS:
            with self.subTest(method=method):
                result = detect_outliers(series, method)
                self.assertEqual(result["n"], 31)
                self.assertEqual(result["rows"].index.tolist(), list(range(100, 131)))
                self.assertEqual(result["count"], int(result["rows"]["sinalizado"].sum()))
                self.assertTrue(result["parameters"])

    def test_non_applicability_and_invalid_method(self):
        for method in METHODS:
            for series in [pd.Series([], dtype=float), pd.Series([1]), pd.Series([2, 2, 2])]:
                with self.subTest(method=method, n=len(series)):
                    result = detect_outliers(series, method)
                    self.assertEqual(result["status"], "não aplicável")
                    self.assertIsNone(result["count"])
        with self.assertRaises(ValueError):
            detect_outliers(pd.Series([1, 2]), "inexistente")


if __name__ == "__main__":
    unittest.main()

