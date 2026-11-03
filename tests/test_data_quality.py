import unittest

import pandas as pd

from analytics.data import load_dataset, prepare_dataset
from analytics.quality import audit_quality
from analytics.schema import SCHEMA
from config import DATA_PATH, TEAMS


class DataQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = load_dataset(DATA_PATH)
        cls.raw, cls.prepared, cls.changes = prepare_dataset(cls.original, TEAMS)

    def test_scope_and_dictionary(self):
        self.assertEqual(len(SCHEMA), 28)
        self.assertEqual(set(self.prepared["time_id"]), set(TEAMS))
        self.assertEqual(self.prepared.groupby("time_id").size().tolist(), [74, 74])
        self.assertEqual(self.prepared["partida_id"].nunique(), 146)
        self.assertEqual(int(self.prepared.duplicated(["time_id", "partida_id"]).sum()), 0)
        self.assertEqual(self.prepared["data_partida"].min(), pd.Timestamp("2025-10-22"))

    def test_current_source_passes_quality(self):
        audit = audit_quality(self.prepared, self.raw, TEAMS, self.original)
        self.assertTrue(audit["issues"].empty)
        self.assertEqual(audit["summary"]["Ausentes na leitura"].sum(), 0)
        self.assertFalse(audit["cleaned"]["tocos"].isna().any())

    def test_invalid_values_are_flagged_without_overwriting_original(self):
        original = self.original.copy()
        index = self.prepared.index[0]
        original.at[index, "BLK"] = "-1"
        original.at[index, "FG_PCT"] = "1.2"
        original.at[index, "GAME_DATE"] = "data inválida"
        raw, prepared, _ = prepare_dataset(original, TEAMS)
        audit = audit_quality(prepared, raw, TEAMS)
        self.assertEqual(raw.at[index, "tocos"], "-1")
        self.assertTrue(pd.isna(audit["cleaned"].at[index, "tocos"]))
        self.assertTrue(pd.isna(audit["cleaned"].at[index, "aproveitamento_quadra"]))
        self.assertTrue(pd.isna(audit["cleaned"].at[index, "data_partida"]))
        self.assertIn("Falha de conversão", audit["issues"]["Regra"].tolist())

    def test_valid_zero_and_fractional_count(self):
        original = self.original.copy()
        index = self.prepared.index[0]
        original.at[index, "BLK"] = "0"
        original.at[index, "STL"] = "1.5"
        raw, prepared, _ = prepare_dataset(original, TEAMS)
        audit = audit_quality(prepared, raw, TEAMS)
        self.assertEqual(audit["cleaned"].at[index, "tocos"], 0)
        self.assertTrue(pd.isna(audit["cleaned"].at[index, "roubos_bola"]))

    def test_missing_and_duplicate_keys(self):
        raw = self.raw.iloc[:2].copy()
        prepared = self.prepared.iloc[:2].copy()
        first, second = prepared.index
        prepared.at[second, "partida_id"] = prepared.at[first, "partida_id"]
        prepared.at[first, "tocos"] = pd.NA
        raw.at[first, "tocos"] = pd.NA
        audit = audit_quality(prepared, raw, TEAMS)
        self.assertEqual(audit["duplicate_key_rows"], 2)
        row = audit["summary"].set_index("Nome").loc["tocos"]
        self.assertEqual(row["Ausentes na leitura"], 1)
        self.assertEqual(row["Ausentes (%)"], 50)


if __name__ == "__main__":
    unittest.main()

