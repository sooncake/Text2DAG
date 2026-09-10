"""Validate the hand-specified expert DAG adjacency artifact."""

from __future__ import annotations

import unittest
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


DAG_PATH = Path(__file__).with_name("expert_dag_adjacency.csv")


@unittest.skipUnless(pd is not None, "pandas is not installed")
class ExpertDAGAdjacencyTests(unittest.TestCase):
    def test_exact_named_edge_set(self) -> None:
        raw = pd.read_csv(DAG_PATH)
        names = raw.iloc[:, 0].astype(str).tolist()
        adjacency = raw.iloc[:, 1:].copy()
        adjacency.index = names
        self.assertEqual(names, adjacency.columns.tolist())
        self.assertEqual(adjacency.shape, (16, 16))
        self.assertTrue((adjacency.to_numpy().diagonal() == 0).all())

        actual = {
            (source, target)
            for source in names
            for target in names
            if int(adjacency.loc[source, target]) == 1
        }
        expected = {
            ("smoking", "COPD"),
            ("smoking", "dysp"),
            ("smoking", "cough"),
            ("COPD", "dysp"),
            ("COPD", "cough"),
            ("COPD", "pain"),
            ("COPD", "pneu"),
            ("asthma", "dysp"),
            ("asthma", "cough"),
            ("asthma", "pneu"),
            ("season", "pneu"),
            ("season", "cold"),
            ("pneu", "dysp"),
            ("pneu", "cough"),
            ("pneu", "pain"),
            ("pneu", "fever"),
            ("cold", "cough"),
            ("cold", "pain"),
            ("cold", "fever"),
            ("cold", "nasal"),
            ("hay_fever", "dysp"),
            ("hay_fever", "nasal"),
            ("dysp", "antibiotics"),
            ("dysp", "days_at_home"),
            ("cough", "pain"),
            ("cough", "antibiotics"),
            ("cough", "days_at_home"),
            ("pain", "antibiotics"),
            ("pain", "days_at_home"),
            ("fever", "antibiotics"),
            ("fever", "days_at_home"),
            ("nasal", "days_at_home"),
            ("policy", "antibiotics"),
            ("antibiotics", "days_at_home"),
            ("self_empl", "days_at_home"),
        }
        self.assertEqual(len(expected), 35)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
