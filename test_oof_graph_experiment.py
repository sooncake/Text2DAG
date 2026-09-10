"""Tests for OOF leakage guards, reconstruction, and graph evaluation."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:
    np = pd = None


@unittest.skipUnless(
    np is not None and pd is not None,
    "The repository runtime dependencies are not installed.",
)
class OOFGraphExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from run_oof_graph_experiment import (
            OuterFold,
            PCConfig,
            align_adjacencies,
            build_graph_dataset,
            endpoint_matrix_to_adjacency,
            evaluate_graph,
            load_reference_adjacency,
            run_pc_learn,
            validate_nested_subsets,
            validate_oof_integrity,
            validate_outer_folds,
        )

        cls.OuterFold = OuterFold
        cls.PCConfig = PCConfig
        cls.align_adjacencies = staticmethod(align_adjacencies)
        cls.build_graph_dataset = staticmethod(build_graph_dataset)
        cls.endpoint_matrix_to_adjacency = staticmethod(endpoint_matrix_to_adjacency)
        cls.evaluate_graph = staticmethod(evaluate_graph)
        cls.load_reference_adjacency = staticmethod(load_reference_adjacency)
        cls.run_pc_learn = staticmethod(run_pc_learn)
        cls.validate_nested_subsets = staticmethod(validate_nested_subsets)
        cls.validate_oof_integrity = staticmethod(validate_oof_integrity)
        cls.validate_outer_folds = staticmethod(validate_outer_folds)

    def test_fixed_pc_configuration(self) -> None:
        config = self.PCConfig()
        self.assertEqual(config.algorithm, "PC-learn")
        self.assertEqual(config.conditional_independence_test, "gsq")
        self.assertEqual(config.alpha, 0.05)
        self.assertTrue(config.stable)
        self.assertIsNone(config.max_k)

    def test_outer_fold_and_oof_leakage_guards(self) -> None:
        patient_ids = np.arange(8, dtype=np.int64) + 100
        folds = [
            self.OuterFold(1, np.arange(2, 8), np.arange(0, 2)),
            self.OuterFold(2, np.array([0, 1, 4, 5, 6, 7]), np.arange(2, 4)),
            self.OuterFold(3, np.array([0, 1, 2, 3, 6, 7]), np.arange(4, 6)),
            self.OuterFold(4, np.arange(0, 6), np.arange(6, 8)),
        ]
        self.validate_outer_folds(patient_ids, folds)
        nested = {}
        for fold in folds:
            train = fold.train_indices
            nested[fold.fold_id] = {
                0.05: train[:1],
                0.10: train[:2],
                0.20: train[:3],
                1.00: train,
            }
        fold_ids = np.repeat(np.arange(1, 5), 2)
        table = pd.DataFrame({"patient_id": patient_ids, "outer_fold": fold_ids})
        oof_tables = {fraction: table.copy() for fraction in (0.05, 0.10, 0.20, 1.00)}
        self.validate_oof_integrity(patient_ids, folds, nested, oof_tables)

        leaked = oof_tables[0.05].copy()
        leaked.loc[0, "outer_fold"] = 2
        broken = dict(oof_tables)
        broken[0.05] = leaked
        with self.assertRaises(AssertionError):
            self.validate_oof_integrity(patient_ids, folds, nested, broken)

    def test_nested_subsets_are_strict_prefix_sets(self) -> None:
        outer = np.arange(20)
        valid = {
            0.05: outer[:1],
            0.10: outer[:2],
            0.20: outer[:4],
            1.00: outer,
        }
        self.validate_nested_subsets(outer, valid)
        invalid = dict(valid)
        invalid[0.10] = np.array([1, 2])
        with self.assertRaises(AssertionError):
            self.validate_nested_subsets(outer, invalid)

    def test_graph_reconstruction_uses_ids_and_changes_only_symptoms(self) -> None:
        original = pd.DataFrame(
            {
                "patient_id": [3, 1, 2],
                "age_group": ["old", "young", "middle"],
                "dysp": ["yes", "no", "yes"],
                "cough": ["no", "yes", "yes"],
                "pain": ["yes", "no", "no"],
                "fever": ["high", "none", "low"],
                "nasal": ["no", "yes", "no"],
            }
        )
        # Deliberately shuffled to prove reconstruction is ID-based, not positional.
        oof = pd.DataFrame(
            {
                "patient_id": [2, 3, 1],
                "dysp_prediction": [0, 1, 0],
                "cough_prediction": [1, 0, 1],
                "pain_prediction": [0, 1, 0],
                "fever_prediction": [1, 0, 0],
                "nasal_prediction": [1, 1, 0],
            }
        )
        rebuilt = self.build_graph_dataset(original, oof)
        self.assertEqual(rebuilt["patient_id"].tolist(), [3, 1, 2])
        self.assertEqual(rebuilt["fever"].tolist(), [0, 0, 1])
        pd.testing.assert_frame_equal(
            rebuilt[["patient_id", "age_group"]],
            original[["patient_id", "age_group"]],
        )

    def test_name_alignment_precedes_metrics(self) -> None:
        reference = pd.DataFrame(
            [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        learned_canonical = pd.DataFrame(
            [[0, 1, 0], [0, 0, 0], [0, 1, 0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        learned_shuffled = learned_canonical.loc[["C", "A", "B"], ["C", "A", "B"]]
        aligned_reference, aligned_learned = self.align_adjacencies(
            reference, learned_shuffled
        )
        self.assertEqual(list(aligned_reference.index), ["A", "B", "C"])
        self.assertEqual(list(aligned_learned.index), ["A", "B", "C"])
        metrics = self.evaluate_graph(reference, learned_shuffled)
        self.assertEqual(metrics["SHD"], 2)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["F1"], 0.5)
        self.assertAlmostEqual(metrics["correlation"], 0.25)

        missing_node = learned_shuffled.drop(index="C", columns="C")
        with self.assertRaises(AssertionError):
            self.align_adjacencies(reference, missing_node)

    def test_causal_learn_endpoint_conversion_is_explicit(self) -> None:
        # A -> B is [-1 at (A,B), +1 at (B,A)]; B -- C is [-1, -1].
        endpoints = pd.DataFrame(
            [[0, -1, 0], [1, 0, -1], [0, -1, 0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        adjacency, edges = self.endpoint_matrix_to_adjacency(endpoints)
        self.assertEqual(int(adjacency.loc["A", "B"]), 1)
        self.assertEqual(int(adjacency.loc["B", "A"]), 0)
        self.assertEqual(int(adjacency.loc["B", "C"]), 1)
        self.assertEqual(int(adjacency.loc["C", "B"]), 1)
        self.assertEqual(edges["edge_type"].tolist(), ["->", "--"])

    def test_pc_invocation_passes_fixed_gsquare_settings(self) -> None:
        captured = {}

        class FakeNode:
            def __init__(self, name):
                self.name = name

            def get_name(self):
                return self.name

        class FakeGraph:
            graph = np.array([[0, -1], [1, 0]], dtype=np.int64)

            @staticmethod
            def get_nodes():
                return [FakeNode("A"), FakeNode("B")]

        def fake_pc(data, **kwargs):
            captured["data"] = data
            captured.update(kwargs)
            return SimpleNamespace(G=FakeGraph())

        modules = {
            name: ModuleType(name)
            for name in [
                "causallearn",
                "causallearn.search",
                "causallearn.search.ConstraintBased",
                "causallearn.search.ConstraintBased.PC",
                "causallearn.utils",
                "causallearn.utils.cit",
            ]
        }
        modules["causallearn.search.ConstraintBased.PC"].pc = fake_pc
        modules["causallearn.utils.cit"].gsq = "gsq"
        dataset = pd.DataFrame({"A": [0, 0, 1, 1], "B": [0, 1, 0, 1]})
        with patch.dict(sys.modules, modules):
            adjacency, _, _ = self.run_pc_learn(dataset, ["A", "B"])

        self.assertEqual(captured["alpha"], 0.05)
        self.assertEqual(captured["indep_test"], "gsq")
        self.assertTrue(captured["stable"])
        self.assertEqual(captured["uc_rule"], 0)
        self.assertEqual(captured["uc_priority"], 2)
        self.assertIsNone(captured["max_k"])
        self.assertEqual(captured["node_names"], ["A", "B"])
        self.assertEqual(int(adjacency.loc["A", "B"]), 1)

    def test_reference_adjacency_requires_named_square_matrix(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reference.csv"
            pd.DataFrame(
                {"node": ["B", "A"], "A": [0, 0], "B": [0, 1]}
            ).to_csv(path, index=False)
            loaded = self.load_reference_adjacency(path)
        self.assertEqual(list(loaded.index), ["B", "A"])
        self.assertEqual(list(loaded.columns), ["B", "A"])
        self.assertEqual(int(loaded.loc["A", "B"]), 1)


class OOFGraphStaticTests(unittest.TestCase):
    def test_pc_call_and_required_artifacts_are_present(self) -> None:
        source = Path(__file__).with_name("run_oof_graph_experiment.py").read_text(
            encoding="utf-8"
        )
        required = [
            'conditional_independence_test: str = "gsq"',
            "alpha: float = 0.05",
            "indep_test=gsq",
            "alpha=config.alpha",
            '"outer_fold_assignments.csv"',
            '"classifier_fold_metrics.csv"',
            '"classifier_oof_metrics.csv"',
            '"classifier_per_symptom_metrics.csv"',
            '"graph_metrics.csv"',
            "build_graph_dataset(",
            "align_adjacencies(",
            "validate_oof_integrity(",
        ]
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, source)


if __name__ == "__main__":
    unittest.main()
