"""Tests for OOF leakage guards, reconstruction, and graph evaluation."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
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
            bayesian_network_to_adjacency,
            build_graph_dataset,
            evaluate_graph,
            load_reference_adjacency,
            dag_to_discrete_bayesian_network,
            run_pc_learn,
            validate_nested_subsets,
            validate_oof_integrity,
            validate_outer_folds,
        )

        cls.OuterFold = OuterFold
        cls.PCConfig = PCConfig
        cls.align_adjacencies = staticmethod(align_adjacencies)
        cls.bayesian_network_to_adjacency = staticmethod(bayesian_network_to_adjacency)
        cls.build_graph_dataset = staticmethod(build_graph_dataset)
        cls.evaluate_graph = staticmethod(evaluate_graph)
        cls.load_reference_adjacency = staticmethod(load_reference_adjacency)
        cls.dag_to_discrete_bayesian_network = staticmethod(
            dag_to_discrete_bayesian_network
        )
        cls.run_pc_learn = staticmethod(run_pc_learn)
        cls.validate_nested_subsets = staticmethod(validate_nested_subsets)
        cls.validate_oof_integrity = staticmethod(validate_oof_integrity)
        cls.validate_outer_folds = staticmethod(validate_outer_folds)

    def test_fixed_pc_configuration(self) -> None:
        config = self.PCConfig()
        self.assertEqual(config.algorithm, "PC-learn")
        self.assertEqual(config.implementation, "pgmpy")
        self.assertEqual(config.model_class, "DiscreteBayesianNetwork")
        self.assertEqual(config.conditional_independence_test, "g_sq")
        self.assertEqual(config.alpha, 0.05)
        self.assertTrue(config.stable)
        self.assertEqual(config.return_type, "dag")
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
        self.assertEqual(metrics["SHD"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["F1"], 0.5)
        self.assertAlmostEqual(metrics["correlation"], 0.25)

        missing_node = learned_shuffled.drop(index="C", columns="C")
        with self.assertRaises(AssertionError):
            self.align_adjacencies(reference, missing_node)

    def test_pgmpy_bayesian_network_conversion_is_explicit(self) -> None:
        class FakeBayesianNetwork:
            @staticmethod
            def nodes():
                return ["A", "B", "C"]

            @staticmethod
            def edges():
                return [("A", "B"), ("B", "C")]

        adjacency, edges = self.bayesian_network_to_adjacency(
            FakeBayesianNetwork(), ["A", "B", "C"]
        )
        self.assertEqual(int(adjacency.loc["A", "B"]), 1)
        self.assertEqual(int(adjacency.loc["B", "A"]), 0)
        self.assertEqual(int(adjacency.loc["B", "C"]), 1)
        self.assertEqual(int(adjacency.loc["C", "B"]), 0)
        self.assertEqual(edges["edge_type"].tolist(), ["->", "->"])

    def test_pc_invocation_passes_fixed_gsquare_settings(self) -> None:
        captured = {}

        class FakeDAG:
            @staticmethod
            def nodes():
                return ["A", "B"]

            @staticmethod
            def edges():
                return [("A", "B")]

        class FakePC:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.causal_graph_ = FakeDAG()

            def fit(self, data):
                captured["data"] = data
                return self

        class FakeDiscreteBayesianNetwork:
            def __init__(self):
                self._nodes = []
                self._edges = []
                self.graph = {}

            def add_nodes_from(self, nodes):
                self._nodes.extend(nodes)

            def add_edges_from(self, edges):
                self._edges.extend(edges)

            def nodes(self):
                return self._nodes

            def edges(self):
                return self._edges

        modules = {
            name: ModuleType(name)
            for name in [
                "pgmpy",
                "pgmpy.causal_discovery",
                "pgmpy.models",
            ]
        }
        modules["pgmpy.causal_discovery"].PC = FakePC
        modules["pgmpy.models"].DiscreteBayesianNetwork = FakeDiscreteBayesianNetwork
        dataset = pd.DataFrame({"A": [0, 0, 1, 1], "B": [0, 1, 0, 1]})
        with patch.dict(sys.modules, modules):
            adjacency, _, model = self.run_pc_learn(dataset, ["A", "B"])

        self.assertEqual(captured["significance_level"], 0.05)
        self.assertEqual(captured["ci_test"], "g_sq")
        self.assertEqual(captured["variant"], "stable")
        self.assertEqual(captured["return_type"], "dag")
        self.assertEqual(captured["max_cond_vars"], 0)
        self.assertEqual(captured["data"].columns.tolist(), ["A", "B"])
        self.assertIsInstance(model, FakeDiscreteBayesianNetwork)
        self.assertEqual(int(adjacency.loc["A", "B"]), 1)

    def test_cyclic_pc_dag_is_projected_to_a_true_dag(self) -> None:
        class CyclicDAG:
            @staticmethod
            def nodes():
                return ["A", "B", "C", "D"]

            @staticmethod
            def edges():
                return [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")]

        class StrictFakeBayesianNetwork:
            def __init__(self):
                self._nodes = []
                self._edges = []
                self.graph = {}

            def add_nodes_from(self, nodes):
                self._nodes.extend(nodes)

            def add_edges_from(self, edges):
                self._edges.extend(edges)

            def nodes(self):
                return self._nodes

            def edges(self):
                return self._edges

        model = self.dag_to_discrete_bayesian_network(
            CyclicDAG(),
            ["A", "B", "C", "D"],
            StrictFakeBayesianNetwork,
        )
        completion = model.graph["pc_dag_validation"]
        rank = {name: index for index, name in enumerate(completion["node_order"])}
        self.assertTrue(all(rank[source] < rank[target] for source, target in model.edges()))
        self.assertEqual(len(model.edges()), 4)
        self.assertGreaterEqual(completion["cycle_breaks"], 1)
        self.assertGreaterEqual(completion["reversed_directed_edges"], 1)

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
    def test_torch_is_imported_at_module_scope(self) -> None:
        source = Path(__file__).with_name("run_oof_graph_experiment.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        imported_names = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertIn("torch", imported_names)

    def test_pc_call_and_required_artifacts_are_present(self) -> None:
        source = Path(__file__).with_name("run_oof_graph_experiment.py").read_text(
            encoding="utf-8"
        )
        required = [
            'implementation: str = "pgmpy"',
            'model_class: str = "DiscreteBayesianNetwork"',
            'conditional_independence_test: str = "g_sq"',
            "alpha: float = 0.05",
            'variant="stable" if config.stable else "orig"',
            "ci_test=config.conditional_independence_test",
            "significance_level=config.alpha",
            "dag_to_discrete_bayesian_network(",
            "return_type=config.return_type",
            '"--graph-only"',
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
