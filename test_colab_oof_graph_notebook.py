"""Static validation for the complete Colab OOF-to-graph notebook."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).with_name("colab_oof_graph_experiment.ipynb")


class ColabOOFGraphNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.code_sources = [
            "".join(cell["source"])
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        cls.all_source = "\n".join(
            "".join(cell["source"]) for cell in cls.notebook["cells"]
        )
        cls.code_source = "\n".join(cls.code_sources)

    def test_notebook_is_valid_gpu_notebook_and_code_compiles(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(self.notebook["metadata"]["accelerator"], "GPU")
        for index, source in enumerate(self.code_sources, start=1):
            compile(source, f"colab-oof-code-cell-{index}", "exec")

    def test_complete_colab_workflow_is_present(self) -> None:
        required = [
            "drive.mount('/content/drive')",
            "REPOSITORY_BRANCH = 'main'",
            "build_gfs_sentence_mapping.py",
            "prepare_patient_embeddings.py",
            "run_oof_graph_experiment.py",
            "SUPERVISION_FRACTIONS == (0.05, 0.10, 0.20, 1.00)",
            "pc_config.implementation == 'pgmpy'",
            "pc_config.model_class == 'DiscreteBayesianNetwork'",
            "pc_config.conditional_independence_test == 'g_sq'",
            "pc_config.alpha == 0.05",
            "pc_config.pc_return_type == 'pdag'",
            "classifier_oof_metrics.csv",
            "classifier_per_symptom_metrics.csv",
            "classifier_fold_metrics.csv",
            "graph_metrics.csv",
            "pc_learn_config.json",
            "outer_fold_assignments.csv",
            "oof_predictions_{suffix}.csv",
            "graph_edges_{suffix}.csv",
            "graph_adjacency_{suffix}.csv",
            "[2000] * 5",
        ]
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.all_source)

    def test_notebook_never_reads_in_sample_prediction_artifacts(self) -> None:
        self.assertNotIn("all_dataset_predictions", self.code_source)


if __name__ == "__main__":
    unittest.main()
