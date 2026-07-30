"""Static, dependency-free validation for the Google Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


NOTEBOOK_PATH = Path(__file__).with_name("colab_embedding_and_training.ipynb")


class ColabNotebookTests(unittest.TestCase):
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

    def test_notebook_is_v4_and_all_code_cells_compile(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(self.notebook["metadata"]["accelerator"], "GPU")
        for index, source in enumerate(self.code_sources, start=1):
            compile(source, f"colab-code-cell-{index}", "exec")

    def test_required_colab_workflow_is_present(self) -> None:
        required_markers = [
            "YOUR_USERNAME/YOUR_REPOSITORY",
            'drive.mount("/content/drive")',
            "FORCE_RECOMPUTE = False",
            '"--device", "cuda"',
            "patient_embeddings_and_labels.npz",
            "X.shape[1] != 768",
            "y.shape[1] != 5",
            "best_model.pt",
            "training_config.json",
            "random_seed.txt",
            "test_predictions.npz",
            "evaluation_metrics.json",
        ]
        for marker in required_markers:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.all_source)


if __name__ == "__main__":
    unittest.main()
