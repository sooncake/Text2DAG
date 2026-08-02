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
            "LEGACY_REPRODUCTION_MODE = True",
            "LEGACY_LEARNING_CURVE_MODE = True",
            "LEGACY_TRAINING_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]",
            'LEGACY_ROOT = DRIVE_ROOT / "legacy_reproduction"',
            'legacy_script = PROJECT_DIR / "legacy_reproduction.py"',
            '"legacy_reproduction_metadata.json"',
            '"--learning-curve"',
            '"--training-fractions"',
            "legacy_variablewise_f1.csv",
            "legacy_macro_f1.csv",
            'CURRENT_FRAMEWORK_HEAD = "modern_mlp"',
            '"legacy_two_layer_linear"',
            "YOUR_USERNAME/YOUR_REPOSITORY",
            'drive.mount("/content/drive")',
            'MAPPING_PATH = DRIVE_ROOT / "inputs/gfs_sentence_mapping.csv"',
            'EMBEDDING_OUTPUT_DIR = DRIVE_ROOT / "generated_sentence_embeddings"',
            "FORCE_RECOMPUTE = False",
            '"--device", "cuda"',
            "patient_embeddings_and_labels.npz",
            "X.shape[1] != 768",
            "y_original.shape[1] != 5",
            "best_model.pt",
            "training_config.json",
            "random_seed.txt",
            "test_predictions.npz",
            "evaluation_metrics.json",
            '"test_fraction": 0.20',
            '"training_pool_fractions"',
            "0.05, 0.10, 0.20",
            "0.80, 0.90, 1.00",
            '"cv_folds": 5',
            '"experiment_seeds": [42, 43, 44, 45, 46]',
            '"hidden_dim": 128',
            "nn.Linear(input_dim, hidden_dim)",
            "nn.Linear(hidden_dim, len(label_names))",
            "MultilabelStratifiedKFold",
            "MultilabelStratifiedShuffleSplit",
            'y[:, 3] = (y_original[:, 3] > 0).astype(np.int64)',
            "learning_curve_summary.csv",
            "learning_curve_per_label.csv",
            "all_dataset_per_label_metrics.csv",
            "test_per_label_metrics.csv",
            "all_dataset_predictions_trainpool_",
            "learning_curve.png",
        ]
        for marker in required_markers:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.all_source)

    def test_legacy_mode_skips_modern_execution_cells(self) -> None:
        skip_markers = [
            "Modern array loading and fever binarization skipped",
            "Workflow B configuration skipped",
            "Workflow B split construction skipped",
            "Workflow B learning-curve training skipped",
            "Workflow B result aggregation skipped",
        ]
        for marker in skip_markers:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.all_source)

    def test_supervised_workflow_has_no_old_three_way_or_multiclass_fever_logic(self) -> None:
        forbidden_markers = [
            '"validation_fraction": 0.15',
            '"test_fraction": 0.15',
            "fever_head",
            "fever_probabilities",
            "CrossEntropyLoss",
            "hidden_dim // 2",
        ]
        for marker in forbidden_markers:
            with self.subTest(marker=marker):
                self.assertNotIn(marker, self.all_source)

    def test_preprocessing_uses_sentence_level_contract(self) -> None:
        forbidden_markers = [
            "data_row_idx",
            "original_sentence_idx",
            "subsentence_idx",
            "subsentence mapping",
            "subsentence traceability",
        ]
        for marker in forbidden_markers:
            with self.subTest(marker=marker):
                self.assertNotIn(marker, self.all_source.lower())


if __name__ == "__main__":
    unittest.main()
