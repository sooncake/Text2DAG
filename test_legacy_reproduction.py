"""Dependency-light invariants for the isolated legacy reproduction path."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # The repository requirements install torch in Colab.
    torch = None

if torch is not None:
    from legacy_reproduction import (
        EMBEDDING_DIM,
        LABEL_NAMES,
        LEGACY_CONFIG,
        LegacyHeadOnlyModel,
        corrected_masked_sentence_mean,
        legacy_collate_fn,
        legacy_get_sentence,
    )


@unittest.skipUnless(torch is not None, "torch is not installed in this local runtime")
class LegacyReproductionTests(unittest.TestCase):
    def test_exact_legacy_head_has_two_linears_and_parameter_count(self) -> None:
        model = LegacyHeadOnlyModel(EMBEDDING_DIM)
        self.assertEqual(list(model._modules), ["projector", "classifier"])
        self.assertEqual(model.projector.in_features, 768)
        self.assertEqual(model.projector.out_features, 256)
        self.assertEqual(model.classifier.in_features, 256)
        self.assertEqual(model.classifier.out_features, 5)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 198_149)
        self.assertEqual(tuple(model(torch.zeros(2, 768)).shape), (2, 5))

    def test_legacy_configuration_keeps_executed_values(self) -> None:
        self.assertEqual(LABEL_NAMES, ["dysp", "cough", "pain", "fever", "nasal"])
        self.assertEqual(LEGACY_CONFIG["seed"], 5)
        self.assertEqual(LEGACY_CONFIG["head_dim"], 256)
        self.assertEqual(LEGACY_CONFIG["batch_size"], 32)
        self.assertEqual(LEGACY_CONFIG["learning_rate"], 3e-5)
        self.assertEqual(LEGACY_CONFIG["maximum_epochs"], 120)
        self.assertEqual(LEGACY_CONFIG["early_stopping_patience"], 5)
        self.assertEqual(LEGACY_CONFIG["threshold_operator"], ">")

    def test_legacy_cleaner_preserves_literal_sentence_rules(self) -> None:
        cleaned = legacy_get_sentence(
            ["**History** Fever, HIGH.  More   text!\n**Physical Examination**"]
        )
        self.assertEqual(cleaned, [["fever high", "more text"]])

    def test_collate_uses_zero_sentence_slots(self) -> None:
        short = {
            "input_ids": torch.tensor([[10, 11, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
            "labels": torch.zeros(5),
            "patient_id": 1,
            "source_patient_id": 101,
        }
        long = {
            "input_ids": torch.tensor([[20, 21, 0], [30, 31, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 0]]),
            "labels": torch.ones(5),
            "patient_id": 2,
            "source_patient_id": 102,
        }
        batch = legacy_collate_fn([short, long])
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 2, 3))
        self.assertTrue(torch.equal(batch["input_ids"][0, 1], torch.zeros(3, dtype=torch.long)))
        self.assertTrue(torch.equal(batch["attention_mask"][0, 1], torch.zeros(3, dtype=torch.long)))

    def test_corrected_pooling_is_separate_from_legacy_unmasked_mean(self) -> None:
        sentence_embeddings = torch.tensor([[[2.0], [10.0]]])
        sentence_mask = torch.tensor([[1, 0]])
        legacy_mean = sentence_embeddings.mean(dim=1)
        corrected_mean = corrected_masked_sentence_mean(sentence_embeddings, sentence_mask)
        self.assertEqual(float(legacy_mean.item()), 6.0)
        self.assertEqual(float(corrected_mean.item()), 2.0)


class LegacyReproductionStaticTests(unittest.TestCase):
    def test_script_contains_required_reproduction_invariants(self) -> None:
        source = Path(__file__).with_name("legacy_reproduction.py").read_text(
            encoding="utf-8"
        )
        required = [
            'LABEL_NAMES: Final[list[str]] = ["dysp", "cough", "pain", "fever", "nasal"]',
            '"seed": 5',
            '"head_dim": 256',
            '"learning_rate": 3e-5',
            '"maximum_epochs": 120',
            '"early_stopping_patience": 5',
            "class LegacyHeadOnlyModel(nn.Module):",
            "self.projector = nn.Linear(input_dim, head_dim)",
            "self.classifier = nn.Linear(head_dim, num_labels)",
            "return sentence_embeddings.mean(dim=1)",
            "criterion = nn.BCEWithLogitsLoss()",
            "probabilities > float(LEGACY_CONFIG",
        ]
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, source)


if __name__ == "__main__":
    unittest.main()
