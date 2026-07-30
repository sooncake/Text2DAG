"""Lightweight unit tests that do not download or run the embedding model."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from prepare_patient_embeddings import (
    EMBEDDING_DIM,
    LABEL_NAMES,
    encode_labels,
    join_embeddings_and_labels,
    mean_pool_by_patient,
    validate_final_dataset,
)


class PatientEmbeddingPipelineTests(unittest.TestCase):
    def test_label_encoding_and_direct_patient_mean_pooling(self) -> None:
        mapping = pd.DataFrame(
            {
                "subsentence": ["a", "b", "c"],
                "original_sentence_idx": [10, 11, 10],
                "subsentence_idx": [0, 0, 1],
                "data_row_idx": [2, 1, 2],
            }
        )
        subsentence_embeddings = np.stack(
            [
                np.full(EMBEDDING_DIM, 1.0, dtype=np.float32),
                np.full(EMBEDDING_DIM, 4.0, dtype=np.float32),
                np.full(EMBEDDING_DIM, 3.0, dtype=np.float32),
            ]
        )
        labels = pd.DataFrame(
            {
                "patient_id": [1, 2],
                "dysp": [" no ", "YES"],
                "cough": ["yes", "no"],
                "pain": ["no", "yes"],
                "fever": ["none", "HIGH"],
                "nasal": ["no", "yes"],
            }
        )

        patient_ids, pooled = mean_pool_by_patient(mapping, subsentence_embeddings)
        self.assertEqual(patient_ids.tolist(), [1, 2])
        np.testing.assert_allclose(pooled[0], 4.0)
        np.testing.assert_allclose(pooled[1], 2.0)
        self.assertEqual(pooled.dtype, np.float32)

        final = join_embeddings_and_labels(
            patient_ids, pooled, encode_labels(labels)
        )
        validate_final_dataset(final)
        self.assertEqual(final[LABEL_NAMES].to_numpy().dtype, np.int64)
        self.assertEqual(final["fever"].tolist(), [0, 2])

    def test_unknown_label_is_rejected(self) -> None:
        labels = pd.DataFrame(
            {
                "patient_id": [1],
                "dysp": ["maybe"],
                "cough": ["no"],
                "pain": ["no"],
                "fever": ["none"],
                "nasal": ["no"],
            }
        )
        with self.assertRaisesRegex(ValueError, "unexpected value"):
            encode_labels(labels)


if __name__ == "__main__":
    unittest.main()
