"""Lightweight unit tests that do not download or run the embedding model."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from prepare_patient_embeddings import (
    EMBEDDING_DIM,
    LABEL_NAMES,
    MAPPING_COLUMNS,
    encode_labels,
    join_embeddings_and_labels,
    load_and_validate_mapping,
    mean_pool_by_patient,
    parse_args,
    save_outputs,
    validate_final_dataset,
)


class PatientEmbeddingPipelineTests(unittest.TestCase):
    def test_label_encoding_and_direct_patient_mean_pooling(self) -> None:
        mapping = pd.DataFrame(
            {
                "patient_id": [2, 1, 2],
                "sentence_idx": [0, 0, 1],
                "global_sentence_idx": [0, 1, 2],
                "sentence": ["a", "b", "c"],
            }
        )
        sentence_embeddings = np.stack(
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

        patient_ids, pooled = mean_pool_by_patient(mapping, sentence_embeddings)
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

    def test_sentence_mapping_loading_filters_blank_text_and_coerces_ids(self) -> None:
        mapping = pd.DataFrame(
            {
                "patient_id": ["2", "2", "1", "3"],
                "sentence_idx": [0, 1, 0, 0],
                "global_sentence_idx": [0, 1, 2, 3],
                "sentence": [" first sentence ", None, "   ", "second sentence"],
            }
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "gfs_sentence_mapping.csv"
            mapping.to_csv(path, index=False)
            cleaned, input_sentence_count = load_and_validate_mapping(path)

        self.assertEqual(input_sentence_count, 4)
        self.assertEqual(list(cleaned.columns), MAPPING_COLUMNS)
        self.assertEqual(cleaned["patient_id"].tolist(), [2, 3])
        self.assertEqual(cleaned["sentence"].tolist(), ["first sentence", "second sentence"])
        self.assertTrue(pd.api.types.is_integer_dtype(cleaned["patient_id"]))

    def test_default_mapping_path_is_sentence_mapping(self) -> None:
        self.assertEqual(parse_args([]).mapping_path, "gfs_sentence_mapping.csv")

    def test_old_mapping_columns_are_not_accepted(self) -> None:
        old_mapping = pd.DataFrame(
            {
                "subsentence": ["legacy text"],
                "original_sentence_idx": [0],
                "subsentence_idx": [0],
                "data_row_idx": [1],
            }
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "old_mapping.csv"
            old_mapping.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing required column"):
                load_and_validate_mapping(path)

    def test_missing_mapping_error_uses_sentence_terminology(self) -> None:
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.csv"
            with self.assertRaisesRegex(
                FileNotFoundError, "Sentence mapping file not found"
            ):
                load_and_validate_mapping(missing)

    def test_saved_traceability_output_uses_sentence_contract(self) -> None:
        mapping = pd.DataFrame(
            {
                "patient_id": [1, 2],
                "sentence_idx": [0, 0],
                "global_sentence_idx": [0, 1],
                "sentence": ["first", "second"],
            }
        )
        sentence_embeddings = np.stack(
            [
                np.full(EMBEDDING_DIM, 1.0, dtype=np.float32),
                np.full(EMBEDDING_DIM, 2.0, dtype=np.float32),
            ]
        )
        labels = encode_labels(
            pd.DataFrame(
                {
                    "patient_id": [1, 2],
                    "dysp": ["no", "yes"],
                    "cough": ["no", "yes"],
                    "pain": ["no", "yes"],
                    "fever": ["none", "high"],
                    "nasal": ["no", "yes"],
                }
            )
        )
        final = join_embeddings_and_labels(
            np.array([1, 2], dtype=np.int64), sentence_embeddings, labels
        )

        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_outputs(final, mapping, sentence_embeddings, output_dir)
            trace = pd.read_parquet(output_dir / "sentence_embeddings.parquet")
            with np.load(
                output_dir / "patient_embeddings_and_labels.npz", allow_pickle=False
            ) as saved:
                self.assertEqual(
                    set(saved.files), {"patient_ids", "X", "y", "label_names"}
                )

            self.assertFalse((output_dir / "subsentence_embeddings.parquet").exists())

        expected_embedding_columns = [
            f"emb_{index:03d}" for index in range(EMBEDDING_DIM)
        ]
        self.assertEqual(
            list(trace.columns), [*MAPPING_COLUMNS, *expected_embedding_columns]
        )

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
