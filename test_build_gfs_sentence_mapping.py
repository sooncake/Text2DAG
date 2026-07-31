"""Lightweight tests for deterministic G-FS sentence mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd

from build_gfs_sentence_mapping import (
    MAPPING_COLUMNS,
    build_patient_summary,
    build_sentence_mapping,
    get_sentence,
    normalize_document,
    validate_outputs,
    validate_source_data,
)


def test_header_removal_retains_section_contents() -> None:
    document = (
        "**History**\nHistory content. Still history.\n\n"
        "**Physical Examination**\nExam content."
    )
    assert get_sentence(document) == [
        "history content",
        "still history",
        "exam content",
    ]


def test_sentence_splitting_uses_dot_space_only() -> None:
    assert get_sentence("One. Two.Three. Four") == [
        "one",
        "twothree",
        "four",
    ]


def test_lowercasing() -> None:
    assert get_sentence("Hello WORLD") == ["hello world"]


def test_punctuation_removal() -> None:
    assert get_sentence("Hello, world! (Again?)") == ["hello world again"]


def test_whitespace_normalization() -> None:
    assert get_sentence("  First\t  sentence.   Second sentence.  ") == [
        "first sentence",
        "second sentence",
    ]


def test_empty_document_handling() -> None:
    assert get_sentence("") == []


def test_none_and_nan_document_handling() -> None:
    assert normalize_document(None) == []
    assert normalize_document(np.nan) == []
    assert normalize_document(pd.NA) == []


def test_non_string_document_is_safely_converted() -> None:
    assert normalize_document(12345) == ["12345"]


def test_sentence_order_patient_mapping_and_indices_are_preserved() -> None:
    source = pd.DataFrame(
        {
            "Unnamed: 0": [42, 7],
            "text": ["First. Second.", "Third. Fourth."],
        }
    )
    source = validate_source_data(source, "Unnamed: 0", "text")
    mapping = build_sentence_mapping(source, "text")

    assert list(mapping.columns) == MAPPING_COLUMNS
    assert mapping.to_dict("records") == [
        {
            "patient_id": 42,
            "sentence_idx": 0,
            "global_sentence_idx": 0,
            "sentence": "first",
        },
        {
            "patient_id": 42,
            "sentence_idx": 1,
            "global_sentence_idx": 1,
            "sentence": "second",
        },
        {
            "patient_id": 7,
            "sentence_idx": 0,
            "global_sentence_idx": 2,
            "sentence": "third",
        },
        {
            "patient_id": 7,
            "sentence_idx": 1,
            "global_sentence_idx": 3,
            "sentence": "fourth",
        },
    ]


def test_repeated_sentence_text_is_not_deduplicated() -> None:
    source = pd.DataFrame({"patient_id": [5], "text": ["Same. Same."]})
    mapping = build_sentence_mapping(source, "text")

    assert mapping["sentence"].tolist() == ["same", "same"]
    assert mapping["sentence_idx"].tolist() == [0, 1]


def test_summary_includes_patients_with_zero_sentences() -> None:
    source = pd.DataFrame(
        {"patient_id": pd.Series([10, 11], dtype="int64"), "text": ["Text.", None]}
    )
    mapping = build_sentence_mapping(source, "text")
    summary = build_patient_summary(source, mapping)
    validate_outputs(source, mapping, summary)

    assert summary.to_dict("records") == [
        {"patient_id": 10, "n_sentences": 1},
        {"patient_id": 11, "n_sentences": 0},
    ]


def test_empty_mapping_preserves_required_columns_and_integer_dtypes() -> None:
    source = pd.DataFrame(
        {"patient_id": pd.Series([1], dtype="int64"), "text": [None]}
    )
    mapping = build_sentence_mapping(source, "text")
    summary = build_patient_summary(source, mapping)
    validate_outputs(source, mapping, summary)

    assert mapping.empty
    assert list(mapping.columns) == MAPPING_COLUMNS
    assert all(
        pd.api.types.is_integer_dtype(mapping[column])
        for column in ["patient_id", "sentence_idx", "global_sentence_idx"]
    )
