#!/usr/bin/env python3
"""Build the deterministic sentence-to-patient mapping for the G-FS workflow.

This module only normalizes natural sentences and records their patient-level
and global positions. It does not create embeddings, attach labels, or pool
sentence representations.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd


MAPPING_COLUMNS: Final[list[str]] = [
    "patient_id",
    "sentence_idx",
    "global_sentence_idx",
    "sentence",
]
SUMMARY_COLUMNS: Final[list[str]] = ["patient_id", "n_sentences"]


def get_sentence(document: str) -> list[str]:
    """
    Args:
        document (str): document to convert into normalized sentences

    Returns:
        sentence (list): list of normalized sentences in the document
    """

    if not document:
        return []

    # Remove section headers while retaining the section contents.
    lines = re.split(r'\n|\s*\*\*History\*\*\s*', document)

    cleaned_lines = [
        line
        for line in lines
        if line.strip()
        and not line.startswith("**History**")
        and not line.startswith("**Physical Examination**")
    ]
    document = " ".join(cleaned_lines)

    # Split into natural sentences.
    sentences = document.split(". ")

    cleaned_sentence = []

    for sentence in sentences:
        normalized = sentence.lower().strip()

        # Remove punctuation.
        normalized = re.sub(r"[^\w\s]", "", normalized)

        # Remove duplicated whitespace.
        normalized = re.sub(r"\s+", " ", normalized).strip()

        if normalized:
            cleaned_sentence.append(normalized)

    return cleaned_sentence


def load_source_data(input_path: str | Path) -> pd.DataFrame:
    """Load a semicolon-delimited SynSUM source file without reordering rows."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    try:
        return pd.read_csv(path, sep=";")
    except Exception as exc:
        raise ValueError(f"Could not read semicolon-delimited input file: {path}") from exc


def _coerce_integer_ids(values: pd.Series, column_name: str) -> pd.Series:
    """Convert patient IDs to int64 without truncating invalid values."""
    if values.isna().any():
        rows = values.index[values.isna()].tolist()[:10]
        raise ValueError(
            f"ID column '{column_name}' contains missing patient IDs at row(s) {rows}."
        )

    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"ID column '{column_name}' contains values that cannot be converted "
            "to integers."
        ) from exc

    numeric_array = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_array).all():
        raise ValueError(f"ID column '{column_name}' contains a non-finite patient ID.")
    if not np.equal(numeric_array, np.floor(numeric_array)).all():
        invalid = values.loc[
            ~pd.Series(
                np.equal(numeric_array, np.floor(numeric_array)), index=values.index
            )
        ].head(10).tolist()
        raise ValueError(
            f"ID column '{column_name}' contains non-integer patient IDs, "
            f"for example: {invalid}."
        )

    int64_info = np.iinfo(np.int64)
    if ((numeric_array < int64_info.min) | (numeric_array > int64_info.max)).any():
        raise ValueError(f"ID column '{column_name}' contains an ID outside int64 range.")

    return pd.Series(
        numeric_array.astype(np.int64), index=values.index, name="patient_id"
    )


def validate_source_data(
    source: pd.DataFrame, id_column: str, text_column: str
) -> pd.DataFrame:
    """Validate required source fields and return a patient-ID-normalized copy."""
    missing_columns = [
        column for column in (id_column, text_column) if column not in source.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Input data is missing required column(s): {missing_columns}. "
            f"Available columns: {list(source.columns)}"
        )
    if id_column != "patient_id" and "patient_id" in source.columns:
        raise ValueError(
            "Input data already contains 'patient_id'; renaming the configured ID "
            f"column '{id_column}' would create an ambiguous duplicate."
        )

    validated = source.copy()
    validated = validated.rename(columns={id_column: "patient_id"})
    validated["patient_id"] = _coerce_integer_ids(
        validated["patient_id"], id_column
    )

    if validated["patient_id"].duplicated().any():
        duplicates = (
            validated.loc[
                validated["patient_id"].duplicated(keep=False), "patient_id"
            ]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(f"Input data contains duplicate patient IDs: {duplicates}")

    return validated


def normalize_document(document: object) -> list[str]:
    """Safely normalize one possibly missing or non-string document."""
    if document is None:
        return []

    try:
        is_missing = bool(pd.isna(document))
    except (TypeError, ValueError):
        is_missing = False
    if is_missing:
        return []

    return get_sentence(str(document))


def build_sentence_mapping(
    source: pd.DataFrame, text_column: str
) -> pd.DataFrame:
    """Create one ordered mapping row per normalized natural sentence."""
    records: list[dict[str, int | str]] = []
    global_sentence_idx = 0

    for patient_id, document in source.loc[
        :, ["patient_id", text_column]
    ].itertuples(index=False, name=None):
        for sentence_idx, sentence in enumerate(normalize_document(document)):
            records.append(
                {
                    "patient_id": int(patient_id),
                    "sentence_idx": sentence_idx,
                    "global_sentence_idx": global_sentence_idx,
                    "sentence": sentence,
                }
            )
            global_sentence_idx += 1

    mapping = pd.DataFrame.from_records(records, columns=MAPPING_COLUMNS)
    for column in ("patient_id", "sentence_idx", "global_sentence_idx"):
        mapping[column] = mapping[column].astype("int64")
    return mapping


def build_patient_summary(
    source: pd.DataFrame, mapping: pd.DataFrame
) -> pd.DataFrame:
    """Count mapped sentences for every input patient, including zero counts."""
    counts = mapping.groupby("patient_id", sort=False).size()
    summary = pd.DataFrame(
        {"patient_id": source["patient_id"].to_numpy(dtype=np.int64, copy=True)}
    )
    summary["n_sentences"] = (
        summary["patient_id"].map(counts).fillna(0).astype("int64")
    )
    return summary.loc[:, SUMMARY_COLUMNS]


def validate_outputs(
    source: pd.DataFrame, mapping: pd.DataFrame, summary: pd.DataFrame
) -> None:
    """Enforce the mapping, ordering, index, and patient-summary contracts."""
    if list(mapping.columns) != MAPPING_COLUMNS:
        raise ValueError(
            f"Mapping columns must be exactly {MAPPING_COLUMNS}; "
            f"found {list(mapping.columns)}."
        )
    if list(summary.columns) != SUMMARY_COLUMNS:
        raise ValueError(
            f"Summary columns must be exactly {SUMMARY_COLUMNS}; "
            f"found {list(summary.columns)}."
        )

    integer_columns = [
        (mapping, "patient_id"),
        (mapping, "sentence_idx"),
        (mapping, "global_sentence_idx"),
        (summary, "patient_id"),
        (summary, "n_sentences"),
    ]
    for frame, column in integer_columns:
        if not pd.api.types.is_integer_dtype(frame[column].dtype):
            raise ValueError(f"Output column '{column}' must use an integer dtype.")

    if mapping["sentence"].isna().any():
        raise ValueError("Mapping contains a missing sentence.")
    sentence_text = mapping["sentence"].astype(str)
    if sentence_text.str.strip().eq("").any():
        raise ValueError("Mapping contains an empty sentence.")
    if not sentence_text.eq(sentence_text.str.strip()).all():
        raise ValueError("Mapping contains leading or trailing sentence whitespace.")
    if sentence_text.str.contains(r"\s{2,}", regex=True).any():
        raise ValueError("Mapping contains repeated sentence whitespace.")
    if not sentence_text.eq(sentence_text.str.lower()).all():
        raise ValueError("Mapping contains a sentence that is not lowercase.")

    expected_local_indices = mapping.groupby(
        "patient_id", sort=False
    ).cumcount().to_numpy(dtype=np.int64)
    actual_local_indices = mapping["sentence_idx"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_local_indices, expected_local_indices):
        raise ValueError(
            "Sentence indices must start at zero and remain contiguous for each patient."
        )

    expected_global_indices = np.arange(len(mapping), dtype=np.int64)
    actual_global_indices = mapping["global_sentence_idx"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_global_indices, expected_global_indices):
        raise ValueError(
            "Global sentence indices must be unique, zero-based, and contiguous."
        )

    if mapping.duplicated(subset=["patient_id", "sentence_idx"]).any():
        raise ValueError("Mapping contains duplicate patient_id + sentence_idx rows.")

    source_ids = source["patient_id"].to_numpy(dtype=np.int64)
    summary_ids = summary["patient_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(summary_ids, source_ids):
        raise ValueError(
            "Patient summary must contain every input patient in original row order."
        )
    if summary["patient_id"].duplicated().any():
        raise ValueError("Patient summary contains duplicate patient IDs.")
    if (summary["n_sentences"] < 0).any():
        raise ValueError("Patient summary contains a negative sentence count.")

    source_id_set = set(source_ids.tolist())
    if not set(mapping["patient_id"].tolist()).issubset(source_id_set):
        raise ValueError("Mapping contains a patient ID absent from the input data.")

    expected_mapping_ids = np.repeat(
        summary_ids, summary["n_sentences"].to_numpy(dtype=np.int64)
    )
    actual_mapping_ids = mapping["patient_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_mapping_ids, expected_mapping_ids):
        raise ValueError("Mapping does not preserve the original patient row order.")

    actual_counts = (
        mapping.groupby("patient_id", sort=False)
        .size()
        .reindex(summary["patient_id"], fill_value=0)
        .to_numpy(dtype=np.int64)
    )
    summary_counts = summary["n_sentences"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_counts, summary_counts):
        raise ValueError("Patient summary counts do not match the mapping rows.")
    if len(mapping) != int(summary["n_sentences"].sum()):
        raise ValueError(
            "Mapping row count does not equal the sum of patient sentence counts."
        )


def save_outputs(
    mapping: pd.DataFrame, summary: pd.DataFrame, output_dir: str | Path
) -> tuple[Path, Path, Path]:
    """Save the mapping as CSV and Parquet plus the patient-level CSV summary."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mapping_csv = output_path / "gfs_sentence_mapping.csv"
    mapping_parquet = output_path / "gfs_sentence_mapping.parquet"
    counts_csv = output_path / "gfs_sentence_counts.csv"

    mapping.to_csv(mapping_csv, index=False)
    try:
        mapping.to_parquet(mapping_parquet, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Writing the required Parquet output needs 'pyarrow' or "
            "'fastparquet'. Install the project requirements and rerun."
        ) from exc
    summary.to_csv(counts_csv, index=False)
    return mapping_csv, mapping_parquet, counts_csv


def print_execution_summary(
    input_path: Path,
    text_column: str,
    source: pd.DataFrame,
    mapping: pd.DataFrame,
    summary: pd.DataFrame,
    output_paths: tuple[Path, Path, Path],
) -> None:
    """Print the requested deterministic execution statistics and preview."""
    counts = summary["n_sentences"]
    nonzero_patients = int(counts.gt(0).sum())
    zero_patients = int(counts.eq(0).sum())

    print(f"Input file path: {input_path}")
    print(f"Input dataframe shape: {source.shape}")
    print(f"Selected text column: {text_column}")
    print(f"Total number of patients: {len(summary)}")
    print(f"Patients with at least one sentence: {nonzero_patients}")
    print(f"Patients with zero valid sentences: {zero_patients}")
    print(f"Total number of generated sentences: {len(mapping)}")
    if len(summary):
        print(f"Minimum sentences per patient: {int(counts.min())}")
        print(f"Maximum sentences per patient: {int(counts.max())}")
        print(f"Mean sentences per patient: {counts.mean():.4f}")
        print(f"Median sentences per patient: {counts.median():.4f}")
    else:
        print("Minimum sentences per patient: n/a")
        print("Maximum sentences per patient: n/a")
        print("Mean sentences per patient: n/a")
        print("Median sentences per patient: n/a")
    print(f"Output dataframe shape: {mapping.shape}")
    print(f"Mapping CSV: {output_paths[0]}")
    print(f"Mapping Parquet: {output_paths[1]}")
    print(f"Patient counts CSV: {output_paths[2]}")
    print("First five mapping rows:")
    print(mapping.head(5).to_string(index=False))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the G-FS mapping build."""
    parser = argparse.ArgumentParser(
        description="Build a deterministic sentence-level patient mapping for G-FS."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("SynSUM.csv"),
        help="Semicolon-delimited SynSUM input path (default: SynSUM.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for mapping outputs (default: outputs).",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Clinical narrative column to process (default: text).",
    )
    parser.add_argument(
        "--id-column",
        default="Unnamed: 0",
        help="Source patient ID column (default: Unnamed: 0).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the complete sentence mapping pipeline from the command line."""
    args = parse_args(argv)
    source = load_source_data(args.input_path)
    validated_source = validate_source_data(
        source, id_column=args.id_column, text_column=args.text_column
    )
    mapping = build_sentence_mapping(validated_source, args.text_column)
    summary = build_patient_summary(validated_source, mapping)
    validate_outputs(validated_source, mapping, summary)
    output_paths = save_outputs(mapping, summary, args.output_dir)
    print_execution_summary(
        input_path=args.input_path,
        text_column=args.text_column,
        source=source,
        mapping=mapping,
        summary=summary,
        output_paths=output_paths,
    )


if __name__ == "__main__":
    main()
