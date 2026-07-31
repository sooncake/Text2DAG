#!/usr/bin/env python3
"""Create patient-level mean text embeddings with symptom labels.

Each valid sentence is embedded independently with ModernBERT. The resulting
sentence vectors are then averaged directly by ``patient_id`` and joined to
labels from SynSUM.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd


EMBEDDING_DIM: Final = 768
LABEL_NAMES: Final[list[str]] = ["dysp", "cough", "pain", "fever", "nasal"]
BINARY_LABELS: Final[list[str]] = ["dysp", "cough", "pain", "nasal"]
MAPPING_COLUMNS: Final[list[str]] = [
    "patient_id",
    "sentence_idx",
    "global_sentence_idx",
    "sentence",
]
SOURCE_ID_COLUMN: Final = "Unnamed: 0"
BINARY_ENCODING: Final[dict[str, int]] = {"no": 0, "yes": 1}
FEVER_ENCODING: Final[dict[str, int]] = {"none": 0, "low": 1, "high": 2}


def _require_columns(
    frame: pd.DataFrame, required: Sequence[str], source_name: str
) -> None:
    """Raise an informative error if a dataframe lacks required columns."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source_name} is missing required column(s): {missing}. "
            f"Available columns: {list(frame.columns)}"
        )


def _coerce_integer_ids(values: pd.Series, column_name: str) -> pd.Series:
    """Convert an ID series to int64 without silently truncating bad values."""
    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column '{column_name}' contains a non-numeric ID.") from exc

    if numeric.isna().any():
        rows = numeric.index[numeric.isna()].tolist()[:10]
        raise ValueError(
            f"Column '{column_name}' contains missing patient IDs at row(s) {rows}."
        )

    numeric_array = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_array).all():
        raise ValueError(f"Column '{column_name}' contains a non-finite patient ID.")
    if not np.equal(numeric_array, np.floor(numeric_array)).all():
        examples = values[~np.equal(numeric_array, np.floor(numeric_array))].head().tolist()
        raise ValueError(
            f"Column '{column_name}' contains non-integer patient IDs, e.g. {examples}."
        )

    int64_info = np.iinfo(np.int64)
    if ((numeric_array < int64_info.min) | (numeric_array > int64_info.max)).any():
        raise ValueError(f"Column '{column_name}' contains an ID outside int64 range.")
    return pd.Series(numeric_array.astype(np.int64), index=values.index, name=values.name)


def load_and_validate_mapping(path: str | Path) -> tuple[pd.DataFrame, int]:
    """Load mapping data, discard blank text, and standardize patient IDs.

    Returns the cleaned mapping and the number of rows before text filtering.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Sentence mapping file not found: {path}")

    mapping = pd.read_csv(path)
    _require_columns(mapping, MAPPING_COLUMNS, str(path))
    input_count = len(mapping)

    valid_text = mapping["sentence"].notna() & mapping["sentence"].astype(
        str
    ).str.strip().ne("")
    mapping = mapping.loc[valid_text].copy()
    if mapping.empty:
        raise ValueError(f"{path} contains no valid, non-blank sentences.")

    mapping["sentence"] = mapping["sentence"].astype(str).str.strip()
    mapping["patient_id"] = _coerce_integer_ids(
        mapping["patient_id"], "patient_id"
    )
    return mapping.reset_index(drop=True), input_count


def load_and_validate_labels(path: str | Path) -> pd.DataFrame:
    """Load semicolon-delimited SynSUM labels and rename its first ID column."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SynSUM source file not found: {path}")

    labels = pd.read_csv(path, sep=";")
    _require_columns(labels, [SOURCE_ID_COLUMN, *LABEL_NAMES], str(path))
    if labels.columns[0] != SOURCE_ID_COLUMN:
        raise ValueError(
            f"Expected the first column of {path} to be '{SOURCE_ID_COLUMN}', "
            f"but found '{labels.columns[0]}'."
        )

    labels = labels.rename(columns={SOURCE_ID_COLUMN: "patient_id"})
    labels["patient_id"] = _coerce_integer_ids(labels["patient_id"], "patient_id")
    if labels["patient_id"].duplicated().any():
        duplicates = (
            labels.loc[labels["patient_id"].duplicated(keep=False), "patient_id"]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(f"SynSUM.csv contains duplicate patient IDs: {duplicates}")
    return labels.loc[:, ["patient_id", *LABEL_NAMES]].copy()


def encode_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Normalize and strictly encode binary symptom and ordinal fever labels."""
    encoded = labels.copy()
    encodings = {**{name: BINARY_ENCODING for name in BINARY_LABELS}, "fever": FEVER_ENCODING}

    for name in LABEL_NAMES:
        raw = encoded[name]
        if raw.isna().any():
            rows = raw.index[raw.isna()].tolist()[:10]
            raise ValueError(f"Label '{name}' contains missing values at row(s) {rows}.")
        normalized = raw.astype(str).str.strip().str.lower()
        unexpected = sorted(set(normalized.unique()) - set(encodings[name]))
        if unexpected:
            raise ValueError(
                f"Label '{name}' contains unexpected value(s): {unexpected}. "
                f"Expected one of {sorted(encodings[name])}."
            )
        encoded[name] = normalized.map(encodings[name]).astype(np.int64)

    return encoded.loc[:, ["patient_id", *LABEL_NAMES]]


def resolve_device(requested_device: str) -> str:
    """Resolve ``auto`` to CUDA when available and otherwise to CPU."""
    import torch

    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device '{requested_device}' was requested, but CUDA is not available. "
            "Use '--device auto' or '--device cpu'."
        )
    return requested_device


def embed_sentences(
    texts: Sequence[str], model_name: str, batch_size: int, device: str
) -> np.ndarray:
    """Embed prefixed sentences in batches as a float32 NumPy matrix."""
    import torch
    from sentence_transformers import SentenceTransformer

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; received {batch_size}.")
    if len(texts) == 0:
        raise ValueError("Cannot embed an empty collection of sentences.")

    prefixed_texts = [f"search_document: {text}" for text in texts]
    model = SentenceTransformer(model_name, device=device)
    model.eval()

    # Pooling operation 1: SentenceTransformer performs its configured token-level
    # pooling to produce one full 768-dimensional vector per sentence.
    with torch.inference_mode():
        embeddings = model.encode(
            prefixed_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            output_value="sentence_embedding",
            precision="float32",
            convert_to_numpy=True,
            convert_to_tensor=False,
            device=device,
        )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    expected_shape = (len(texts), EMBEDDING_DIM)
    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Model '{model_name}' produced embeddings with shape {embeddings.shape}; "
            f"expected {expected_shape}. Ensure no embedding truncation is configured."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Sentence embeddings contain NaN or infinite values.")
    return embeddings


def mean_pool_by_patient(
    mapping: pd.DataFrame, embeddings: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-pool all sentence embeddings directly by patient ID."""
    expected_shape = (len(mapping), EMBEDDING_DIM)
    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Cannot pool embeddings with shape {embeddings.shape}; expected {expected_shape}."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Cannot pool embeddings containing NaN or infinite values.")

    patient_ids, inverse = np.unique(
        mapping["patient_id"].to_numpy(dtype=np.int64), return_inverse=True
    )
    sums = np.zeros((len(patient_ids), EMBEDDING_DIM), dtype=np.float32)
    np.add.at(sums, inverse, embeddings)
    counts = np.bincount(inverse, minlength=len(patient_ids)).astype(np.float32)
    np.divide(sums, counts[:, None], out=sums)

    # Directly average every sentence vector for each patient. There is no
    # intermediate sentence-level or document-level pooling operation.
    unique_mapping_ids = mapping["patient_id"].nunique()
    if len(patient_ids) != unique_mapping_ids or len(sums) != unique_mapping_ids:
        raise RuntimeError(
            "Patient-level mean pooling did not produce exactly one row per unique "
            "patient_id."
        )
    if not np.isfinite(sums).all():
        raise ValueError("Patient-level embeddings contain NaN or infinite values.")
    return patient_ids.astype(np.int64, copy=False), sums


def join_embeddings_and_labels(
    patient_ids: np.ndarray, patient_embeddings: np.ndarray, labels: pd.DataFrame
) -> pd.DataFrame:
    """Join patient embeddings to ground-truth labels with strict ID matching."""
    embedding_columns = [f"emb_{index:03d}" for index in range(EMBEDDING_DIM)]
    embedding_frame = pd.DataFrame(patient_embeddings, columns=embedding_columns)
    embedding_frame.insert(0, "patient_id", patient_ids)

    known_ids = set(labels["patient_id"].tolist())
    missing_ids = sorted(set(patient_ids.tolist()) - known_ids)
    if missing_ids:
        preview = missing_ids[:20]
        suffix = " ..." if len(missing_ids) > len(preview) else ""
        raise ValueError(
            f"{len(missing_ids)} patient_id value(s) are missing from SynSUM.csv: "
            f"{preview}{suffix}"
        )

    final = embedding_frame.merge(
        labels, on="patient_id", how="left", validate="one_to_one", sort=False
    )
    return final.loc[:, ["patient_id", *embedding_columns, *LABEL_NAMES]].sort_values(
        "patient_id", kind="stable", ignore_index=True
    )


def validate_final_dataset(final: pd.DataFrame) -> None:
    """Validate final dimensions, values, dtypes, ordering, and uniqueness."""
    embedding_columns = [f"emb_{index:03d}" for index in range(EMBEDDING_DIM)]
    expected_columns = ["patient_id", *embedding_columns, *LABEL_NAMES]
    if list(final.columns) != expected_columns:
        raise ValueError("Final dataset columns or their order do not match the contract.")
    if final.empty:
        raise ValueError("Final dataset contains no patients.")
    if final["patient_id"].duplicated().any():
        raise ValueError("Final dataset contains duplicate patient IDs.")
    if not final["patient_id"].is_monotonic_increasing:
        raise ValueError("Final dataset is not sorted by patient_id.")

    embedding_values = final[embedding_columns].to_numpy(dtype=np.float32)
    if embedding_values.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Final embedding dimension is {embedding_values.shape[1]}; "
            f"expected {EMBEDDING_DIM}."
        )
    if not np.isfinite(embedding_values).all():
        raise ValueError("Final embedding values contain NaN or infinity.")

    if final[LABEL_NAMES].isna().any().any():
        missing_columns = final[LABEL_NAMES].columns[
            final[LABEL_NAMES].isna().any()
        ].tolist()
        raise ValueError(
            f"Final dataset contains missing values in label(s): {missing_columns}."
        )
    non_integer_labels = [
        name
        for name in LABEL_NAMES
        if not pd.api.types.is_integer_dtype(final[name].dtype)
    ]
    if non_integer_labels:
        raise ValueError(
            f"Final labels must have integer dtype; found non-integer label(s): "
            f"{non_integer_labels}."
        )

    for name in BINARY_LABELS:
        actual = set(final[name].unique().tolist())
        if not actual <= {0, 1}:
            raise ValueError(f"Final binary label '{name}' has invalid values: {actual}")
    fever_values = set(final["fever"].unique().tolist())
    if not fever_values <= {0, 1, 2}:
        raise ValueError(f"Final ordinal label 'fever' has invalid values: {fever_values}")


def save_outputs(
    final: pd.DataFrame,
    clean_mapping: pd.DataFrame,
    sentence_embeddings: np.ndarray,
    output_dir: str | Path,
) -> None:
    """Write patient tables, training arrays, and sentence traceability data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_columns = [f"emb_{index:03d}" for index in range(EMBEDDING_DIM)]

    final.to_parquet(
        output_dir / "patient_mean_embeddings_with_labels.parquet", index=False
    )
    final.to_csv(output_dir / "patient_mean_embeddings_with_labels.csv", index=False)

    patient_ids = final["patient_id"].to_numpy(dtype=np.int64)
    x = final[embedding_columns].to_numpy(dtype=np.float32)
    y = final[LABEL_NAMES].to_numpy(dtype=np.int64)
    if x.shape[0] != y.shape[0] or x.shape[0] != len(patient_ids):
        raise RuntimeError("Patient IDs, X, and y have inconsistent row counts.")
    if x.shape[1] != EMBEDDING_DIM or y.shape[1] != len(LABEL_NAMES):
        raise RuntimeError(
            f"Unexpected training array shapes: X={x.shape}, y={y.shape}."
        )
    np.savez_compressed(
        output_dir / "patient_embeddings_and_labels.npz",
        patient_ids=patient_ids,
        X=x,
        y=y,
        label_names=np.asarray(LABEL_NAMES),
    )

    trace = clean_mapping.loc[:, MAPPING_COLUMNS].copy()
    trace_embeddings = pd.DataFrame(sentence_embeddings, columns=embedding_columns)
    trace = pd.concat([trace.reset_index(drop=True), trace_embeddings], axis=1)
    trace.to_parquet(output_dir / "sentence_embeddings.parquet", index=False)


def print_summary(
    input_sentence_count: int,
    clean_mapping: pd.DataFrame,
    final: pd.DataFrame,
) -> None:
    """Print concise data counts and non-zero symptom prevalence."""
    print("Preprocessing summary")
    print(f"  Input sentences: {input_sentence_count:,}")
    print(f"  Valid sentences: {len(clean_mapping):,}")
    print(f"  Unique global sentences: {clean_mapping['global_sentence_idx'].nunique():,}")
    print(f"  Unique patients: {clean_mapping['patient_id'].nunique():,}")
    print(f"  Embedding dimension: {EMBEDDING_DIM}")
    print(f"  Patients in final joined dataset: {len(final):,}")
    print("  Positive-label prevalence:")
    for name in LABEL_NAMES:
        # Fever is ordinal; both low (1) and high (2) count as positive here.
        prevalence = final[name].gt(0).mean()
        print(f"    {name}: {prevalence:.2%}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create patient-level mean embeddings and symptom labels."
    )
    parser.add_argument("--mapping-path", default="gfs_sentence_mapping.csv")
    parser.add_argument("--source-path", default="SynSUM.csv")
    parser.add_argument("--output-dir", default="outputs/supervised_embeddings")
    parser.add_argument(
        "--model-name", default="nomic-ai/modernbert-embed-base"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device, e.g. auto, cpu, cuda, cuda:0. Auto chooses CUDA or CPU.",
    )
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Run the end-to-end preprocessing pipeline."""
    args = parse_args(argv)
    mapping, input_sentence_count = load_and_validate_mapping(args.mapping_path)
    labels = encode_labels(load_and_validate_labels(args.source_path))

    missing_ids = sorted(
        set(mapping["patient_id"].tolist()) - set(labels["patient_id"].tolist())
    )
    if missing_ids:
        preview = missing_ids[:20]
        suffix = " ..." if len(missing_ids) > len(preview) else ""
        raise ValueError(
            f"{len(missing_ids)} patient_id value(s) are missing from SynSUM.csv: "
            f"{preview}{suffix}"
        )

    device = resolve_device(args.device)
    print(f"Embedding {len(mapping):,} valid sentences on device: {device}")
    sentence_embeddings = embed_sentences(
        mapping["sentence"].tolist(),
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=device,
    )
    patient_ids, patient_embeddings = mean_pool_by_patient(
        mapping, sentence_embeddings
    )
    final = join_embeddings_and_labels(patient_ids, patient_embeddings, labels)
    validate_final_dataset(final)
    save_outputs(final, mapping, sentence_embeddings, args.output_dir)
    print_summary(input_sentence_count, mapping, final)
    print(f"Outputs written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
