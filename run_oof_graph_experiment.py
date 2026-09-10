#!/usr/bin/env python3
"""Strict patient-level OOF symptom classification and causal graph evaluation.

This workflow deliberately never consumes the notebook's ``all_dataset_predictions``
artifacts. Each prediction used downstream is produced by a classifier whose outer
training pool excludes that patient.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from prepare_patient_embeddings import EMBEDDING_DIM, LABEL_NAMES


SUPERVISION_FRACTIONS: Final[tuple[float, ...]] = (0.05, 0.10, 0.20, 1.00)
CONDITION_NAMES: Final[dict[float, str]] = {
    0.05: "FS_5",
    0.10: "FS_10",
    0.20: "FS_20",
    1.00: "FS_100",
}
FILE_SUFFIXES: Final[dict[float, str]] = {
    0.05: "005",
    0.10: "010",
    0.20: "020",
    1.00: "100",
}


@dataclass(frozen=True)
class PCConfig:
    """One immutable causal-discovery configuration for every condition."""

    algorithm: str = "PC-learn"
    implementation: str = "pgmpy"
    model_class: str = "DiscreteBayesianNetwork"
    conditional_independence_test: str = "g_sq"
    conditional_independence_test_description: str = "likelihood-ratio G-square (G^2)"
    alpha: float = 0.05
    stable: bool = True
    pc_return_type: str = "pdag"
    return_type: str = "dag"
    max_k: int | None = None


@dataclass(frozen=True)
class OuterFold:
    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def load_embedding_arrays(
    path: str | Path, expected_patients: int | None = 10_000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the existing patient embeddings and modern binary targets."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Patient embedding NPZ not found: {path}")
    with np.load(path, allow_pickle=False) as saved:
        required = {"patient_ids", "X", "y", "label_names"}
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"Embedding NPZ is missing keys: {sorted(missing)}")
        patient_ids = saved["patient_ids"].astype(np.int64, copy=False)
        X = saved["X"].astype(np.float32, copy=False)
        y_original = saved["y"].astype(np.int64, copy=False)
        label_names = saved["label_names"].astype(str)

    expected_labels = np.asarray(LABEL_NAMES)
    if not np.array_equal(label_names, expected_labels):
        raise ValueError(
            f"Expected label order {expected_labels.tolist()}, found {label_names.tolist()}"
        )
    if X.shape != (len(patient_ids), EMBEDDING_DIM):
        raise ValueError(
            f"Expected embedding shape ({len(patient_ids)}, {EMBEDDING_DIM}), found {X.shape}"
        )
    if y_original.shape != (len(patient_ids), len(LABEL_NAMES)):
        raise ValueError(f"Unexpected target shape: {y_original.shape}")
    if expected_patients is not None and len(patient_ids) != expected_patients:
        raise ValueError(
            f"Strict experiment requires {expected_patients:,} patients; found {len(patient_ids):,}."
        )
    if len(np.unique(patient_ids)) != len(patient_ids):
        raise ValueError("Embedding data contains duplicate patient IDs.")
    if not np.isfinite(X).all() or not np.isfinite(y_original).all():
        raise ValueError("Embedding data contains NaN or infinite values.")

    y = y_original.copy()
    y[:, LABEL_NAMES.index("fever")] = (
        y_original[:, LABEL_NAMES.index("fever")] > 0
    ).astype(np.int64)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("All five modern classifier targets must be binary.")
    return patient_ids, X, y, label_names


def create_outer_folds(
    patient_ids: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    n_splits: int = 5,
) -> list[OuterFold]:
    """Create one shared patient-level multilabel-stratified outer split."""
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "iterative-stratification is required for the outer folds. "
            "Install the repository requirements."
        ) from exc

    if len(patient_ids) != len(y):
        raise ValueError("patient_ids and y have different row counts.")
    splitter = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    folds = [
        OuterFold(
            fold_id=fold_id,
            train_indices=np.asarray(train, dtype=np.int64),
            test_indices=np.asarray(test, dtype=np.int64),
        )
        for fold_id, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y), start=1)
    ]
    validate_outer_folds(patient_ids, folds)
    return folds


def validate_outer_folds(patient_ids: np.ndarray, folds: Sequence[OuterFold]) -> None:
    """Assert disjoint training/test pools and exactly one test fold per patient."""
    all_positions = set(range(len(patient_ids)))
    test_positions: list[int] = []
    for fold in folds:
        train_set = set(fold.train_indices.tolist())
        test_set = set(fold.test_indices.tolist())
        assert train_set.isdisjoint(test_set), (
            f"Outer fold {fold.fold_id} has patients in both train and test."
        )
        assert train_set | test_set == all_positions, (
            f"Outer fold {fold.fold_id} does not partition every patient."
        )
        test_positions.extend(fold.test_indices.tolist())
    counts = np.bincount(np.asarray(test_positions), minlength=len(patient_ids))
    assert len(test_positions) == len(patient_ids)
    assert np.array_equal(counts, np.ones(len(patient_ids), dtype=counts.dtype)), (
        "Every patient must appear in exactly one outer test fold."
    )


def create_nested_training_subsets(
    train_indices: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    fractions: Sequence[float] = SUPERVISION_FRACTIONS,
) -> dict[float, np.ndarray]:
    """Create stratified nested prefixes of an outer training pool."""
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "iterative-stratification is required for nested training subsets."
        ) from exc

    fractions = tuple(sorted(float(value) for value in fractions))
    if not fractions or fractions[-1] != 1.0:
        raise ValueError("Nested fractions must include 100% of the outer training pool.")
    if fractions[0] <= 0 or len(set(fractions)) != len(fractions):
        raise ValueError(f"Invalid supervision fractions: {fractions}")
    chunk_count = int(round(1.0 / fractions[0]))
    chunk_count = min(chunk_count, len(train_indices))
    splitter = MultilabelStratifiedKFold(
        n_splits=chunk_count, shuffle=True, random_state=seed
    )
    chunks = [
        train_indices[test_positions]
        for _, test_positions in splitter.split(
            np.zeros(len(train_indices)), y[train_indices]
        )
    ]
    order = np.concatenate(chunks)
    assert len(order) == len(train_indices)
    assert set(order.tolist()) == set(train_indices.tolist())

    subsets: dict[float, np.ndarray] = {}
    for fraction in fractions:
        size = int(round(len(train_indices) * fraction))
        subsets[fraction] = order[:size].copy()
    validate_nested_subsets(train_indices, subsets)
    return subsets


def validate_nested_subsets(
    outer_train_indices: np.ndarray, subsets: Mapping[float, np.ndarray]
) -> None:
    """Assert 5% subset 10% subset 20% subset 100% for one fold."""
    previous: set[int] = set()
    outer = set(outer_train_indices.tolist())
    for fraction in sorted(subsets):
        current = set(np.asarray(subsets[fraction]).tolist())
        assert len(current) == len(subsets[fraction]), "A nested subset contains duplicates."
        assert previous.issubset(current), "Training subsets are not nested."
        assert current.issubset(outer), "A training subset escapes the outer training pool."
        previous = current
    assert previous == outer, "The 100% subset must equal the outer training pool."


def _fold_assignment_frame(
    patient_ids: np.ndarray, folds: Sequence[OuterFold]
) -> pd.DataFrame:
    assignment = np.zeros(len(patient_ids), dtype=np.int64)
    for fold in folds:
        assignment[fold.test_indices] = fold.fold_id
    assert np.all(assignment > 0)
    return pd.DataFrame({"patient_id": patient_ids, "outer_fold": assignment})


def _prediction_frame(
    patient_ids: np.ndarray,
    y: np.ndarray,
    label_names: Sequence[str],
    fold_ids: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame({"patient_id": patient_ids, "outer_fold": fold_ids})
    for index, label in enumerate(label_names):
        frame[f"{label}_true"] = y[:, index]
        frame[f"{label}_probability"] = probabilities[:, index]
        frame[f"{label}_prediction"] = predictions[:, index]
    return frame


def generate_oof_predictions(
    patient_ids: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    label_names: Sequence[str],
    folds: Sequence[OuterFold],
    nested_subsets: Mapping[int, Mapping[float, np.ndarray]],
    *,
    classifier_config: ClassifierConfig,
    device: torch.device,
    base_seed: int,
    checkpoint_dir: Path,
) -> tuple[dict[float, pd.DataFrame], list[dict[str, Any]]]:
    """Train all outer models and return one unseen prediction per patient."""
    from modern_symptom_classifier import (
        calculate_metrics,
        predict_probabilities,
        select_training_epochs,
        train_final_classifier,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fold_ids = _fold_assignment_frame(patient_ids, folds)["outer_fold"].to_numpy()
    probability_arrays = {
        fraction: np.full((len(patient_ids), len(label_names)), np.nan, dtype=np.float32)
        for fraction in SUPERVISION_FRACTIONS
    }
    prediction_arrays = {
        fraction: np.full((len(patient_ids), len(label_names)), -1, dtype=np.int64)
        for fraction in SUPERVISION_FRACTIONS
    }
    inner_rows: list[dict[str, Any]] = []

    for fold in folds:
        for fraction in SUPERVISION_FRACTIONS:
            subset = np.asarray(nested_subsets[fold.fold_id][fraction], dtype=np.int64)
            percent = int(round(fraction * 100))
            model_seed = base_seed * 100_000 + fold.fold_id * 1_000 + percent
            final_epochs, cv_rows = select_training_epochs(
                X,
                y,
                subset,
                label_names,
                classifier_config,
                device,
                model_seed,
            )
            for row in cv_rows:
                inner_rows.append(
                    {
                        "condition": CONDITION_NAMES[fraction],
                        "outer_fold": fold.fold_id,
                        "training_pool_fraction": fraction,
                        "subset_patients": len(subset),
                        "final_refit_epochs": final_epochs,
                        **row,
                    }
                )
            model, mean, std = train_final_classifier(
                X,
                y,
                subset,
                final_epochs,
                classifier_config,
                device,
                model_seed,
            )
            probabilities = predict_probabilities(
                model,
                X,
                y,
                fold.test_indices,
                mean,
                std,
                classifier_config,
                device,
                model_seed,
            )
            _, predictions = calculate_metrics(
                y[fold.test_indices],
                probabilities,
                label_names,
                classifier_config.binary_threshold,
            )
            probability_arrays[fraction][fold.test_indices] = probabilities
            prediction_arrays[fraction][fold.test_indices] = predictions

            checkpoint_path = checkpoint_dir / (
                f"model_{FILE_SUFFIXES[fraction]}_outer_fold_{fold.fold_id}.pt"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_mean": mean,
                    "feature_std": std,
                    "label_names": list(label_names),
                    "condition": CONDITION_NAMES[fraction],
                    "training_pool_fraction": fraction,
                    "outer_fold": fold.fold_id,
                    "outer_training_patient_ids": patient_ids[fold.train_indices],
                    "selected_training_patient_ids": patient_ids[subset],
                    "outer_test_patient_ids": patient_ids[fold.test_indices],
                    "final_epochs": final_epochs,
                    "seed": model_seed,
                    "classifier_config": classifier_config.to_dict(),
                },
                checkpoint_path,
            )

    tables: dict[float, pd.DataFrame] = {}
    for fraction in SUPERVISION_FRACTIONS:
        if np.isnan(probability_arrays[fraction]).any():
            raise AssertionError(f"Missing OOF probabilities for {CONDITION_NAMES[fraction]}.")
        if not set(np.unique(prediction_arrays[fraction])).issubset({0, 1}):
            raise AssertionError(f"Missing OOF predictions for {CONDITION_NAMES[fraction]}.")
        tables[fraction] = _prediction_frame(
            patient_ids,
            y,
            label_names,
            fold_ids,
            probability_arrays[fraction],
            prediction_arrays[fraction],
        )
    return tables, inner_rows


def validate_oof_integrity(
    patient_ids: np.ndarray,
    folds: Sequence[OuterFold],
    nested_subsets: Mapping[int, Mapping[float, np.ndarray]],
    oof_tables: Mapping[float, pd.DataFrame],
) -> None:
    """Run the patient-level leakage and consistency assertions."""
    validate_outer_folds(patient_ids, folds)
    expected_ids = set(patient_ids.tolist())
    assignment = _fold_assignment_frame(patient_ids, folds).set_index("patient_id")
    reference_folds: np.ndarray | None = None
    fold_by_id = assignment["outer_fold"].to_dict()

    for fold in folds:
        validate_nested_subsets(fold.train_indices, nested_subsets[fold.fold_id])
    for fraction in SUPERVISION_FRACTIONS:
        table = oof_tables[fraction]
        assert len(table) == len(patient_ids)
        assert table["patient_id"].is_unique
        assert set(table["patient_id"].tolist()) == expected_ids
        actual_folds = table["patient_id"].map(fold_by_id).to_numpy()
        assert np.array_equal(table["outer_fold"].to_numpy(), actual_folds)
        if reference_folds is None:
            reference_folds = actual_folds
        else:
            assert np.array_equal(reference_folds, actual_folds), (
                "All supervision conditions must use the same outer folds."
            )
        for fold in folds:
            predicted_ids = set(
                table.loc[table["outer_fold"] == fold.fold_id, "patient_id"].tolist()
            )
            training_ids = set(patient_ids[fold.train_indices].tolist())
            assert predicted_ids.isdisjoint(training_ids), (
                "An OOF prediction came from a model whose outer pool contained that patient."
            )


def calculate_classifier_outputs(
    oof_tables: Mapping[float, pd.DataFrame], label_names: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create fold, complete-OOF, and per-symptom classifier metric tables."""
    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    symptom_rows: list[dict[str, Any]] = []
    for fraction in SUPERVISION_FRACTIONS:
        table = oof_tables[fraction]
        condition = CONDITION_NAMES[fraction]
        for fold_id, fold_frame in table.groupby("outer_fold", sort=True):
            per_fold: list[tuple[float, float, float]] = []
            for label in label_names:
                metrics = _binary_metrics(
                    fold_frame[f"{label}_true"], fold_frame[f"{label}_prediction"]
                )
                per_fold.append((metrics["precision"], metrics["recall"], metrics["f1"]))
                fold_rows.append(
                    {
                        "condition": condition,
                        "outer_fold": int(fold_id),
                        "label": str(label),
                        "patients": len(fold_frame),
                        **metrics,
                    }
                )
            macro = np.asarray(per_fold).mean(axis=0)
            fold_rows.append(
                {
                    "condition": condition,
                    "outer_fold": int(fold_id),
                    "label": "macro",
                    "patients": len(fold_frame),
                    "precision": float(macro[0]),
                    "recall": float(macro[1]),
                    "f1": float(macro[2]),
                }
            )

        condition_folds = pd.DataFrame(fold_rows)
        condition_folds = condition_folds[condition_folds["condition"] == condition]
        macro_folds = condition_folds[condition_folds["label"] == "macro"]
        overall_per_label: list[tuple[float, float, float]] = []
        for label in label_names:
            metrics = _binary_metrics(table[f"{label}_true"], table[f"{label}_prediction"])
            overall_per_label.append((metrics["precision"], metrics["recall"], metrics["f1"]))
            label_folds = condition_folds[condition_folds["label"] == label]
            symptom_rows.append(
                {
                    "condition": condition,
                    "label": str(label),
                    "patients": len(table),
                    **metrics,
                    **_mean_sd_columns(label_folds),
                }
            )
        overall_macro = np.asarray(overall_per_label).mean(axis=0)
        oof_rows.append(
            {
                "condition": condition,
                "patients": len(table),
                "oof_macro_precision": float(overall_macro[0]),
                "oof_macro_recall": float(overall_macro[1]),
                "oof_macro_f1": float(overall_macro[2]),
                **_mean_sd_columns(macro_folds, prefix="fold_macro_"),
            }
        )
    return pd.DataFrame(fold_rows), pd.DataFrame(oof_rows), pd.DataFrame(symptom_rows)


def _binary_metrics(targets: Sequence[int], predictions: Sequence[int]) -> dict[str, float]:
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average="binary", zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _mean_sd_columns(frame: pd.DataFrame, prefix: str = "fold_") -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in ("precision", "recall", "f1"):
        result[f"{prefix}{metric}_mean"] = float(frame[metric].mean())
        result[f"{prefix}{metric}_sd"] = float(frame[metric].std(ddof=1))
    return result


def save_oof_predictions(
    oof_tables: Mapping[float, pd.DataFrame], output_dir: Path, label_names: Sequence[str]
) -> None:
    for fraction, table in oof_tables.items():
        suffix = FILE_SUFFIXES[fraction]
        table.to_csv(output_dir / f"oof_predictions_{suffix}.csv", index=False)
        np.savez_compressed(
            output_dir / f"oof_predictions_{suffix}.npz",
            patient_ids=table["patient_id"].to_numpy(),
            outer_fold=table["outer_fold"].to_numpy(dtype=np.int64),
            y_true=table[[f"{name}_true" for name in label_names]].to_numpy(dtype=np.int64),
            probabilities=table[
                [f"{name}_probability" for name in label_names]
            ].to_numpy(dtype=np.float32),
            y_pred=table[
                [f"{name}_prediction" for name in label_names]
            ].to_numpy(dtype=np.int64),
            label_names=np.asarray(label_names),
            training_pool_fraction=np.float64(fraction),
        )


def load_structured_dataset(
    path: str | Path, patient_id_column: str, separator: str = ";"
) -> pd.DataFrame:
    """Load SynSUM and normalize its ID column without changing other values."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Structured SynSUM data not found: {path}")
    frame = pd.read_csv(path, sep=separator)
    if patient_id_column not in frame.columns:
        raise ValueError(
            f"Structured data lacks patient ID column '{patient_id_column}'. "
            f"Available columns: {list(frame.columns)}"
        )
    if patient_id_column != "patient_id":
        if "patient_id" in frame.columns:
            raise ValueError("Cannot rename the ID column because 'patient_id' already exists.")
        frame = frame.rename(columns={patient_id_column: "patient_id"})
    frame["patient_id"] = pd.to_numeric(frame["patient_id"], errors="raise").astype(np.int64)
    if frame["patient_id"].duplicated().any():
        raise ValueError("Structured SynSUM data contains duplicate patient IDs.")
    return frame


def build_graph_dataset(
    original: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    symptom_columns: Sequence[str] = LABEL_NAMES,
) -> pd.DataFrame:
    """Replace only symptoms using an explicit one-to-one patient-ID merge."""
    required_oof = ["patient_id", *[f"{name}_prediction" for name in symptom_columns]]
    missing = [column for column in required_oof if column not in oof_predictions.columns]
    if missing:
        raise ValueError(f"OOF table is missing columns: {missing}")
    if any(column not in original.columns for column in ["patient_id", *symptom_columns]):
        raise ValueError("Structured data is missing patient_id or a symptom column.")
    if original["patient_id"].duplicated().any() or oof_predictions["patient_id"].duplicated().any():
        raise ValueError("ID-based graph reconstruction requires unique patient IDs.")
    if set(original["patient_id"]) != set(oof_predictions["patient_id"]):
        raise ValueError("Structured and OOF patient-ID sets differ.")

    replacements = oof_predictions[required_oof].rename(
        columns={f"{name}_prediction": f"__oof_{name}" for name in symptom_columns}
    )
    tagged = original.copy()
    tagged["__original_row_order"] = np.arange(len(tagged))
    rebuilt = tagged.merge(
        replacements, on="patient_id", how="left", validate="one_to_one", sort=False
    ).sort_values("__original_row_order", kind="stable")
    assert np.array_equal(rebuilt["patient_id"].to_numpy(), original["patient_id"].to_numpy())
    for name in symptom_columns:
        rebuilt[name] = rebuilt.pop(f"__oof_{name}").astype(np.int64)
    rebuilt = rebuilt.drop(columns="__original_row_order").reset_index(drop=True)
    validate_graph_reconstruction(original.reset_index(drop=True), rebuilt, symptom_columns)
    return rebuilt


def validate_graph_reconstruction(
    original: pd.DataFrame,
    rebuilt: pd.DataFrame,
    symptom_columns: Sequence[str] = LABEL_NAMES,
) -> None:
    """Assert that reconstruction preserved every non-symptom value exactly."""
    assert list(original.columns) == list(rebuilt.columns)
    assert len(original) == len(rebuilt)
    assert np.array_equal(original["patient_id"].to_numpy(), rebuilt["patient_id"].to_numpy())
    untouched = [column for column in original.columns if column not in symptom_columns]
    pd.testing.assert_frame_equal(
        original[untouched], rebuilt[untouched], check_dtype=True, check_exact=True
    )


def load_reference_adjacency(path: str | Path) -> pd.DataFrame:
    """Load a named binary DAG adjacency matrix (row=source, column=target)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Reference adjacency CSV not found: {path}")
    raw = pd.read_csv(path)
    if raw.shape[1] != raw.shape[0] + 1:
        raise ValueError(
            "Reference adjacency must have one node-name column followed by a square matrix."
        )
    node_names = raw.iloc[:, 0].astype(str).tolist()
    adjacency = raw.iloc[:, 1:].copy()
    adjacency.columns = adjacency.columns.astype(str)
    adjacency.index = node_names
    if len(set(node_names)) != len(node_names):
        raise ValueError("Reference adjacency has duplicate row node names.")
    if set(adjacency.columns) != set(node_names):
        raise ValueError("Reference adjacency row and column node sets differ.")
    adjacency = adjacency.loc[node_names, node_names].apply(pd.to_numeric, errors="raise")
    values = adjacency.to_numpy()
    if not set(np.unique(values)).issubset({0, 1}):
        raise ValueError("Reference adjacency must contain only 0/1 values.")
    if np.any(np.diag(values) != 0):
        raise ValueError("Reference adjacency diagonal must be zero.")
    return adjacency.astype(np.int64)


def encode_discrete_graph_data(
    graph_dataset: pd.DataFrame, node_names: Sequence[str]
) -> pd.DataFrame:
    """Encode every named SynSUM graph variable as a discrete integer category."""
    missing = [name for name in node_names if name not in graph_dataset.columns]
    if missing:
        raise ValueError(f"Structured data is missing reference graph nodes: {missing}")
    encoded_columns: dict[str, np.ndarray] = {}
    for name in node_names:
        values = graph_dataset[name]
        if values.isna().any():
            raise ValueError(f"Graph node '{name}' contains missing values.")
        codes, _ = pd.factorize(values, sort=True)
        if np.any(codes < 0):
            raise ValueError(f"Graph node '{name}' could not be discretely encoded.")
        encoded_columns[name] = codes.astype(np.int64, copy=False)
    encoded = pd.DataFrame(encoded_columns, columns=list(node_names))
    if encoded.shape != (len(graph_dataset), len(node_names)):
        raise AssertionError("Encoded graph data has an unexpected shape.")
    return encoded


def run_pc_learn(
    graph_dataset: pd.DataFrame,
    node_names: Sequence[str],
    config: PCConfig = PCConfig(),
    *,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Learn a DAG with pgmpy PC and wrap it in a DiscreteBayesianNetwork."""
    if config != PCConfig():
        raise ValueError("All graph conditions must use the immutable PCConfig defaults.")
    try:
        from pgmpy.causal_discovery import PC
        from pgmpy.models import DiscreteBayesianNetwork
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pgmpy is required for PC+G-square. Install repository requirements."
        ) from exc

    encoded = encode_discrete_graph_data(graph_dataset, node_names)
    max_cond_vars = len(node_names) - 2 if config.max_k is None else config.max_k
    estimator = PC(
        variant="stable" if config.stable else "orig",
        ci_test=config.conditional_independence_test,
        return_type=config.pc_return_type,
        significance_level=config.alpha,
        max_cond_vars=max_cond_vars,
        show_progress=show_progress,
    ).fit(encoded)

    learned_pdag = estimator.causal_graph_
    bayesian_network = pdag_to_discrete_bayesian_network(
        learned_pdag,
        node_names,
        DiscreteBayesianNetwork,
    )
    learned_names = set(bayesian_network.nodes())
    assert learned_names == set(node_names), "PC returned an unexpected node set."
    adjacency, edges = bayesian_network_to_adjacency(bayesian_network, node_names)
    return adjacency, edges, bayesian_network


def pdag_to_discrete_bayesian_network(
    pdag: Any,
    node_names: Sequence[str],
    model_class: Any,
) -> Any:
    """Create a deterministic acyclic BN extension of a pgmpy PC PDAG.

    A finite-sample PC result can contain conflicting directed preferences and
    therefore have no faithful DAG extension. We retain the complete learned
    skeleton, preserve every compatible directed preference, and break any
    remaining preference cycle deterministically by node name. Orienting all
    skeleton edges forward in the resulting total order guarantees a true DAG.
    """
    names = list(node_names)
    node_set = set(names)
    if set(pdag.nodes()) != node_set:
        raise ValueError("PC PDAG and reference node sets differ.")

    directed_edges = set(pdag.directed_edges)
    undirected_edges = set(pdag.undirected_edges)
    for source, target in directed_edges | undirected_edges:
        if source not in node_set or target not in node_set:
            raise ValueError("PC returned an edge with an unknown endpoint.")
        if source == target:
            raise ValueError("PC returned a self-loop.")

    incoming: dict[str, set[str]] = {name: set() for name in names}
    for source, target in directed_edges:
        incoming[target].add(source)

    remaining = set(names)
    total_order: list[str] = []
    cycle_breaks = 0
    while remaining:
        sources = [name for name in remaining if not (incoming[name] & remaining)]
        if sources:
            selected = min(sources, key=lambda value: (str(value), repr(value)))
        else:
            selected = min(remaining, key=lambda value: (str(value), repr(value)))
            cycle_breaks += 1
        total_order.append(selected)
        remaining.remove(selected)

    rank = {name: index for index, name in enumerate(total_order)}
    skeleton_pairs = {
        frozenset((source, target))
        for source, target in directed_edges | undirected_edges
    }
    oriented_edges = []
    for pair in skeleton_pairs:
        source, target = sorted(pair, key=rank.__getitem__)
        oriented_edges.append((source, target))
    oriented_edges.sort(key=lambda edge: (rank[edge[0]], rank[edge[1]]))

    reversed_preferences = sum(
        rank[source] > rank[target] for source, target in directed_edges
    )
    bayesian_network = model_class()
    bayesian_network.add_nodes_from(names)
    bayesian_network.add_edges_from(oriented_edges)
    bayesian_network.graph["pc_pdag_conversion"] = {
        "method": "deterministic_acyclic_total_order",
        "node_order": total_order,
        "pdag_directed_edges": len(directed_edges),
        "pdag_undirected_edges": len(undirected_edges),
        "cycle_breaks": cycle_breaks,
        "reversed_directed_preferences": int(reversed_preferences),
    }
    return bayesian_network


def bayesian_network_to_adjacency(
    bayesian_network: Any,
    node_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a pgmpy DiscreteBayesianNetwork DAG to named graph artifacts."""
    names = list(node_names)
    if set(bayesian_network.nodes()) != set(names):
        raise ValueError("Bayesian-network and reference node sets differ.")
    adjacency = pd.DataFrame(0, index=names, columns=names, dtype=np.int64)
    edge_rows: list[dict[str, str]] = []
    for source, target in bayesian_network.edges():
        adjacency.loc[source, target] = 1
        edge_rows.append({"source": source, "target": target, "edge_type": "->"})
    return (
        adjacency,
        pd.DataFrame(edge_rows, columns=["source", "target", "edge_type"]),
    )


def align_adjacencies(
    reference: pd.DataFrame, learned: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align matrices by node name and assert the exact intended node set."""
    reference_rows = set(reference.index.astype(str))
    reference_columns = set(reference.columns.astype(str))
    learned_rows = set(learned.index.astype(str))
    learned_columns = set(learned.columns.astype(str))
    assert reference_rows == reference_columns, "Reference row/column node sets differ."
    assert learned_rows == learned_columns, "Learned row/column node sets differ."
    assert reference_rows == learned_rows, (
        f"Graph node sets differ: reference-only={sorted(reference_rows - learned_rows)}, "
        f"learned-only={sorted(learned_rows - reference_rows)}"
    )
    node_order = reference.index.astype(str).tolist()
    aligned_reference = reference.copy()
    aligned_reference.index = aligned_reference.index.astype(str)
    aligned_reference.columns = aligned_reference.columns.astype(str)
    aligned_learned = learned.copy()
    aligned_learned.index = aligned_learned.index.astype(str)
    aligned_learned.columns = aligned_learned.columns.astype(str)
    aligned_reference = aligned_reference.loc[node_order, node_order]
    aligned_learned = aligned_learned.loc[node_order, node_order]
    assert list(aligned_reference.index) == list(aligned_learned.index)
    assert list(aligned_reference.columns) == list(aligned_learned.columns)
    return aligned_reference, aligned_learned


def evaluate_graph(reference: pd.DataFrame, learned: pd.DataFrame) -> dict[str, float | int]:
    """Evaluate aligned directed binary adjacency entries off the diagonal.

    SHD uses the standard graph-edit definition: an addition, deletion, or edge
    reversal each costs one. Precision, recall, and F1 treat each ordered directed
    edge as one item. Correlation is Pearson correlation over the same entries.
    """
    reference, learned = align_adjacencies(reference, learned)
    reference_values = reference.to_numpy(dtype=np.int64)
    learned_values = learned.to_numpy(dtype=np.int64)
    mask = ~np.eye(len(reference), dtype=bool)
    truth = reference_values[mask]
    prediction = learned_values[mask]
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if np.array_equal(truth, prediction):
        correlation = 1.0
    elif np.std(truth) == 0 or np.std(prediction) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(truth, prediction)[0, 1])
    shd = 0
    for i in range(len(reference)):
        for j in range(i + 1, len(reference)):
            reference_edge = (int(reference_values[i, j]), int(reference_values[j, i]))
            learned_edge = (int(learned_values[i, j]), int(learned_values[j, i]))
            if reference_edge != learned_edge:
                shd += 1
    return {
        "SHD": shd,
        "correlation": correlation,
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1),
    }


def _save_named_matrix(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=True, index_label="node")


def run_graph_conditions(
    graph_datasets: Mapping[str, pd.DataFrame],
    reference: pd.DataFrame,
    output_dir: Path,
    *,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, PCConfig]:
    """Run the fixed pgmpy PC configuration for every graph condition."""
    node_names = reference.index.astype(str).tolist()
    expected_conditions = ["oracle", "FS_5", "FS_10", "FS_20", "FS_100"]
    if set(graph_datasets) != set(expected_conditions):
        raise ValueError(
            f"Expected graph conditions {expected_conditions}; found {list(graph_datasets)}."
        )

    pc_config = PCConfig()
    (output_dir / "pc_learn_config.json").write_text(
        json.dumps(asdict(pc_config), indent=2), encoding="utf-8"
    )
    condition_suffixes = {
        "oracle": "oracle",
        **{CONDITION_NAMES[fraction]: FILE_SUFFIXES[fraction] for fraction in SUPERVISION_FRACTIONS},
    }
    graph_rows: list[dict[str, Any]] = []
    for condition in expected_conditions:
        adjacency, edges, bayesian_network = run_pc_learn(
            graph_datasets[condition],
            node_names,
            pc_config,
            show_progress=show_progress,
        )
        aligned_reference, adjacency = align_adjacencies(reference, adjacency)
        assert list(aligned_reference.index) == list(adjacency.index)
        suffix = condition_suffixes[condition]
        edges.to_csv(output_dir / f"graph_edges_{suffix}.csv", index=False)
        _save_named_matrix(adjacency, output_dir / f"graph_adjacency_{suffix}.csv")
        pdag_conversion = bayesian_network.graph["pc_pdag_conversion"]
        graph_rows.append(
            {
                "condition": condition,
                **evaluate_graph(reference, adjacency),
                "algorithm": pc_config.algorithm,
                "conditional_independence_test": pc_config.conditional_independence_test,
                "alpha": pc_config.alpha,
                "stable": pc_config.stable,
                "pc_return_type": pc_config.pc_return_type,
                "return_type": pc_config.return_type,
                "max_k": "None",
                "effective_max_cond_vars": len(node_names) - 2,
                "dag_completion_method": pdag_conversion["method"],
                "pdag_directed_edges": pdag_conversion["pdag_directed_edges"],
                "pdag_undirected_edges": pdag_conversion["pdag_undirected_edges"],
                "dag_completion_cycle_breaks": pdag_conversion["cycle_breaks"],
                "reversed_directed_preferences": pdag_conversion[
                    "reversed_directed_preferences"
                ],
            }
        )
    graph_metrics = pd.DataFrame(graph_rows)
    graph_metrics.to_csv(output_dir / "graph_metrics.csv", index=False)
    return graph_metrics, pc_config


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the complete classifier OOF and graph evaluation workflow."""
    from modern_symptom_classifier import ClassifierConfig

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_ids, X, y, label_names = load_embedding_arrays(
        args.embedding_npz, expected_patients=10_000
    )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable.")

    # Validate every downstream input before starting the expensive nested-CV run.
    original = load_structured_dataset(
        args.structured_data, args.patient_id_column, args.structured_separator
    )
    if len(original) != len(patient_ids) or set(original["patient_id"]) != set(patient_ids):
        raise ValueError("Structured SynSUM and embedding patient-ID sets must match exactly.")
    reference = load_reference_adjacency(args.reference_adjacency)
    node_names = reference.index.astype(str).tolist()
    missing_symptoms = [name for name in LABEL_NAMES if name not in node_names]
    if missing_symptoms:
        raise ValueError(
            f"Reference graph must contain all five symptom nodes: missing {missing_symptoms}"
        )
    missing_graph_columns = [name for name in node_names if name not in original.columns]
    if missing_graph_columns:
        raise ValueError(
            "Reference graph node names must exactly match SynSUM columns; missing "
            f"columns: {missing_graph_columns}"
        )

    folds = create_outer_folds(patient_ids, y, seed=args.seed, n_splits=5)
    if len(patient_ids) == 10_000:
        assert all(len(fold.train_indices) == 8_000 for fold in folds)
        assert all(len(fold.test_indices) == 2_000 for fold in folds)
    nested_subsets = {
        fold.fold_id: create_nested_training_subsets(
            fold.train_indices, y, seed=args.seed * 100 + fold.fold_id
        )
        for fold in folds
    }
    if len(patient_ids) == 10_000:
        expected_sizes = {0.05: 400, 0.10: 800, 0.20: 1_600, 1.00: 8_000}
        for subsets in nested_subsets.values():
            assert {key: len(value) for key, value in subsets.items()} == expected_sizes

    fold_assignments = _fold_assignment_frame(patient_ids, folds)
    fold_assignments.to_csv(output_dir / "outer_fold_assignments.csv", index=False)
    subset_rows = []
    for fold in folds:
        for fraction, indices in nested_subsets[fold.fold_id].items():
            subset_rows.extend(
                {
                    "outer_fold": fold.fold_id,
                    "condition": CONDITION_NAMES[fraction],
                    "training_pool_fraction": fraction,
                    "patient_id": int(patient_id),
                }
                for patient_id in patient_ids[indices]
            )
    pd.DataFrame(subset_rows).to_csv(
        output_dir / "nested_training_subset_membership.csv", index=False
    )

    classifier_config = ClassifierConfig()
    oof_tables, inner_rows = generate_oof_predictions(
        patient_ids,
        X,
        y,
        label_names,
        folds,
        nested_subsets,
        classifier_config=classifier_config,
        device=device,
        base_seed=args.seed,
        checkpoint_dir=checkpoint_dir,
    )
    validate_oof_integrity(patient_ids, folds, nested_subsets, oof_tables)
    save_oof_predictions(oof_tables, output_dir, label_names)
    pd.DataFrame(inner_rows).to_csv(
        output_dir / "classifier_inner_cv_metrics.csv", index=False
    )
    fold_metrics, oof_metrics, symptom_metrics = calculate_classifier_outputs(
        oof_tables, label_names
    )
    fold_metrics.to_csv(output_dir / "classifier_fold_metrics.csv", index=False)
    oof_metrics.to_csv(output_dir / "classifier_oof_metrics.csv", index=False)
    symptom_metrics.to_csv(
        output_dir / "classifier_per_symptom_metrics.csv", index=False
    )

    graph_datasets: dict[str, pd.DataFrame] = {"oracle": original}
    for fraction, table in oof_tables.items():
        suffix = FILE_SUFFIXES[fraction]
        dataset = build_graph_dataset(original, table, LABEL_NAMES)
        dataset.to_csv(output_dir / f"graph_dataset_{suffix}.csv", index=False)
        graph_datasets[CONDITION_NAMES[fraction]] = dataset

    _, pc_config = run_graph_conditions(
        graph_datasets,
        reference,
        output_dir,
        show_progress=args.show_pc_progress,
    )

    metadata = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "strict patient-level 5-fold OOF symptom-to-causal-graph experiment",
        "patients": len(patient_ids),
        "outer_folds": 5,
        "outer_fold_assignment_sha256": _sha256(output_dir / "outer_fold_assignments.csv"),
        "supervision_fractions_of_outer_training_pool": list(SUPERVISION_FRACTIONS),
        "classifier": classifier_config.to_dict(),
        "classifier_source": "modern_symptom_classifier.py factored from colab_embedding_and_training.ipynb",
        "fever_definition": "classifier target 0=no fever; 1=original low or high fever",
        "oof_threshold_operator": ">=",
        "graph_discovery": asdict(pc_config),
        "graph_conditions_share_exact_config": True,
        "graph_input": "all graph nodes independently factorized to sorted discrete category codes",
        "graph_metric_definition": {
            "adjacency": "row=source, column=target; pgmpy PC PDAG deterministically extended to an acyclic DiscreteBayesianNetwork",
            "SHD": "standard add/delete/reverse edit count; reversal costs one",
            "correlation": "Pearson correlation of off-diagonal binary adjacency entries",
            "precision_recall_F1": "ordered directed-edge entries",
            "alignment": "learned matrix explicitly reordered by reference node names before metrics",
        },
        "input_files": {
            "embedding_npz": str(Path(args.embedding_npz).resolve()),
            "embedding_npz_sha256": _sha256(Path(args.embedding_npz)),
            "structured_data": str(Path(args.structured_data).resolve()),
            "structured_data_sha256": _sha256(Path(args.structured_data)),
            "reference_adjacency": str(Path(args.reference_adjacency).resolve()),
            "reference_adjacency_sha256": _sha256(Path(args.reference_adjacency)),
        },
        "node_order": node_names,
        "seed": args.seed,
        "device": str(device),
        "runtime": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit_learn": _package_version("scikit-learn"),
            "torch": _package_version("torch"),
            "iterative_stratification": _package_version("iterative-stratification"),
            "pgmpy": _package_version("pgmpy"),
            "cuda": torch.version.cuda,
        },
    }
    metadata_path = output_dir / "experiment_metadata.json"
    metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")
    return metadata


def run_graph_only(args: argparse.Namespace) -> dict[str, Any]:
    """Recompute graph artifacts from saved OOF predictions without retraining."""
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"OOF output directory not found: {output_dir}")
    original = load_structured_dataset(
        args.structured_data, args.patient_id_column, args.structured_separator
    )
    reference = load_reference_adjacency(args.reference_adjacency)
    node_names = reference.index.astype(str).tolist()
    missing_graph_columns = [name for name in node_names if name not in original.columns]
    if missing_graph_columns:
        raise ValueError(
            "Reference graph node names must exactly match structured-data columns; "
            f"missing: {missing_graph_columns}"
        )

    graph_datasets: dict[str, pd.DataFrame] = {"oracle": original}
    for fraction in SUPERVISION_FRACTIONS:
        suffix = FILE_SUFFIXES[fraction]
        oof_path = output_dir / f"oof_predictions_{suffix}.csv"
        if not oof_path.is_file():
            raise FileNotFoundError(f"Saved OOF predictions not found: {oof_path}")
        oof_table = pd.read_csv(oof_path)
        graph_datasets[CONDITION_NAMES[fraction]] = build_graph_dataset(
            original, oof_table, LABEL_NAMES
        )

    graph_metrics, pc_config = run_graph_conditions(
        graph_datasets,
        reference,
        output_dir,
        show_progress=args.show_pc_progress,
    )
    metadata_path = output_dir / "experiment_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    else:
        metadata = {}
    runtime = dict(metadata.get("runtime", {}))
    runtime.pop("causal_learn", None)
    runtime["pgmpy"] = _package_version("pgmpy")
    metadata.update(
        {
            "status": "completed",
            "graph_recomputed_at_utc": datetime.now(timezone.utc).isoformat(),
            "graph_recomputed_without_classifier_retraining": True,
            "graph_discovery": asdict(pc_config),
            "graph_conditions_share_exact_config": True,
            "graph_metric_definition": {
                "adjacency": "row=source, column=target; pgmpy PC PDAG deterministically extended to an acyclic DiscreteBayesianNetwork",
                "SHD": "standard add/delete/reverse edit count; reversal costs one",
                "correlation": "Pearson correlation of off-diagonal binary adjacency entries",
                "precision_recall_F1": "ordered directed-edge entries",
                "alignment": "learned matrix explicitly reordered by reference node names before metrics",
            },
            "runtime": runtime,
        }
    )
    metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")
    return {
        "status": "completed",
        "mode": "graph-only",
        "conditions": graph_metrics["condition"].tolist(),
        "graph_metrics": str(output_dir / "graph_metrics.csv"),
        "graph_discovery": asdict(pc_config),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-npz",
        default="outputs/supervised_embeddings/patient_embeddings_and_labels.npz",
    )
    parser.add_argument("--structured-data", default="SynSUM.csv")
    parser.add_argument("--reference-adjacency", required=True)
    parser.add_argument("--output-dir", default="outputs/oof_graph_experiment")
    parser.add_argument("--patient-id-column", default="Unnamed: 0")
    parser.add_argument("--structured-separator", default=";")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--show-pc-progress", action="store_true")
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Reuse saved OOF prediction CSVs and recompute only pgmpy graph artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = run_graph_only(args) if args.graph_only else run_experiment(args)
    print(json.dumps(_json_safe(metadata), indent=2))


if __name__ == "__main__":
    main()
