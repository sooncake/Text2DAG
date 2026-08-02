#!/usr/bin/env python3
"""Faithful reproduction of the legacy cached-ModernBERT symptom classifier.

This module intentionally retains several unconventional behaviors from
``symtoms_classifier_mean.ipynb``.  It is isolated from the modern pipeline so
that fixing or improving those behaviors cannot silently change reproduction
results.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


LABEL_NAMES: Final[list[str]] = ["dysp", "cough", "pain", "fever", "nasal"]
EMBEDDING_DIM: Final = 768
LEGACY_REPLACEMENTS: Final[dict[str, int]] = {
    "yes": 1,
    "no": 0,
    "none": 0,
    "low": 1,
    "high": 2,
    "winter": 1,
    "summer": 0,
}
LEGACY_CONFIG: Final[dict[str, Any]] = {
    "seed": 5,
    "outer_test_fraction": 0.20,
    "training_subset_fraction": 0.20,
    "validation_fraction": 0.10,
    "max_token_length": 128,
    "embedding_batch_size": 32,
    "head_dim": 256,
    "batch_size": 32,
    "learning_rate": 3e-5,
    "maximum_epochs": 120,
    "early_stopping_patience": 5,
    "prediction_threshold": 0.5,
    "threshold_operator": ">",
    "loss": "BCEWithLogitsLoss(unweighted)",
    "optimizer": "torch.optim.AdamW(lr=3e-5; default weight_decay)",
    "feature_standardization": False,
    "sentence_pooling": "unmasked mean including padded sentence slots",
}


def seed_everything(seed: int) -> None:
    """Set every seed requested for the reproduction run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def legacy_get_sentence(documents: Sequence[str]) -> list[list[str]]:
    """Copy the legacy notebook's list-of-documents sentence cleaner."""
    if not documents:
        return []
    all_cleaned_sentences: list[list[str]] = []
    for paragraph in documents:
        # These rules, including the literal '. ' split, are preserved exactly.
        lines = re.split(r"\n|\s*\*\*History\*\*\s*", paragraph)
        cleaned_lines = [
            line
            for line in lines
            if line.strip()
            and not line.startswith("**History**")
            and not line.startswith("**Physical Examination**")
        ]
        document = " ".join(cleaned_lines)
        cleaned_sentence: list[str] = []
        for sentence in document.split(". "):
            normalized = sentence.lower().strip()
            normalized = re.sub(r"[^\w\s]", "", normalized)
            normalized = re.sub(r"\s+", " ", normalized).strip()
            if normalized:
                cleaned_sentence.append(normalized)
        all_cleaned_sentences.append(cleaned_sentence)
    return all_cleaned_sentences


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_legacy_source(source_path: str | Path) -> dict[str, Any]:
    """Load text and labels using the exact legacy replacement path."""
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Legacy source file not found: {source_path}")
    frame = pd.read_csv(source_path, delimiter=";", index_col=0)
    missing = [name for name in ["text", *LABEL_NAMES] if name not in frame.columns]
    if missing:
        raise ValueError(f"SynSUM source is missing columns: {missing}")
    if frame["text"].isna().any() or not frame["text"].map(lambda x: isinstance(x, str)).all():
        raise ValueError(
            "Legacy get_sentence expects every text value to be a non-missing string. "
            "The source violates that legacy assumption."
        )

    # The broad DataFrame replacement is intentional: it matches the notebook.
    frame.replace(LEGACY_REPLACEMENTS, inplace=True)
    raw_labels = frame[LABEL_NAMES].to_numpy()
    try:
        labels = raw_labels.astype(np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Legacy label replacement left a non-numeric target. Do not normalize or "
            "silently remap it in reproduction mode."
        ) from exc
    texts = legacy_get_sentence(frame["text"].tolist())
    if any(len(sentences) == 0 for sentences in texts):
        empty_rows = [i for i, sentences in enumerate(texts) if not sentences][:20]
        raise ValueError(
            "The legacy tokenizer cannot batch a patient with zero cleaned sentences; "
            f"affected row IDs include {empty_rows}."
        )

    row_ids = np.arange(len(frame), dtype=np.int64)
    source_ids = frame.index.to_numpy(copy=True)
    return {
        "frame": frame,
        "texts": texts,
        "labels": labels,
        "patient_ids": row_ids,
        "source_patient_ids": source_ids,
        "source_sha256": _sha256(source_path),
    }


class LegacyMultiLabelDataset(Dataset):
    """Legacy per-patient, per-sentence tokenization contract."""

    def __init__(
        self,
        texts: Sequence[list[str]],
        labels: np.ndarray,
        patient_ids: np.ndarray,
        source_patient_ids: np.ndarray,
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.patient_ids = np.asarray(patient_ids, dtype=np.int64)
        self.source_patient_ids = np.asarray(source_patient_ids)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(self.labels[index], dtype=torch.float32),
            "patient_id": int(self.patient_ids[index]),
            "source_patient_id": self.source_patient_ids[index],
        }


def legacy_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pad sentence slots with zero IDs/masks exactly as the old notebook did."""
    max_n = max(item["input_ids"].size(0) for item in batch)
    ids_list, masks_list = [], []
    for item in batch:
        n_sentences, token_length = item["input_ids"].shape
        pad_length = max_n - n_sentences
        if pad_length:
            # Intentionally not tokenizer.pad_token_id: the legacy notebook used 0.
            pad_ids = torch.zeros((pad_length, token_length), dtype=torch.long)
            pad_mask = torch.zeros((pad_length, token_length), dtype=torch.long)
            input_ids = torch.cat([item["input_ids"], pad_ids], dim=0)
            attention_mask = torch.cat([item["attention_mask"], pad_mask], dim=0)
        else:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
        ids_list.append(input_ids)
        masks_list.append(attention_mask)
    return {
        "input_ids": torch.stack(ids_list),
        "attention_mask": torch.stack(masks_list),
        "labels": torch.stack([item["labels"] for item in batch]),
        "patient_ids": np.asarray([item["patient_id"] for item in batch], dtype=np.int64),
        "source_patient_ids": np.asarray(
            [item["source_patient_id"] for item in batch]
        ),
    }


class LegacyEncoderModel(nn.Module):
    """Legacy encoder wrapper, including unused linears that consume RNG state."""

    def __init__(self, pretrained_name: str, num_labels: int = 5, embed_dim: int = 256):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(pretrained_name)
        self.projector = nn.Linear(self.encoder.config.hidden_size, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_labels)

    def encode_sentences(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, sentence_count, token_length = input_ids.shape
        outputs = self.encoder(
            input_ids=input_ids.view(-1, token_length),
            attention_mask=attention_mask.view(-1, token_length),
        )
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0, :]
        return pooled.view(batch_size, sentence_count, -1)

    @staticmethod
    def pool_sentences(sentence_embeddings: torch.Tensor) -> torch.Tensor:
        # REPRODUCTION QUIRK: padded sentence slots deliberately influence the mean.
        return sentence_embeddings.mean(dim=1)


class LegacyHeadOnlyModel(nn.Module):
    """Two-layer linear classification head from the legacy notebook."""

    def __init__(
        self,
        input_dim: int,
        num_labels: int = 5,
        head_dim: int = 256,
    ) -> None:
        super().__init__()
        self.projector = nn.Linear(input_dim, head_dim)
        self.classifier = nn.Linear(head_dim, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.projector(features)
        logits = self.classifier(hidden)
        return logits


def corrected_masked_sentence_mean(
    sentence_embeddings: torch.Tensor, sentence_mask: torch.Tensor
) -> torch.Tensor:
    """Optional corrected pooling; never called by legacy reproduction mode."""
    weights = sentence_mask.to(sentence_embeddings.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (sentence_embeddings * weights).sum(dim=1) / denominator


def _subset_loader(
    dataset: Dataset,
    ratio: float,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, np.ndarray]:
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    selected = np.asarray(indices[: int(len(dataset) * ratio)], dtype=np.int64)
    loader = DataLoader(
        Subset(dataset, selected.tolist()),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=legacy_collate_fn,
    )
    return loader, selected


def _cache_scope(
    model: LegacyEncoderModel,
    loader: DataLoader,
    device: torch.device,
    prefix: Path,
) -> dict[str, np.ndarray]:
    from tqdm.auto import tqdm

    model.to(device).eval()
    embeddings, labels, patient_ids, source_ids = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Caching {prefix.name}"):
            sentence_embeddings = model.encode_sentences(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            pooled = model.pool_sentences(sentence_embeddings)
            embeddings.append(pooled.cpu())
            labels.append(batch["labels"])
            patient_ids.append(batch["patient_ids"])
            source_ids.append(batch["source_patient_ids"])
    arrays = {
        "X": torch.cat(embeddings).numpy().astype(np.float32, copy=False),
        "y": torch.cat(labels).numpy().astype(np.float32, copy=False),
        "patient_ids": np.concatenate(patient_ids).astype(np.int64, copy=False),
        "source_patient_ids": np.concatenate(source_ids),
    }
    if arrays["X"].ndim != 2 or arrays["X"].shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Expected cached shape (N, 768), found {arrays['X'].shape}")
    np.save(prefix.with_name(prefix.name + "_patient_embeddings.npy"), arrays["X"])
    np.save(prefix.with_name(prefix.name + "_patient_embeddings_labels.npy"), arrays["y"])
    np.save(prefix.with_name(prefix.name + "_patient_ids.npy"), arrays["patient_ids"])
    np.save(prefix.with_name(prefix.name + "_source_patient_ids.npy"), arrays["source_patient_ids"])
    return arrays


def _embedding_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "train_X": output_dir / "legacy_train_subset_patient_embeddings.npy",
        "train_y": output_dir / "legacy_train_subset_patient_embeddings_labels.npy",
        "train_ids": output_dir / "legacy_train_subset_patient_ids.npy",
        "train_source_ids": output_dir / "legacy_train_subset_source_patient_ids.npy",
        "selection_ids": output_dir / "legacy_training_subset_selection_patient_ids.npy",
        "selection_source_ids": output_dir / "legacy_training_subset_selection_source_patient_ids.npy",
        "test_X": output_dir / "legacy_test_patient_embeddings.npy",
        "test_y": output_dir / "legacy_test_patient_embeddings_labels.npy",
        "test_ids": output_dir / "legacy_test_patient_ids.npy",
        "test_source_ids": output_dir / "legacy_test_source_patient_ids.npy",
        "all_X": output_dir / "legacy_patient_embeddings.npy",
        "all_y": output_dir / "legacy_patient_embeddings_labels.npy",
        "all_ids": output_dir / "legacy_patient_ids.npy",
        "all_source_ids": output_dir / "legacy_source_patient_ids.npy",
        "rng": output_dir / "legacy_rng_state_after_embedding.pt",
        "metadata": output_dir / "legacy_embedding_metadata.json",
    }


def _save_rng_state(path: Path) -> None:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def _restore_rng_state(path: Path) -> None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was added.
        payload = torch.load(path, map_location="cpu")
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in payload:
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


def _package_versions() -> dict[str, str | None]:
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "tqdm",
    ]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def generate_or_load_embeddings(
    source: dict[str, Any],
    output_dir: Path,
    model_name: str,
    device: torch.device,
    force_recompute: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Generate the three legacy caches and preserve post-cache RNG state."""
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _embedding_paths(output_dir)
    reusable = all(path.is_file() for path in paths.values()) and not force_recompute
    if reusable:
        previous = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if previous.get("source_sha256") != source["source_sha256"]:
            raise ValueError(
                "Existing legacy embeddings were generated from a different SynSUM "
                "file. Set FORCE_RECOMPUTE=True instead of reusing them."
            )
        if previous.get("model_name") != model_name:
            raise ValueError(
                "Existing legacy embeddings use a different model name. Set "
                "FORCE_RECOMPUTE=True instead of reusing them."
            )
        current_code_sha256 = _sha256(Path(__file__))
        if previous.get("reproduction_code_sha256") != current_code_sha256:
            raise ValueError(
                "The legacy reproduction code changed after these embeddings were "
                "created. Set FORCE_RECOMPUTE=True to avoid stale cache reuse."
            )
        scopes = {
            name: {
                "X": np.load(paths[f"{name}_X"], allow_pickle=False),
                "y": np.load(paths[f"{name}_y"], allow_pickle=False),
                "patient_ids": np.load(paths[f"{name}_ids"], allow_pickle=False),
                "source_patient_ids": np.load(
                    paths[f"{name}_source_ids"], allow_pickle=True
                ),
            }
            for name in ("train", "test", "all")
        }
        scopes["train"]["selection_patient_ids"] = np.load(
            paths["selection_ids"], allow_pickle=False
        )
        scopes["train"]["selection_source_patient_ids"] = np.load(
            paths["selection_source_ids"], allow_pickle=True
        )
        _restore_rng_state(paths["rng"])
        print(f"Reusing legacy-compatible embeddings from {output_dir}")
        return scopes, previous

    seed_everything(int(LEGACY_CONFIG["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder_model = LegacyEncoderModel(model_name)
    all_positions = np.arange(len(source["patient_ids"]))
    train_positions, test_positions = train_test_split(
        all_positions,
        test_size=LEGACY_CONFIG["outer_test_fraction"],
        random_state=LEGACY_CONFIG["seed"],
    )

    def make_dataset(positions: np.ndarray) -> LegacyMultiLabelDataset:
        return LegacyMultiLabelDataset(
            [source["texts"][int(i)] for i in positions],
            source["labels"][positions],
            source["patient_ids"][positions],
            source["source_patient_ids"][positions],
            tokenizer,
            max_length=int(LEGACY_CONFIG["max_token_length"]),
        )

    train_dataset = make_dataset(train_positions)
    train_loader, subset_local_positions = _subset_loader(
        train_dataset,
        ratio=float(LEGACY_CONFIG["training_subset_fraction"]),
        batch_size=int(LEGACY_CONFIG["embedding_batch_size"]),
        seed=int(LEGACY_CONFIG["seed"]),
    )
    test_loader = DataLoader(
        make_dataset(test_positions),
        batch_size=int(LEGACY_CONFIG["embedding_batch_size"]),
        shuffle=False,
        collate_fn=legacy_collate_fn,
    )
    all_loader = DataLoader(
        make_dataset(all_positions),
        batch_size=int(LEGACY_CONFIG["embedding_batch_size"]),
        shuffle=False,
        collate_fn=legacy_collate_fn,
    )

    scopes = {
        "train": _cache_scope(encoder_model, train_loader, device, output_dir / "legacy_train_subset"),
        "test": _cache_scope(encoder_model, test_loader, device, output_dir / "legacy_test"),
        "all": _cache_scope(encoder_model, all_loader, device, output_dir / "legacy"),
    }
    scopes["train"]["selection_patient_ids"] = source["patient_ids"][
        train_positions[subset_local_positions]
    ]
    scopes["train"]["selection_source_patient_ids"] = source["source_patient_ids"][
        train_positions[subset_local_positions]
    ]
    np.save(paths["selection_ids"], scopes["train"]["selection_patient_ids"])
    np.save(
        paths["selection_source_ids"],
        scopes["train"]["selection_source_patient_ids"],
    )
    _save_rng_state(paths["rng"])
    resolved_model_revision = getattr(encoder_model.encoder.config, "_commit_hash", None)
    resolved_tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash")
    metadata = {
        "cache_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source["source_sha256"],
        "reproduction_code_sha256": _sha256(Path(__file__)),
        "model_name": model_name,
        "requested_revision": None,
        "resolved_model_revision": resolved_model_revision,
        "resolved_tokenizer_revision": resolved_tokenizer_revision,
        "token_pooling": "pooler_output when available, otherwise last_hidden_state[:, 0, :]",
        "patient_pooling": LEGACY_CONFIG["sentence_pooling"],
        "train_pool_positions_first_five": train_positions[:5].tolist(),
        "test_positions_first_five": test_positions[:5].tolist(),
        "subset_local_positions_first_five": subset_local_positions[:5].tolist(),
        "package_versions": _package_versions(),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Generated separate legacy-compatible embeddings in {output_dir}")
    return scopes, metadata


def _label_counts(labels: np.ndarray) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for index, name in enumerate(LABEL_NAMES):
        values, frequencies = np.unique(labels[:, index], return_counts=True)
        counts[name] = {
            str(int(value) if float(value).is_integer() else float(value)): int(frequency)
            for value, frequency in zip(values, frequencies)
        }
    return counts


def _train_legacy_head(
    arrays: dict[str, np.ndarray],
    device: torch.device,
    checkpoint_dir: Path,
) -> tuple[LegacyHeadOnlyModel, dict[str, Any], dict[str, np.ndarray]]:
    from sklearn.model_selection import train_test_split

    X = arrays["X"].astype(np.float32, copy=False)
    y = arrays["y"].astype(np.float32, copy=False)
    patient_ids = arrays["patient_ids"].astype(np.int64, copy=False)
    source_ids = arrays["source_patient_ids"]
    assert X.ndim == 2
    assert X.shape[1] == EMBEDDING_DIM
    assert y.ndim == 2
    assert y.shape[1] == len(LABEL_NAMES)
    if 2 in np.unique(y[:, 3]):
        warnings.warn(
            "LEGACY REPRODUCTION: fever target value 2 is being passed unchanged to "
            "BCEWithLogitsLoss. This is intentionally not corrected.",
            RuntimeWarning,
        )

    positions = np.arange(len(X))
    train_positions, validation_positions = train_test_split(
        positions,
        test_size=LEGACY_CONFIG["validation_fraction"],
        random_state=LEGACY_CONFIG["seed"],
    )
    train_dataset = TensorDataset(
        torch.tensor(X[train_positions], dtype=torch.float32),
        torch.tensor(y[train_positions], dtype=torch.float32),
    )
    validation_dataset = TensorDataset(
        torch.tensor(X[validation_positions], dtype=torch.float32),
        torch.tensor(y[validation_positions], dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=int(LEGACY_CONFIG["batch_size"]), shuffle=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=int(LEGACY_CONFIG["batch_size"]), shuffle=False
    )

    model = LegacyHeadOnlyModel(
        input_dim=X.shape[1],
        num_labels=len(LABEL_NAMES),
        head_dim=int(LEGACY_CONFIG["head_dim"]),
    ).to(device)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_parameters == 198_149, trainable_parameters
    dummy = torch.zeros(2, EMBEDDING_DIM, device=device)
    assert model(dummy).shape == (2, len(LABEL_NAMES))

    # Do not pass weight_decay: the legacy call used AdamW's PyTorch default.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(LEGACY_CONFIG["learning_rate"])
    )
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    patience_counter = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(LEGACY_CONFIG["maximum_epochs"]) + 1):
        model.train()
        train_losses = []
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for features, targets in validation_loader:
                features, targets = features.to(device), targets.to(device)
                validation_losses.append(float(criterion(model(features), targets).item()))
        average_validation_loss = float(np.mean(validation_losses))
        history.append(
            {
                "epoch": epoch,
                "training_loss_mean_of_batches": float(np.mean(train_losses)),
                "validation_loss_mean_of_batches": average_validation_loss,
            }
        )
        print(f"Epoch {epoch:03d}: validation loss={average_validation_loss:.6f}")
        if abs(average_validation_loss) < abs(best_loss):
            best_loss = average_validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= int(LEGACY_CONFIG["early_stopping_patience"]):
                print(f"Early stopping at epoch {epoch}")
                break
    if best_state is None or best_epoch is None:
        raise RuntimeError("Legacy training did not produce a best checkpoint.")
    model.load_state_dict(best_state)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "legacy_best_validation_loss_model.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "config": LEGACY_CONFIG,
            "label_names": LABEL_NAMES,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
        },
        checkpoint_path,
    )
    pd.DataFrame(history).to_csv(checkpoint_dir / "legacy_training_history.csv", index=False)
    training_metadata = {
        "best_epoch": best_epoch,
        "epochs_executed": len(history),
        "best_validation_loss": best_loss,
        "trainable_parameters": trainable_parameters,
        "model_architecture": str(model),
        "optimizer_defaults": {
            key: value
            for key, value in optimizer.defaults.items()
            if isinstance(value, (str, int, float, bool, type(None), tuple))
        },
        "loss_configuration": str(criterion),
        "checkpoint": str(checkpoint_path),
    }
    split_arrays = {
        "training_patient_ids": patient_ids[train_positions],
        "validation_patient_ids": patient_ids[validation_positions],
        "training_source_patient_ids": source_ids[train_positions],
        "validation_source_patient_ids": source_ids[validation_positions],
    }
    return model, training_metadata, split_arrays


def _metric_rows(
    targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
) -> list[dict[str, Any]]:
    from sklearn.metrics import precision_recall_fscore_support

    rows: list[dict[str, Any]] = []
    for index, name in enumerate(LABEL_NAMES):
        raw_truth = targets[:, index]
        binary_truth = (raw_truth > 0).astype(np.int64)  # Evaluation diagnostic only.
        binary_prediction = predictions[:, index].astype(np.int64)
        tp = int(np.sum((binary_truth == 1) & (binary_prediction == 1)))
        tn = int(np.sum((binary_truth == 0) & (binary_prediction == 0)))
        fp = int(np.sum((binary_truth == 0) & (binary_prediction == 1)))
        fn = int(np.sum((binary_truth == 1) & (binary_prediction == 0)))
        precision, recall, f1, _ = precision_recall_fscore_support(
            binary_truth,
            binary_prediction,
            average="binary",
            zero_division=0,
        )
        rows.append(
            {
                "label": name,
                "accuracy": float(np.mean(binary_truth == binary_prediction)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "legacy_raw_target_accuracy": float(np.mean(raw_truth == binary_prediction)),
                "target_value_2_count": int(np.sum(raw_truth == 2)),
                "mean_predicted_probability": float(probabilities[:, index].mean()),
                "metric_target_note": (
                    "positive-presence target (raw target > 0); training targets remain raw"
                ),
            }
        )
    return rows


def _evaluate_and_save(
    model: LegacyHeadOnlyModel,
    arrays: dict[str, np.ndarray],
    device: torch.device,
    output_dir: Path,
    scope: str,
) -> dict[str, Any]:
    X = arrays["X"].astype(np.float32, copy=False)
    targets = arrays["y"].astype(np.float32, copy=False)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        probabilities = torch.sigmoid(logits).cpu().numpy()
    predictions = (
        probabilities > float(LEGACY_CONFIG["prediction_threshold"])
    ).astype(np.int64)
    legacy_exact_match = float(np.mean(np.all(predictions == targets, axis=1)))
    positive_targets = (targets > 0).astype(np.int64)
    presence_exact_match = float(np.mean(np.all(predictions == positive_targets, axis=1)))
    rows = _metric_rows(targets, predictions, probabilities)
    metrics_path = output_dir / f"legacy_{scope}_per_symptom_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    prediction_frame = pd.DataFrame(
        {
            "patient_id": arrays["patient_ids"],
            "source_patient_id": arrays["source_patient_ids"],
        }
    )
    for index, name in enumerate(LABEL_NAMES):
        prediction_frame[f"{name}_target"] = targets[:, index]
        prediction_frame[f"{name}_prediction"] = predictions[:, index]
        prediction_frame[f"{name}_probability"] = probabilities[:, index]
    csv_path = output_dir / f"legacy_{scope}_predictions.csv"
    npz_path = output_dir / f"legacy_{scope}_predictions.npz"
    prediction_frame.to_csv(csv_path, index=False)
    np.savez_compressed(
        npz_path,
        patient_ids=arrays["patient_ids"],
        source_patient_ids=arrays["source_patient_ids"],
        y_true=targets,
        y_pred=predictions,
        probabilities=probabilities.astype(np.float32),
        label_names=np.asarray(LABEL_NAMES),
    )
    return {
        "scope": scope,
        "patients": len(X),
        "legacy_raw_exact_match_accuracy": legacy_exact_match,
        "positive_presence_exact_match_accuracy": presence_exact_match,
        "legacy_raw_per_label_accuracy": {
            row["label"]: row["legacy_raw_target_accuracy"] for row in rows
        },
        "predictions_csv": str(csv_path),
        "predictions_npz": str(npz_path),
        "per_symptom_metrics_csv": str(metrics_path),
        "per_symptom_metrics": rows,
        "macro_f1": float(np.mean([row["f1"] for row in rows])),
    }


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


def run_legacy_reproduction(
    source_path: str | Path,
    embedding_dir: str | Path,
    checkpoint_dir: str | Path,
    results_dir: str | Path,
    model_name: str = "nomic-ai/modernbert-embed-base",
    device_name: str = "cuda",
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Run Workflow A from source text through both legacy evaluation scopes."""
    seed_everything(int(LEGACY_CONFIG["seed"]))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_name}, but CUDA is unavailable.")
    device = torch.device(device_name)
    embedding_dir, checkpoint_dir, results_dir = map(
        Path, (embedding_dir, checkpoint_dir, results_dir)
    )
    for directory in (embedding_dir, checkpoint_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source = load_legacy_source(source_path)
    label_counts = _label_counts(source["labels"])
    print(f"Patients: {len(source['patient_ids'])}")
    print(f"Label order: {LABEL_NAMES}")
    print("Unique label values/counts:", json.dumps(label_counts, indent=2))

    scopes, embedding_metadata = generate_or_load_embeddings(
        source,
        embedding_dir,
        model_name,
        device,
        force_recompute,
    )
    # The cached RNG state is restored on reuse; on generation it is already current.
    model, training_metadata, validation_ids = _train_legacy_head(
        scopes["train"], device, checkpoint_dir
    )

    from sklearn.model_selection import train_test_split

    all_ids = source["patient_ids"]
    outer_train, outer_test = train_test_split(
        all_ids,
        test_size=LEGACY_CONFIG["outer_test_fraction"],
        random_state=LEGACY_CONFIG["seed"],
    )
    split_payload = {
        "outer_train_patient_ids": outer_train,
        "outer_test_patient_ids": outer_test,
        "outer_train_source_patient_ids": source["source_patient_ids"][outer_train],
        "outer_test_source_patient_ids": source["source_patient_ids"][outer_test],
        "training_subset_selection_order_patient_ids": scopes["train"][
            "selection_patient_ids"
        ],
        "training_subset_selection_order_source_patient_ids": scopes["train"][
            "selection_source_patient_ids"
        ],
        "training_subset_cache_order_patient_ids": scopes["train"]["patient_ids"],
        "test_cache_order_patient_ids": scopes["test"]["patient_ids"],
        "all_cache_order_patient_ids": scopes["all"]["patient_ids"],
        **validation_ids,
    }
    split_path = results_dir / "legacy_split_patient_ids.npz"
    np.savez_compressed(split_path, **split_payload)

    print(f"Embedding shape: {scopes['train']['X'].shape}")
    print(
        f"Embedding mean/std: {scopes['train']['X'].mean():.8f} / "
        f"{scopes['train']['X'].std():.8f}"
    )
    print(f"Model architecture:\n{model}")
    print(f"Trainable parameters: {training_metadata['trainable_parameters']}")
    print(f"Optimizer: {training_metadata['optimizer_defaults']}")
    print(f"Loss: {training_metadata['loss_configuration']}")
    for key, ids in split_payload.items():
        print(f"{key}: n={len(ids)}, first five={np.asarray(ids)[:5].tolist()}")

    evaluations = {
        scope: _evaluate_and_save(model, scopes[scope], device, results_dir, scope)
        for scope in ("test", "all")
    }
    historical_references = {
        "test_exact_match_accuracy": 0.6535,
        "test_per_label_accuracy": [0.934, 0.9175, 0.9075, 0.828, 0.966],
        "all_exact_match_accuracy": 0.6459,
        "all_per_label_accuracy": [0.9424, 0.9121, 0.908, 0.8191, 0.9629],
        "row_level_reference_files_available": False,
    }
    comparison_status = {
        "data_rows": "not comparable: historical patient-ID file unavailable",
        "labels": "not comparable: historical target array unavailable",
        "patient_split": "reconstructed from code; historical split-ID file unavailable",
        "embeddings": "not comparable: historical embedding arrays unavailable",
        "model_configuration": "matches audited two-layer linear legacy code",
        "predictions": "not comparable: historical row-level predictions unavailable",
        "metrics": {
            "test_exact_match_rounded_4_decimals": (
                round(evaluations["test"]["legacy_raw_exact_match_accuracy"], 4)
                == historical_references["test_exact_match_accuracy"]
            ),
            "test_per_label_accuracy_rounded_4_decimals": bool(
                np.array_equal(
                    np.round(
                        list(
                            evaluations["test"][
                                "legacy_raw_per_label_accuracy"
                            ].values()
                        ),
                        4,
                    ),
                    np.asarray(historical_references["test_per_label_accuracy"]),
                )
            ),
            "all_exact_match_rounded_4_decimals": (
                round(evaluations["all"]["legacy_raw_exact_match_accuracy"], 4)
                == historical_references["all_exact_match_accuracy"]
            ),
            "all_per_label_accuracy_rounded_4_decimals": bool(
                np.array_equal(
                    np.round(
                        list(
                            evaluations["all"][
                                "legacy_raw_per_label_accuracy"
                            ].values()
                        ),
                        4,
                    ),
                    np.asarray(historical_references["all_per_label_accuracy"]),
                )
            ),
            "note": "Rounded aggregate matches do not establish row-level reproduction.",
        },
    }
    metadata = {
        "status": "completed",
        "workflow": "A: exact legacy reproduction",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(Path(source_path)),
        "source_sha256": source["source_sha256"],
        "legacy_config": LEGACY_CONFIG,
        "label_names": LABEL_NAMES,
        "label_counts": label_counts,
        "fever_training_values": sorted(np.unique(source["labels"][:, 3]).tolist()),
        "fever_warning": (
            "Raw fever value 2 is intentionally passed to BCEWithLogitsLoss."
            if 2 in np.unique(source["labels"][:, 3])
            else None
        ),
        "dataset_sizes": {
            "all": len(source["patient_ids"]),
            "outer_train": len(outer_train),
            "training_subset": len(scopes["train"]["X"]),
            "head_train": len(validation_ids["training_patient_ids"]),
            "validation": len(validation_ids["validation_patient_ids"]),
            "test": len(outer_test),
        },
        "embedding_statistics": {
            name: {
                "shape": list(arrays["X"].shape),
                "mean": float(arrays["X"].mean()),
                "standard_deviation": float(arrays["X"].std()),
            }
            for name, arrays in scopes.items()
        },
        "embedding_metadata": embedding_metadata,
        "training": training_metadata,
        "evaluations": evaluations,
        "split_patient_ids": str(split_path),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "packages": _package_versions(),
        },
        "historical_notebook_references": historical_references,
        "comparison_status": comparison_status,
        "reproduction_assessment": (
            "Compare patient IDs, raw targets, predictions, probabilities, and metrics. "
            "Similar aggregate accuracy alone is not sufficient."
        ),
    }
    metadata_path = results_dir / "legacy_reproduction_metadata.json"
    metadata_path.write_text(
        json.dumps(_json_safe(metadata), indent=2), encoding="utf-8"
    )
    report_path = results_dir / "legacy_reproduction_report.md"
    report_path.write_text(
        "# Legacy reproduction run\n\n"
        f"Status: completed. Metadata: `{metadata_path.name}`.\n\n"
        "The workflow intentionally retained unmasked padded-sentence averaging, "
        "raw ordinal fever targets in BCE, shuffled cache construction, the two-layer "
        "linear head, and both legacy evaluation scopes. Reproduction must be judged "
        "from the saved row-level arrays, not aggregate accuracy alone.\n\n"
        "## Comparison status\n\n"
        f"```json\n{json.dumps(_json_safe(comparison_status), indent=2)}\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(evaluations, indent=2))
    print(f"Metadata: {metadata_path}")
    return metadata


def _learning_curve_cache_paths(prefix: Path) -> dict[str, Path]:
    """Return the four files written by ``_cache_scope`` for one prefix."""
    return {
        "X": prefix.with_name(prefix.name + "_patient_embeddings.npy"),
        "y": prefix.with_name(prefix.name + "_patient_embeddings_labels.npy"),
        "patient_ids": prefix.with_name(prefix.name + "_patient_ids.npy"),
        "source_patient_ids": prefix.with_name(
            prefix.name + "_source_patient_ids.npy"
        ),
    }


def _load_learning_curve_cache(prefix: Path) -> dict[str, np.ndarray]:
    paths = _learning_curve_cache_paths(prefix)
    return {
        "X": np.load(paths["X"], allow_pickle=False),
        "y": np.load(paths["y"], allow_pickle=False),
        "patient_ids": np.load(paths["patient_ids"], allow_pickle=False),
        "source_patient_ids": np.load(
            paths["source_patient_ids"], allow_pickle=True
        ),
    }


def _validate_learning_curve_fractions(
    training_fractions: Sequence[float],
) -> list[float]:
    fractions = [float(value) for value in training_fractions]
    if not fractions:
        raise ValueError("At least one legacy training fraction is required.")
    if len(set(fractions)) != len(fractions):
        raise ValueError(f"Legacy training fractions contain duplicates: {fractions}")
    if any(value <= 0 or value > 1 for value in fractions):
        raise ValueError(
            f"Legacy training fractions must be in (0, 1], found {fractions}"
        )
    return fractions


def generate_or_load_learning_curve_embeddings(
    source: dict[str, Any],
    output_dir: Path,
    model_name: str,
    device: torch.device,
    training_fractions: Sequence[float],
    force_recompute: bool,
) -> tuple[
    dict[float, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """Create separate legacy-compatible caches for every nested train fraction.

    Each fraction is embedded through its own shuffled DataLoader. This retains
    the legacy batch-local sentence padding behavior instead of slicing a single
    corrected or globally pooled cache.
    """
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer

    fractions = _validate_learning_curve_fractions(training_fractions)
    seed = int(LEGACY_CONFIG["seed"])
    curve_dir = output_dir / "learning_curve_seed5"
    curve_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = curve_dir / "legacy_learning_curve_embedding_metadata.json"
    split_path = curve_dir / "legacy_learning_curve_split_patient_ids.npz"
    test_prefix = curve_dir / "legacy_fixed_test"
    fraction_prefixes = {
        fraction: curve_dir / f"legacy_trainpool_{int(round(fraction * 100)):03d}pct"
        for fraction in fractions
    }
    expected_paths = [
        path
        for prefix in [test_prefix, *fraction_prefixes.values()]
        for path in _learning_curve_cache_paths(prefix).values()
    ]
    reusable = (
        metadata_path.is_file()
        and split_path.is_file()
        and all(path.is_file() for path in expected_paths)
        and not force_recompute
    )
    if reusable:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_fraction_list = [float(value) for value in metadata["training_fractions"]]
        invalid_reasons = []
        if metadata.get("source_sha256") != source["source_sha256"]:
            invalid_reasons.append("source file changed")
        if metadata.get("model_name") != model_name:
            invalid_reasons.append("model name changed")
        if metadata.get("reproduction_code_sha256") != _sha256(Path(__file__)):
            invalid_reasons.append("reproduction code changed")
        if expected_fraction_list != fractions:
            invalid_reasons.append("training fractions changed")
        if invalid_reasons:
            raise ValueError(
                "Cannot reuse legacy learning-curve embeddings because "
                + ", ".join(invalid_reasons)
                + ". Set FORCE_RECOMPUTE=True."
            )
        with np.load(split_path, allow_pickle=True) as saved_split:
            split_payload = {name: saved_split[name] for name in saved_split.files}
        fraction_arrays = {
            fraction: _load_learning_curve_cache(prefix)
            for fraction, prefix in fraction_prefixes.items()
        }
        test_arrays = _load_learning_curve_cache(test_prefix)
        print(f"Reusing legacy learning-curve embeddings from {curve_dir}")
        return fraction_arrays, test_arrays, split_payload, metadata

    seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder_model = LegacyEncoderModel(model_name)
    all_positions = np.arange(len(source["patient_ids"]), dtype=np.int64)
    train_positions, test_positions = train_test_split(
        all_positions,
        test_size=LEGACY_CONFIG["outer_test_fraction"],
        random_state=seed,
    )
    local_order = list(range(len(train_positions)))
    random.seed(seed)
    random.shuffle(local_order)
    nested_train_positions = train_positions[np.asarray(local_order, dtype=np.int64)]

    def make_dataset(positions: np.ndarray) -> LegacyMultiLabelDataset:
        return LegacyMultiLabelDataset(
            [source["texts"][int(index)] for index in positions],
            source["labels"][positions],
            source["patient_ids"][positions],
            source["source_patient_ids"][positions],
            tokenizer,
            max_length=int(LEGACY_CONFIG["max_token_length"]),
        )

    # Fixed test embeddings are shared by all supervised models.
    seed_everything(seed)
    test_loader = DataLoader(
        make_dataset(test_positions),
        batch_size=int(LEGACY_CONFIG["embedding_batch_size"]),
        shuffle=False,
        collate_fn=legacy_collate_fn,
    )
    test_arrays = _cache_scope(encoder_model, test_loader, device, test_prefix)

    fraction_arrays: dict[float, dict[str, np.ndarray]] = {}
    split_payload: dict[str, np.ndarray] = {
        "outer_train_patient_ids": source["patient_ids"][train_positions],
        "outer_test_patient_ids": source["patient_ids"][test_positions],
        "nested_training_order_patient_ids": source["patient_ids"][
            nested_train_positions
        ],
        "outer_train_source_patient_ids": source["source_patient_ids"][
            train_positions
        ],
        "outer_test_source_patient_ids": source["source_patient_ids"][test_positions],
    }
    for fraction in fractions:
        percent = int(round(fraction * 100))
        subset_size = max(2, min(len(train_positions), int(len(train_positions) * fraction)))
        subset_positions = nested_train_positions[:subset_size]
        split_payload[f"trainpool_{percent:03d}pct_patient_ids"] = source[
            "patient_ids"
        ][subset_positions]
        split_payload[f"trainpool_{percent:03d}pct_source_patient_ids"] = source[
            "source_patient_ids"
        ][subset_positions]

        # One experiment seed (5) is reset for every fraction. The subset membership
        # remains nested, while each fraction gets its own legacy shuffled cache.
        seed_everything(seed)
        loader = DataLoader(
            make_dataset(subset_positions),
            batch_size=int(LEGACY_CONFIG["embedding_batch_size"]),
            shuffle=True,
            collate_fn=legacy_collate_fn,
        )
        fraction_arrays[fraction] = _cache_scope(
            encoder_model, loader, device, fraction_prefixes[fraction]
        )

    np.savez_compressed(split_path, **split_payload)
    metadata = {
        "cache_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "legacy learning curve",
        "seed": seed,
        "training_fractions": fractions,
        "source_sha256": source["source_sha256"],
        "reproduction_code_sha256": _sha256(Path(__file__)),
        "model_name": model_name,
        "resolved_model_revision": getattr(
            encoder_model.encoder.config, "_commit_hash", None
        ),
        "resolved_tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "outer_train_patients": len(train_positions),
        "fixed_test_patients": len(test_positions),
        "selection": "one seed-5 shuffled nested order; each fraction is a prefix",
        "embedding_behavior": (
            "each fraction uses its own shuffled legacy DataLoader, zero sentence-slot "
            "padding, pooler/CLS sentence embeddings, and unmasked sentence mean"
        ),
        "split_patient_ids": str(split_path),
        "package_versions": _package_versions(),
    }
    metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")
    print(f"Generated legacy learning-curve embeddings in {curve_dir}")
    return fraction_arrays, test_arrays, split_payload, metadata


def run_legacy_learning_curve(
    source_path: str | Path,
    embedding_dir: str | Path,
    checkpoint_dir: str | Path,
    results_dir: str | Path,
    training_fractions: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.50, 1.00),
    model_name: str = "nomic-ai/modernbert-embed-base",
    device_name: str = "cuda",
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Train every requested legacy supervised model and summarize test F1."""
    fractions = _validate_learning_curve_fractions(training_fractions)
    seed = int(LEGACY_CONFIG["seed"])
    seed_everything(seed)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_name}, but CUDA is unavailable.")
    device = torch.device(device_name)
    embedding_dir, checkpoint_dir, results_dir = map(
        Path, (embedding_dir, checkpoint_dir, results_dir)
    )
    curve_checkpoint_dir = checkpoint_dir / "learning_curve_seed5"
    curve_results_dir = results_dir / "learning_curve_seed5"
    per_model_results_dir = curve_results_dir / "per_model"
    for directory in (embedding_dir, curve_checkpoint_dir, per_model_results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source = load_legacy_source(source_path)
    fraction_arrays, test_arrays, split_payload, embedding_metadata = (
        generate_or_load_learning_curve_embeddings(
            source=source,
            output_dir=embedding_dir,
            model_name=model_name,
            device=device,
            training_fractions=fractions,
            force_recompute=force_recompute,
        )
    )

    variablewise_rows: list[dict[str, Any]] = []
    macro_rows: list[dict[str, Any]] = []
    model_runs: list[dict[str, Any]] = []
    for fraction in fractions:
        percent = int(round(fraction * 100))
        print(
            f"\nTraining legacy supervised model: {percent}% of train pool "
            f"({len(fraction_arrays[fraction]['X'])} patients), seed={seed}"
        )
        # Every supervised model uses the same single experiment seed.
        seed_everything(seed)
        model_checkpoint_dir = curve_checkpoint_dir / f"trainpool_{percent:03d}pct"
        model_results_dir = per_model_results_dir / f"trainpool_{percent:03d}pct"
        model_results_dir.mkdir(parents=True, exist_ok=True)
        model, training_metadata, validation_ids = _train_legacy_head(
            fraction_arrays[fraction], device, model_checkpoint_dir
        )
        evaluation = _evaluate_and_save(
            model,
            test_arrays,
            device,
            model_results_dir,
            "test",
        )
        np.savez_compressed(
            model_results_dir / "legacy_model_split_patient_ids.npz",
            selected_trainpool_patient_ids=split_payload[
                f"trainpool_{percent:03d}pct_patient_ids"
            ],
            fixed_test_patient_ids=split_payload["outer_test_patient_ids"],
            **validation_ids,
        )

        per_label_f1 = {
            row["label"]: float(row["f1"])
            for row in evaluation["per_symptom_metrics"]
        }
        common = {
            "seed": seed,
            "trainset_fraction": fraction,
            "trainset_percent": percent,
            "full_dataset_fraction": fraction
            * (1.0 - float(LEGACY_CONFIG["outer_test_fraction"])),
            "subset_patients": len(fraction_arrays[fraction]["X"]),
            "fixed_test_patients": len(test_arrays["X"]),
        }
        variablewise_rows.append(
            {
                **common,
                **{f"{label}_f1": per_label_f1[label] for label in LABEL_NAMES},
            }
        )
        macro_rows.append({**common, "macro_f1": evaluation["macro_f1"]})
        model_runs.append(
            {
                **common,
                "training": training_metadata,
                "test_evaluation": evaluation,
                "checkpoint_directory": str(model_checkpoint_dir),
                "results_directory": str(model_results_dir),
            }
        )

    variablewise_frame = pd.DataFrame(variablewise_rows).sort_values(
        "trainset_fraction"
    )
    macro_frame = pd.DataFrame(macro_rows).sort_values("trainset_fraction")
    variablewise_path = curve_results_dir / "legacy_variablewise_f1.csv"
    macro_path = curve_results_dir / "legacy_macro_f1.csv"
    variablewise_frame.to_csv(variablewise_path, index=False)
    macro_frame.to_csv(macro_path, index=False)

    metadata = {
        "status": "completed",
        "workflow": "legacy learning curve",
        "seed": seed,
        "training_fractions": fractions,
        "fraction_definition": "fraction of the fixed 80% outer training pool",
        "fixed_test_fraction": LEGACY_CONFIG["outer_test_fraction"],
        "label_names": LABEL_NAMES,
        "fever_training_values": sorted(np.unique(source["labels"][:, 3]).tolist()),
        "f1_target_note": (
            "F1 treats every raw target > 0 as positive for evaluation only; raw fever "
            "values, including 2, remain unchanged during BCE training."
        ),
        "variablewise_f1": str(variablewise_path),
        "macro_f1": str(macro_path),
        "embedding_metadata": embedding_metadata,
        "model_runs": model_runs,
        "runtime": {
            "python": platform.python_version(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "packages": _package_versions(),
        },
    }
    metadata_path = curve_results_dir / "legacy_learning_curve_metadata.json"
    metadata_path.write_text(
        json.dumps(_json_safe(metadata), indent=2), encoding="utf-8"
    )
    report_path = curve_results_dir / "legacy_learning_curve_report.md"
    fraction_text = ", ".join(
        f"{int(round(fraction * 100))}%" for fraction in fractions
    )
    report_path.write_text(
        "# Legacy supervised learning curve\n\n"
        f"{len(fractions)} independent supervised models were trained with seed 5 "
        f"using {fraction_text} of the fixed outer training pool. All models were "
        "evaluated on the same held-out 20% test set.\n\n"
        f"- Variable-wise F1: `{variablewise_path.name}`\n"
        f"- Macro-F1: `{macro_path.name}`\n"
        f"- Metadata: `{metadata_path.name}`\n",
        encoding="utf-8",
    )
    print("\nVariable-wise F1")
    print(variablewise_frame.to_string(index=False))
    print("\nMacro-F1")
    print(macro_frame.to_string(index=False))
    print(f"\nVariable-wise F1: {variablewise_path}")
    print(f"Macro-F1: {macro_path}")
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--model-name", default="nomic-ai/modernbert-embed-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--learning-curve",
        action="store_true",
        help="Train one legacy model for every --training-fractions value.",
    )
    parser.add_argument(
        "--training-fractions",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.20, 0.30, 0.50, 1.00],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    common = {
        "source_path": args.source_path,
        "embedding_dir": args.embedding_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "results_dir": args.results_dir,
        "model_name": args.model_name,
        "device_name": args.device,
        "force_recompute": args.force_recompute,
    }
    if args.learning_curve:
        run_legacy_learning_curve(
            **common, training_fractions=args.training_fractions
        )
    else:
        run_legacy_reproduction(**common)


if __name__ == "__main__":
    main()
