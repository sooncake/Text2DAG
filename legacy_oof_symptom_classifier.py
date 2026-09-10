#!/usr/bin/env python3
"""Legacy-reproduction classifier head adapted to leakage-safe OOF training.

The feature matrix is supplied by the OOF experiment. The supervised model and
optimization choices match ``legacy_reproduction.py``: a 768->256->5 linear
head without an activation, unweighted BCE, AdamW at 3e-5 with its default
weight decay, no feature standardization, validation-loss epoch selection, and
a strict probability threshold greater than 0.5.
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from legacy_reproduction import LegacyHeadOnlyModel


@dataclass(frozen=True)
class ClassifierConfig:
    """Legacy classifier settings used independently in every outer fold."""

    implementation: str = "legacy_reproduction"
    model_class: str = "LegacyHeadOnlyModel"
    training_seed: int = 5
    head_dim: int = 256
    learning_rate: float = e-3
    batch_size: int = 32
    max_epochs: int = 120
    early_stopping_patience: int = 5
    validation_fraction: float = 0.10
    binary_threshold: float = 0.5
    threshold_operator: str = ">"
    loss: str = "BCEWithLogitsLoss(unweighted)"
    optimizer: str = "torch.optim.AdamW(lr=3e-5; default weight_decay)"
    feature_standardization: bool = False
    fever_training_target: str = "raw 0/1/2"
    oof_adaptation: str = (
        "select epochs on a legacy 10% validation split, then refit on the complete "
        "outer-training subset"
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seed_everything(seed: int) -> None:
    """Deterministically seed the legacy head training run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def feature_scaler(X: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return identity scaling because legacy reproduction standardizes nothing."""
    del indices
    return (
        np.zeros(X.shape[1], dtype=np.float32),
        np.ones(X.shape[1], dtype=np.float32),
    )


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    config: ClassifierConfig,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X[indices].astype(np.float32, copy=False)),
        torch.from_numpy(y[indices].astype(np.float32, copy=False)),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _new_model_and_optimizer(
    input_dim: int,
    num_labels: int,
    config: ClassifierConfig,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, torch.optim.Optimizer]:
    seed_everything(seed)
    model = LegacyHeadOnlyModel(
        input_dim=input_dim,
        num_labels=num_labels,
        head_dim=config.head_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    return model, optimizer


def _run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    model.train()
    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features), targets)
        loss.backward()
        optimizer.step()


def _mean_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            losses.append(float(criterion(model(features), targets).item()))
    if not losses:
        raise ValueError("Legacy validation split produced no batches.")
    return float(np.mean(losses))


def calculate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_names: Sequence[str],
    threshold: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate symptom presence while retaining raw fever values for training."""
    from sklearn.metrics import precision_recall_fscore_support

    binary_targets = (np.asarray(targets) > 0).astype(np.int64)
    predictions = (np.asarray(probabilities) > threshold).astype(np.int64)
    per_label: dict[str, dict[str, float]] = {}
    rows: list[tuple[float, float, float]] = []
    for index, name in enumerate(label_names):
        precision, recall, f1, _ = precision_recall_fscore_support(
            binary_targets[:, index],
            predictions[:, index],
            average="binary",
            zero_division=0,
        )
        values = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        per_label[str(name)] = values
        rows.append((values["precision"], values["recall"], values["f1"]))
    metric_array = np.asarray(rows, dtype=float)
    return {
        "macro_precision": float(metric_array[:, 0].mean()),
        "macro_recall": float(metric_array[:, 1].mean()),
        "macro_f1": float(metric_array[:, 2].mean()),
        "per_label": per_label,
    }, predictions


def predict_probabilities(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    config: ClassifierConfig,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    """Predict in index order without standardizing the supplied embeddings."""
    del mean, std
    loader = _make_loader(X, y, indices, config, shuffle=False, seed=seed)
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for features, _ in loader:
            logits = model(features.to(device, non_blocking=True))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float32, copy=False)


def select_training_epochs(
    X: np.ndarray,
    y: np.ndarray,
    subset_indices: np.ndarray,
    label_names: Sequence[str],
    config: ClassifierConfig,
    device: torch.device,
    seed: int,
) -> tuple[int, list[dict[str, Any]]]:
    """Select duration by the legacy 90/10 split and validation-loss rule."""
    from sklearn.model_selection import train_test_split

    if len(subset_indices) < 2:
        raise ValueError("A legacy training subset must contain at least two patients.")
    train_indices, validation_indices = train_test_split(
        np.asarray(subset_indices, dtype=np.int64),
        test_size=config.validation_fraction,
        random_state=seed,
    )
    train_loader = _make_loader(
        X, y, train_indices, config, shuffle=True, seed=seed
    )
    validation_loader = _make_loader(
        X, y, validation_indices, config, shuffle=False, seed=seed
    )
    model, optimizer = _new_model_and_optimizer(
        X.shape[1], y.shape[1], config, device, seed
    )
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    for epoch in range(1, config.max_epochs + 1):
        _run_training_epoch(model, train_loader, criterion, optimizer, device)
        validation_loss = _mean_loss(model, validation_loader, criterion, device)
        if abs(validation_loss) < abs(best_loss):
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("Legacy epoch selection did not produce a checkpoint.")
    model.load_state_dict(best_state)
    identity_mean, identity_std = feature_scaler(X, train_indices)
    probabilities = predict_probabilities(
        model,
        X,
        y,
        validation_indices,
        identity_mean,
        identity_std,
        config,
        device,
        seed,
    )
    metrics, _ = calculate_metrics(
        y[validation_indices], probabilities, label_names, config.binary_threshold
    )
    return best_epoch, [
        {
            "inner_fold": 1,
            "inner_train_patients": len(train_indices),
            "inner_validation_patients": len(validation_indices),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "validation_macro_f1": metrics["macro_f1"],
            "selection_metric": "minimum absolute validation loss",
        }
    ]


def train_final_classifier(
    X: np.ndarray,
    y: np.ndarray,
    subset_indices: np.ndarray,
    final_epochs: int,
    config: ClassifierConfig,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    """Refit the legacy head on the complete outer-training subset."""
    mean, std = feature_scaler(X, subset_indices)
    loader = _make_loader(X, y, subset_indices, config, shuffle=True, seed=seed)
    model, optimizer = _new_model_and_optimizer(
        X.shape[1], y.shape[1], config, device, seed
    )
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(final_epochs):
        _run_training_epoch(model, loader, criterion, optimizer, device)
    return model, mean, std
