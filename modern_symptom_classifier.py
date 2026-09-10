#!/usr/bin/env python3
"""Importable version of the modern symptom classifier used by the notebook.

The architecture, scaling, class weighting, optimizer, inner-CV epoch selection,
and decision threshold intentionally match ``colab_embedding_and_training.ipynb``.
Keeping them here lets non-notebook experiments reuse one implementation.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class ClassifierConfig:
    """Settings copied from the notebook's modern learning-curve workflow."""

    hidden_dim: int = 256
    dropout: float = 0.25
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 120
    early_stopping_patience: int = 12
    binary_threshold: float = 0.5
    inner_cv_folds: int = 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SymptomMultiTaskMLP(nn.Module):
    """One 768-to-256 hidden layer and five binary symptom outputs."""

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_head = nn.Linear(hidden_dim, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output_head(self.shared(features))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch as in the modern notebook."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def feature_scaler(X: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit z-score parameters using only the supplied training indices."""
    mean = X[indices].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X[indices].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def positive_class_weights(
    y: np.ndarray, indices: np.ndarray, device: torch.device
) -> torch.Tensor:
    """Return the notebook's finite negative/positive label weights."""
    positive = y[indices].sum(axis=0).astype(np.float32)
    negative = len(indices) - positive
    return torch.from_numpy(negative / np.maximum(positive, 1.0)).to(device)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    config: ClassifierConfig,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Build a deterministic loader after applying training-fitted scaling."""
    scaled_features = ((X[indices] - mean) / std).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(scaled_features),
        torch.from_numpy(y[indices].astype(np.int64, copy=False)),
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


def calculate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_names: Sequence[str],
    threshold: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Calculate per-label and macro precision, recall, and F1."""
    from sklearn.metrics import precision_recall_fscore_support

    predictions = (probabilities >= threshold).astype(np.int64)
    per_label: dict[str, dict[str, float]] = {}
    values: list[tuple[float, float, float]] = []
    for label_index, label_name in enumerate(label_names):
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets[:, label_index],
            predictions[:, label_index],
            average="binary",
            zero_division=0,
        )
        row = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        per_label[str(label_name)] = row
        values.append((row["precision"], row["recall"], row["f1"]))
    metric_array = np.asarray(values, dtype=float)
    return {
        "macro_precision": float(metric_array[:, 0].mean()),
        "macro_recall": float(metric_array[:, 1].mean()),
        "macro_f1": float(metric_array[:, 2].mean()),
        "per_label": per_label,
    }, predictions


def _run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> None:
    model.train()
    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(features), targets.float())
        loss.backward()
        optimizer.step()


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
    """Predict probabilities in the exact order of ``indices``."""
    loader = make_loader(
        X, y, indices, mean, std, config, shuffle=False, seed=seed
    )
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for features, _ in loader:
            logits = model(features.to(device, non_blocking=True))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float32, copy=False)


def _new_model_and_optimizer(
    input_dim: int,
    num_labels: int,
    config: ClassifierConfig,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, torch.optim.Optimizer]:
    seed_everything(seed)
    model = SymptomMultiTaskMLP(
        input_dim=input_dim,
        num_labels=num_labels,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    return model, optimizer


def select_training_epochs(
    X: np.ndarray,
    y: np.ndarray,
    subset_indices: np.ndarray,
    label_names: Sequence[str],
    config: ClassifierConfig,
    device: torch.device,
    seed: int,
) -> tuple[int, list[dict[str, Any]]]:
    """Select the final duration using the notebook's inner-CV procedure."""
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "iterative-stratification is required for multilabel inner CV. "
            "Install the repository requirements."
        ) from exc

    n_splits = min(config.inner_cv_folds, len(subset_indices))
    if n_splits < 2:
        raise ValueError("A training subset must contain at least two patients for CV.")
    splitter = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    best_epochs: list[int] = []
    fold_rows: list[dict[str, Any]] = []
    local_X, local_y = X[subset_indices], y[subset_indices]

    for inner_fold, (train_positions, validation_positions) in enumerate(
        splitter.split(local_X, local_y), start=1
    ):
        train_indices = subset_indices[train_positions]
        validation_indices = subset_indices[validation_positions]
        fold_seed = seed * 100 + inner_fold
        mean, std = feature_scaler(X, train_indices)
        train_loader = make_loader(
            X, y, train_indices, mean, std, config, shuffle=True, seed=fold_seed
        )
        loss_function = nn.BCEWithLogitsLoss(
            pos_weight=positive_class_weights(y, train_indices, device)
        )
        model, optimizer = _new_model_and_optimizer(
            X.shape[1], y.shape[1], config, device, fold_seed
        )
        best_score = -np.inf
        best_epoch = 1
        epochs_without_improvement = 0

        for epoch in range(1, config.max_epochs + 1):
            _run_training_epoch(model, train_loader, loss_function, device, optimizer)
            probabilities = predict_probabilities(
                model,
                X,
                y,
                validation_indices,
                mean,
                std,
                config,
                device,
                fold_seed,
            )
            metrics, _ = calculate_metrics(
                y[validation_indices], probabilities, label_names, config.binary_threshold
            )
            score = metrics["macro_f1"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.early_stopping_patience:
                    break

        best_epochs.append(best_epoch)
        fold_rows.append(
            {
                "inner_fold": inner_fold,
                "inner_train_patients": len(train_indices),
                "inner_validation_patients": len(validation_indices),
                "best_epoch": best_epoch,
                "validation_macro_f1": float(best_score),
            }
        )

    return max(1, int(np.median(best_epochs))), fold_rows


def train_final_classifier(
    X: np.ndarray,
    y: np.ndarray,
    subset_indices: np.ndarray,
    final_epochs: int,
    config: ClassifierConfig,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    """Refit from scratch on the complete selected outer-training subset."""
    mean, std = feature_scaler(X, subset_indices)
    loader = make_loader(
        X, y, subset_indices, mean, std, config, shuffle=True, seed=seed
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=positive_class_weights(y, subset_indices, device)
    )
    model, optimizer = _new_model_and_optimizer(
        X.shape[1], y.shape[1], config, device, seed
    )
    for _ in range(final_epochs):
        _run_training_epoch(model, loader, loss_function, device, optimizer)
    return model, mean, std
