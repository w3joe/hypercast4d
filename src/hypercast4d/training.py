"""Deterministic training and evaluation helpers."""

from __future__ import annotations

import copy
import platform
import random
import resource
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class FitResult:
    epochs_ran: int
    best_validation_loss: float
    train_seconds: float
    process_peak_rss_mb: float


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if platform.system() == "Darwin" else 1024.0
    return value / divisor


def fit_model(
    model: nn.Module,
    train_data: TensorDataset,
    validation_data: TensorDataset,
    *,
    seed: int | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
    adam_amsgrad: bool,
    loss_name: str,
    shuffle: bool,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    restore_best_weights: bool,
    device: torch.device,
    epoch_callback: Callable[[int, float, float], None] | None = None,
) -> FitResult:
    seed_everything(seed)
    model.to(device)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
    validation_loader = DataLoader(validation_data, batch_size=batch_size)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
        amsgrad=adam_amsgrad,
    )
    if loss_name == "mse":
        criterion: nn.Module = nn.MSELoss()
    elif loss_name == "mae":
        criterion = nn.L1Loss()
    else:
        raise ValueError("loss must be 'mse' or 'mae'")
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    start = time.perf_counter()
    epochs_ran = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_train_items = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(features)
            total_train_items += len(features)

        model.eval()
        total_loss = 0.0
        total_items = 0
        with torch.no_grad():
            for features, targets in validation_loader:
                features, targets = features.to(device), targets.to(device)
                batch_loss = criterion(model(features), targets)
                total_loss += batch_loss.item() * len(features)
                total_items += len(features)
        validation_loss = total_loss / total_items
        epochs_ran = epoch + 1
        if epoch_callback is not None:
            epoch_callback(
                epochs_ran,
                total_train_loss / total_train_items,
                validation_loss,
            )
        if validation_loss < best_loss - early_stopping_min_delta:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if (
                early_stopping_patience is not None
                and stale_epochs >= early_stopping_patience
            ):
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    if restore_best_weights:
        model.load_state_dict(best_state)
    return FitResult(
        epochs_ran=epochs_ran,
        best_validation_loss=best_loss,
        train_seconds=time.perf_counter() - start,
        process_peak_rss_mb=_peak_rss_mb(),
    )


def predict(
    model: nn.Module,
    dataset: TensorDataset,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for features, _ in DataLoader(dataset, batch_size=batch_size):
            batches.append(model(features.to(device)).cpu().numpy())
    return np.concatenate(batches)


def error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    residual = prediction - target
    return {
        "mae": float(np.mean(np.abs(residual))),
        "mse": float(np.mean(np.square(residual))),
    }
