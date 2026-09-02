"""Deterministic training and evaluation helpers."""

from __future__ import annotations

import copy
import platform
import random
import resource
import time
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


def seed_everything(seed: int) -> None:
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
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    device: torch.device,
) -> FitResult:
    seed_everything(seed)
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(validation_data, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    start = time.perf_counter()
    epochs_ran = 0

    for epoch in range(epochs):
        model.train()
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()

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
        if validation_loss < best_loss - 1e-10:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
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
