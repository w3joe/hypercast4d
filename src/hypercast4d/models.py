"""Comparable real and hypercomplex forecasting models."""

from __future__ import annotations

import torch
from torch import nn

from .algebras import COMPONENT_COUNT
from .layers import HyperDense


class LinearForecaster(nn.Module):
    def __init__(
        self, window: int, horizon: int, features: int = COMPONENT_COUNT
    ) -> None:
        super().__init__()
        self.output = nn.Linear(window * features, horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(inputs.flatten(start_dim=1))


class CNNForecaster(nn.Module):
    def __init__(
        self,
        horizon: int,
        channels: int = 16,
        features: int = COMPONENT_COUNT,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("CNN kernel_size must be a positive odd number")
        self.features = nn.Sequential(
            nn.Conv1d(
                features,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.output = nn.Linear(channels, horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.features(inputs.transpose(1, 2)).squeeze(-1)
        return self.output(encoded)


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        horizon: int,
        hidden: int = 16,
        features: int = COMPONENT_COUNT,
    ) -> None:
        super().__init__()
        self.recurrent = nn.LSTM(features, hidden, batch_first=True)
        self.output = nn.Linear(hidden, horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.recurrent(inputs)
        return self.output(sequence[:, -1])


class HyperForecaster(nn.Module):
    """Paper-shaped hypercomplex front end followed by real forecasting layers."""

    def __init__(
        self,
        window: int,
        horizon: int,
        algebra: str,
        hidden: int = 8,
        dropout: float = 0.1,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        if pool_size < 1:
            raise ValueError("HyperForecaster pool_size must be positive")
        self.hyper = HyperDense(1, hidden, algebra=algebra)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        pooled_steps = window // pool_size
        if pooled_steps < 1:
            raise ValueError(
                f"HyperForecaster window must be at least pool_size ({pool_size})"
            )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(COMPONENT_COUNT * hidden * pooled_steps, horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.activation(self.hyper(inputs))
        encoded = self.pool(encoded.transpose(1, 2)).flatten(start_dim=1)
        return self.output(self.dropout(encoded))


def build_model(
    name: str,
    window: int,
    horizon: int,
    *,
    hyper_hidden: int = 8,
    cnn_channels: int = 16,
    cnn_kernel_size: int = 3,
    lstm_hidden: int = 16,
    dropout: float = 0.1,
    hyper_pool_size: int = 2,
) -> nn.Module:
    if name == "linear":
        return LinearForecaster(window, horizon)
    if name == "cnn":
        return CNNForecaster(horizon, cnn_channels, kernel_size=cnn_kernel_size)
    if name == "lstm":
        return LSTMForecaster(horizon, lstm_hidden)
    if name.startswith("hyper_"):
        algebra = name.removeprefix("hyper_")
        return HyperForecaster(
            window,
            horizon,
            algebra,
            hyper_hidden,
            dropout,
            hyper_pool_size,
        )
    raise ValueError(f"Unknown model {name!r}")


def parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
