"""Comparable real and hypercomplex forecasting models."""

from __future__ import annotations

import torch
from torch import nn

from .layers import HyperDense


class LinearForecaster(nn.Module):
    def __init__(self, window: int, horizon: int, features: int = 4) -> None:
        super().__init__()
        self.output = nn.Linear(window * features, horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(inputs.flatten(start_dim=1))


class CNNForecaster(nn.Module):
    def __init__(self, horizon: int, channels: int = 16, features: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(features, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.output = nn.Linear(channels, horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.features(inputs.transpose(1, 2)).squeeze(-1)
        return self.output(encoded)


class LSTMForecaster(nn.Module):
    def __init__(self, horizon: int, hidden: int = 16, features: int = 4) -> None:
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
    ) -> None:
        super().__init__()
        self.hyper = HyperDense(1, hidden, algebra=algebra)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        pooled_steps = window // 2
        if pooled_steps < 1:
            raise ValueError("HyperForecaster requires window >= 2")
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(4 * hidden * pooled_steps, horizon)

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
    lstm_hidden: int = 16,
    dropout: float = 0.1,
) -> nn.Module:
    if name == "linear":
        return LinearForecaster(window, horizon)
    if name == "cnn":
        return CNNForecaster(horizon, cnn_channels)
    if name == "lstm":
        return LSTMForecaster(horizon, lstm_hidden)
    if name.startswith("hyper_"):
        algebra = name.removeprefix("hyper_")
        return HyperForecaster(window, horizon, algebra, hyper_hidden, dropout)
    raise ValueError(f"Unknown model {name!r}")


def parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
