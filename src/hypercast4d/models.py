"""Paper-aligned real and hypercomplex forecasting models."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from .algebras import COMPONENT_COUNT
from .layers import HyperDense


def _activation(name: str) -> nn.Module:
    if name == "linear":
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    raise ValueError("activation must be 'linear' or 'relu'")


class LinearForecaster(nn.Module):
    """Extra real-valued baseline; this model is not part of the paper."""

    def __init__(
        self, window: int, horizon: int, features: int = COMPONENT_COUNT
    ) -> None:
        super().__init__()
        self.output = nn.Linear(window * features, horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(inputs.flatten(start_dim=1))


class CausalConv1d(nn.Conv1d):
    """Conv1d with the paper notebook's Keras ``padding='causal'`` behavior."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        left_padding = self.dilation[0] * (self.kernel_size[0] - 1)
        return super().forward(functional.pad(inputs, (left_padding, 0)))


class PaperForecaster(nn.Module):
    """Shared seven-stage architecture shown in Figure 3 of the paper.

    The tested first layer is followed by an optional time-distributed dense
    layer, max pooling, flattening, an optional dense layer, dropout, and the
    real-valued forecast head. The first layer is selected with ``kind``.
    """

    def __init__(
        self,
        kind: str,
        window: int,
        horizon: int,
        *,
        first_layer_units: int,
        dense_before_pool: bool,
        dense_after_pool: bool,
        dense_units: int,
        activation: str,
        dropout: float,
        pool_size: int,
        conv_kernel_size: int | None,
        algebra: str | None = None,
        features: int = COMPONENT_COUNT,
    ) -> None:
        super().__init__()
        if window < 1 or horizon < 1:
            raise ValueError("window and horizon must be positive")
        if first_layer_units < 1 or dense_units < 1:
            raise ValueError("model unit counts must be positive")
        if pool_size < 1 or window < pool_size:
            raise ValueError("pool_size must be positive and no larger than window")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.kind = kind
        first_width: int
        if kind == "cnn":
            if conv_kernel_size is None or conv_kernel_size < 1:
                raise ValueError("A CNN requires a positive conv_kernel_size")
            self.first = CausalConv1d(
                features,
                first_layer_units,
                kernel_size=conv_kernel_size,
                stride=1,
            )
            first_width = first_layer_units
        elif kind == "lstm":
            self.first = nn.LSTM(features, first_layer_units, batch_first=True)
            first_width = first_layer_units
        elif kind == "hyper":
            if algebra is None:
                raise ValueError("A hypercomplex model requires an algebra")
            self.first = HyperDense(1, first_layer_units, algebra=algebra)
            first_width = COMPONENT_COUNT * first_layer_units
        else:
            raise ValueError("kind must be 'cnn', 'lstm', or 'hyper'")

        self.first_activation = _activation(activation) if kind != "lstm" else None
        self.before_pool = (
            nn.Sequential(nn.Linear(first_width, dense_units), _activation(activation))
            if dense_before_pool
            else nn.Identity()
        )
        pooled_width = dense_units if dense_before_pool else first_width
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        pooled_steps = (window - pool_size) // pool_size + 1
        flattened_width = pooled_steps * pooled_width
        self.after_pool = (
            nn.Sequential(
                nn.Linear(flattened_width, dense_units), _activation(activation)
            )
            if dense_after_pool
            else nn.Identity()
        )
        output_width = dense_units if dense_after_pool else flattened_width
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(output_width, horizon)

        self._reset_paper_initialization()

    def _reset_paper_initialization(self) -> None:
        """Match Keras defaults used by the supplementary notebook."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                nn.init.xavier_uniform_(module.weight_ih_l0)
                nn.init.orthogonal_(module.weight_hh_l0)
                nn.init.zeros_(module.bias_ih_l0)
                nn.init.zeros_(module.bias_hh_l0)
                hidden = module.hidden_size
                module.bias_ih_l0.data[hidden : 2 * hidden].fill_(1.0)
                # Keras has one LSTM bias vector; PyTorch has two whose sum is
                # equivalent. Keep the second fixed at zero for matching
                # behavior and trainable-parameter counts.
                module.bias_hh_l0.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.kind == "cnn":
            sequence = self.first(inputs.transpose(1, 2)).transpose(1, 2)
            sequence = self.first_activation(sequence)
        elif self.kind == "lstm":
            sequence, _ = self.first(inputs)
        else:
            sequence = self.first_activation(self.first(inputs))

        sequence = self.before_pool(sequence)
        sequence = self.pool(sequence.transpose(1, 2)).transpose(1, 2)
        flattened = sequence.flatten(start_dim=1)
        encoded = self.after_pool(flattened)
        return self.output(self.dropout(encoded))


def build_model(
    name: str,
    window: int,
    horizon: int,
    *,
    first_layer_units: int | None = None,
    conv_kernel_size: int | None = None,
    dense_before_pool: bool | None = None,
    dense_after_pool: bool | None = None,
    dense_units: int | None = None,
    activation: str | None = None,
    dropout: float | None = None,
    pool_size: int | None = None,
) -> nn.Module:
    """Build a configured baseline or Figure 3 forecasting architecture."""
    if name == "linear":
        return LinearForecaster(window, horizon)
    required = {
        "first_layer_units": first_layer_units,
        "dense_before_pool": dense_before_pool,
        "dense_after_pool": dense_after_pool,
        "dense_units": dense_units,
        "activation": activation,
        "dropout": dropout,
        "pool_size": pool_size,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing model settings: {', '.join(missing)}")
    if name == "cnn":
        kind, algebra = "cnn", None
    elif name == "lstm":
        kind, algebra = "lstm", None
    elif name.startswith("hyper_"):
        kind = "hyper"
        algebra = name.removeprefix("hyper_")
    else:
        raise ValueError(f"Unknown model {name!r}")
    return PaperForecaster(
        kind,
        window,
        horizon,
        first_layer_units=int(first_layer_units),
        dense_before_pool=bool(dense_before_pool),
        dense_after_pool=bool(dense_after_pool),
        dense_units=int(dense_units),
        activation=str(activation),
        dropout=float(dropout),
        pool_size=int(pool_size),
        conv_kernel_size=conv_kernel_size,
        algebra=algebra,
    )


def parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
