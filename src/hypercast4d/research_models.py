"""Compact, target-only adaptations; not reproductions of published benchmarks."""

import torch
from torch import nn
from torch.nn import functional as F


class DLinear(nn.Module):
    """Moving-average decomposition with separate seasonal/trend projections."""

    def __init__(self, window: int, horizon: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.seasonal = nn.Linear(window, horizon)
        self.trend = nn.Linear(window, horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        target = inputs[:, :, 0]
        padding = self.kernel_size // 2
        trend = F.avg_pool1d(
            F.pad(target.unsqueeze(1), (padding, padding), mode="replicate"),
            self.kernel_size, stride=1,
        ).squeeze(1)
        return self.seasonal(target - trend) + self.trend(trend)


class AttentionForecast(nn.Module):
    """Patch tokens within the target, or variable tokens across all inputs.

    Uses PyTorch LayerNorm encoders, non-affine per-window normalization,
    no calendar covariates, and a target-only training loss.
    """

    def __init__(self, kind: str, window: int, horizon: int, *, d_model: int,
                 n_heads: int, num_layers: int, dropout: float,
                 patch_size: int = 4, stride: int = 2) -> None:
        super().__init__()
        self.kind = kind
        self.patch_size = patch_size
        self.stride = stride
        self.padding = max(0, patch_size - window) + stride
        if kind == "patchtst":
            patches = (window + self.padding - patch_size) // stride + 1
            self.embedding = nn.Linear(patch_size, d_model)
            self.position = nn.Parameter(torch.empty(1, patches, d_model))
            nn.init.normal_(self.position, std=0.02)
            self.projection = nn.Linear(patches * d_model, horizon)
        else:
            self.embedding = nn.Linear(window, d_model)
            self.register_parameter("position", None)
            self.projection = nn.Linear(d_model, horizon)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                       dropout, activation="gelu", batch_first=True),
            num_layers, enable_nested_tensor=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=1, keepdim=True).detach()
        scale = (inputs.var(dim=1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        normalized = (inputs - mean) / scale
        if self.kind == "patchtst":
            # Channel independence means exogenous channels cannot affect the target.
            target = F.pad(normalized[:, :, 0].unsqueeze(1),
                           (0, self.padding), mode="replicate").squeeze(1)
            tokens = target.unfold(-1, self.patch_size, self.stride)
            encoded = self.encoder(self.embedding(tokens) + self.position)
            forecast = self.projection(encoded.flatten(1))
        else:
            tokens = self.embedding(normalized.transpose(1, 2))
            forecast = self.projection(self.encoder(tokens)[:, 0])
        return forecast * scale[:, 0, 0:1] + mean[:, 0, 0:1]
