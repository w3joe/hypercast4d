"""Compact, target-only adaptations; not reproductions of published benchmarks."""

import torch
from torch import nn
from torch.nn import functional as F


class Decomposition(nn.Module):
    """Expose seasonal and trend channels side by side: [B,T,C] -> [B,T,2C]."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padding = self.kernel_size // 2
        trend = F.avg_pool1d(F.pad(inputs.transpose(1, 2), (padding, padding),
                                  mode="replicate"), self.kernel_size, stride=1).transpose(1, 2)
        return torch.cat((inputs - trend, trend), dim=-1)


class TemporalProjection(nn.Module):
    """Learn a separate time-axis projection per feature, retaining a sequence."""

    def __init__(self, steps: int, width: int, output_steps: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, steps, output_steps))
        self.bias = nn.Parameter(torch.zeros(width, output_steps))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (torch.einsum("btc,ctu->bcu", inputs, self.weight) + self.bias).transpose(1, 2)


class PatchEmbedding(nn.Module):
    """Joint feature/time patches, projected into a sequence of learned tokens."""

    def __init__(self, steps: int, width: int, d_model: int, patch_size: int, stride: int) -> None:
        super().__init__()
        self.patch_size, self.stride = patch_size, stride
        self.padding = max(0, patch_size - steps) + stride
        self.output_steps = (steps + self.padding - patch_size) // stride + 1
        self.projection = nn.Linear(width * patch_size, d_model)
        self.position = nn.Parameter(torch.empty(1, self.output_steps, d_model))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padded = F.pad(inputs.transpose(1, 2), (0, self.padding), mode="replicate")
        patches = padded.unfold(-1, self.patch_size, self.stride).permute(0, 2, 1, 3)
        return self.projection(patches.flatten(2)) + self.position


class SequenceAttention(nn.Module):
    """Attention along time/patches, or variables with a projection back to time."""

    def __init__(self, steps: int, width: int, d_model: int, n_heads: int,
                 num_layers: int, dropout: float, *, variables: bool = False) -> None:
        super().__init__()
        self.variables = variables
        self.embedding = nn.Linear(steps if variables else width, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout,
                                       activation="gelu", batch_first=True),
            num_layers, enable_nested_tensor=False,
        )
        self.projection = nn.Linear(d_model, steps) if variables else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = inputs.transpose(1, 2) if self.variables else inputs
        encoded = self.projection(self.encoder(self.embedding(tokens)))
        return encoded.transpose(1, 2) if self.variables else encoded


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
