"""PyTorch layers backed by explicit hypercomplex structure constants."""

from __future__ import annotations

import torch
from torch import nn

from .algebras import Algebra, get_algebra


class HyperDense(nn.Module):
    """A dense layer using input-by-weight hypercomplex products.

    Inputs use component-major layout: all real features, followed by each
    subsequent basis component. The returned tensor follows the same convention.
    For an algebra with ``d`` components, ``d`` learned real matrices replace
    the ``d²`` independent matrices of a matching real dense map.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        algebra: str | Algebra,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.algebra = get_algebra(algebra) if isinstance(algebra, str) else algebra
        self.component_count = self.algebra.component_count
        self.weight = nn.Parameter(
            torch.empty(self.component_count, in_features, out_features)
        )
        self.bias = (
            nn.Parameter(torch.empty(self.component_count, out_features))
            if bias
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # The supplementary TensorFlow layer initializes each component kernel
        # independently with Glorot normal and initializes its bias to zero.
        for component in self.weight:
            nn.init.xavier_normal_(component)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        expected = self.component_count * self.in_features
        if inputs.shape[-1] != expected:
            raise ValueError(
                f"Expected final dimension {expected}, got {inputs.shape[-1]}"
            )
        components = inputs.reshape(
            *inputs.shape[:-1], self.component_count, self.in_features
        )
        constants = self.algebra.constants.to(inputs.device, inputs.dtype)
        # ``components`` supplies the left factor and ``weight`` the right one,
        # matching the block matrix in the paper's archived HyperDense layer.
        outputs = torch.einsum(
            "bac,aio,...bi->...co", constants, self.weight, components
        )
        if self.bias is not None:
            outputs = outputs + self.bias
        return outputs.reshape(
            *inputs.shape[:-1], self.component_count * self.out_features
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"algebra={self.algebra.name}, bias={self.bias is not None}"
        )
