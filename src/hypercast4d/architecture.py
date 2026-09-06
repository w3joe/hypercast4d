"""Composable forecasting architectures used by the local playground."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .algebras import COMPONENT_COUNT
from .layers import HyperDense
from .upstream_models import UPSTREAM_MODELS, UpstreamSequence, upstream_defaults
from .models import CausalConv1d, parameter_count
from .research_models import (
    AttentionForecast, DLinear, Decomposition, PatchEmbedding,
    SequenceAttention, TemporalProjection,
)


SCHEMA_VERSION = 1
ALGEBRAS = ("quaternion", "coquaternion", "cl11")
ACTIVATIONS = ("relu", "gelu", "silu", "tanh", "linear")
INPUT_REPRESENTATIONS = ("levels", "centered", "differences")
HEAD_TYPES = ("direct", "persistence_residual", "cumulative_residual")
RESEARCH_MODELS = {"dlinear", "patchtst", "itransformer"}
RESEARCH_BLOCKS = {"decomposition", "patch_embedding", "temporal_attention",
                   "variable_attention", "temporal_projection"}


class ArchitectureError(ValueError):
    """Raised when an architecture specification cannot be compiled."""


@dataclass(frozen=True)
class TensorShape:
    kind: str
    width: int
    steps: int | None = None

    def label(self) -> str:
        if self.kind == "sequence":
            return f"[B, {self.steps}, {self.width}]"
        return f"[B, {self.width}]"


@dataclass(frozen=True)
class TraceEntry:
    layer_id: str
    layer_type: str
    input_shape: str
    output_shape: str
    parameters: int
    receptive_field: int


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ArchitectureError(f"{name} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise ArchitectureError(f"{name} must be an integer") from error
    if converted != value or not minimum <= converted <= maximum:
        raise ArchitectureError(f"{name} must be between {minimum} and {maximum}")
    return converted


def _bounded_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ArchitectureError(f"{name} must be a number") from error
    if not minimum <= converted <= maximum:
        raise ArchitectureError(f"{name} must be between {minimum} and {maximum}")
    return converted


def normalize_architecture_spec(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the serializable portion of a model spec."""
    if not isinstance(raw, dict):
        raise ArchitectureError("architecture must be an object")
    if raw.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ArchitectureError(f"schema_version must be {SCHEMA_VERSION}")
    name = str(raw.get("name", "Untitled architecture")).strip()
    if not name or len(name) > 80:
        raise ArchitectureError("name must contain between 1 and 80 characters")

    input_raw = raw.get("input", {})
    if not isinstance(input_raw, dict):
        raise ArchitectureError("input must be an object")
    representation = str(input_raw.get("representation", "levels"))
    if representation not in INPUT_REPRESENTATIONS:
        raise ArchitectureError(
            f"input representation must be one of {INPUT_REPRESENTATIONS}"
        )
    order_raw = input_raw.get("feature_order", list(range(COMPONENT_COUNT)))
    if not isinstance(order_raw, list) or not order_raw:
        raise ArchitectureError("feature_order must be a non-empty list")
    feature_order = [
        _bounded_int(item, "feature index", 0, COMPONENT_COUNT - 1)
        for item in order_raw
    ]
    if len(feature_order) != len(set(feature_order)):
        raise ArchitectureError("feature_order cannot contain duplicate indices")

    layers_raw = raw.get("layers", [])
    if not isinstance(layers_raw, list):
        raise ArchitectureError("layers must be a list")
    layers: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, item in enumerate(layers_raw):
        if not isinstance(item, dict):
            raise ArchitectureError(f"layer {index + 1} must be an object")
        layer_type = str(item.get("type", ""))
        if layer_type not in LAYER_TYPES:
            raise ArchitectureError(f"unknown layer type {layer_type!r}")
        layer_id = str(item.get("id", f"layer-{index + 1}")).strip()
        if not layer_id or len(layer_id) > 64:
            raise ArchitectureError(f"layer {index + 1} has an invalid id")
        if layer_id in ids:
            raise ArchitectureError(f"duplicate layer id {layer_id!r}")
        ids.add(layer_id)
        params = item.get("params", {})
        if not isinstance(params, dict):
            raise ArchitectureError(f"parameters for {layer_id!r} must be an object")
        layers.append(
            {
                "id": layer_id,
                "type": layer_type,
                "params": _normalize_layer_params(layer_type, params),
            }
        )

    head_raw = raw.get("head", {})
    if not isinstance(head_raw, dict):
        raise ArchitectureError("head must be an object")
    head_type = str(head_raw.get("type", "direct"))
    if head_type not in HEAD_TYPES:
        raise ArchitectureError(f"head type must be one of {HEAD_TYPES}")
    zero_default = head_type != "direct"
    zero_initialize = bool(head_raw.get("zero_initialize", zero_default))

    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "input": {
            "representation": representation,
            "feature_order": feature_order,
        },
        "layers": layers,
        "head": {"type": head_type, "zero_initialize": zero_initialize},
    }


def _normalize_layer_params(layer_type: str, params: dict[str, Any]) -> dict[str, Any]:
    if layer_type.startswith("tslib_"):
        defaults = upstream_defaults(layer_type[6:])
        bounds = {"d_model": (8, 256), "num_layers": (1, 3), "n_heads": (1, 16)}
        normalized = {key: (_bounded_float(params.get(key, value), key, 0, 0.95)
                            if key == "dropout" else _bounded_int(params.get(key, value), key, *bounds[key]))
                      for key, value in defaults.items()}
        if normalized.get("d_model", 32) % 8:
            raise ArchitectureError("TSLib d_model must be a multiple of 8")
        if normalized.get("d_model", 32) % normalized.get("n_heads", 4):
            raise ArchitectureError("d_model must be divisible by n_heads")
        return normalized
    if layer_type == "temporal_projection":
        return {"output_steps": _bounded_int(params.get("output_steps", 8), "output_steps", 1, 512)}
    if layer_type == "patch_embedding":
        patch = _bounded_int(params.get("patch_size", 4), "patch_size", 1, 128)
        stride = _bounded_int(params.get("stride", 2), "stride", 1, 128)
        if stride > patch:
            raise ArchitectureError("stride must not exceed patch_size")
        return {"d_model": _bounded_int(params.get("d_model", 32), "d_model", 1, 512),
                "patch_size": patch, "stride": stride}
    if layer_type in {"dlinear", "decomposition"}:
        kernel = _bounded_int(params.get("kernel_size", 25), "kernel_size", 1, 101)
        if kernel % 2 == 0:
            raise ArchitectureError("Decomposition kernel_size must be odd")
        return {"kernel_size": kernel}
    if layer_type in {"patchtst", "itransformer", "temporal_attention", "variable_attention"}:
        normalized = {
            "d_model": _bounded_int(params.get("d_model", 32), "d_model", 4, 256),
            "n_heads": _bounded_int(params.get("n_heads", 4), "n_heads", 1, 16),
            "num_layers": _bounded_int(params.get("num_layers", 2), "num_layers", 1, 4),
            "dropout": _bounded_float(params.get("dropout", 0.1), "dropout", 0, 0.95),
        }
        if normalized["d_model"] % normalized["n_heads"]:
            raise ArchitectureError("d_model must be divisible by n_heads")
        if layer_type == "patchtst":
            normalized["patch_size"] = _bounded_int(params.get("patch_size", 4), "patch_size", 1, 128)
            normalized["stride"] = _bounded_int(params.get("stride", 2), "stride", 1, 128)
            if normalized["stride"] > normalized["patch_size"]:
                raise ArchitectureError("PatchTST stride must not exceed patch_size")
        return normalized
    if layer_type == "dense":
        return {"units": _bounded_int(params.get("units", 32), "units", 1, 512)}
    if layer_type == "hyper_dense":
        algebra = str(params.get("algebra", "quaternion"))
        if algebra not in ALGEBRAS:
            raise ArchitectureError(f"algebra must be one of {ALGEBRAS}")
        return {
            "units": _bounded_int(params.get("units", 8), "units", 1, 128),
            "algebra": algebra,
        }
    if layer_type == "causal_conv":
        return {
            "channels": _bounded_int(
                params.get("channels", 16), "channels", 1, 512
            ),
            "kernel_size": _bounded_int(
                params.get("kernel_size", 3), "kernel_size", 1, 31
            ),
            "dilation": _bounded_int(
                params.get("dilation", 1), "dilation", 1, 64
            ),
        }
    if layer_type == "tcn":
        dilations_raw = params.get("dilations", [1, 2, 4, 8])
        if not isinstance(dilations_raw, list) or not dilations_raw:
            raise ArchitectureError("dilations must be a non-empty list")
        if len(dilations_raw) > 8:
            raise ArchitectureError("dilations can contain at most 8 values")
        return {
            "channels": _bounded_int(
                params.get("channels", 16), "channels", 1, 512
            ),
            "kernel_size": _bounded_int(
                params.get("kernel_size", 3), "kernel_size", 1, 31
            ),
            "dilations": [
                _bounded_int(value, "dilation", 1, 64) for value in dilations_raw
            ],
            "dropout": _bounded_float(
                params.get("dropout", 0.1), "dropout", 0.0, 0.95
            ),
        }
    if layer_type in {"lstm", "gru"}:
        return {
            "hidden_size": _bounded_int(
                params.get("hidden_size", 16), "hidden_size", 1, 512
            ),
            "num_layers": _bounded_int(
                params.get("num_layers", 1), "num_layers", 1, 3
            ),
            "dropout": _bounded_float(
                params.get("dropout", 0.0), "dropout", 0.0, 0.95
            ),
        }
    if layer_type == "activation":
        kind = str(params.get("kind", "relu"))
        if kind not in ACTIVATIONS:
            raise ArchitectureError(f"activation must be one of {ACTIVATIONS}")
        return {"kind": kind}
    if layer_type == "dropout":
        return {"p": _bounded_float(params.get("p", 0.1), "p", 0.0, 0.95)}
    if layer_type == "max_pool":
        return {
            "size": _bounded_int(params.get("size", 2), "size", 1, 64),
            "stride": _bounded_int(
                params.get("stride", params.get("size", 2)), "stride", 1, 64
            ),
        }
    if layer_type in {"layer_norm", "last_state", "mean_pool", "flatten"}:
        return {}
    raise ArchitectureError(f"unknown layer type {layer_type!r}")


class SequenceConv(nn.Module):
    def __init__(self, in_channels: int, channels: int, kernel: int, dilation: int):
        super().__init__()
        self.conv = CausalConv1d(
            in_channels, channels, kernel_size=kernel, dilation=dilation
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs.transpose(1, 2)).transpose(1, 2)


class RecurrentSequence(nn.Module):
    def __init__(
        self,
        kind: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        recurrent = nn.LSTM if kind == "lstm" else nn.GRU
        self.recurrent = recurrent(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if kind == "lstm":
            # Keras exposes one LSTM bias vector. Freezing PyTorch's second
            # vector retains equivalent behavior and paper-comparable counts.
            for layer in range(num_layers):
                getattr(self.recurrent, f"bias_hh_l{layer}").requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.recurrent(inputs)
        return sequence


class TemporalConvNet(nn.Module):
    """Small causal residual TCN that retains the input sequence length."""

    def __init__(
        self,
        input_width: int,
        channels: int,
        kernel_size: int,
        dilations: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = (
            nn.Linear(input_width, channels)
            if input_width != channels
            else nn.Identity()
        )
        self.convolutions = nn.ModuleList(
            [
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for dilation in dilations
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in dilations])
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = self.input_projection(inputs)
        for convolution, norm in zip(self.convolutions, self.norms, strict=True):
            residual = sequence
            encoded = convolution(sequence.transpose(1, 2)).transpose(1, 2)
            sequence = norm(residual + self.dropout(self.activation(encoded)))
        return sequence


class SequencePool(nn.Module):
    def __init__(self, size: int, stride: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool1d(size, stride=stride)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pool(inputs.transpose(1, 2)).transpose(1, 2)


class LastState(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, -1]


class MeanPool(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.mean(dim=1)


class FlattenSequence(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.flatten(start_dim=1)


def _activation(kind: str) -> nn.Module:
    return {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "linear": nn.Identity,
    }[kind]()


def _build_layer(
    layer: dict[str, Any], shape: TensorShape
) -> tuple[nn.Module, TensorShape, int]:
    layer_type = layer["type"]
    params = layer["params"]
    receptive_addition = 0
    if layer_type.startswith("tslib_"):
        if shape.kind != "sequence" or shape.steps is None:
            raise ArchitectureError(f"{layer_type} requires a sequence; place it before sequence reduction")
        return UpstreamSequence(layer_type[6:], shape.steps, shape.width, **params), shape, shape.steps - 1
    if layer_type in RESEARCH_BLOCKS | RESEARCH_MODELS:
        if shape.kind != "sequence" or shape.steps is None:
            raise ArchitectureError(f"{layer_type} requires a sequence; place it before sequence reduction")
        steps, width = shape.steps, shape.width
        if layer_type == "decomposition":
            return Decomposition(**params), TensorShape("sequence", width * 2, steps), params["kernel_size"] - 1
        if layer_type == "temporal_projection":
            return TemporalProjection(steps, width, **params), TensorShape("sequence", width, params["output_steps"]), steps - 1
        if layer_type == "patch_embedding":
            module = PatchEmbedding(steps, width, **params)
            return module, TensorShape("sequence", params["d_model"], module.output_steps), params["patch_size"] - 1
        if layer_type in {"temporal_attention", "variable_attention", "itransformer"}:
            variables = layer_type != "temporal_attention"
            return (SequenceAttention(steps, width, **params, variables=variables),
                    TensorShape("sequence", width if variables else params["d_model"], steps), steps - 1)
        # Legacy whole-model names remain usable in hybrid pipelines as sequence encoders.
        if layer_type == "dlinear":
            return nn.Sequential(Decomposition(**params), TemporalProjection(steps, width * 2, steps)), TensorShape("sequence", width * 2, steps), steps - 1
        patch_params = {key: params[key] for key in ("d_model", "patch_size", "stride")}
        patch = PatchEmbedding(steps, width, **patch_params)
        attention_params = {key: value for key, value in params.items() if key not in {"patch_size", "stride"}}
        return nn.Sequential(patch, SequenceAttention(patch.output_steps, params["d_model"], **attention_params)), TensorShape("sequence", params["d_model"], patch.output_steps), steps - 1
    if layer_type == "dense":
        module = nn.Linear(shape.width, params["units"])
        return module, TensorShape(shape.kind, params["units"], shape.steps), 0
    if layer_type == "hyper_dense":
        if shape.kind != "sequence":
            raise ArchitectureError("HyperDense requires a sequence")
        if shape.width % COMPONENT_COUNT:
            raise ArchitectureError("HyperDense input width must be divisible by four")
        module = HyperDense(
            shape.width // COMPONENT_COUNT,
            params["units"],
            algebra=params["algebra"],
        )
        return (
            module,
            TensorShape("sequence", COMPONENT_COUNT * params["units"], shape.steps),
            0,
        )
    if layer_type == "causal_conv":
        if shape.kind != "sequence":
            raise ArchitectureError("Causal convolution requires a sequence")
        module = SequenceConv(
            shape.width,
            params["channels"],
            params["kernel_size"],
            params["dilation"],
        )
        receptive_addition = params["dilation"] * (params["kernel_size"] - 1)
        return (
            module,
            TensorShape("sequence", params["channels"], shape.steps),
            receptive_addition,
        )
    if layer_type == "tcn":
        if shape.kind != "sequence":
            raise ArchitectureError("TCN requires a sequence")
        module = TemporalConvNet(
            shape.width,
            params["channels"],
            params["kernel_size"],
            params["dilations"],
            params["dropout"],
        )
        receptive_addition = sum(params["dilations"]) * (
            params["kernel_size"] - 1
        )
        return (
            module,
            TensorShape("sequence", params["channels"], shape.steps),
            receptive_addition,
        )
    if layer_type in {"lstm", "gru"}:
        if shape.kind != "sequence":
            raise ArchitectureError(f"{layer_type.upper()} requires a sequence")
        module = RecurrentSequence(
            layer_type,
            shape.width,
            params["hidden_size"],
            params["num_layers"],
            params["dropout"],
        )
        return (
            module,
            TensorShape("sequence", params["hidden_size"], shape.steps),
            max((shape.steps or 1) - 1, 0),
        )
    if layer_type == "layer_norm":
        return nn.LayerNorm(shape.width), shape, 0
    if layer_type == "activation":
        return _activation(params["kind"]), shape, 0
    if layer_type == "dropout":
        return nn.Dropout(params["p"]), shape, 0
    if layer_type == "max_pool":
        if shape.kind != "sequence" or shape.steps is None:
            raise ArchitectureError("Max pooling requires a sequence")
        if params["size"] > shape.steps:
            raise ArchitectureError("Max-pool size exceeds the remaining time steps")
        steps = (shape.steps - params["size"]) // params["stride"] + 1
        return SequencePool(params["size"], params["stride"]), TensorShape(
            "sequence", shape.width, steps
        ), 0
    if layer_type == "last_state":
        if shape.kind != "sequence":
            raise ArchitectureError("Last-state selection requires a sequence")
        return LastState(), TensorShape("vector", shape.width), 0
    if layer_type == "mean_pool":
        if shape.kind != "sequence":
            raise ArchitectureError("Mean pooling requires a sequence")
        return MeanPool(), TensorShape("vector", shape.width), 0
    if layer_type == "flatten":
        if shape.kind != "sequence" or shape.steps is None:
            raise ArchitectureError("Flattening requires a sequence")
        width = shape.steps * shape.width
        return FlattenSequence(), TensorShape("vector", width), 0
    raise ArchitectureError(f"unknown layer type {layer_type!r}")


class ExperimentalForecaster(nn.Module):
    """Forecasting network compiled from an ArchitectureSpec."""

    def __init__(
        self,
        spec: dict[str, Any],
        window: int,
        horizon: int,
        features: int = COMPONENT_COUNT,
    ) -> None:
        super().__init__()
        if window < 2 or horizon < 1:
            raise ArchitectureError("window must be at least 2 and horizon positive")
        self.spec = normalize_architecture_spec(spec)
        self.representation = self.spec["input"]["representation"]
        self.feature_order = tuple(self.spec["input"]["feature_order"])
        if max(self.feature_order) >= features:
            raise ArchitectureError("feature_order references an unavailable feature")
        steps = window - 1 if self.representation == "differences" else window
        shape = TensorShape("sequence", len(self.feature_order), steps)
        if (len(self.spec["layers"]) == 1 and self.spec["layers"][0]["type"] in RESEARCH_MODELS
                and self.spec["head"] == {"type": "direct", "zero_initialize": False}):
            layer = self.spec["layers"][0]
            kind = layer["type"]
            module = (DLinear(window, horizon, **layer["params"]) if kind == "dlinear"
                      else AttentionForecast(kind, window, horizon, **layer["params"]))
            self.layers = nn.ModuleList([module])
            self.output = nn.Identity()
            self.head_type = "direct"
            self.output_shape = TensorShape("vector", horizon)
            self.receptive_field = window
            self._reset_initialization()
            self.trace = [TraceEntry(layer["id"], kind, shape.label(),
                                     self.output_shape.label(), parameter_count(module), window)]
            return
        modules: list[nn.Module] = []
        self.trace: list[TraceEntry] = []
        receptive_field = 1
        temporal_jump = 1
        for layer in self.spec["layers"]:
            input_shape = shape
            module, shape, receptive_addition = _build_layer(layer, shape)
            receptive_field = min(
                steps, receptive_field + receptive_addition * temporal_jump
            )
            if layer["type"] == "max_pool":
                receptive_field = min(
                    steps,
                    receptive_field
                    + (layer["params"]["size"] - 1) * temporal_jump,
                )
                temporal_jump *= layer["params"]["stride"]
            if layer["type"] == "patch_embedding":
                temporal_jump *= layer["params"]["stride"]
            if layer["type"] in {"flatten", "mean_pool"}:
                # The reduced output can depend on every retained token.
                receptive_field = min(steps, receptive_field + ((input_shape.steps or 1) - 1) * temporal_jump)
            modules.append(module)
            self.trace.append(
                TraceEntry(
                    layer_id=layer["id"],
                    layer_type=layer["type"],
                    input_shape=input_shape.label(),
                    output_shape=shape.label(),
                    parameters=parameter_count(module),
                    receptive_field=receptive_field,
                )
            )
        if shape.kind != "vector":
            raise ArchitectureError(
                "architecture must reduce the sequence with last_state, mean_pool, "
                "or flatten before the forecast head"
            )
        self.layers = nn.ModuleList(modules)
        self.output = nn.Linear(shape.width, horizon)
        self.head_type = self.spec["head"]["type"]
        self._reset_initialization()
        if self.spec["head"]["zero_initialize"]:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        self.output_shape = TensorShape("vector", horizon)
        self.receptive_field = receptive_field

    def _reset_initialization(self) -> None:
        """Use the paper/Keras conventions for comparable editable presets."""
        upstream_modules = {child for module in self.modules()
                            if isinstance(module, UpstreamSequence) for child in module.modules()}
        for module in self.modules():
            if module in upstream_modules:
                continue
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                for name, parameter in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(parameter)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(parameter)
                    elif "bias" in name:
                        nn.init.zeros_(parameter)
                if isinstance(module, nn.LSTM):
                    for layer in range(module.num_layers):
                        bias = getattr(module, f"bias_ih_l{layer}")
                        hidden = module.hidden_size
                        bias.data[hidden : 2 * hidden].fill_(1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        anchor = inputs[:, -1, 0:1]
        sequence = inputs[..., list(self.feature_order)]
        if self.representation == "centered":
            sequence = sequence - sequence[:, -1:, :]
        elif self.representation == "differences":
            sequence = sequence[:, 1:] - sequence[:, :-1]
        encoded = sequence
        for layer in self.layers:
            encoded = layer(encoded)
        offsets = self.output(encoded)
        if self.head_type == "direct":
            return offsets
        if self.head_type == "cumulative_residual":
            offsets = torch.cumsum(offsets, dim=-1)
        return anchor + offsets


def build_architecture(
    spec: dict[str, Any], window: int, horizon: int, features: int = COMPONENT_COUNT
) -> ExperimentalForecaster:
    return ExperimentalForecaster(spec, window, horizon, features)


def validate_architecture(
    spec: dict[str, Any], window: int, horizon: int, features: int = COMPONENT_COUNT
) -> dict[str, Any]:
    normalized = normalize_architecture_spec(spec)
    model = build_architecture(normalized, window, horizon, features)
    input_steps = window - 1 if normalized["input"]["representation"] == "differences" else window
    return {
        "valid": True,
        "spec": normalized,
        "input_shape": f"[B, {input_steps}, {len(normalized['input']['feature_order'])}]",
        "output_shape": f"[B, {horizon}]",
        "parameters": parameter_count(model),
        "receptive_field": model.receptive_field,
        "trace": [asdict(item) for item in model.trace],
    }


def architecture_hash(spec: dict[str, Any], evaluation: dict[str, Any] | None = None) -> str:
    canonical = normalize_architecture_spec(spec)
    canonical.pop("name", None)
    payload: dict[str, Any] = {"architecture": canonical}
    if evaluation is not None:
        payload["evaluation"] = evaluation
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


LAYER_TYPES = {
    *(f"tslib_{method}" for method in UPSTREAM_MODELS),
    *RESEARCH_MODELS,
    *RESEARCH_BLOCKS,
    "dense",
    "hyper_dense",
    "causal_conv",
    "tcn",
    "lstm",
    "gru",
    "layer_norm",
    "activation",
    "dropout",
    "max_pool",
    "last_state",
    "mean_pool",
    "flatten",
}


def layer_catalog() -> dict[str, Any]:
    """Return UI metadata without exposing arbitrary Python construction."""
    return {
        "schema_version": SCHEMA_VERSION,
        "categories": [
            {"name": "TSLib model blocks", "layers": [
                {"type": f"tslib_{key}", "label": f"{name} (TSLib)", "defaults": upstream_defaults(key)}
                for key, name in UPSTREAM_MODELS.items()
            ]},
            {"name": "Research blocks", "layers": [
                {"type": "decomposition", "label": "Trend / seasonal split", "defaults": {"kernel_size": 25}},
                {"type": "patch_embedding", "label": "Patch embedding", "defaults": {"d_model": 32, "patch_size": 4, "stride": 2}},
                {"type": "temporal_attention", "label": "Temporal attention", "defaults": {"d_model": 32, "n_heads": 4, "num_layers": 2, "dropout": 0.1}},
                {"type": "variable_attention", "label": "Cross-variable attention", "defaults": {"d_model": 32, "n_heads": 4, "num_layers": 2, "dropout": 0.1}},
                {"type": "temporal_projection", "label": "Temporal projection", "defaults": {"output_steps": 8}},
            ]},
            {
                "name": "Feature mixing",
                "layers": [
                    {"type": "dense", "label": "Dense", "defaults": {"units": 32}},
                    {
                        "type": "hyper_dense",
                        "label": "HyperDense",
                        "defaults": {"units": 8, "algebra": "quaternion"},
                    },
                ],
            },
            {
                "name": "Temporal",
                "layers": [
                    {
                        "type": "causal_conv",
                        "label": "Causal Conv1D",
                        "defaults": {"channels": 16, "kernel_size": 3, "dilation": 1},
                    },
                    {
                        "type": "tcn",
                        "label": "Residual TCN",
                        "defaults": {
                            "channels": 16,
                            "kernel_size": 3,
                            "dilations": [1, 2, 4, 8],
                            "dropout": 0.1,
                        },
                    },
                    {
                        "type": "lstm",
                        "label": "LSTM",
                        "defaults": {"hidden_size": 16, "num_layers": 1, "dropout": 0.0},
                    },
                    {
                        "type": "gru",
                        "label": "GRU",
                        "defaults": {"hidden_size": 16, "num_layers": 1, "dropout": 0.0},
                    },
                ],
            },
            {
                "name": "Processing",
                "layers": [
                    {"type": "layer_norm", "label": "Layer norm", "defaults": {}},
                    {
                        "type": "activation",
                        "label": "Activation",
                        "defaults": {"kind": "relu"},
                    },
                    {"type": "dropout", "label": "Dropout", "defaults": {"p": 0.1}},
                    {
                        "type": "max_pool",
                        "label": "Max pool",
                        "defaults": {"size": 2, "stride": 2},
                    },
                ],
            },
            {
                "name": "Sequence reduction",
                "layers": [
                    {"type": "last_state", "label": "Last state", "defaults": {}},
                    {"type": "mean_pool", "label": "Mean pool", "defaults": {}},
                    {"type": "flatten", "label": "Flatten", "defaults": {}},
                ],
            },
        ],
        "input_representations": list(INPUT_REPRESENTATIONS),
        "head_types": list(HEAD_TYPES),
        "algebras": list(ALGEBRAS),
        "activations": list(ACTIVATIONS),
    }


def _layer(layer_id: str, layer_type: str, **params: Any) -> dict[str, Any]:
    return {"id": layer_id, "type": layer_type, "params": params}


def presets() -> list[dict[str, Any]]:
    """Return reference and recommended architecture starting points."""
    common_input = {"representation": "levels", "feature_order": [0, 1, 2, 3]}
    output: list[dict[str, Any]] = []
    for first_type, title, params in (
        ("causal_conv", "Paper CNN", {"channels": 16, "kernel_size": 3, "dilation": 1}),
        ("lstm", "Paper LSTM", {"hidden_size": 16, "num_layers": 1, "dropout": 0.0}),
    ):
        output.append(
            normalize_architecture_spec(
                {
                    "name": title,
                    "input": common_input,
                    "layers": [
                        _layer("first", first_type, **params),
                        *([] if first_type == "lstm" else [_layer("first-relu", "activation", kind="relu")]),
                        _layer("before-pool", "dense", units=32),
                        _layer("before-relu", "activation", kind="relu"),
                        _layer("pool", "max_pool", size=2, stride=2),
                        _layer("flatten", "flatten"),
                        _layer("after-pool", "dense", units=32),
                        _layer("after-relu", "activation", kind="relu"),
                        _layer("dropout", "dropout", p=0.5),
                    ],
                    "head": {"type": "direct", "zero_initialize": False},
                }
            )
            | {"locked": True, "preset_id": title.lower().replace(" ", "-")}
        )
    for algebra, title in (
        ("quaternion", "Paper Quaternion"),
        ("coquaternion", "Paper Coquaternion"),
        ("cl11", "Paper Cl(1,1)"),
    ):
        output.append(
            normalize_architecture_spec(
                {
                    "name": title,
                    "input": common_input,
                    "layers": [
                        _layer("first", "hyper_dense", units=8, algebra=algebra),
                        _layer("first-relu", "activation", kind="relu"),
                        _layer("before-pool", "dense", units=32),
                        _layer("before-relu", "activation", kind="relu"),
                        _layer("pool", "max_pool", size=2, stride=2),
                        _layer("flatten", "flatten"),
                        _layer("after-pool", "dense", units=32),
                        _layer("after-relu", "activation", kind="relu"),
                        _layer("dropout", "dropout", p=0.5),
                    ],
                    "head": {"type": "direct", "zero_initialize": False},
                }
            )
            | {"locked": True, "preset_id": f"paper-{algebra}"}
        )

    output.extend(
        [
            normalize_architecture_spec(
                {
                    "name": "Residual TCN",
                    "input": {"representation": "differences", "feature_order": [0, 1, 2, 3]},
                    "layers": [
                        _layer(
                            "tcn",
                            "tcn",
                            channels=16,
                            kernel_size=3,
                            dilations=[1, 2, 4, 8],
                            dropout=0.1,
                        ),
                        _layer("last", "last_state"),
                    ],
                    "head": {"type": "persistence_residual", "zero_initialize": True},
                }
            )
            | {"locked": False, "preset_id": "residual-tcn"},
            normalize_architecture_spec(
                {
                    "name": "Quaternion GRU Residual",
                    "input": {"representation": "differences", "feature_order": [0, 1, 2, 3]},
                    "layers": [
                        _layer("hyper", "hyper_dense", units=8, algebra="quaternion"),
                        _layer("norm", "layer_norm"),
                        _layer("gru", "gru", hidden_size=16, num_layers=1, dropout=0.0),
                        _layer("last", "last_state"),
                    ],
                    "head": {"type": "persistence_residual", "zero_initialize": True},
                }
            )
            | {"locked": False, "preset_id": "quaternion-gru-residual"},
        ]
    )
    for kind, title in (("dlinear", "DLinear"), ("patchtst", "PatchTST"),
                        ("itransformer", "iTransformer")):
        output.append(normalize_architecture_spec({
            "name": f"{title}-inspired (editable)",
            "input": {"representation": "levels", "feature_order": [0] if kind in {"dlinear", "patchtst"} else [0, 1, 2, 3]},
            "layers": ({
                "dlinear": [_layer("split", "decomposition"), _layer("project", "temporal_projection"), _layer("flatten", "flatten")],
                "patchtst": [_layer("patches", "patch_embedding"), _layer("attention", "temporal_attention"), _layer("flatten", "flatten")],
                "itransformer": [_layer("variables", "variable_attention"), _layer("flatten", "flatten")],
            })[kind],
            "head": {"type": "direct", "zero_initialize": False},
        }) | {"locked": False, "preset_id": f"research-{kind}"})
    for key, name in UPSTREAM_MODELS.items():
        output.append(normalize_architecture_spec({
            "name": f"{name} (TSLib core, editable)",
            "input": {"representation": "levels", "feature_order": [0, 1, 2, 3]},
            "layers": [_layer("core", f"tslib_{key}"), _layer("flatten", "flatten")],
            "head": {"type": "direct", "zero_initialize": False},
        }) | {"locked": False, "preset_id": f"tslib-{key}"})
    return copy.deepcopy(output)
