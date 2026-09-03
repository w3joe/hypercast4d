import copy

import pytest
import torch

from hypercast4d.architecture import (
    ArchitectureError,
    architecture_hash,
    build_architecture,
    presets,
    validate_architecture,
)
from hypercast4d.models import build_model, parameter_count


def _preset(preset_id: str) -> dict:
    return next(item for item in presets() if item["preset_id"] == preset_id)


@pytest.mark.parametrize(
    ("preset_id", "model_name", "first_units"),
    [
        ("paper-cnn", "cnn", 16),
        ("paper-lstm", "lstm", 16),
        ("paper-quaternion", "hyper_quaternion", 8),
        ("paper-coquaternion", "hyper_coquaternion", 8),
        ("paper-cl11", "hyper_cl11", 8),
    ],
)
def test_paper_presets_match_existing_parameter_counts(
    preset_id: str, model_name: str, first_units: int
) -> None:
    options = {
        "first_layer_units": first_units,
        "conv_kernel_size": 3 if model_name == "cnn" else None,
        "dense_before_pool": True,
        "dense_after_pool": True,
        "dense_units": 32,
        "activation": "relu",
        "dropout": 0.5,
        "pool_size": 2,
    }
    expected = build_model(model_name, window=10, horizon=5, **options)
    compiled = build_architecture(_preset(preset_id), window=10, horizon=5)
    assert parameter_count(compiled) == parameter_count(expected)
    assert compiled(torch.randn(3, 10, 4)).shape == (3, 5)


def test_zero_initialized_residual_head_is_exact_persistence() -> None:
    model = build_architecture(_preset("residual-tcn"), window=20, horizon=5)
    inputs = torch.randn(4, 20, 4)
    expected = inputs[:, -1, 0:1].repeat(1, 5)
    assert torch.equal(model(inputs), expected)


def test_cumulative_head_accumulates_offsets() -> None:
    spec = copy.deepcopy(_preset("residual-tcn"))
    spec["head"] = {"type": "cumulative_residual", "zero_initialize": False}
    model = build_architecture(spec, window=10, horizon=3)
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
    inputs = torch.zeros(2, 10, 4)
    assert torch.equal(model(inputs), torch.tensor([[1.0, 3.0, 6.0]]).repeat(2, 1))


def test_invalid_layer_order_reports_a_useful_error() -> None:
    spec = copy.deepcopy(_preset("residual-tcn"))
    spec["layers"] = [
        {"id": "last", "type": "last_state", "params": {}},
        {"id": "conv", "type": "causal_conv", "params": {}},
    ]
    with pytest.raises(ArchitectureError, match="requires a sequence"):
        build_architecture(spec, window=10, horizon=1)


def test_shape_trace_captures_difference_length_and_receptive_field() -> None:
    result = validate_architecture(_preset("residual-tcn"), window=20, horizon=5)
    assert result["input_shape"] == "[B, 19, 4]"
    assert result["output_shape"] == "[B, 5]"
    assert result["receptive_field"] == 19
    assert result["trace"][-1]["output_shape"] == "[B, 16]"


def test_architecture_hash_ignores_display_name() -> None:
    first = _preset("residual-tcn")
    second = copy.deepcopy(first)
    second["name"] = "Renamed"
    assert architecture_hash(first) == architecture_hash(second)
