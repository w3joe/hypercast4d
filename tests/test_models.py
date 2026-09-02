import pytest
import torch
from torch import nn

from hypercast4d.models import CausalConv1d, PaperForecaster, build_model


PAPER_MODEL_OPTIONS = {
    "first_layer_units": 16,
    "conv_kernel_size": 3,
    "dense_before_pool": True,
    "dense_after_pool": True,
    "dense_units": 32,
    "activation": "relu",
    "dropout": 0.5,
    "pool_size": 2,
}


@pytest.mark.parametrize(
    "name",
    [
        "linear",
        "cnn",
        "lstm",
        "hyper_quaternion",
        "hyper_coquaternion",
        "hyper_cl11",
    ],
)
def test_model_output_shape(name: str) -> None:
    model = build_model(name, window=10, horizon=5, **PAPER_MODEL_OPTIONS)
    assert model(torch.randn(7, 10, 4)).shape == (7, 5)


def test_hyper_model_uses_requested_algebra() -> None:
    model = build_model("hyper_cl11", window=10, horizon=1, **PAPER_MODEL_OPTIONS)
    assert isinstance(model, PaperForecaster)
    assert model.first.algebra.name == "cl11"


def test_paper_scaffold_contains_both_optional_dense_layers() -> None:
    model = build_model("cnn", window=10, horizon=1, **PAPER_MODEL_OPTIONS)
    assert isinstance(model, PaperForecaster)
    assert isinstance(model.before_pool, nn.Sequential)
    assert isinstance(model.after_pool, nn.Sequential)
    assert model.dropout.p == 0.5


def test_optional_dense_layers_can_be_disabled() -> None:
    options = PAPER_MODEL_OPTIONS | {
        "dense_before_pool": False,
        "dense_after_pool": False,
    }
    model = build_model("hyper_quaternion", window=10, horizon=1, **options)
    assert isinstance(model.before_pool, nn.Identity)
    assert isinstance(model.after_pool, nn.Identity)
    assert model(torch.randn(2, 10, 4)).shape == (2, 1)


def test_configurable_architecture_values_are_used() -> None:
    options = PAPER_MODEL_OPTIONS | {
        "conv_kernel_size": 5,
        "pool_size": 3,
    }
    cnn = build_model("cnn", window=12, horizon=1, **options)
    assert cnn.first.kernel_size == (5,)
    assert cnn.pool.kernel_size == 3


def test_causal_convolution_does_not_use_future_values() -> None:
    layer = CausalConv1d(1, 1, kernel_size=3, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    original = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    changed_future = torch.tensor([[[1.0, 2.0, 300.0, 400.0]]])
    assert torch.equal(layer(original)[:, :, :2], layer(changed_future)[:, :, :2])
    assert layer(original).shape[-1] == original.shape[-1]


@pytest.mark.parametrize("kernel_size", [0, -1])
def test_invalid_cnn_kernel_size_is_rejected(kernel_size: int) -> None:
    options = PAPER_MODEL_OPTIONS | {"conv_kernel_size": kernel_size}
    with pytest.raises(ValueError, match="kernel_size"):
        build_model("cnn", window=10, horizon=1, **options)


@pytest.mark.parametrize("activation", ["sigmoid", "tanh"])
def test_activation_outside_paper_grid_is_rejected(activation: str) -> None:
    options = PAPER_MODEL_OPTIONS | {"activation": activation}
    with pytest.raises(ValueError, match="activation"):
        build_model("cnn", window=10, horizon=1, **options)
