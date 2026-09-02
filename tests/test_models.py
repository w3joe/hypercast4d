import pytest
import torch

from hypercast4d.models import build_model


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
    model = build_model(name, window=10, horizon=5)
    assert model(torch.randn(7, 10, 4)).shape == (7, 5)


def test_hyper_model_uses_requested_algebra() -> None:
    model = build_model("hyper_cl11", window=10, horizon=1)
    assert model.hyper.algebra.name == "cl11"


def test_configurable_architecture_values_are_used() -> None:
    cnn = build_model("cnn", window=10, horizon=1, cnn_kernel_size=5)
    hyper = build_model("hyper_quaternion", window=12, horizon=1, hyper_pool_size=3)
    assert cnn.features[0].kernel_size == (5,)
    assert hyper.pool.kernel_size == 3
    assert hyper(torch.randn(2, 12, 4)).shape == (2, 1)


@pytest.mark.parametrize("kernel_size", [0, 2, 4])
def test_invalid_cnn_kernel_size_is_rejected(kernel_size: int) -> None:
    with pytest.raises(ValueError, match="kernel_size"):
        build_model("cnn", window=10, horizon=1, cnn_kernel_size=kernel_size)
