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
