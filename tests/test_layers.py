import torch
import pytest

from hypercast4d.algebras import ALGEBRAS
from hypercast4d.layers import HyperDense


def test_hyperdense_matches_input_by_weight_basis_product() -> None:
    layer = HyperDense(1, 1, algebra="quaternion", bias=False).double()
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[1, 0, 0] = 1.0  # right factor i
    inputs = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float64)  # left factor j
    expected = torch.tensor([[0.0, 0.0, 0.0, -1.0]], dtype=torch.float64)
    assert torch.equal(layer(inputs), expected)


def test_algebra_argument_changes_computation() -> None:
    outputs = []
    inputs = torch.tensor([[0.2, -0.7, 1.1, 0.4]])
    fixed_weight = torch.tensor([[[0.3]], [[0.5]], [[-0.2]], [[0.9]]])
    for algebra in ("quaternion", "coquaternion", "cl11"):
        layer = HyperDense(1, 1, algebra=algebra, bias=False)
        with torch.no_grad():
            layer.weight.copy_(fixed_weight)
        outputs.append(layer(inputs))
    assert not torch.allclose(outputs[0], outputs[1])
    assert not torch.allclose(outputs[0], outputs[2])
    assert not torch.allclose(outputs[1], outputs[2])


def test_gradients_are_finite() -> None:
    layer = HyperDense(2, 3, algebra="coquaternion")
    inputs = torch.randn(5, 8, requires_grad=True)
    layer(inputs).square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(layer.weight.grad).all()


def test_expected_parameter_reduction() -> None:
    layer = HyperDense(3, 5, algebra="quaternion")
    hyper_parameters = sum(parameter.numel() for parameter in layer.parameters())
    real_parameters = (4 * 3) * (4 * 5) + 4 * 5
    assert hyper_parameters == 4 * 3 * 5 + 4 * 5
    assert hyper_parameters < real_parameters


@pytest.mark.parametrize("name", ALGEBRAS)
def test_hyperdense_uses_selected_algebra_dimension(name: str) -> None:
    algebra = ALGEBRAS[name]
    dimension = algebra.component_count
    layer = HyperDense(2, 3, algebra=name)
    inputs = torch.randn(4, 5, dimension * 2, requires_grad=True)
    outputs = layer(inputs)
    assert outputs.shape == (4, 5, dimension * 3)
    assert layer.weight.shape == (dimension, 2, 3)
    assert layer.bias.shape == (dimension, 3)
    outputs.square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(layer.weight.grad).all()


@pytest.mark.parametrize("name", ALGEBRAS)
def test_hyperdense_matches_direct_algebra_product(name: str) -> None:
    algebra = ALGEBRAS[name]
    dimension = algebra.component_count
    layer = HyperDense(1, 1, algebra=name, bias=False).double()
    left = torch.linspace(-0.4, 0.6, dimension, dtype=torch.float64)
    right = torch.linspace(0.7, -0.2, dimension, dtype=torch.float64)
    with torch.no_grad():
        layer.weight[:, 0, 0].copy_(right)
    torch.testing.assert_close(
        layer(left.unsqueeze(0))[0],
        algebra.multiply(left, right),
        rtol=0,
        atol=1e-15,
    )


def test_hyperdense_reports_algebra_specific_input_width() -> None:
    layer = HyperDense(2, 1, algebra="tricomplex")
    with pytest.raises(ValueError, match="Expected final dimension 6"):
        layer(torch.randn(2, 8))
