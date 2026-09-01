import torch

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
    layer = HyperDense(3, 5)
    hyper_parameters = sum(parameter.numel() for parameter in layer.parameters())
    real_parameters = (4 * 3) * (4 * 5) + 4 * 5
    assert hyper_parameters == 4 * 3 * 5 + 4 * 5
    assert hyper_parameters < real_parameters
