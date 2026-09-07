import pytest
import torch

from hypercast4d.algebras import ALGEBRAS, Algebra, get_algebra


def basis(index: int, dimension: int = 4) -> torch.Tensor:
    value = torch.zeros(dimension, dtype=torch.float64)
    value[index] = 1
    return value


@pytest.mark.parametrize(
    ("name", "basis_index", "expected_sign"),
    [
        ("quaternion", 1, -1),
        ("quaternion", 2, -1),
        ("quaternion", 3, -1),
        ("coquaternion", 1, -1),
        ("coquaternion", 2, 1),
        ("coquaternion", 3, 1),
        ("cl11", 1, 1),
        ("cl11", 2, -1),
        ("cl11", 3, 1),
    ],
)
def test_basis_squares(name: str, basis_index: int, expected_sign: int) -> None:
    algebra = get_algebra(name)
    expected = expected_sign * basis(0)
    assert torch.equal(
        algebra.multiply(basis(basis_index), basis(basis_index)), expected
    )


@pytest.mark.parametrize("name", ["quaternion", "coquaternion", "cl11"])
def test_ij_is_k_and_ji_is_minus_k(name: str) -> None:
    algebra = get_algebra(name)
    assert torch.equal(algebra.multiply(basis(1), basis(2)), basis(3))
    assert torch.equal(algebra.multiply(basis(2), basis(1)), -basis(3))


def test_unknown_algebra_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown algebra"):
        get_algebra("not-an-algebra")


@pytest.mark.parametrize(
    ("name", "dimension"),
    [
        ("complex", 2),
        ("split_complex", 2),
        ("tricomplex", 3),
        ("quaternion", 4),
        ("coquaternion", 4),
        ("cl11", 4),
        ("octonion", 8),
    ],
)
def test_algebra_dimensions_and_identity(name: str, dimension: int) -> None:
    algebra = get_algebra(name)
    assert algebra.component_count == dimension
    value = torch.arange(1, dimension + 1, dtype=torch.float64)
    identity = basis(0, dimension)
    assert torch.equal(algebra.multiply(identity, value), value)
    assert torch.equal(algebra.multiply(value, identity), value)


def test_two_dimensional_products() -> None:
    complex_numbers = get_algebra("complex")
    split_complex = get_algebra("split_complex")
    unit = basis(1, 2)
    assert torch.equal(complex_numbers.multiply(unit, unit), -basis(0, 2))
    assert torch.equal(split_complex.multiply(unit, unit), basis(0, 2))


def test_cyclic_tricomplex_products() -> None:
    algebra = get_algebra("tricomplex")
    h, k = basis(1, 3), basis(2, 3)
    assert torch.equal(algebra.multiply(h, h), k)
    assert torch.equal(algebra.multiply(k, k), h)
    assert torch.equal(algebra.multiply(h, k), basis(0, 3))
    assert torch.equal(algebra.multiply(k, h), basis(0, 3))


def test_octonion_orientation_and_nonassociativity() -> None:
    algebra = get_algebra("octonion")
    units = [basis(index, 8) for index in range(8)]
    assert torch.equal(algebra.multiply(units[1], units[2]), units[3])
    assert torch.equal(algebra.multiply(units[2], units[1]), -units[3])
    left = algebra.multiply(algebra.multiply(units[1], units[2]), units[4])
    right = algebra.multiply(units[1], algebra.multiply(units[2], units[4]))
    assert torch.equal(left, units[7])
    assert torch.equal(right, -units[7])


def test_octonion_norm_composes() -> None:
    algebra = get_algebra("octonion")
    left = torch.tensor([0.2, -0.3, 0.5, 0.7, -1.1, 0.4, 0.9, -0.6], dtype=torch.float64)
    right = torch.tensor([-0.8, 0.1, 0.6, -0.2, 0.3, 1.2, -0.4, 0.5], dtype=torch.float64)
    product = algebra.multiply(left, right)
    torch.testing.assert_close(
        product.square().sum(),
        left.square().sum() * right.square().sum(),
        rtol=1e-14,
        atol=1e-14,
    )


def test_algebra_rejects_non_cubic_constants() -> None:
    with pytest.raises(ValueError, match="cubic"):
        Algebra("invalid", torch.zeros(2, 2, 3))


def test_registry_contains_only_supported_dimensions() -> None:
    assert {algebra.component_count for algebra in ALGEBRAS.values()} == {2, 3, 4, 8}
