import pytest
import torch

from hypercast4d.algebras import get_algebra


def basis(index: int) -> torch.Tensor:
    value = torch.zeros(4, dtype=torch.float64)
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
