"""Structure constants for the three 4D algebras used in the paper.

The basis order is ``(1, i, j, k)``. ``constants[a, b, c]`` is the
coefficient of basis element ``c`` in ``e_a * e_b``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


COMPONENT_COUNT = 4


@dataclass(frozen=True)
class Algebra:
    """A named real four-dimensional algebra."""

    name: str
    constants: torch.Tensor

    def multiply(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Multiply tensors whose final axis stores four basis coefficients."""
        if left.shape[-1] != COMPONENT_COUNT or right.shape[-1] != COMPONENT_COUNT:
            raise ValueError(
                f"Hypercomplex operands must have final dimension {COMPONENT_COUNT}"
            )
        constants = self.constants.to(device=left.device, dtype=left.dtype)
        return torch.einsum("...a,...b,abc->...c", left, right, constants)


def _constants(products: dict[tuple[int, int], tuple[float, int]]) -> torch.Tensor:
    result = torch.zeros(
        COMPONENT_COUNT, COMPONENT_COUNT, COMPONENT_COUNT, dtype=torch.float64
    )
    for basis in range(COMPONENT_COUNT):
        result[0, basis, basis] = 1.0
        result[basis, 0, basis] = 1.0
    for (left, right), (sign, output) in products.items():
        result[left, right, output] = sign
    return result


_QUATERNION = _constants(
    {
        (1, 1): (-1, 0),
        (1, 2): (1, 3),
        (1, 3): (-1, 2),
        (2, 1): (-1, 3),
        (2, 2): (-1, 0),
        (2, 3): (1, 1),
        (3, 1): (1, 2),
        (3, 2): (-1, 1),
        (3, 3): (-1, 0),
    }
)

_COQUATERNION = _constants(
    {
        (1, 1): (-1, 0),
        (1, 2): (1, 3),
        (1, 3): (-1, 2),
        (2, 1): (-1, 3),
        (2, 2): (1, 0),
        (2, 3): (-1, 1),
        (3, 1): (1, 2),
        (3, 2): (1, 1),
        (3, 3): (1, 0),
    }
)

_CLIFFORD_11 = _constants(
    {
        (1, 1): (1, 0),
        (1, 2): (1, 3),
        (1, 3): (1, 2),
        (2, 1): (-1, 3),
        (2, 2): (-1, 0),
        (2, 3): (1, 1),
        (3, 1): (-1, 2),
        (3, 2): (-1, 1),
        (3, 3): (1, 0),
    }
)

ALGEBRAS: dict[str, Algebra] = {
    "quaternion": Algebra("quaternion", _QUATERNION),
    "coquaternion": Algebra("coquaternion", _COQUATERNION),
    "cl11": Algebra("cl11", _CLIFFORD_11),
}

_ALIASES = {
    "quaternions": "quaternion",
    "coquaternions": "coquaternion",
    "cl(1,1)": "cl11",
    "clifford11": "cl11",
}


def get_algebra(name: str) -> Algebra:
    """Return an algebra by its canonical name or a documented alias."""
    key = _ALIASES.get(name.lower(), name.lower())
    try:
        return ALGEBRAS[key]
    except KeyError as error:
        choices = ", ".join(sorted(ALGEBRAS))
        raise ValueError(
            f"Unknown algebra {name!r}; choose one of {choices}"
        ) from error
