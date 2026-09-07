"""Structure constants for supported finite-dimensional real algebras.

``constants[a, b, c]`` is the coefficient of basis element ``c`` in
``e_a * e_b``.  The paper-facing data path remains four-dimensional; the
algebras in this module may have two, three, four, or eight components.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# Kept for the four-series paper data/model API. HyperDense obtains its own
# component count from the selected algebra instead.
COMPONENT_COUNT = 4
PAPER_COMPONENT_COUNT = COMPONENT_COUNT


@dataclass(frozen=True)
class Algebra:
    """A named finite-dimensional real algebra with a bilinear product."""

    name: str
    constants: torch.Tensor

    def __post_init__(self) -> None:
        if self.constants.ndim != 3 or len(set(self.constants.shape)) != 1:
            raise ValueError("structure constants must be a non-empty cubic tensor")
        if self.constants.shape[0] < 1:
            raise ValueError("structure constants must contain at least one component")

    @property
    def component_count(self) -> int:
        return self.constants.shape[0]

    def multiply(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Multiply tensors whose final axis stores basis coefficients."""
        if (
            left.shape[-1] != self.component_count
            or right.shape[-1] != self.component_count
        ):
            raise ValueError(
                "Hypercomplex operands must have final dimension "
                f"{self.component_count} for {self.name}"
            )
        constants = self.constants.to(device=left.device, dtype=left.dtype)
        return torch.einsum("...a,...b,abc->...c", left, right, constants)


def _constants(
    dimension: int,
    products: dict[tuple[int, int], tuple[float, int]],
) -> torch.Tensor:
    result = torch.zeros(dimension, dimension, dimension, dtype=torch.float64)
    for basis in range(dimension):
        result[0, basis, basis] = 1.0
        result[basis, 0, basis] = 1.0
    for (left, right), (sign, output) in products.items():
        result[left, right, output] = sign
    return result


def _imaginary_units(
    dimension: int, oriented_triples: tuple[tuple[int, int, int], ...]
) -> torch.Tensor:
    """Build constants for anticommuting units with square ``-1``.

    Each oriented triple ``(a, b, c)`` defines ``e_a e_b = e_c`` and its
    cyclic permutations. Reversing either factor changes the sign.
    """
    products: dict[tuple[int, int], tuple[float, int]] = {
        (basis, basis): (-1, 0) for basis in range(1, dimension)
    }
    for a, b, c in oriented_triples:
        for left, right, output in ((a, b, c), (b, c, a), (c, a, b)):
            products[left, right] = (1, output)
            products[right, left] = (-1, output)
    return _constants(dimension, products)


_COMPLEX = _constants(2, {(1, 1): (-1, 0)})

_SPLIT_COMPLEX = _constants(2, {(1, 1): (1, 0)})

# Cyclic tricomplex numbers R[h] / (h^3 - 1), with k = h^2.
_TRICOMPLEX = _constants(
    3,
    {
        (1, 1): (1, 2),
        (1, 2): (1, 0),
        (2, 1): (1, 0),
        (2, 2): (1, 1),
    },
)


_QUATERNION = _constants(
    4,
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
    4,
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
    4,
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

# Baez/Fano-plane orientation. Octonion multiplication conventions differ by
# basis signs; this explicit orientation makes saved models reproducible.
_OCTONION = _imaginary_units(
    8,
    (
        (1, 2, 3),
        (1, 4, 5),
        (1, 7, 6),
        (2, 4, 6),
        (2, 5, 7),
        (3, 4, 7),
        (3, 6, 5),
    ),
)

ALGEBRAS: dict[str, Algebra] = {
    "complex": Algebra("complex", _COMPLEX),
    "split_complex": Algebra("split_complex", _SPLIT_COMPLEX),
    "tricomplex": Algebra("tricomplex", _TRICOMPLEX),
    "quaternion": Algebra("quaternion", _QUATERNION),
    "coquaternion": Algebra("coquaternion", _COQUATERNION),
    "cl11": Algebra("cl11", _CLIFFORD_11),
    "octonion": Algebra("octonion", _OCTONION),
}

_ALIASES = {
    "complexes": "complex",
    "split-complex": "split_complex",
    "split complex": "split_complex",
    "trinion": "tricomplex",
    "trinions": "tricomplex",
    "tricomplexes": "tricomplex",
    "quaternions": "quaternion",
    "coquaternions": "coquaternion",
    "cl(1,1)": "cl11",
    "clifford11": "cl11",
    "octonions": "octonion",
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
