"""HyperCast4D: leakage-safe checks for 4D hypercomplex forecasting."""

from .algebras import COMPONENT_COUNT, Algebra, get_algebra
from .architecture import ArchitectureError, build_architecture, validate_architecture
from .layers import HyperDense

__all__ = [
    "Algebra",
    "ArchitectureError",
    "COMPONENT_COUNT",
    "HyperDense",
    "build_architecture",
    "get_algebra",
    "validate_architecture",
]
__version__ = "0.2.0"
