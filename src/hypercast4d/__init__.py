"""HyperCast4D: leakage-safe checks for 4D hypercomplex forecasting."""

from .algebras import COMPONENT_COUNT, Algebra, get_algebra
from .layers import HyperDense

__all__ = ["Algebra", "COMPONENT_COUNT", "HyperDense", "get_algebra"]
__version__ = "0.1.0"
