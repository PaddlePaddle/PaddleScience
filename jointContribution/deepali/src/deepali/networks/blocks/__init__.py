r"""Building blocks to construct subnetworks, and predefined subnetworks (blocks)."""

from .residual import ResidualUnit
from .skip import DenseBlock
from .skip import Shortcut
from .skip import SkipConnection

# fmt: off
__all__ = (
    "DenseBlock",
    "ResidualUnit",
    "Shortcut",
    "SkipConnection",
)
# fmt: on
