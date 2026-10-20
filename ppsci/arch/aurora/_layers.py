"""Paddle equivalents of the small timm utilities used by Aurora."""

from ppsci.arch.paddle_timm.drop import DropPath
from ppsci.arch.paddle_timm.helpers import to_2tuple
from ppsci.arch.paddle_timm.helpers import to_ntuple
from ppsci.utils.initializer import trunc_normal_

to_3tuple = to_ntuple(3)

__all__ = ["DropPath", "to_2tuple", "to_3tuple", "trunc_normal_"]
