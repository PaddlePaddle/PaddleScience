r"""Definition of common enumerations."""

from __future__ import annotations  # noqa

import itertools
import re
from enum import Enum
from enum import IntEnum
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Union
from typing import overload


class Sampling(Enum):
    r"""Enumeration of image interpolation modes."""

    AREA = "area"
    BICUBIC = "bicubic"
    BSPLINE = "bspline"  # cubic B-spline
    LINEAR = "linear"  # bilinear or trilinear
    NEAREST = "nearest"

    @classmethod
    def from_arg(cls, arg: Union[Sampling, str, None]) -> Sampling:
        r"""Create enumeration value from function argument."""
        if isinstance(arg, str):
            arg = arg.lower()
        if arg is None or arg in ("default", "bilinear", "trilinear"):
            return cls.LINEAR
        if arg == "nn":
            return cls.NEAREST
        return cls(arg)

    def grid_sample_mode(self, num_spatial_dim: int) -> str:
        r"""Interpolation mode argument for paddle.nn.functional.grid_sample() for given number of spatial dimensions."""
        if self == self.LINEAR:
            return "bilinear"
        if self == self.NEAREST:
            return "nearest"
        raise ValueError(
            f"paddle.nn.functional.grid_sample() does not support padding mode '{self.value}' for {num_spatial_dim}-dimensional images"
        )

    def interpolate_mode(self, num_spatial_dim: int) -> str:
        r"""Interpolation mode argument for paddle.nn.functional.interpolate() for given number of spatial dimensions."""
        if self == self.AREA:
            return "area"
        if self == self.BICUBIC:
            return "bicubic"
        if self == self.LINEAR:
            if num_spatial_dim == 1:
                return "linear"
            if num_spatial_dim == 2:
                return "bilinear"
            if num_spatial_dim == 3:
                return "trilinear"
        if self == self.NEAREST:
            return "nearest"
        raise ValueError(
            f"paddle.nn.functional.interpolate() does not support padding mode '{self.value}' for {num_spatial_dim}-dimensional images"
        )


class PaddingMode(Enum):
    r"""Enumeration of image extrapolation modes."""

    NONE = "none"
    CONSTANT = "constant"
    BORDER = "border"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    ZEROS = "zeros"

    @classmethod
    def from_arg(cls, arg: Union[PaddingMode, str, None]) -> PaddingMode:
        r"""Create enumeration value from function argument."""
        if isinstance(arg, str):
            arg = arg.lower()
        if arg is None or arg == "default":
            return cls.ZEROS
        if arg in ("mirror", "reflection"):
            return cls.REFLECT
        if arg == "circular":
            return cls.REPLICATE
        return cls(arg)

    def conv_mode(self, num_spatial_dim: int = 3) -> str:
        r"""Padding mode argument for paddle.nn.ConvNd()."""
        if self in (self.CONSTANT, self.ZEROS):
            return "zeros"
        elif self == self.REFLECT:
            return "reflect"
        elif self == self.REPLICATE:
            return "replicate"
        raise ValueError(
            f"paddle.nn.Conv{num_spatial_dim}d() does not support padding mode '{self.value}'"
        )

    def grid_sample_mode(self, num_spatial_dim: int) -> str:
        r"""Padding mode argument for paddle.nn.functional.grid_sample()."""
        if 2 <= num_spatial_dim <= 3:
            if self in (self.CONSTANT, self.ZEROS):
                return "zeros"
            if self == self.BORDER:
                return "border"
            if self == self.REFLECT:
                return "reflection"
        raise ValueError(
            f"paddle.nn.functional.grid_sample() does not support padding mode '{self.value}' for {num_spatial_dim}-dimensional images"
        )

    def pad_mode(self, num_spatial_dim: int) -> str:
        r"""Padding mode argument for paddle.nn.functional.pad() for given number of spatial dimensions."""
        if self == self.CONSTANT:
            return "constant"
        elif self == self.REFLECT:
            if 1 <= num_spatial_dim <= 2:
                return "reflect"
        elif self == self.REPLICATE:
            if 1 <= num_spatial_dim <= 3:
                return "replicate"
        raise ValueError(
            f"paddle.nn.functional.pad() does not support padding mode '{self.value}' for {num_spatial_dim}-dimensional images"
        )


class SpatialDim(IntEnum):
    r"""Spatial image dimension selector."""

    X = 0
    Y = 1
    Z = 2
    T = 3

    @classmethod
    def from_arg(cls, arg: Union[int, str, SpatialDim]) -> SpatialDim:
        r"""Get enumeration value from function argument."""
        if arg in ("x", "X"):
            return cls.X
        if arg in ("y", "Y"):
            return cls.Y
        if arg in ("z", "Z"):
            return cls.Z
        if arg in ("t", "T"):
            return cls.T
        return cls(arg)

    def symbol(self) -> str:
        r"""Letter of spatial dimension."""
        return ("x", "y", "z", "t")[self.value]

    def tensor_dim(self, ndim: int, channels_last: bool = False) -> int:
        r"""Map spatial dimension identifier to image data tensor dimension."""
        dim = ndim - (2 if channels_last else 1) - self.value
        if (
            channels_last
            and (dim < 1 or dim > ndim - 2)
            or not channels_last
            and (dim < 2 or dim > ndim - 1)
        ):
            raise ValueError("SpatialDim.tensor_dim() is out-of-bounds")
        return dim

    def __str__(self) -> str:
        r"""Letter of spatial dimension."""
        return self.symbol()


SpatialDimArg = Union[int, str, SpatialDim]

SpatialDerivativeKey = str


class SpatialDerivativeKeys(object):
    r"""Auxiliary functions for identifying and enumerating spatial derivatives.

    Spatial derivatives are encoded by a sequence of letters, where each letter
    identifies the spatial dimension (cf. ``SpatialDim``) along which to take the
    derivative. The length of the string encoding determines the order of the
    derivative, i.e., how many times the input image is being derived with
    respect to one or more spatial dimensions.

    """

    @staticmethod
    def check(arg: Union[str, Iterable[str]]):
        r"""Check if given derivatives key is valid."""
        if isinstance(arg, str):
            arg = (arg,)
        for key in arg:
            if not isinstance(key, str):
                raise TypeError("Spatial derivatives key must be str")
            if re.search("[^xyzt]", key):
                raise ValueError(
                    "Spatial derivatives key must only contain letters 'x', 'y', 'z', or 't'"
                )

    @classmethod
    def is_valid(cls, arg: Union[str, Iterable[str]]) -> bool:
        r"""Check if given derivatives key is valid."""
        try:
            cls.check(arg)
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    def is_mixed(key: SpatialDerivativeKey) -> bool:
        r"""Whether derivative contains mixed terms."""
        return len(set(key)) > 1

    @staticmethod
    def all(spatial_dims: int, order: Union[int, Sequence[int]]) -> List[SpatialDerivativeKey]:
        r"""Unmixed spatial derivatives of specified order."""
        if isinstance(order, int):
            order = [order]
        keys = []
        dims = [str(SpatialDim(d)) for d in range(spatial_dims)]
        for n in order:
            if n > 0:
                codes = dims
                for _ in range(1, n):
                    codes = [(code + letter) for code, letter in itertools.product(codes, dims)]
                keys.extend(codes)
        return keys

    @staticmethod
    def unmixed(spatial_dims: int, order: int) -> List[SpatialDerivativeKey]:
        r"""Unmixed spatial derivatives of specified order."""
        if order <= 0:
            return []
        return [(SpatialDim(d).symbol() * order) for d in range(spatial_dims)]

    @classmethod
    def unique(cls, keys: Iterable[SpatialDerivativeKey]) -> Set[SpatialDerivativeKey]:
        r"""Unique spatial derivatives."""
        return set(cls.sorted(key) for key in keys)

    @classmethod
    def sorted(cls, key: SpatialDerivativeKey) -> SpatialDerivativeKey:
        r"""Sort letters of spatial dimensions in spatial derivative key."""
        return cls.join(sorted(cls.split(key)))

    @staticmethod
    def order(arg: SpatialDerivativeKey) -> int:
        r"""Order of the spatial derivative."""
        return len(arg)

    @classmethod
    def max_order(cls, keys: Iterable[SpatialDerivativeKey]) -> int:
        return max([0] + [cls.order(key) for key in keys])

    @staticmethod
    def split(arg: SpatialDerivativeKey) -> List[SpatialDim]:
        r"""Split spatial derivative key into spatial dimensions enum values."""
        return [SpatialDim.from_arg(letter) for letter in arg]

    @staticmethod
    def join(arg: Sequence[SpatialDim]) -> SpatialDerivativeKey:
        r"""Join spatial dimensions to spatial derivative key."""
        return "".join(sdim.symbol() for sdim in arg)


class FlowChannelIndex(IntEnum):
    r"""Flow vector component index."""

    U = 0
    V = 1
    W = 2

    @classmethod
    def from_arg(cls, arg: Union[int, str, FlowChannelIndex]) -> FlowChannelIndex:
        r"""Get enumeration value from function argument."""
        if arg in ("u", "U"):
            return cls.U
        if arg in ("v", "V"):
            return cls.V
        if arg in ("w", "W"):
            return cls.W
        return cls(arg)

    def index(self) -> int:
        r"""Map flow component identifier to tensor channel index."""
        return self.value

    def symbol(self) -> str:
        r"""Single lowercase letter identifying this flow component."""
        return ("u", "v", "w")[self.value]

    def __str__(self) -> str:
        return self.symbol()


FlowChannelIndexArg = Union[int, str, FlowChannelIndex]

FlowDerivativeKey = str


class FlowDerivativeKeys(object):
    r"""Auxiliary functions for identifying and enumerating spatial derivatives of a flow field.

    The flow field components are denoted as "u" (c=0), "v"  (c=1), and "w"  (c=2), respectively,
    where c is the channel index. The spatial dimensions along which to take derivatives are
    identified by the letters "x", "y", and "z" (cf. :class:`SpatialDerivativeKeys`). The partial
    derivative of a given vector field component along one or more spatial dimensions is denoted
    by a quotient string such as "dv/dz" or "du/dxy". When multiple flow field components are
    specified, the spatial derivative of each is computed, i.e., "duw/dx" is shorthand for
    ["du/dx", "dw/dx"].

    """

    @classmethod
    def from_arg(
        cls,
        spatial_dims: int,
        which: Optional[Union[str, Sequence[str]]] = None,
        order: Optional[int] = None,
    ) -> List[FlowDerivativeKey]:
        r"""Get flow derivative keys from function arguments.

        See also ``which`` parameter of :func:`flow_derivatives()` function.

        Args:
            spatial_dims: Number of spatial dimensions.
            which: String codes of spatial deriviatives to compute. When only a sequence of spatial
                dimension keys is given (cf. :class:`SpatialDerivateKeys`), the respective spatial
                derivative is computed for all vector field components, i.e., for ``spatial_dims=3``,
                "x" is shorthand for "du/dx", "dv/dx", and "dw/dx".
            order: Order of spatial derivatives. When both ``which`` and ``order`` are specified,
                only the flow field derivatives listed in ``which`` and are of the given order
                are returned.

        Returns:
            Flow derivative keys, one for each partial derivative scalar field.

        """
        if spatial_dims < 2 or spatial_dims > 3:
            raise ValueError("Spatial flow field dimensions must be 2 or 3")
        if which is None:
            if order is None:
                order = 1
            keys = cls.all(spatial_dims=spatial_dims, order=order)
        else:
            keys = []
            regex = re.compile("^(d(?P<channels>[uvw]+)/d)?(?P<derivs>[xyzt]+)$")
            if isinstance(which, str):
                which = [which]
            for arg in which:
                match = regex.match(arg)
                if not match:
                    raise ValueError(f"Invalid flow derivative specification: {arg}")
                derivs = match["derivs"]
                SpatialDerivativeKeys.check(derivs)
                if order is None or SpatialDerivativeKeys.order(derivs) == order:
                    if match["channels"] is None:
                        channels = ["u", "v", "w"][:spatial_dims]
                    else:
                        channels = [str(c).lower() for c in match["channels"]]
                    for channel in channels:
                        keys.append(f"d{channel}/d{derivs}")
        return keys

    @overload
    @classmethod
    def unique(cls, arg: FlowDerivativeKey) -> FlowDerivativeKey:
        r"""Unique spatial flow derivative key, where spatial dimension codes are sorted alphabetically."""
        ...

    @overload
    @classmethod
    def unique(cls, arg: Iterable[FlowDerivativeKey]) -> Set[FlowDerivativeKey]:
        r"""Set of unique spatial flow derivative keys."""
        ...

    @classmethod
    def unique(
        cls, arg: Union[FlowDerivativeKey, Iterable[FlowDerivativeKey]]
    ) -> Union[FlowDerivativeKey, Set[FlowDerivativeKey]]:
        r"""Unique spatial flow derivative."""
        unique_keys = set()
        keys = [arg] if isinstance(arg, str) else arg
        for channel, derivative_key in cls.split(keys):
            derivative_key = SpatialDerivativeKeys.sorted(derivative_key)
            unique_key = cls.symbol(channel, derivative_key)
            unique_keys.add(unique_key)
        if arg is not keys:
            return next(iter(unique_keys))
        return unique_keys

    @classmethod
    def sorted(cls, keys: Iterable[FlowDerivativeKey]) -> List[FlowDerivativeKey]:
        r"""Ordered list of flow derivative keys."""
        return sorted(keys)

    @classmethod
    def is_mixed(cls, arg: FlowDerivativeKey) -> bool:
        r"""Whether flow derivative is taken along more than one spatial dimension."""
        return SpatialDerivativeKeys.is_mixed(cls.split(arg)[1])

    @classmethod
    def order(cls, arg: FlowDerivativeKey) -> int:
        r"""Order of the spatial flow derivative."""
        return SpatialDerivativeKeys.order(cls.split(arg)[1])

    @classmethod
    def max_order(cls, keys: Iterable[SpatialDerivativeKey]) -> int:
        return SpatialDerivativeKeys.max_order(derivs for _, derivs in cls.split(keys))

    @overload
    @staticmethod
    def split(arg: FlowDerivativeKey) -> Tuple[FlowChannelIndex, SpatialDerivativeKey]:
        ...

    @overload
    @staticmethod
    def split(
        arg: Iterable[FlowDerivativeKey],
    ) -> List[Tuple[FlowChannelIndex, SpatialDerivativeKey]]:
        ...

    @staticmethod
    def split(
        arg: Union[FlowDerivativeKey, Iterable[FlowDerivativeKey]]
    ) -> List[Tuple[FlowChannelIndex, SpatialDerivativeKey]]:
        r"""Split flow derivative keys into flow channel index and spatial derivative key."""
        keys = []
        regex = re.compile("^d(?P<channel_key>[uvw])/d(?P<derivative_key>[xXyYzZtT]+)$")
        args = [arg] if isinstance(arg, str) else arg
        for key in args:
            match = regex.match(key)
            if not match:
                raise ValueError(f"Invalid flow derivative key: {key}")
            channel_key = match["channel_key"]
            derivative_key = match["derivative_key"]
            channel_index = FlowChannelIndex.from_arg(channel_key)
            SpatialDerivativeKeys.check(derivative_key)
            keys.append((channel_index, derivative_key))
        if arg is not args:
            return keys[0]
        return keys

    @staticmethod
    def symbol(
        channel: FlowChannelIndexArg, *args: Union[str, int, SpatialDim]
    ) -> FlowDerivativeKey:
        channel = FlowChannelIndex.from_arg(channel)
        derivs = []
        for arg in args:
            if isinstance(arg, str):
                derivs.extend(SpatialDerivativeKeys.split(arg))
            elif isinstance(arg, int):
                derivs.append(SpatialDim(arg))
            elif isinstance(arg, SpatialDim):
                derivs.append(arg)
            else:
                raise TypeError("Spatial derivative identifier must be int, str, or SpatialDim")
        return f"d{channel.symbol()}/d{SpatialDerivativeKeys.join(derivs)}"

    @classmethod
    def all(
        cls,
        spatial_dims: int,
        channel: Optional[Union[FlowChannelIndexArg, Sequence[FlowChannelIndexArg]]] = None,
        order: Union[int, Sequence[int]] = 1,
    ) -> List[FlowDerivativeKey]:
        r"""All spatial derivatives of specified order."""
        if order < 0:
            raise ValueError("Spatial derivatives order must be non-negative")
        if order == 0:
            return []
        channels = cls._channels(spatial_dims, channel)
        derivs = SpatialDerivativeKeys.all(spatial_dims=spatial_dims, order=order)
        return [cls.symbol(c, d) for c, d in itertools.product(channels, derivs)]

    @classmethod
    def unmixed(
        cls,
        spatial_dims: int,
        channel: Optional[Union[FlowChannelIndexArg, Sequence[FlowChannelIndexArg]]] = None,
        order: Union[int, Sequence[int]] = 1,
    ) -> List[FlowDerivativeKey]:
        r"""Unmixed spatial derivatives of specified order."""
        if order < 0:
            raise ValueError("Spatial derivatives order must be non-negative")
        if order == 0:
            return []
        channels = cls._channels(spatial_dims, channel)
        derivs = SpatialDerivativeKeys.unmixed(spatial_dims=spatial_dims, order=order)
        return [cls.symbol(c, d) for c, d in itertools.product(channels, derivs)]

    @classmethod
    def gradient(
        cls,
        spatial_dims: int,
        channel: Optional[Union[FlowChannelIndexArg, Sequence[FlowChannelIndexArg]]] = None,
    ) -> List[FlowDerivativeKey]:
        return cls.all(spatial_dims=spatial_dims, channel=channel, order=1)

    @classmethod
    def jacobian(cls, spatial_dims: int) -> List[FlowDerivativeKey]:
        return cls.all(spatial_dims=spatial_dims, order=1)

    @classmethod
    def divergence(cls, spatial_dims: int) -> List[FlowDerivativeKey]:
        return [cls.symbol(i, i) for i in range(spatial_dims)]

    @classmethod
    def curvature(cls, spatial_dims: int) -> List[FlowDerivativeKey]:
        channels = range(spatial_dims)
        derivs = SpatialDerivativeKeys.unmixed(spatial_dims=spatial_dims, order=2)
        return [cls.symbol(c, d) for c, d in itertools.product(channels, derivs)]

    @classmethod
    def hessian(
        cls,
        spatial_dims: int,
        channel: Optional[Union[FlowChannelIndexArg, Sequence[FlowChannelIndexArg]]] = None,
    ) -> List[FlowDerivativeKey]:
        return cls.all(spatial_dims=spatial_dims, channel=channel, order=2)

    @staticmethod
    def _channels(
        spatial_dims: int,
        channel: Optional[Union[FlowChannelIndexArg, Sequence[FlowChannelIndexArg]]] = None,
    ) -> List[FlowChannelIndex]:
        if channel is None:
            channel = range(spatial_dims)
        elif isinstance(channel, (int, str, FlowChannelIndex)):
            channel = [channel]
        return [FlowChannelIndex.from_arg(c) for c in channel]
