# Copyright 2023 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spherical harmonics basis evaluation, and differential operators."""

import dataclasses
import functools
import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import dinosaur.associated_legendre as associated_legendre
import dinosaur.fourier as fourier
import numpy as np
import paddle

LATITUDE_SPACINGS = dict(
    gauss=associated_legendre.gauss_legendre_nodes,
    equiangular=associated_legendre.equiangular_nodes,
    equiangular_with_poles=associated_legendre.equiangular_nodes_with_poles,
)


def get_latitude_nodes(n: int, spacing: str) -> tuple[np.ndarray, np.ndarray]:
    """Computes latitude nodes using the given spacing."""
    get_nodes = LATITUDE_SPACINGS.get(spacing)
    if get_nodes is None:
        raise ValueError(
            f"Unknown spacing: {spacing}"
            f"available spacings are {list(LATITUDE_SPACINGS.keys())}"
        )
    return get_nodes(n)


@dataclasses.dataclass
class _SphericalHarmonicBasis:
    """Data structure representing a basis for spherical harmonics.

    Attributes:
      f: Fourier matrix.
      p: Legendre transform coefficients.
      w: nodal quadrature weights.
    """

    f: np.ndarray
    p: np.ndarray
    w: np.ndarray


@dataclasses.dataclass(frozen=True)
class SphericalHarmonics:
    """Base class for spherical harmonics implementations.

    Attributes:
        longitude_wavenumbers: the maximum (exclusive) wavenumber in the
          longitudinal direction. Indexes along longitudinal wavenumber are
          typically denoted by `m`.
        total_wavenumbers: the maximum (exclusive) sum of the latitudinal and
          longitudinal wavenumbers. Indices along total wavenumber are typically
          denoted by `l`.
        longitude_nodes: the number of nodes in the longitudinal direction. The
          selected nodes will be the equally spaced points in [0, 2π).
        latitude_nodes: the number of nodes in the latitudinal direction. The
          selected nodes will be the Gauss-Legendre quadrature points.
        latitude_spacing: a string indicating the spacing of latitude nodes. If
          'gauss' is passed, then Gauss-Legendre nodes are used. If 'equiangular' or
          'equiangular_with_poles' is passed, then the nodes are equally spaced in
          latitude (without or with points at the poles, respectively).
    """

    longitude_wavenumbers: int = 0
    total_wavenumbers: int = 0
    longitude_nodes: int = 0
    latitude_nodes: int = 0
    latitude_spacing: str = "gauss"

    @property
    def nodal_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Longitude and sin(latitude) coordinates of the nodal basis."""
        raise NotImplementedError

    @property
    def nodal_shape(self) -> tuple[int, int]:
        """Shape in the nodal basis."""
        raise NotImplementedError

    @property
    def nodal_padding(self) -> tuple[int, int]:
        """Padding in the nodal basis."""
        raise NotImplementedError

    @property
    def modal_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Longitudinal and total wavenumbers (m, l) of the modal basis."""
        raise NotImplementedError

    @property
    def modal_shape(self) -> tuple[int, int]:
        """Shape in the modal basis."""
        raise NotImplementedError

    @property
    def modal_padding(self) -> tuple[int, int]:
        """Padding in the modal basis."""
        raise NotImplementedError

    @property
    def modal_dtype(self) -> paddle.dtype:
        """Dtype in the modal state."""
        raise NotImplementedError

    @property
    def mask(self) -> np.ndarray:
        """Mask of valid values in modal representation."""
        raise NotImplementedError

    @property
    def basis(self):
        """Basis functions for these spherical harmonics."""
        raise NotImplementedError

    def inverse_transform(self, x: paddle.Tensor) -> paddle.Tensor:
        """Maps `x` from a modal to nodal representation."""
        raise NotImplementedError

    def transform(self, x: paddle.Tensor) -> paddle.Tensor:
        """Maps `x` from a nodal to modal representation."""
        raise NotImplementedError

    def longitudinal_derivative(self, x: paddle.Tensor) -> paddle.Tensor:
        """Computes `∂x/∂λ` in the modal basis, where λ denotes longitude."""
        raise NotImplementedError


class RealSphericalHarmonics(SphericalHarmonics):
    """Pedagogical implementation of spherical harmonics transforms.

    This transform represents spherical harmonic (modal) coefficients as a two
    dimensional grid of longtitudinal wavenumber (m) and total wavenumber (l)
    values:
        m = [0, +1, -1, +2, -2, ..., +M, -M]
        l = [0, 1, 2, ..., L]
    where `M = longitude_wavenumbers - 1` and `L = total_wavenumbers`.

    Entries with `abs(m) > l` are structural zeros,

    For better performance when using computing forward and inverse transforms,
    but no guaranteed stable representation, use FastSphericalHarmonics, which
    also supports parallelism.
    """

    @functools.cached_property
    def nodal_axes(self) -> tuple[np.ndarray, np.ndarray]:
        longitude, _ = fourier.quadrature_nodes(self.longitude_nodes)
        sin_latitude, _ = get_latitude_nodes(self.latitude_nodes, self.latitude_spacing)
        return longitude, sin_latitude

    @functools.cached_property
    def nodal_shape(self) -> tuple[int, int]:
        return (self.longitude_nodes, self.latitude_nodes)

    @functools.cached_property
    def nodal_padding(self) -> tuple[int, int]:
        return (0, 0)

    @functools.cached_property
    def modal_axes(self) -> tuple[np.ndarray, np.ndarray]:
        m_pos = np.arange(1, self.longitude_wavenumbers)
        m_pos_neg = np.stack([m_pos, -m_pos], axis=1).ravel()
        lon_wavenumbers = np.concatenate([[0], m_pos_neg])
        tot_wavenumbers = np.arange(self.total_wavenumbers)
        return lon_wavenumbers, tot_wavenumbers

    @functools.cached_property
    def modal_shape(self) -> tuple[int, int]:
        return (2 * self.longitude_wavenumbers - 1, self.total_wavenumbers)

    @functools.cached_property
    def modal_padding(self) -> tuple[int, int]:
        return (0, 0)

    @functools.cached_property
    def modal_dtype(self) -> np.dtype:
        return np.dtype(np.float32)

    @functools.cached_property
    def mask(self) -> np.ndarray:
        m, l = np.meshgrid(*self.modal_axes, indexing="ij")
        return np.abs(m) <= l

    @functools.cached_property
    def basis(self) -> _SphericalHarmonicBasis:
        # The product of the arrays `f` and `p` gives the real normalized spherical
        # harmonic basis evaluated on a grid of longitudes λ and latitudes θ:
        #
        #   f[i, 0]      p[0     , j, l] = cₗ₀ P⁰ₗ(sin θⱼ)
        #   f[i, 2m - 1] p[2m - 1, j, l] = cₗₘ cos(m λᵢ) Pᵐₗ(sin θⱼ)
        #   f[i, 2m]     p[2m,     j, l] = cₗₘ sin(m λᵢ) Pᵐₗ(sin θⱼ)
        #
        # where the constants cₗₘ are chosen such that each function has unit L²
        # norm on the unit sphere. The longitudes λᵢ are `longitude_nodes` equally
        # spaced points in [0, 2π). The latitude nodes θⱼ are chosen such that
        # (sin θⱼ) are the Gauss-Legendre quadrature points if
        # `latitude_spacing = 'gauss'`, or θⱼ are `latitude_nodes` equally spaced
        # points if `latitude_spacing = 'equiangular'` (or
        # `'equiangular_with_poles'` for equally spaced points including points at
        # the poles).
        #
        # The shapes of the returned arrays are
        #
        #   f.shape == [longitude_nodes, (2 * longitude_wavenumbers - 1)]
        #   p.shape == [2 * longitude_wavenumbers - 1,
        #               latitude_nodes,
        #               total_wavenumbers]
        f = fourier.real_basis(
            wavenumbers=self.longitude_wavenumbers,
            nodes=self.longitude_nodes,
        )
        _, wf = fourier.quadrature_nodes(self.longitude_nodes)
        x, wp = get_latitude_nodes(self.latitude_nodes, self.latitude_spacing)
        w = wf * wp
        p = associated_legendre.evaluate(
            n_m=self.longitude_wavenumbers, n_l=self.total_wavenumbers, x=x
        )
        # Each associated Legendre polynomial Pᵐₗ with m > 0 is paired with both
        # the sin and cos components of the Fourier basis. As a result, we have to
        # duplicate the rows of the associated Legendre matrix.
        p = np.repeat(p, 2, axis=0)
        # When m = 0, the associated Legendre polynomial is paired only with the
        # constant component of the Fourier matrix, so we only need one copy.
        p = p[1:]
        return _SphericalHarmonicBasis(f=f, p=p, w=w)

    def inverse_transform(self, x):
        p = paddle.to_tensor(self.basis.p)
        f = paddle.to_tensor(self.basis.f)
        x = paddle.to_tensor(x)
        px = paddle.einsum("mjl,...ml->...mj", p, x)
        # note: explicitly matrix multiplication seems to be faster than using an
        # explicit FFT at the resolutions we use.
        fpx = paddle.einsum("im,...mj->...ij", f, px)
        return fpx

    def transform(self, x):
        w = self.basis.w
        f = self.basis.f
        p = self.basis.p
        wx = w * x
        fwx = paddle.einsum("im,...ij->...mj", f, wx)
        pfwx = paddle.einsum("mjl,...mj->...ml", p, fwx)
        return pfwx

    def longitudinal_derivative(self, x: paddle.Tensor) -> paddle.Tensor:
        return fourier.real_basis_derivative(x, axis=-2)

    def _round_to_multiple(x: int, multiple: int) -> int:
        return multiple * math.ceil(x / multiple)


def _unstack_m(x: paddle.Tensor, mesh: None) -> paddle.Tensor:
    """Unstack positive and negative values of `m` along a separate dimension."""

    def unstack(x):
        shape = x.shape[:-2] + (2, x.shape[-2] // 2) + x.shape[-1:]
        return paddle.reshape(x, shape)

    if mesh is None:
        return unstack(x)

    assert x.ndim in {2, 3}, x.shape
    return unstack(x)


def _stack_m(x: paddle.Tensor, mesh: None) -> paddle.Tensor:
    """Stack a separate "sign" dimension into single dimension for `m`."""

    def stack(x):
        shape = x.shape[:-3] + (-1,) + x.shape[-1:]
        return paddle.reshape(x, shape)

    if mesh is None:
        return stack(x)

    assert x.ndim in {3, 4}, x.shape
    return stack(x)


def _fourier_derivative_for_real_basis_with_zero_imag(
    x: paddle.Tensor, mesh: None
) -> paddle.Tensor:
    """Calculate a Fourier basis derivative."""

    if mesh is None:
        return fourier.real_basis_derivative_with_zero_imag(x, axis=-2)

    assert x.ndim in {2, 3}, x.shape
    return fourier.real_basis_derivative_with_zero_imag(x, axis=-2)


@dataclasses.dataclass(frozen=True)
class FastSphericalHarmonics(SphericalHarmonics):
    """Fast implementation of spherical harmonic transformation for PaddlePaddle.

    No stability guarantees are made about the shapes of arrays in the modal
    representation.

    Currently uses an extra imaginary term for m=-0. This can be more efficient
    because the array of Legendre transform coefficients is the same for positive
    and negative coefficients, so this halves the size of the `p` array on the
    MXU.

    """

    base_shape_multiple: int | None = None
    reverse_einsum_arg_order: bool | None = None
    stacked_fourier_transforms: bool | None = None
    spmd_mesh: None = None
    transform_precision: str = "tensorfloat32"

    def __post_init__(self):
        model_parallelism = self.spmd_mesh is not None and any(
            self.spmd_mesh.shape[dim] > 1 for dim in "zxy"
        )

        if self.base_shape_multiple is None:
            shape_multiple = 8 if model_parallelism else 1
            object.__setattr__(self, "base_shape_multiple", shape_multiple)

        if self.reverse_einsum_arg_order is None:
            object.__setattr__(self, "reverse_einsum_arg_order", model_parallelism)

        if self.stacked_fourier_transforms is None:
            unstacked_matmuls = math.ceil(self.longitude_wavenumbers / 128)
            stacked_matmuls = 2 * math.ceil(self.longitude_wavenumbers / 256)
            stack = stacked_matmuls <= unstacked_matmuls
            object.__setattr__(self, "stacked_fourier_transforms", stack)

    @functools.cached_property
    def nodal_limits(self) -> tuple[int, int]:
        return (self.longitude_nodes, self.latitude_nodes)

    @functools.cached_property
    def modal_limits(self) -> tuple[int, int]:
        return (2 * self.longitude_wavenumbers, self.total_wavenumbers)

    def _mesh_shape(self) -> tuple[int, int]:
        if self.spmd_mesh is not None:
            return (self.spmd_mesh.shape["x"], self.spmd_mesh.shape["y"])
        else:
            return (1, 1)

    @functools.cached_property
    def nodal_padding(self) -> tuple[int, int]:
        return tuple(x - y for x, y in zip(self.nodal_shape, self.nodal_limits))

    @functools.cached_property
    def modal_padding(self) -> tuple[int, int]:
        return tuple(x - y for x, y in zip(self.modal_shape, self.modal_limits))

    @functools.cached_property
    def nodal_axes(self) -> tuple[paddle.Tensor, paddle.Tensor]:
        nodal_pad_x, nodal_pad_y = self.nodal_padding
        longitude, _ = fourier.quadrature_nodes(self.longitude_nodes)
        longitude = paddle.to_tensor(np.pad(longitude, [(0, nodal_pad_x)]))
        sin_latitude, _ = get_latitude_nodes(self.latitude_nodes, self.latitude_spacing)
        sin_latitude = paddle.to_tensor(np.pad(sin_latitude, [(0, nodal_pad_y)]))
        return longitude, sin_latitude

    @functools.cached_property
    def modal_axes(self) -> tuple[paddle.Tensor, paddle.Tensor]:
        modal_pad_x, modal_pad_y = self.modal_padding
        m_pos = paddle.arange(1, self.longitude_wavenumbers)
        m_pos_neg = paddle.concat([m_pos, -m_pos], axis=0).reshape([-1, 2]).flatten()
        lon_wavenumbers = paddle.to_tensor(
            np.pad(np.concatenate([[0, 0], m_pos_neg.numpy()]), [(0, modal_pad_x)])
        )
        tot_wavenumbers = paddle.to_tensor(
            np.pad(np.arange(self.total_wavenumbers), [(0, modal_pad_y)])
        )
        return lon_wavenumbers, tot_wavenumbers

    @functools.cached_property
    def modal_dtype(self) -> paddle.dtype:
        return paddle.float32

    @functools.cached_property
    def mask(self) -> paddle.Tensor:
        m, l = paddle.meshgrid(*self.modal_axes, indexing="ij")
        i, j = paddle.meshgrid(
            *(paddle.arange(s) for s in self.modal_shape), indexing="ij"
        )
        i_lim, j_lim = self.modal_limits
        return (paddle.abs(m) <= l) & (i != 1) & (i < i_lim) & (j < j_lim)

    @functools.cached_property
    def basis(self) -> _SphericalHarmonicBasis:
        # The product of the arrays `f` and `p` gives the real normalized spherical
        # harmonic basis evaluated on a grid of longitudes λ and latitudes θ:
        #
        #   f[i, 2m    ]  p[2m,     j, l] = cₗₘ cos(m λᵢ) Pᵐₗ(sin θⱼ)
        #   f[i, 2m + 1]  p[2m + 1, j, l] = cₗₘ sin(m λᵢ) Pᵐₗ(sin θⱼ)
        #
        # where the constants cₗₘ are chosen such that each function has unit L²
        # norm on the unit sphere. The longitudes λᵢ are `longitude_nodes` equally
        # spaced points in [0, 2π). The latitude nodes θⱼ are chosen such that
        # (sin θⱼ) are the Gauss-Legendre quadrature points if
        # `latitude_spacing = 'gauss'`, or θⱼ are `latitude_nodes` equally spaced
        # points if `latitude_spacing = 'equiangular'` (or
        # `'equiangular_with_poles'` for equally spaced points including points at
        # the poles).
        #
        # The shapes of the returned arrays are
        #
        #   f.shape == (longitude_nodes, 2*longitude_wavenumbers)
        #   p.shape == (2*longitude_wavenumbers, latitude_nodes, total_wavenumbers)
        nodal_pad_x, nodal_pad_y = self.nodal_padding
        modal_pad_x, modal_pad_y = self.modal_padding

        f = fourier.real_basis_with_zero_imag(
            wavenumbers=self.longitude_wavenumbers,
            nodes=self.longitude_nodes,
        )
        f = np.pad(f, [(0, nodal_pad_x), (0, modal_pad_x)])
        if self.stacked_fourier_transforms:
            f = np.reshape(f, (-1, 2, f.shape[-1] // 2), order="F")

        _, wf = fourier.quadrature_nodes(self.longitude_nodes)
        x, wp = get_latitude_nodes(self.latitude_nodes, self.latitude_spacing)
        w = wf * wp
        w = np.pad(w, [(0, nodal_pad_y)])

        p = associated_legendre.evaluate(
            n_m=self.longitude_wavenumbers, n_l=self.total_wavenumbers, x=x
        )
        p = np.pad(p, [(0, modal_pad_x // 2), (0, nodal_pad_y), (0, modal_pad_y)])

        return _SphericalHarmonicBasis(f=f, p=p, w=w)

    def inverse_transform(self, x):
        p = self.basis.p
        f = self.basis.f
        mesh = self.spmd_mesh

        # TODO(shoyer): consider supporting a "stacked" modal representation with
        # positive & negative values of `m` separated. This would allow for omitting
        # this call to _unstack_m().
        x = _unstack_m(x, mesh)
        x = paddle.einsum("mjl,...sml->...smj", p, x)

        if self.stacked_fourier_transforms:
            # note: explicit matrix multiplication seems to be faster than using an
            # explicit FFT at the resolutions we use.
            x = paddle.einsum("ism,...smj->...ij", f, x)
        else:
            x = _stack_m(x, mesh)
            x = paddle.einsum("im,...mj->...ij", f, x)

        return x

    def transform(self, x):
        w = self.basis.w
        f = self.basis.f
        p = self.basis.p
        mesh = self.spmd_mesh

        x = w * x
        if self.stacked_fourier_transforms:
            x = paddle.einsum("ism,...ij->...smj", f, x)
        else:
            x = paddle.einsum("im,...ij->...mj", f, x)
            x = _unstack_m(x, mesh)
        x = paddle.einsum("mjl,...smj->...sml", p, x)
        x = _stack_m(x, mesh)
        return x

    def longitudinal_derivative(self, x: paddle.Tensor) -> paddle.Tensor:
        return _fourier_derivative_for_real_basis_with_zero_imag(x, self.spmd_mesh)


@dataclasses.dataclass(frozen=True)
class RealSphericalHarmonicsWithZeroImag(FastSphericalHarmonics):
    """Deprecated alias for `FastSphericalHarmonics`."""


def _vertical_pad(
    field: paddle.Tensor, mesh: None  # PaddlePaddle 不支持 mesh 分片
) -> Tuple[paddle.Tensor, int | None]:
    if field.ndim < 3 or field.shape[0] == 1 or mesh is None:
        return field, None


def _vertical_crop(field: paddle.Tensor, padding: int | None) -> paddle.Tensor:
    if not padding:
        return field
    assert field.ndim == 3, field.shape
    return field[: field.shape[0] - padding, :, :]


def _with_vertical_padding(
    f: Callable[[paddle.Tensor], paddle.Tensor], mesh: None
) -> Callable[[paddle.Tensor], paddle.Tensor]:
    """Apply a function with vertical padding on a mesh.

    This is useful for implementing sharded Grid operations even the case where
    the z dimension has some irregular size.

    Args:
        f: function to apply on padded data.
        mesh: Placeholder for SPMD mesh, not used in PaddlePaddle.

    Returns:
        Function that can be applied to non-padded arrays.
    """

    def g(x):
        x, padding = _vertical_pad(x, mesh)
        return _vertical_crop(f(x), padding)

    return g


def numpy_tree_map(f: Callable[[np.ndarray], np.ndarray], x: Any) -> Any:
    if isinstance(x, dict):
        return {k: numpy_tree_map(f, v) for k, v in x.items()}
    elif isinstance(x, list):
        return [numpy_tree_map(f, item) for item in x]
    elif isinstance(x, tuple):
        return tuple(numpy_tree_map(f, item) for item in x)
    else:
        return f(x)


def tree_map_over_nonscalars(
    f: Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]],
    x: Any,
    *,
    scalar_fn: Callable[
        [Union[np.ndarray, float]], Union[np.ndarray, float]
    ] = lambda x: x,
    backend: str = "numpy",
) -> Any:
    """Map `f` over nonscalar pytree leaves, but use `scalar_fn` on scalars."""
    as_array_fn = np.asarray

    def g(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        x = as_array_fn(x)
        return f(x) if x.ndim else scalar_fn(x)

    return numpy_tree_map(g, x)


SPHERICAL_HARMONICS_IMPL_KEY = "spherical_harmonics_impl"
SPMD_MESH_KEY = "spmd_mesh"


SphericalHarmonicsImpl = Callable[..., SphericalHarmonics]


@dataclasses.dataclass(frozen=True)
class Grid:
    """A class that represents real-space and spectral grids over the sphere.

    The number of wavenumbers and nodes is entirely flexible, although in practice
    one should use one of the established conventions used by the constructors
    below. Typically both wavenumbers and nodes should be specified, unless you
    only need operations in real or spectral space.

    Attributes:
      longitude_wavenumbers: the maximum (exclusive) wavenumber in the
        longitudinal direction. Indexes along longitudinal wavenumber are
        typically denoted by `m`. Must satisfy `longitude_wavenumbers <=
        total_wavenumbers`.
      total_wavenumbers: the maximum (exclusive) sum of the latitudinal and
        longitudinal wavenumbers. Indices along total wavenumber are typically
        denoted by `l`. Must satisfy `longitude_wavenumbers <= total_wavenumbers`.
      longitude_nodes: the number of nodes in the longitudinal direction. The
        selected nodes will be the equally spaced points in [0, 2π) incremented by
        longitude_offset.
      latitude_nodes: the number of nodes in the latitudinal direction. The
        selected nodes will be the Gauss-Legendre quadrature points.
      latitude_spacing: a string indicating the spacing of latitude nodes. If
        'gauss' is passed, then Gauss-Legendre nodes are used. If 'equiangular' or
        'equiangular_with_poles' is passed, then the nodes are equally spaced in
        latitude (without or with points at the poles, respectively).
      longitude_offset: the value of the first longitude node, in radians.
      radius: radius of the sphere. If `None` a default value of `1` is used.
      spherical_harmonics_impl: class providing an implementation of spherical
        harmonics.
      spmd_mesh: mesh to use for parallelism in the single program multiple device
        (SPMD) paradigm with distributed JAX arrays, if any. Required if using
        model parallelism.
    """

    longitude_wavenumbers: int = 0
    total_wavenumbers: int = 0
    longitude_nodes: int = 0
    latitude_nodes: int = 0
    latitude_spacing: str = "gauss"
    longitude_offset: float = 0.0
    radius: float | None = None
    spherical_harmonics_impl: SphericalHarmonicsImpl = RealSphericalHarmonics
    spmd_mesh: None = None

    def __post_init__(self):
        if self.radius is None:
            # 默认半径设置为 1.0
            object.__setattr__(self, "radius", 1.0)

        if self.latitude_spacing not in LATITUDE_SPACINGS:
            raise ValueError(
                f'Unsupported `latitude_spacing` "{self.latitude_spacing}". '
                f"Supported values are: {list(LATITUDE_SPACINGS)}."
            )

    @classmethod
    def with_wavenumbers(
        cls,
        longitude_wavenumbers: int,
        dealiasing: str = "quadratic",
        latitude_spacing: str = "gauss",
        longitude_offset: float = 0.0,
        spherical_harmonics_impl: SphericalHarmonicsImpl = RealSphericalHarmonics,
        radius: Optional[float] = None,
    ) -> "Grid":
        """Construct a `Grid` by specifying only wavenumbers."""

        order = {"linear": 2, "quadratic": 3, "cubic": 4}.get(dealiasing)

        longitude_nodes = order * longitude_wavenumbers + 1
        latitude_nodes = math.ceil(longitude_nodes / 2)

        return cls(
            longitude_wavenumbers=longitude_wavenumbers,
            total_wavenumbers=longitude_wavenumbers + 1,
            longitude_nodes=longitude_nodes,
            latitude_nodes=latitude_nodes,
            latitude_spacing=latitude_spacing,
            longitude_offset=longitude_offset,
            spherical_harmonics_impl=spherical_harmonics_impl,
            radius=radius,
        )

    @classmethod
    def construct(
        cls,
        max_wavenumber: int,
        gaussian_nodes: int,
        latitude_spacing: str = "gauss",
        longitude_offset: float = 0.0,
        radius: Optional[float] = None,
        spherical_harmonics_impl: SphericalHarmonicsImpl = RealSphericalHarmonics,
    ) -> "Grid":
        """Construct a `Grid` by specifying max wavenumber & the number of nodes.

        Args:
          max_wavenumber: maximum wavenumber to resolve.
          gaussian_nodes: number of nodes on the Gaussian grid between the equator
            and a pole.
          latitude_spacing: either 'gauss' or 'equiangular'. This determines the
            spacing of nodal grid points in the latitudinal (north-south) direction.
          longitude_offset: the value of the first longitude node, in radians.
          radius: radius of the sphere. If `None` a default values of `1` is used.
          spherical_harmonics_impl: class providing an implementation of spherical
            harmonics.

        Returns:
          Constructed Grid object.
        """

        return cls(
            longitude_wavenumbers=max_wavenumber + 1,
            total_wavenumbers=max_wavenumber + 2,
            longitude_nodes=4 * gaussian_nodes,
            latitude_nodes=2 * gaussian_nodes,
            latitude_spacing=latitude_spacing,
            longitude_offset=longitude_offset,
            spherical_harmonics_impl=spherical_harmonics_impl,
            radius=radius,
        )

    # The factory methods below return "standard" grids that appear in the
    # literature. See, e.g. https://doi.org/10.5194/tc-12-1499-2018 and
    # https://www.ecmwf.int/en/forecasts/documentation-and-support/data-spatial-coordinate-systems

    # The number in these names correspond to the maximum resolved wavenumber,
    # which is one less than the number of wavenumbers used in the Grid
    # constructor. An additional total wavenumber is added because the top
    # wavenumber is clipped from the initial state and each calculation of
    # explicit tendencies.

    # The names for these factory methods (including capilatization) are
    # standard in the literature.
    # pylint:disable=invalid-name

    # T* grids can model quadratic terms without aliasing, because the maximum
    # total wavenumber is <= 2/3 of the number of latitudinal nodes. ECMWF
    # sometimes calls these "TQ" (truncated quadratic) grids.

    @classmethod
    def T21(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=21, gaussian_nodes=16, **kwargs)

    @classmethod
    def T31(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=31, gaussian_nodes=24, **kwargs)

    @classmethod
    def T42(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=42, gaussian_nodes=32, **kwargs)

    @classmethod
    def T85(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=85, gaussian_nodes=64, **kwargs)

    @classmethod
    def T106(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=106, gaussian_nodes=80, **kwargs)

    @classmethod
    def T119(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=119, gaussian_nodes=90, **kwargs)

    @classmethod
    def T170(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=170, gaussian_nodes=128, **kwargs)

    @classmethod
    def T213(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=213, gaussian_nodes=160, **kwargs)

    @classmethod
    def T340(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=340, gaussian_nodes=256, **kwargs)

    @classmethod
    def T425(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=425, gaussian_nodes=320, **kwargs)

    # TL* grids do not truncate any frequencies, and hence can only model linear
    # terms exactly. ECMWF used "TL" (truncated linear) grids for semi-Lagrangian
    # advection (which eliminates quadratic terms) up to 2016, which it switched
    # to "cubic" grids for resolutions above TL1279:
    # https://www.ecmwf.int/sites/default/files/elibrary/2016/17262-new-grid-ifs.pdf

    @classmethod
    def TL31(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=31, gaussian_nodes=16, **kwargs)

    @classmethod
    def TL47(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=47, gaussian_nodes=24, **kwargs)

    @classmethod
    def TL63(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=63, gaussian_nodes=32, **kwargs)

    @classmethod
    def TL95(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=95, gaussian_nodes=48, **kwargs)

    @classmethod
    def TL127(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=127, gaussian_nodes=64, **kwargs)

    @classmethod
    def TL159(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=159, gaussian_nodes=80, **kwargs)

    @classmethod
    def TL179(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=179, gaussian_nodes=90, **kwargs)

    @classmethod
    def TL255(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=255, gaussian_nodes=128, **kwargs)

    @classmethod
    def TL639(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=639, gaussian_nodes=320, **kwargs)

    @classmethod
    def TL1279(cls, **kwargs) -> "Grid":
        return cls.construct(max_wavenumber=1279, gaussian_nodes=640, **kwargs)

    # pylint:enable=invalid-name

    def asdict(self) -> Dict[str, Any]:
        """Returns grid attributes as a dictionary."""
        items = dataclasses.asdict(self)
        items[SPHERICAL_HARMONICS_IMPL_KEY] = self.spherical_harmonics_impl.__name__
        items[SPMD_MESH_KEY] = ""  # PaddlePaddle 不支持 mesh
        return items

    @functools.lru_cache(maxsize=None)
    def spherical_harmonics(self) -> SphericalHarmonics:
        """Implementation of spherical harmonic transformations."""
        kwargs = {}
        return self.spherical_harmonics_impl(
            longitude_wavenumbers=self.longitude_wavenumbers,
            total_wavenumbers=self.total_wavenumbers,
            longitude_nodes=self.longitude_nodes,
            latitude_nodes=self.latitude_nodes,
            latitude_spacing=self.latitude_spacing,
            **kwargs,
        )

    @property
    def longitudes(self) -> np.ndarray:
        return self.nodal_axes[0]

    @property
    def latitudes(self) -> np.ndarray:
        return np.arcsin(self.nodal_axes[1])

    @functools.lru_cache(maxsize=None)
    def nodal_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Longitude and sin(latitude) coordinates of the nodal basis."""
        lon, sin_lat = self.spherical_harmonics().nodal_axes
        return lon + self.longitude_offset, sin_lat

    @functools.lru_cache(maxsize=None)
    def nodal_shape(self) -> Tuple[int, int]:
        return self.spherical_harmonics().nodal_shape

    @functools.lru_cache(maxsize=None)
    def nodal_padding(self) -> Tuple[int, int]:
        return self.spherical_harmonics().nodal_padding

    @functools.lru_cache(maxsize=None)
    def nodal_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(*self.nodal_axes(), indexing="ij")

    @functools.lru_cache(maxsize=None)
    def modal_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Longitudinal and total wavenumbers (m, l) of the modal basis."""
        return self.spherical_harmonics().modal_axes

    @functools.lru_cache(maxsize=None)
    def modal_shape(self) -> Tuple[int, int]:
        return self.spherical_harmonics().modal_shape

    @functools.lru_cache(maxsize=None)
    def modal_padding(self) -> Tuple[int, int]:
        return self.spherical_harmonics().modal_padding

    @functools.lru_cache(maxsize=None)
    def mask(self) -> np.ndarray:
        """Modal mask."""
        return self.spherical_harmonics().mask

    @functools.lru_cache(maxsize=None)
    def modal_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Mesh of longitudinal and total wavenumbers (m, l) for the modal basis."""
        return np.meshgrid(*self.spherical_harmonics.modal_axes, indexing="ij")

    @functools.lru_cache(maxsize=None)
    def cos_lat(self) -> np.ndarray:
        _, sin_lat = self.nodal_axes
        return np.sqrt(1 - sin_lat**2)

    @functools.lru_cache(maxsize=None)
    def sec2_lat(self) -> np.ndarray:
        _, sin_lat = self.nodal_axes
        return 1 / (1 - sin_lat**2)

    @functools.lru_cache(maxsize=None)
    def laplacian_eigenvalues(self) -> np.ndarray:
        _, l = self.modal_axes()
        return -l * (l + 1) / (self.radius**2)

    def to_nodal(self, x: Any) -> Any:
        """Maps `x` from a modal to nodal representation."""
        f = _with_vertical_padding(
            self.spherical_harmonics().inverse_transform, self.spmd_mesh
        )
        return tree_map_over_nonscalars(f, x)

    # def to_modal(self, z: Any) -> Any:
    #     """Maps `x` from a nodal to modal representation."""
    #     f = _with_vertical_padding(
    #         self.spherical_harmonics.transform, self.spmd_mesh
    #     )
    #     return self._tree_map_over_nonscalars(f, z)

    def laplacian(self, x: paddle.Tensor) -> paddle.Tensor:
        """Computes `∇²(x)` in the spectral basis."""
        return x * self.laplacian_eigenvalues

    def inverse_laplacian(self, x: paddle.Tensor) -> paddle.Tensor:
        """Computes `(∇²)⁻¹(x)` in the spectral basis."""
        inverse_eigenvalues = paddle.to_tensor(1 / self.laplacian_eigenvalues)
        inverse_eigenvalues[0] = 0
        inverse_eigenvalues[self.total_wavenumbers :] = 0
        assert not paddle.isnan(inverse_eigenvalues).any()
        return x * inverse_eigenvalues

    # def clip_wavenumbers(self, x: Pytree, n: int = 1) -> Pytree:
    #     """Zeros out the highest `n` total wavenumbers."""
    #     if n <= 0:
    #         raise ValueError(f'`n` must be >= 0; got {n}.')

    #     def clip(x):
    #         # 创建一个用于掩码的 Paddle 张量
    #         num_zeros = n + self.modal_padding[-1]
    #         mask = paddle.ones(self.modal_shape[-1], dtype=x.dtype)
    #         mask[-num_zeros:] = 0
    #         return x * mask

    #     return tree_map_over_nonscalars(clip, x)

    @functools.lru_cache(maxsize=None)
    def _derivative_recurrence_weights(self) -> Tuple[paddle.Tensor, paddle.Tensor]:
        m, l = self.modal_mesh
        mask = self.mask.astype(float)
        a = np.sqrt(mask * (l**2 - m**2) / (4 * l**2 - 1))
        a[:, 0] = 0
        b = np.sqrt(mask * ((l + 1) ** 2 - m**2) / (4 * (l + 1) ** 2 - 1))
        b[:, -1] = 0
        return a, b

    def d_dlon(self, x: paddle.Tensor) -> paddle.Tensor:
        """Computes `∂x/∂λ` where λ denotes longitude."""
        return _with_vertical_padding(
            self.spherical_harmonics.longitudinal_derivative, self.spmd_mesh
        )(x)

    def cos_lat_d_dlat(self, x: paddle.Tensor) -> paddle.Tensor:
        """Computes `cosθ ∂x/∂θ`, where θ denotes latitude."""
        _, l = self.modal_mesh
        a, b = self._derivative_recurrence_weights
        x_lm1 = paddle.roll(((l + 1) * a) * x, shifts=-1, axis=-1)
        x_lp1 = paddle.roll((-l * b) * x, shifts=1, axis=-1)
        return x_lm1 + x_lp1

    def sec_lat_d_dlat_cos2(self, x: paddle.Tensor) -> paddle.Tensor:
        """Computes `secθ ∂/∂θ(cos²θ x)`, where θ denotes latitude."""
        _, l = self.modal_mesh
        a, b = self._derivative_recurrence_weights
        x_lm1 = paddle.roll(((l - 1) * a) * x, shifts=-1, axis=-1)
        x_lp1 = paddle.roll((-(l + 2) * b) * x, shifts=1, axis=-1)
        return x_lm1 + x_lp1

    def cos_lat_grad(self, x: paddle.Tensor, clip: bool = True) -> paddle.Tensor:
        """Computes `cosθ ∇(x)` where θ denotes latitude."""
        raw = (self.d_dlon(x) / self.radius, self.cos_lat_d_dlat(x) / self.radius)
        if clip:
            return self.clip_wavenumbers(raw)
        return raw

    def k_cross(self, v: paddle.Tensor | tuple[paddle.Tensor, ...]) -> paddle.Tensor:
        """Computes `k ✕ v`, where k is the normal unit vector."""
        return (-v[1], v[0])

    def div_cos_lat(
        self, v: paddle.Tensor | tuple[paddle.Tensor, ...], clip: bool = True
    ) -> paddle.Tensor:
        """Computes `∇ · (v cosθ)` where θ denotes latitude."""
        raw = (self.d_dlon(v[0]) + self.sec_lat_d_dlat_cos2(v[1])) / self.radius
        if clip:
            return self.clip_wavenumbers(raw)
        return raw

    def curl_cos_lat(
        self, v: paddle.Tensor | tuple[paddle.Tensor, ...], clip: bool = True
    ) -> paddle.Tensor:
        """Computes `k · ∇ ✕ (v cosθ)` where θ denotes latitude."""
        raw = (self.d_dlon(v[1]) - self.sec_lat_d_dlat_cos2(v[0])) / self.radius
        if clip:
            return self.clip_wavenumbers(raw)
        return raw

    @property
    def quadrature_weights(self) -> np.ndarray:
        """Calculates quadrature weights in nodal space."""
        return np.broadcast_to(self.spherical_harmonics.basis.w, self.nodal_shape)

    def integrate(self, z: paddle.Tensor) -> paddle.Tensor:
        """Approximates the integral of nodal values `z` over the sphere."""
        w = self.spherical_harmonics.basis.w * self.radius**2
        return paddle.einsum("y,...xy->...", w, z)
