# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for diffusion samplers. Makes use of dinosaur.spherical_harmonic."""

import dataclasses
import functools
from typing import Any
from typing import Optional
from typing import Tuple

import numpy as np
import xarray
import xarray_tree
from dinosaur import spherical_harmonic

# Some useful constants useful when dealing with Earth's geometry.
# The earth isn't really a sphere so these are only approximate, this is the
# average radius according to https://en.wikipedia.org/wiki/Earth_radius,
# with the actual value varying from 6378 to 6357km.
EARTH_RADIUS_KM = 6371.0
# And this is also approximate, but we've chosen to make it consistent with the
# radius above when modelling the earth as a sphere. This gives a value of
# around 40030; the actual value varies from 40008 to 40075.
EARTH_CIRCUMFERENCE_KM = EARTH_RADIUS_KM * 2 * np.pi


@dataclasses.dataclass(frozen=True)
class _ArrayGrid:
    """A class that performs operations and transformations in the spectral basis."""

    longitude_wavenumbers: int
    total_wavenumbers: int
    longitude_nodes: int
    latitude_nodes: int
    latitude_spacing: str

    @classmethod
    def with_lat_lon(
        cls,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> "_ArrayGrid":
        """_ArrayGrid for use with data in specified lat/lon grid (in degrees)."""

        latitude_nodes = lat.shape[0]
        longitude_nodes = lon.shape[0]
        latitude_spacing = _infer_latitude_spacing(lat)
        if latitude_spacing in ["equiangular", "gauss"]:
            if longitude_nodes != 2 * latitude_nodes:
                raise ValueError(
                    "Unexpected number of longitude nodes. "
                    f"Expected {2 * latitude_nodes}, got {longitude_nodes}"
                )
        elif latitude_spacing == "equiangular_with_poles":
            if longitude_nodes != 2 * (latitude_nodes - 1):
                raise ValueError(
                    "Unexpected number of longitude nodes. "
                    f"Expected {2 * (latitude_nodes - 1)}, got {longitude_nodes}"
                )
        else:
            raise ValueError(f"Unexpected latitude_spacing={latitude_spacing}")
        max_wavenumber = int(longitude_nodes // 2) - 1
        grid = cls(
            longitude_wavenumbers=max_wavenumber + 1,
            total_wavenumbers=max_wavenumber + 1,
            longitude_nodes=longitude_nodes,
            latitude_nodes=latitude_nodes,
            latitude_spacing=latitude_spacing,
        )
        _verify_nodal_axes(lat, lon, grid.nodal_axes)
        return grid

    @functools.cached_property
    def _grid(self):
        return spherical_harmonic.Grid(
            spherical_harmonics_impl=spherical_harmonic.RealSphericalHarmonics,
            **dataclasses.asdict(self),
        )

    @functools.cached_property
    def nodal_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Longitude and sin(latitude) coordinates of the nodal basis."""
        return self._grid.nodal_axes()

    @functools.cached_property
    def modal_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Longitudinal and total wavenumbers (m, l) of the modal basis."""
        return self._grid.modal_axes()

    def to_nodal(self, x: np.ndarray) -> np.ndarray:
        """Maps `x` from a modal to nodal representation."""
        return self._grid.to_nodal(x)


def _infer_latitude_spacing(lat: np.ndarray) -> str:
    """Infers the type of latitude spacing given the latitude."""
    if not np.all(np.diff(lat) > 0.0):
        raise ValueError("Latitude values are expected to be sorted.")

    if np.allclose(np.diff(lat), lat[1] - lat[0]):
        if np.isclose(max(lat), 90.0):
            spacing = "equiangular_with_poles"
        else:
            spacing = "equiangular"
    else:
        spacing = "gauss"
    return spacing


def _verify_nodal_axes(
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    nodal_axes: Tuple[np.ndarray, np.ndarray],
):
    nodal_axes_lon, nodal_axes_sin_lat = nodal_axes
    if not np.allclose(nodal_axes_sin_lat, np.sin(np.deg2rad(lat_coords))):
        raise ValueError(
            "Latitude coords don't match those used by "
            "spherical_harmonic.SphericalHarmonicBasis."
        )
    if not np.allclose(nodal_axes_lon, np.deg2rad(lon_coords)):
        raise ValueError(
            "Longitude coords don't match those used by "
            "spherical_harmonic.SphericalHarmonicBasis."
        )


class Grid:
    """xarray wrapper around _ArrayGrid."""

    @classmethod
    def for_nodal_data(
        cls,
        nodal_data: xarray.DataArray,
    ) -> "Grid":
        """A Grid for use with a given shape of nodal (lat/lon grid) data.

        This uses the maximum number of spherical harmonics that the grid is able
        to resolve.

        This class supports data arrays with latitude spacings as defined by
        "dinosaur.spherical_harmonic". In summary:
        * 'equiangular': equally spaced (by `d_lat`) values between -90 + d_lat and
        90 - d_lat / 2. In our case, longitude must also be spaced by `d_lat`.
        * 'equiangular_with_poles': equally spaced (by `d_lat`) values between -90
        and 90. In our case, longitude must also be spaced by `d_lat`.
        * 'gauss': Gauss-Legendre nodes.

        Args:
        nodal_data: An xarray with 'lat' and 'lon' dimensions and coordinates in
            degrees.

        Returns:
        A grid with the specified latitude_nodes, with
        longitude_nodes=2*latitude_nodes and max_wavenumber=latitude_nodes-1.
        """

        grid = _ArrayGrid.with_lat_lon(
            nodal_data.coords["lat"].data, nodal_data.coords["lon"].data
        )
        return cls(grid, nodal_data.coords["lat"].data, nodal_data.coords["lon"].data)

    def __init__(
        self, grid: _ArrayGrid, lat_coords: np.ndarray, lon_coords: np.ndarray
    ):
        _verify_nodal_axes(lat_coords, lon_coords, grid.nodal_axes)
        self._underlying = grid
        # Record the exact original lat/lon coords so we can return them exactly
        # from an inverse transform, avoiding any xarray merge issues if coordinates
        # are off by a rounding error.
        self._lat_coords = lat_coords
        self._lon_coords = lon_coords
        (
            self._longitude_wavenumber_coords,
            self._total_wavenumber_coords,
        ) = grid.modal_axes

    @property
    def total_wavenumber_coords(self) -> xarray.DataArray:
        """Coords that must be used for 'total_wavenumber' dimension."""
        return xarray.DataArray(
            data=self._total_wavenumber_coords,
            dims=("total_wavenumber",),
            coords={"total_wavenumber": self._total_wavenumber_coords},
        )

    @property
    def longitude_wavenumber_coords(self) -> xarray.DataArray:
        """Coords that must be used for 'longitude_wavenumber' dimension."""
        return xarray.DataArray(
            data=self._longitude_wavenumber_coords,
            dims=("longitude_wavenumber",),
            coords={"longitude_wavenumber": self._longitude_wavenumber_coords},
        )

    def to_nodal(self, modal_data: xarray.DataArray) -> xarray.DataArray:
        """Applies the inverse spherical harmonic transform.

        Args:
        modal_data: A tree of xarray.DataArray with 'longitude_wavenumber' and
            'total_wavenumber' dimensions with coords
            `self.longitude_wavenumber_coords` and `self.total_wavenumber_coords`
            respectively, and with the same sparsity pattern described under
            `to_modal`.

        Returns:
        Corresponding tree where the 'longitude_wavenumber' and
        'total_wavenumber' dimensions are replaced by 'lat', 'lon' dimensions.
        """

        def inverse_transform(modal: xarray.DataArray) -> xarray.DataArray:
            if not np.all(
                modal.coords["longitude_wavenumber"]
                == self._longitude_wavenumber_coords
            ) or not np.all(
                modal.coords["total_wavenumber"] == self._total_wavenumber_coords
            ):
                raise ValueError("Wavenumber coords don't follow required convention.")

            return xarray.apply_ufunc(
                self._underlying.to_nodal,
                modal,
                input_core_dims=[["longitude_wavenumber", "total_wavenumber"]],
                output_core_dims=[["lon", "lat"]],
            ).assign_coords(
                lon=self._lon_coords,
                lat=self._lat_coords,
            )

        return xarray_tree.map_structure(inverse_transform, modal_data)


def sample(
    key: np.ndarray,  # PaddlePaddle doesn't have jax.random, so using numpy
    power_spectrum: xarray.DataArray,
    template: xarray.DataArray,
    grid: Optional[Grid] = None,
) -> xarray.DataArray:
    """Samples Gaussian Process noise on a sphere, with a given power spectrum.

    This means the noise will have the given power spectrum *in expectation*; the
    power spectrum of individual samples may vary.

    The noise will be isotropic, meaning the distribution is invariant to
    rotations of the sphere.

    The marginal variance of the returned values will be equal to the total power,
    i.e. the sum of power_spectrum. So if you want unit marginal variance, just
    make sure to normalize the power_spectrum to sum to 1.

    Args:
        key: JAX rng key.
        power_spectrum: An array with shape (total_wavenumber,) giving the power
        which is desired at each total wavenumber (corresponding to a wavelength
        EARTH_CIRCUMFERENCE/total_wavenumber) for total wavenumbers 0 up to some
        maximum. This is in squared units of the quantity being sampled.
        template: An array with the shape that you want the samples in, containing
        'lat' and 'lon' dimensions. If other dimensions are present, we draw
        multiple independent samples along these other dimensions.
        grid: spherical_harmonic.Grid on which to sample the noise. If not specified
        a grid will be created based on `template`, however note you may save some
        RAM and compute by re-using a single Grid instance across multiple calls.

    Returns:
        DataArray with the same shape as template.
    """
    if grid is None:
        grid = Grid.for_nodal_data(template)

    dims = [d for d in template.dims if d not in ("lat", "lon")]
    shape = [template.sizes[d] for d in dims]
    coords = {
        name: coord
        for name, coord in template.coords.items()
        if name not in ("lat", "lon")
    }
    dims.extend(("total_wavenumber", "longitude_wavenumber"))
    shape.extend(
        (len(grid.total_wavenumber_coords), len(grid.longitude_wavenumber_coords))
    )
    coords.update(
        {
            "total_wavenumber": grid.total_wavenumber_coords,
            "longitude_wavenumber": grid.longitude_wavenumber_coords,
        }
    )

    # Use PaddlePaddle to generate random normal data
    random_data = np.random.normal(size=shape)
    coeffs = xarray.DataArray(data=random_data, dims=dims, coords=coords)

    # Mask out coefficients which are out of range
    mask = (abs(coeffs.longitude_wavenumber) <= coeffs.total_wavenumber).astype(
        np.float32
    )

    # Normalize the coefficients
    multiplier = mask * np.sqrt(
        power_spectrum / mask.sum("longitude_wavenumber", skipna=False)
    )
    multiplier *= np.sqrt(4 * np.pi)

    # Multiply by coeffs
    coeffs *= multiplier

    # Convert grid to nodal and cast data type
    result = grid.to_nodal(coeffs)
    result = result.astype(template.dtype)

    return result.transpose(*template.dims)


def spherical_white_noise_like(template: xarray.Dataset) -> xarray.Dataset:
    """Samples isotropic mean 0 variance 1 white noise on the sphere."""

    def spherical_white_noise_like_dataarray(
        data_array: xarray.DataArray,
    ) -> xarray.DataArray:
        num_wavenumbers = data_array.lon.shape[0] // 2
        # PaddlePaddle doesn't have hk.next_rng_key(), using numpy for random key generation
        key = np.random.randint(low=0, high=2**32, size=(2,), dtype=np.uint32)

        # Create power spectrum
        power_spectrum_data = np.array(
            [1 / num_wavenumbers for _ in range(num_wavenumbers)]
        )
        power_spectrum = xarray.DataArray(
            data=power_spectrum_data, dims=["total_wavenumber"]
        )

        return sample(key=key, power_spectrum=power_spectrum, template=data_array)

    return template.map(spherical_white_noise_like_dataarray)


def rho_inverse_cdf(min_value: float, max_value: float, rho: float, cdf: Any) -> Any:
    """Quantiles of rho distribution used for noise levels at sampling time.

    This is parameterised by rho as in Eqn 5 from the Elucidating paper
    (but with max/min flipped so that quantiles are given in ascending not
    descending order). It's equivalent to a Beta[rho, 1] distribution rescaled to
    [min_value, max_value].

    At sampling time we use noise levels at fixed quantiles of this distribution.
    Unlike in the paper, we also use the same distribution for noise levels at
    training time (albeit potentially with different parameters, and sampling from
    it at random).

    Args:
      min_value:
      max_value:
        Define the support of the distribution.
      rho:
        Shape parameter.
      cdf:
        Value or values between 0 and 1 indicating which quantile you want. Can
        be a numpy or jax array.

    Returns:
      Quantiles of the distribution, with same shape/type as `cdf`.
    """
    return (
        min_value ** (1 / rho) + cdf * (max_value ** (1 / rho) - min_value ** (1 / rho))
    ) ** rho


def tree_where(cond: np.ndarray, xs: Any, ys: Any) -> Any:
    """
    Like np.where but works with trees for xs and ys (but not for cond).
    Supports xarray.Dataset as a possible structure for xs and ys.
    """
    if isinstance(xs, xarray.Dataset) and isinstance(ys, xarray.Dataset):
        # Ensure both datasets have the same data variables and dimensions
        assert xs.keys() == ys.keys(), "Datasets must have the same variables"
        return xarray.Dataset({var: tree_where(cond, xs[var], ys[var]) for var in xs})
    elif isinstance(xs, xarray.DataArray) and isinstance(ys, xarray.DataArray):
        # Directly apply np.where to xarray DataArray
        return xarray.where(cond, xs, ys)
    elif isinstance(xs, dict) and isinstance(ys, dict):
        # Recursively handle dictionaries
        assert xs.keys() == ys.keys(), "Dictionaries must have the same keys"
        return {key: tree_where(cond, xs[key], ys[key]) for key in xs}
    elif isinstance(xs, list) and isinstance(ys, list):
        # Recursively handle lists
        assert len(xs) == len(ys), "Lists must have the same length"
        return [tree_where(cond, x, y) for x, y in zip(xs, ys)]
    else:
        # Base case: apply np.where directly
        return np.where(cond, xs, ys)


def noise_schedule(
    max_noise_level: float = 80.0,
    min_noise_level: float = 0.002,
    num_noise_levels: int = 30,
    rho: float = 7.0,
) -> np.ndarray:
    """Computes a descending noise schedule for sampling, ending with zero."""
    noise_levels = rho_inverse_cdf(
        min_value=min_noise_level,
        max_value=max_noise_level,
        rho=rho,
        cdf=np.linspace(1, 0, num_noise_levels),
    )
    return np.append(noise_levels, 0.0)


def stochastic_churn_rate_schedule(
    noise_levels: np.ndarray,
    stochastic_churn_rate: float = 0.0,
    churn_min_noise_level: float = 0.05,
    churn_max_noise_level: float = 50.0,
) -> np.ndarray:
    """Computes a stochastic churn rate for each noise level."""
    num_noise_levels = len(noise_levels) - 1
    per_step_churn_rate = min(stochastic_churn_rate / num_noise_levels, np.sqrt(2) - 1)

    return (
        (churn_min_noise_level <= noise_levels[:-1])
        & (noise_levels[:-1] <= float(churn_max_noise_level))
    ) * per_step_churn_rate


def apply_stochastic_churn(
    x: np.ndarray,
    noise_level: np.ndarray,
    stochastic_churn_rate: np.ndarray,
    noise_level_inflation_factor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns x at higher noise level, and the higher noise level itself."""
    new_noise_level = noise_level * (1.0 + stochastic_churn_rate)
    noise_diff = new_noise_level**2 - noise_level**2
    noise_diff = np.maximum(noise_diff.numpy(), 0)
    extra_noise_stddev = np.sqrt(noise_diff) * noise_level_inflation_factor
    updated_x = x + spherical_white_noise_like(x) * extra_noise_stddev
    return updated_x, new_noise_level
