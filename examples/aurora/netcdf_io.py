"""Framework-neutral, lossless Batch serialization for long verification runs."""

import xarray as xr


def write_batch(batch, path, compression_level=0):
    """Preserve Batch's schema and dtypes with independently readable levels.

    Uncompressed NetCDF avoids making compression dominate a multi-day numerical
    verification. No quantization, dtype conversion or field selection is used.
    """
    if compression_level not in range(10):
        raise ValueError("NetCDF compression level must be between 0 and 9")

    def array(tensor):
        return tensor.detach().cpu().numpy()

    surface_dims = ("batch", "history", "latitude", "longitude")
    atmosphere_dims = ("batch", "history", "level", "latitude", "longitude")
    ds = xr.Dataset(
        {
            **{
                f"surf_{name}": (surface_dims, array(value))
                for name, value in batch.surf_vars.items()
            },
            **{
                f"atmos_{name}": (atmosphere_dims, array(value))
                for name, value in batch.atmos_vars.items()
            },
            **{
                f"static_{name}": (("latitude", "longitude"), array(value))
                for name, value in batch.static_vars.items()
            },
        },
        coords={
            "latitude": array(batch.metadata.lat),
            "longitude": array(batch.metadata.lon),
            "time": list(batch.metadata.time),
            "level": list(batch.metadata.atmos_levels),
            "rollout_step": batch.metadata.rollout_step,
        },
    )
    encoding = {}
    for name, variable in ds.data_vars.items():
        encoding[name] = {
            "zlib": compression_level > 0,
            "chunksizes": tuple(
                1 if dimension in ("batch", "history", "level") else size
                for dimension, size in zip(variable.dims, variable.shape)
            ),
        }
        if compression_level:
            encoding[name]["complevel"] = compression_level
    ds.to_netcdf(path, engine="netcdf4", encoding=encoding)
    ds.close()
