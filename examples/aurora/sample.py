"""Small reproducible inputs shared by the example and cross-framework checks."""

from datetime import datetime

import numpy as np


def small_config():
    """A lightweight architecture that exercises all three U-Net stages."""
    return dict(
        encoder_depths=(2, 2, 2),
        decoder_depths=(2, 2, 2),
        encoder_num_heads=(2, 4, 8),
        decoder_num_heads=(8, 4, 2),
        embed_dim=32,
        num_heads=2,
    )


def make_batch(framework, batch_class, metadata_class, seed=7, batch_size=1):
    """Use identical NumPy draws for both frameworks; no framework RNG involved."""
    rng = np.random.default_rng(seed)

    def tensor(shape, location=0.0, scale=1.0):
        return framework.tensor(
            (location + scale * rng.standard_normal(shape)).astype("float32")
        )

    return batch_class(
        surf_vars={
            k: tensor((batch_size, 2, 17, 32), *stats)
            for k, stats in {
                "2t": (280, 5),
                "10u": (0, 5),
                "10v": (0, 5),
                "msl": (101325, 500),
            }.items()
        },
        static_vars={
            k: tensor((17, 32), *stats)
            for k, stats in {
                "lsm": (0.5, 0.1),
                "z": (1000, 100),
                "slt": (2, 0.1),
            }.items()
        },
        atmos_vars={
            k: tensor((batch_size, 2, 4, 17, 32), *stats)
            for k, stats in {
                "z": (50000, 1000),
                "u": (0, 10),
                "v": (0, 10),
                "t": (250, 5),
                "q": (0.003, 0.0001),
            }.items()
        },
        metadata=metadata_class(
            lat=framework.tensor(np.linspace(90, -90, 17, dtype="float32")),
            lon=framework.tensor(np.arange(32, dtype="float32") * 11.25),
            time=(datetime(2020, 6, 1, 12),) * batch_size,
            atmos_levels=(100, 250, 500, 850),
        ),
    )
