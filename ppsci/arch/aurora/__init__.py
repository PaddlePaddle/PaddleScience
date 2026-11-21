"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from __future__ import annotations

from ppsci.arch.aurora._random import make_generator
from ppsci.arch.aurora.arch import AuroraArch
from ppsci.arch.aurora.batch import Batch
from ppsci.arch.aurora.batch import Metadata
from ppsci.arch.aurora.batch import split_batch
from ppsci.arch.aurora.batch import tile_batch
from ppsci.arch.aurora.insolation import insolation
from ppsci.arch.aurora.model.aurora import Aurora
from ppsci.arch.aurora.model.aurora import Aurora12hPretrained
from ppsci.arch.aurora.model.aurora import AuroraAirPollution
from ppsci.arch.aurora.model.aurora import AuroraHighRes
from ppsci.arch.aurora.model.aurora import AuroraPretrained
from ppsci.arch.aurora.model.aurora import AuroraSmall
from ppsci.arch.aurora.model.aurora import AuroraSmallPretrained
from ppsci.arch.aurora.model.aurora import AuroraV1p5
from ppsci.arch.aurora.model.aurora import AuroraV1p5Ensemble
from ppsci.arch.aurora.model.aurora import AuroraWave
from ppsci.arch.aurora.rollout import rollout
from ppsci.arch.aurora.rollout import rollout_ensemble
from ppsci.arch.aurora.tracker import Tracker

__all__ = [
    "AuroraArch",
    "make_generator",
    "Aurora",
    "AuroraPretrained",
    "AuroraSmallPretrained",
    "AuroraSmall",
    "Aurora12hPretrained",
    "AuroraHighRes",
    "AuroraAirPollution",
    "AuroraWave",
    "AuroraV1p5",
    "AuroraV1p5Ensemble",
    "Batch",
    "Metadata",
    "tile_batch",
    "split_batch",
    "insolation",
    "rollout",
    "rollout_ensemble",
    "Tracker",
]
