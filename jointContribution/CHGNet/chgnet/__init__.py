from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from typing import Literal

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
TrainTask = Literal["ef", "efs", "efsm"]
PredTask = Literal["e", "ef", "em", "efs", "efsm"]
ROOT = os.path.dirname(os.path.dirname(__file__))
