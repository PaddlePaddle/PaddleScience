import warnings
from typing import Literal

import paddle_geometric


def warn(message: str) -> None:
    if paddle_geometric.is_compiling():
        return

    warnings.warn(message)


def filterwarnings(
    action: Literal['default', 'error', 'ignore', 'always', 'module', 'once'],
    message: str,
) -> None:
    if paddle_geometric.is_compiling():
        return

    warnings.filterwarnings(action, message)
