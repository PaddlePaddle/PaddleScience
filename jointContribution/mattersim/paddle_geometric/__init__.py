from collections import defaultdict

import paddle
import paddle_geometric.typing

from ._compile import compile, is_compiling
from ._onnx import is_in_onnx_export
from .index import Index
from .edge_index import EdgeIndex
from .pytree import pytree
from .seed import seed_everything
from .home import get_home_dir, set_home_dir
from .device import is_mps_available, is_xpu_available, device
from .isinstance import is_paddle_instance
from .debug import is_debug_enabled, debug, set_debug

import paddle_geometric.utils
import paddle_geometric.data
import paddle_geometric.sampler
import paddle_geometric.loader
import paddle_geometric.transforms
import paddle_geometric.datasets
import paddle_geometric.nn
import paddle_geometric.explain
import paddle_geometric.profile

from .experimental import (is_experimental_mode_enabled, experimental_mode,
                           set_experimental_mode)
from .lazy_loader import LazyLoader

contrib = LazyLoader('contrib', globals(), 'paddle_geometric.contrib')
graphgym = LazyLoader('graphgym', globals(), 'paddle_geometric.graphgym')

__version__ = '2.7.0'

__all__ = [
    'Index',
    'EdgeIndex',
    'seed_everything',
    'get_home_dir',
    'set_home_dir',
    'compile',
    'is_compiling',
    'is_in_onnx_export',
    'is_mps_available',
    'is_xpu_available',
    'device',
    'is_paddle_instance',
    'is_debug_enabled',
    'debug',
    'set_debug',
    'is_experimental_mode_enabled',
    'experimental_mode',
    'set_experimental_mode',
    'paddle_geometric',
    '__version__',
]

