r"""Common types and functions that operate on tensors representing different types of data.

Besides defining common types and auxiliary functions extending the standard library,
this core library in particular defines functions which operate on objects of type ``paddle.Tensor``.
This functional API defines a set of reusable state-less functions similar to ``paddle.nn.functional``.
Object-oriented APIs use this functional API to realize their functionality. In particular, the
``forward()`` method of ``paddle.nn.Layer`` subclasses, such as for example data transformations
(cf. :mod:`.data.transforms`) and neural network components (cf. :mod:`.modules` and :mod:`.networks`)
are implemented using these functional building blocks.

The following import statement can be used to access the functional API:

.. code-block::

    import deepali.core.functional as U

"""

from importlib import metadata

from .cube import Cube
from .enum import PaddingMode
from .enum import Sampling
from .grid import ALIGN_CORNERS
from .grid import Axes
from .grid import Grid
from .grid import grid_points_transform
from .grid import grid_transform_points
from .grid import grid_transform_vectors
from .grid import grid_vectors_transform
from .pathlib import is_uri
from .pathlib import to_uri
from .storage import StorageObject
from .typing import Array
from .typing import Batch
from .typing import Dataclass
from .typing import Device
from .typing import DeviceStr
from .typing import DType
from .typing import DTypeStr
from .typing import Name
from .typing import PathStr
from .typing import PathUri
from .typing import Sample
from .typing import Scalar
from .typing import ScalarOrTuple
from .typing import ScalarOrTuple1d
from .typing import ScalarOrTuple2d
from .typing import ScalarOrTuple3d
from .typing import Shape
from .typing import Size
from .typing import TensorCollection
from .typing import is_bool_dtype
from .typing import is_float_dtype
from .typing import is_int_dtype
from .typing import is_namedtuple
from .typing import is_path_str
from .typing import is_uint_dtype

__version__ = metadata.version("hf-deepali")
r"""Version string of installed deepali core libraries."""

__all__ = (
    "ALIGN_CORNERS",
    "Array",
    "Axes",
    "Batch",
    "Cube",
    "Dataclass",
    "Device",
    "DeviceStr",
    "DType",
    "DTypeStr",
    "Grid",
    "Name",
    "PaddingMode",
    "PathStr",
    "PathUri",
    "Sample",
    "Sampling",
    "Scalar",
    "ScalarOrTuple",
    "ScalarOrTuple1d",
    "ScalarOrTuple2d",
    "ScalarOrTuple3d",
    "Size",
    "Shape",
    "StorageObject",
    "TensorCollection",
    "grid_points_transform",
    "grid_vectors_transform",
    "grid_transform_points",
    "grid_transform_vectors",
    "is_bool_dtype",
    "is_float_dtype",
    "is_int_dtype",
    "is_uint_dtype",
    "is_namedtuple",
    "is_path_str",
    "is_uri",
    "to_uri",
)
