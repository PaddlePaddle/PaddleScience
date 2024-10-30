r"""The transforms in this Python package generally built on the :mod:`.core` library.
The classes defined by these modules can be used, for example, in a data input pipeline
which is attached to a data loader. The spatial transforms defined in the :mod:`.spatial`
library, on the other hand, can be used to implement either a traditional or machine
learning based co-registration approach.

Note that data transforms are included in the :mod:`.data` library to avoid cyclical
imports between modules defining specialized tensor types such as :mod:`.data.image`
and datasets defined in :mod:`.data.dataset`, which also use these transforms to read
and preprocess the loaded data.

See also: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Transformer_cn.html#transformer.

"""

from typing import Callable

from .image import AvgPoolImage
from .image import CastImage
from .image import CenterCropImage
from .image import CenterPadImage
from .image import ClampImage
from .image import ImageToTensor
from .image import NarrowImage
from .image import NormalizeImage
from .image import ReadImage
from .image import ResampleImage
from .image import RescaleImage
from .image import ResizeImage
from .item import ItemTransform
from .item import ItemwiseTransform

Transform = Callable


__all__ = (
    # Type annotation
    "Transform",
    # Generic transforms
    "ItemTransform",
    "ItemwiseTransform",
    # Image transforms
    "AvgPoolImage",
    "CastImage",
    "CenterCropImage",
    "CenterPadImage",
    "ClampImage",
    "ImageToTensor",
    "NarrowImage",
    "NormalizeImage",
    "ReadImage",
    "ResampleImage",
    "RescaleImage",
    "ResizeImage",
)
