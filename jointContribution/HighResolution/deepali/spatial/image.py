from deprecation import deprecated

from ..core import __version__
from .transformer import ImageTransformer


@deprecated(
    deprecated_in="0.3",
    removed_in="1.0",
    current_version=__version__,
    details="Use deepali.spatial.ImageTransformer instead",
)
class ImageTransform(ImageTransformer):
    ...
