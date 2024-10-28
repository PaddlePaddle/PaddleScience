r"""Point set distance terms."""
import paddle
from deepali.core import functional as U
from paddle import Tensor

from .base import PointSetDistance


class ClosestPointDistance(PointSetDistance):
    r"""Average closest point distance."""

    def __init__(self, scale: float = 10, split_size: int = 100000.0):
        r"""Initialize closest point distance loss.

        Args:
            scale: Constant factor by which to scale average closest point
                distance value such that magnitude is in similar range to
                other registration loss terms, i.e., image similarity losses.
            split_size: Number of points by which to split point sets during
                distance calculation to avoid running out of memory. For each
                split, all pairwise point distances are calculated, followed
                by a reduction of the results by selecting the minimum across
                all splits.

        """
        super().__init__()
        self.scale = float(scale)
        self.split_size = int(split_size)

    def forward(self, x: paddle.Tensor, *ys: paddle.Tensor) -> paddle.Tensor:
        r"""Evaluate point set distance."""
        if not ys:
            raise ValueError(f"{type(self).__name__}.forward() requires at least two point sets")
        x = x.astype(dtype="float32")
        loss = paddle.to_tensor(data=0, dtype=x.dtype, place=x.place)
        for y in ys:
            with paddle.no_grad():
                indices = U.closest_point_indices(x, y, split_size=self.split_size)
            y = U.batched_index_select(y, 1, indices)
            dists: Tensor = paddle.linalg.norm(x=x - y, p=2, axis=2)
            loss += dists.mean()
        return self.scale * loss / len(ys)

    def extra_repr(self) -> str:
        return f"scale={self.scale}, split_size={self.split_size}"


CPD = ClosestPointDistance


class LandmarkPointDistance(PointSetDistance):
    r"""Average distance between corresponding landmarks."""

    def __init__(self, scale: float = 10):
        r"""Initialize point distance loss.

        Args:
            scale: Constant factor by which to scale average point distance value
                such that magnitude is in similar range to other registration loss
                terms, i.e., image similarity losses.

        """
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: paddle.Tensor, *ys: paddle.Tensor) -> paddle.Tensor:
        r"""Evaluate point set distance."""
        if not ys:
            raise ValueError(f"{type(self).__name__}.forward() requires at least two point sets")
        x = x.astype(dtype="float32")
        loss = paddle.to_tensor(data=0, dtype=x.dtype, place=x.place)
        for y in ys:
            dists: Tensor = paddle.linalg.norm(x=x - y, p=2, axis=2)
            loss += dists.mean()
        return self.scale * loss / len(ys)

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


LPD = LandmarkPointDistance
