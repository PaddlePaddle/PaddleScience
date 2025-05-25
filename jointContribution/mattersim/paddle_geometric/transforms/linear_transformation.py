from typing import Union

import paddle
from paddle import Tensor

from paddle_geometric.data import Data, HeteroData
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform


@functional_transform('linear_transformation')
class LinearTransformation(BaseTransform):
    r"""Transforms node positions :obj:`data.pos` with a square transformation
    matrix computed offline (functional name: :obj:`linear_transformation`).

    Args:
        matrix (Tensor): Tensor with shape :obj:`[D, D]` where :obj:`D`
            corresponds to the dimensionality of node positions.
    """
    def __init__(self, matrix: Tensor):
        if not isinstance(matrix, Tensor):
            matrix = paddle.to_tensor(matrix)
        assert matrix.ndim == 2, (
            'Transformation matrix should be two-dimensional.')
        assert matrix.shape[0] == matrix.shape[1], (
            f'Transformation matrix should be square (got {matrix.shape})')

        # Store the matrix as its transpose.
        # We do this to enable post-multiplication in `forward`.
        self.matrix = matrix.T

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            if not hasattr(store, 'pos'):
                continue

            pos = store.pos.reshape([-1, 1]) if store.pos.ndim == 1 else store.pos
            assert pos.shape[-1] == self.matrix.shape[-2], (
                'Node position matrix and transformation matrix have '
                'incompatible shapes')
            # Post-multiply the points by the transformation matrix
            # instead of pre-multiplying, to preserve shape `[N, D]`.
            store.pos = paddle.matmul(pos, self.matrix.cast(pos.dtype))

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(\n{self.matrix.numpy()}\n)'
