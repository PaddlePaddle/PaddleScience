from typing import Any, Dict, Tuple

import numpy as np
import paddle
from paddle import Tensor

from paddle_geometric.data import Data
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform
from paddle_geometric.utils import (
    add_self_loops,
    coalesce,
    get_ppr,
    is_undirected,
    scatter,
    sort_edge_index,
    to_dense_adj,
)


@functional_transform('gdc')
class GDC(BaseTransform):
    r"""Processes the graph via Graph Diffusion Convolution (GDC) from the
    `"Diffusion Improves Graph Learning" <https://arxiv.org/abs/1911.05485>`_
    paper (functional name: :obj:`gdc`).

    Args:
        self_loop_weight (float, optional): Weight of the added self-loop.
            Set to :obj:`None` to add no self-loops. (default: :obj:`1`)
        normalization_in (str, optional): Normalization of the transition
            matrix on the original (input) graph. Options are:
            :obj:`"sym"`, :obj:`"col"`, and :obj:`"row"`.
        normalization_out (str, optional): Normalization of the transition
            matrix on the transformed GDC (output) graph. Options are:
            :obj:`"sym"`, :obj:`"col"`, :obj:`"row"`, and :obj:`None`.
        diffusion_kwargs (dict, optional): Parameters for diffusion method.
            Options for `method` include :obj:`"ppr"`, :obj:`"heat"`,
            and :obj:`"coeff"`. Additional parameters are specific to the method.
        sparsification_kwargs (dict, optional): Parameters for sparsification.
            Options for `method` include :obj:`"threshold"` and :obj:`"topk"`.
        exact (bool, optional): If True, calculate the exact diffusion
            matrix (not scalable for large graphs). (default: :obj:`True`)
    """
    def __init__(
        self,
        self_loop_weight: float = 1.,
        normalization_in: str = 'sym',
        normalization_out: str = 'col',
        diffusion_kwargs: Dict[str, Any] = dict(method='ppr', alpha=0.15),
        sparsification_kwargs: Dict[str, Any] = dict(
            method='threshold',
            avg_degree=64,
        ),
        exact: bool = True,
    ) -> None:
        self.self_loop_weight = self_loop_weight
        self.normalization_in = normalization_in
        self.normalization_out = normalization_out
        self.diffusion_kwargs = diffusion_kwargs
        self.sparsification_kwargs = sparsification_kwargs
        self.exact = exact

        if self_loop_weight:
            assert exact or self_loop_weight == 1

    @paddle.no_grad()
    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        edge_index = data.edge_index
        N = data.num_nodes
        assert N is not None

        if data.edge_attr is None:
            edge_weight = paddle.ones([edge_index.shape[1]])
        else:
            edge_weight = data.edge_attr
            assert self.exact
            assert edge_weight.ndim == 1

        if self.self_loop_weight:
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, fill_value=self.self_loop_weight,
                num_nodes=N)

        edge_index, edge_weight = coalesce(edge_index, edge_weight, N)

        if self.exact:
            edge_index, edge_weight = self.transition_matrix(
                edge_index, edge_weight, N, self.normalization_in)
            diff_mat = self.diffusion_matrix_exact(edge_index, edge_weight, N,
                                                   **self.diffusion_kwargs)
            edge_index, edge_weight = self.sparsify_dense(
                diff_mat, **self.sparsification_kwargs)
        else:
            edge_index, edge_weight = self.diffusion_matrix_approx(
                edge_index, edge_weight, N, self.normalization_in,
                **self.diffusion_kwargs)
            edge_index, edge_weight = self.sparsify_sparse(
                edge_index, edge_weight, N, **self.sparsification_kwargs)

        edge_index, edge_weight = coalesce(edge_index, edge_weight, N)
        edge_index, edge_weight = self.transition_matrix(
            edge_index, edge_weight, N, self.normalization_out)

        data.edge_index = edge_index
        data.edge_attr = edge_weight

        return data

    def transition_matrix(
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        num_nodes: int,
        normalization: str,
    ) -> Tuple[Tensor, Tensor]:
        if normalization == 'sym':
            row, col = edge_index
            deg = scatter(edge_weight, col, dim=0, reduce='sum')
            deg_inv_sqrt = paddle.pow(deg, -0.5)
            deg_inv_sqrt = paddle.where(paddle.isinf(deg_inv_sqrt), paddle.zeros_like(deg_inv_sqrt), deg_inv_sqrt)
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        elif normalization == 'col':
            _, col = edge_index
            deg = scatter(edge_weight, col, dim=0, reduce='sum')
            deg_inv = 1. / deg
            deg_inv = paddle.where(paddle.isinf(deg_inv), paddle.zeros_like(deg_inv), deg_inv)
            edge_weight = edge_weight * deg_inv[col]
        elif normalization == 'row':
            row, _ = edge_index
            deg = scatter(edge_weight, row, dim=0, reduce='sum')
            deg_inv = 1. / deg
            deg_inv = paddle.where(paddle.isinf(deg_inv), paddle.zeros_like(deg_inv), deg_inv)
            edge_weight = edge_weight * deg_inv[row]
        elif normalization is None:
            pass
        else:
            raise ValueError(f"Transition matrix normalization '{normalization}' unknown")

        return edge_index, edge_weight

    def diffusion_matrix_exact(
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        num_nodes: int,
        method: str,
        **kwargs: Any,
    ) -> Tensor:
        if method == 'ppr':
            edge_weight = (kwargs['alpha'] - 1) * edge_weight
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                     fill_value=1,
                                                     num_nodes=num_nodes)
            mat = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()
            diff_matrix = kwargs['alpha'] * paddle.inverse(mat)

        elif method == 'heat':
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                     fill_value=-1,
                                                     num_nodes=num_nodes)
            edge_weight = kwargs['t'] * edge_weight
            mat = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()
            undirected = is_undirected(edge_index, edge_weight, num_nodes)
            diff_matrix = self.__expm__(mat, undirected)

        elif method == 'coeff':
            adj_matrix = to_dense_adj(edge_index,
                                      edge_attr=edge_weight).squeeze()
            mat = paddle.eye(num_nodes)

            diff_matrix = kwargs['coeffs'][0] * mat
            for coeff in kwargs['coeffs'][1:]:
                mat = paddle.matmul(mat, adj_matrix)
                diff_matrix += coeff * mat
        else:
            raise ValueError(f"Exact GDC diffusion '{method}' unknown")

        return diff_matrix

    def diffusion_matrix_approx(
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        num_nodes: int,
        normalization: str,
        method: str,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        if method == 'ppr':
            if normalization == 'sym':
                _, col = edge_index
                deg = scatter(edge_weight, col, dim=0, reduce='sum')

            edge_index, edge_weight = get_ppr(
                edge_index,
                alpha=kwargs['alpha'],
                eps=kwargs['eps'],
                num_nodes=num_nodes,
            )

            if normalization == 'col':
                edge_index, edge_weight = sort_edge_index(
                    edge_index.flip([0]), edge_weight, num_nodes)

            if normalization == 'sym':
                row, col = edge_index
                deg_inv = paddle.sqrt(deg)
                deg_inv_sqrt = paddle.pow(deg, -0.5)
                deg_inv_sqrt = paddle.where(paddle.isinf(deg_inv_sqrt), paddle.zeros_like(deg_inv_sqrt), deg_inv_sqrt)
                edge_weight = deg_inv[row] * edge_weight * deg_inv_sqrt[col]
            elif normalization in ['col', 'row']:
                pass
            else:
                raise ValueError(
                    f"Transition matrix normalization '{normalization}' not "
                    f"implemented for non-exact GDC computation")

        elif method == 'heat':
            raise NotImplementedError('Currently no fast heat kernel is implemented.')
        else:
            raise ValueError(f"Approximate GDC diffusion '{method}' unknown")

        return edge_index, edge_weight

    def sparsify_dense(
        self,
        matrix: Tensor,
        method: str,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        assert matrix.shape[0] == matrix.shape[1]
        N = matrix.shape[1]

        if method == 'threshold':
            if 'eps' not in kwargs.keys():
                kwargs['eps'] = self.__calculate_eps__(matrix, N,
                                                       kwargs['avg_degree'])

            edge_index = paddle.nonzero(matrix >= kwargs['eps'], as_tuple=False).t()
            edge_index_flat = edge_index[0] * N + edge_index[1]
            edge_weight = paddle.flatten(matrix)[edge_index_flat]

        elif method == 'topk':
            k, dim = min(N, kwargs['k']), kwargs['dim']
            assert dim in [0, 1]
            sort_idx = paddle.argsort(matrix, axis=dim, descending=True)
            if dim == 0:
                top_idx = sort_idx[:k]
                edge_weight = paddle.gather(matrix, axis=dim, index=top_idx).flatten()

                row_idx = paddle.arange(0, N).tile([k])
                edge_index = paddle.stack([top_idx.flatten(), row_idx], axis=0)
            else:
                top_idx = sort_idx[:, :k]
                edge_weight = paddle.gather(matrix, axis=dim, index=top_idx).flatten()

                col_idx = paddle.arange(0, N).tile([k])
                edge_index = paddle.stack([col_idx, top_idx.flatten()], axis=0)
        else:
            raise ValueError(f"GDC sparsification '{method}' unknown")

        return edge_index, edge_weight

    def sparsify_sparse(
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        num_nodes: int,
        method: str,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        if method == 'threshold':
            if 'eps' not in kwargs.keys():
                kwargs['eps'] = self.__calculate_eps__(edge_weight, num_nodes, kwargs['avg_degree'])

            remaining_edge_idx = paddle.nonzero(edge_weight >= kwargs['eps'], as_tuple=False).flatten()
            edge_index = edge_index[:, remaining_edge_idx]
            edge_weight = edge_weight[remaining_edge_idx]
        elif method == 'topk':
            raise NotImplementedError('Sparse topk sparsification not implemented')
        else:
            raise ValueError(f"GDC sparsification '{method}' unknown")

        return edge_index, edge_weight

    def __expm__(self, matrix: Tensor, symmetric: bool) -> Tensor:
        from scipy.linalg import expm

        if symmetric:
            e, V = paddle.linalg.eigh(matrix, UPLO='U')
            diff_mat = V @ paddle.diag(paddle.exp(e)) @ V.t()
        else:
            diff_mat = paddle.to_tensor(expm(matrix.numpy()))
        return diff_mat

    def __calculate_eps__(self, matrix: Tensor, num_nodes: int, avg_degree: int) -> float:
        sorted_edges = paddle.sort(paddle.flatten(matrix), descending=True)
        if avg_degree * num_nodes > len(sorted_edges):
            return -np.inf

        left = sorted_edges[avg_degree * num_nodes - 1]
        right = sorted_edges[avg_degree * num_nodes]
        return float((left + right) / 2.0)
