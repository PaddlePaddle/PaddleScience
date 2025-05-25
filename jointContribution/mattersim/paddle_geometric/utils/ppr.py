from itertools import chain
from typing import Callable, List, Optional, Tuple

import numpy as np
import paddle
from paddle import Tensor

try:
    import numba
    WITH_NUMBA = True
except Exception:  # pragma: no cover
    WITH_NUMBA = False


def _get_ppr(  # pragma: no cover
    rowptr: np.ndarray,
    col: np.ndarray,
    alpha: float,
    eps: float,
    target: Optional[np.ndarray] = None,
) -> Tuple[List[List[int]], List[List[float]]]:

    num_nodes = len(rowptr) - 1 if target is None else len(target)
    alpha_eps = alpha * eps
    js = [[0]] * num_nodes
    vals = [[0.]] * num_nodes

    for inode_uint in numba.prange(num_nodes):
        inode = inode_uint if target is None else target[inode_uint]

        p = {inode: 0.0}
        r = {}
        r[inode] = alpha
        q = [inode]

        while len(q) > 0:
            unode = q.pop()

            res = r[unode] if unode in r else 0
            p[unode] = p.get(unode, 0) + res

            r[unode] = 0
            start, end = rowptr[unode], rowptr[unode + 1]
            ucount = end - start

            for vnode in col[start:end]:
                _val = (1 - alpha) * res / ucount
                r[vnode] = r.get(vnode, 0) + _val

                res_vnode = r[vnode]
                vcount = rowptr[vnode + 1] - rowptr[vnode]
                if res_vnode >= alpha_eps * vcount and vnode not in q:
                    q.append(vnode)

        js[inode_uint] = list(p.keys())
        vals[inode_uint] = list(p.values())

    return js, vals


_get_ppr_numba: Optional[Callable] = None


def get_ppr(
    edge_index: Tensor,
    alpha: float = 0.2,
    eps: float = 1e-5,
    target: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Calculates the personalized PageRank (PPR) vector for all or a subset
    of nodes using a variant of the `Andersen algorithm
    <https://mathweb.ucsd.edu/~fan/wp/localpartition.pdf>`_.

    Args:
        edge_index (paddle.Tensor): The indices of the graph.
        alpha (float, optional): The alpha value of the PageRank algorithm.
            (default: :obj:`0.2`)
        eps (float, optional): The threshold for stopping the PPR calculation
            (:obj:`edge_weight >= eps * out_degree`). (default: :obj:`1e-5`)
        target (paddle.Tensor, optional): The target nodes to compute PPR for.
            If not given, calculates PPR vectors for all nodes.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes. (default: :obj:`None`)

    :rtype: (:class:`paddle.Tensor`, :class:`paddle.Tensor`)
    """
    if not WITH_NUMBA:  # pragma: no cover
        raise ImportError("'get_ppr' requires the 'numba' package")

    global _get_ppr_numba
    if _get_ppr_numba is None:
        _get_ppr_numba = numba.jit(nopython=True, parallel=True)(_get_ppr)

    num_nodes = num_nodes or edge_index.shape[1] if num_nodes is None else num_nodes

    rowptr, col = edge_index[0].numpy(), edge_index[1].numpy()

    cols, weights = _get_ppr_numba(
        rowptr,
        col,
        alpha,
        eps,
        None if target is None else target.numpy(),
    )

    device = edge_index.place
    col = paddle.to_tensor(list(chain.from_iterable(cols)), place=device)
    weight = paddle.to_tensor(list(chain.from_iterable(weights)), place=device)
    deg = paddle.to_tensor([len(value) for value in cols], place=device)

    row = paddle.arange(num_nodes) if target is None else target
    row = row.tile([deg.sum() // num_nodes])

    edge_index = paddle.stack([row, col], axis=0)

    return edge_index, weight
