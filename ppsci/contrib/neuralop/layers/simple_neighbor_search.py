"""
Python implementation of neighbor-search algorithm for use on CPU to avoid
breaking torch_cluster's CPU version.
"""

import paddle


def simple_neighbor_search(data: paddle.Tensor, queries: paddle.Tensor, radius: float):
    """

    Parameters
    ----------
    Density-Based Spatial Clustering of Applications with Noise
    data : torch.Tensor
        vector of data points from which to find neighbors
    queries : torch.Tensor
        centers of neighborhoods
    radius : float
        size of each neighborhood
    """

    dists = paddle.cdist(queries, data)  # shaped num query points x num data points
    in_nbr = paddle.where(dists <= radius, 1.0, 0.0)  # i,j is one if j is i's neighbor
    nbr_indices = in_nbr.nonzero()[:, 1:].reshape(
        [
            -1,
        ]
    )  # only keep the column indices
    nbrhd_sizes = paddle.cumsum(
        paddle.sum(in_nbr, axis=1), axis=0
    )  # num points in each neighborhood, summed cumulatively
    nbrhd_sizes = nbrhd_sizes.astype(paddle.float32)
    splits = paddle.concat((paddle.to_tensor([0.0]), nbrhd_sizes))
    nbr_dict = {}
    nbr_dict["neighbors_index"] = nbr_indices.astype("int64")
    nbr_dict["neighbors_row_splits"] = splits.astype("int64")
    return nbr_dict
