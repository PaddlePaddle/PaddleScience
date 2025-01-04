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
    data : paddle.Tensor
        vector of data points from which to find neighbors
    queries : paddle.Tensor
        centers of neighborhoods
    radius : float
        size of each neighborhood
    """

    dists = paddle.cdist(queries, data).to(queries.device) # shaped num query points x num data points
    in_nbr = paddle.where(dists <= radius, 1., 0.) # i,j is one if j is i's neighbor
    nbr_indices = in_nbr.nonzero()[:,1:].reshape(-1,) # only keep the column indices
    nbrhd_sizes = paddle.cumsum(paddle.sum(in_nbr, axis=1), axis=0) # num points in each neighborhood, summed cumulatively
    splits = paddle.concat((paddle.to_tensor([0.]).to(queries.device), nbrhd_sizes))
    nbr_dict = {}
    nbr_dict['neighbors_index'] = nbr_indices.long().to(queries.device)
    nbr_dict['neighbors_row_splits'] = splits.long()
    return nbr_dict