import numpy as np


def reorganize(in_order_points, out_order_points, quantity_to_reordered):
    n = out_order_points.shape[0]
    idx = np.zeros(n)
    for i in range(n):
        cond = out_order_points[i] == in_order_points
        cond = cond[:, 0] * cond[:, 1]
        idx[i] = np.argwhere(cond)[0][0]
    idx = idx.astype('int')
    assert (in_order_points[idx] == out_order_points).all()
    return quantity_to_reordered[idx]
