import paddle


def knn(ref_points, query_points, k=4):
    dists = paddle.norm(
        paddle.unsqueeze(ref_points, axis=1) - paddle.unsqueeze(query_points, axis=0),
        p=2,
        axis=-1,
    )
    _, indices = paddle.topk(dists, k=k, axis=-1, largest=False)
    return indices
