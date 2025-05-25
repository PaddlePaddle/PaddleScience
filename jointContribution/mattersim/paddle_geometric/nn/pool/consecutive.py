import paddle

def consecutive_cluster(src):
    unique, inv = paddle.unique(src, return_inverse=True)
    perm = paddle.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = paddle.zeros_like(unique, dtype=inv.dtype).scatter_(0, inv, perm)
    return inv, perm
