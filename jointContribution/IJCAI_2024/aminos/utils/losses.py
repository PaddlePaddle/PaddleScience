import paddle


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = tuple(x.shape)[0]
        h = 1.0 / (tuple(x.shape)[1] - 1.0)
        all_norms = h ** (self.d / self.p) * paddle.linalg.norm(
            x=x.view(num_examples, -1) - y.view(num_examples, -1), p=self.p, axis=1
        )
        if self.reduction:
            if self.size_average:
                return paddle.mean(x=all_norms)
            else:
                return paddle.sum(x=all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = tuple(x.shape)[0]
        diff_norms = paddle.linalg.norm(
            x=x.reshape(num_examples, -1) - y.reshape(num_examples, -1),
            p=self.p,
            axis=1,
        )
        y_norms = paddle.linalg.norm(x=y.reshape(num_examples, -1), p=self.p, axis=1)
        if self.reduction:
            if self.size_average:
                return paddle.mean(x=diff_norms / y_norms)
            else:
                return paddle.sum(x=diff_norms / y_norms)
        return diff_norms / y_norms

    def rel_batch(self, x, y, batch, num_graphs):
        loss = paddle.to_tensor(data=0.0, dtype=x.dtype, place=x.place)
        for i in range(num_graphs):
            mask = i == batch
            rel_loss = self.rel(
                x[mask][
                    None,
                ],
                y[mask][
                    None,
                ]
                + 1e-08,
            )
            if paddle.isnan(x=rel_loss).astype("bool").any():
                raise ValueError(f"NaN detected in rel_loss for graph {i}")
            loss = loss + rel_loss
        loss /= num_graphs
        return loss

    def __call__(self, x, y, batch=None, num_graphs=None):
        if batch is None:
            return self.rel(x, y)
        else:
            return self.rel_batch(x, y, batch, num_graphs=num_graphs)
