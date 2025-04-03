import argparse
import operator
from functools import reduce

import paddle


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


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

    def __call__(self, x, y):
        return self.rel(x, y)


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(tuple(p.shape)))
    return c
