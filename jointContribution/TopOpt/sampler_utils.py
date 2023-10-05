import numpy as np


def uniform_sampler():
    return lambda: np.random.randint(1, 99)


def poisson_sampler(lam):
    def func():
        iter_ = max(np.random.poisson(lam), 1)
        iter_ = min(iter_, 99)
        return iter_

    return func
