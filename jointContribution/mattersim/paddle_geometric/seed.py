import random

import numpy as np
import paddle


def seed_everything(seed: int) -> None:
    r"""Sets the seed for generating random numbers in :paddle:`PaddlePaddle`,
    :obj:`numpy` and :python:`Python`.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
