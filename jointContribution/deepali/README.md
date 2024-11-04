# deepali(Paddle Backend)

> [!IMPORTANT]
> This branch(paddle) experimentally supports [Paddle backend](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html)
> as almost all the core code has been completely rewritten using the Paddle API.
>
> It is recommended to install **nightly-build(develop)** Paddle before running any code in this branch.

Install:

``` shell
# paddlepaddle develop
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/

# setup config file is pyproject.toml
pip install .

# test
pytest tests/

```

# example code
``` python
import paddle
from deepali.core import bspline as B

kernel = B.cubic_bspline_interpolation_weights(5)
assert isinstance(kernel, paddle.Tensor)
assert tuple(kernel.shape) == (5, 4)

assert paddle.allclose(
        x=kernel, y=B.cubic_bspline_interpolation_weights(5, derivative=0)
    ).item()
```
