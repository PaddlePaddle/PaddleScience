- Build

```sh
cd PaddleScatter
python setup_ops.py install
pip install .
```

- Test

```sh
cd PaddleScatter/paddle_scatter
pytest tests
```

- Usage
```py
import paddle
from paddle_scatter import scatter_max

src = paddle.to_tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
index = paddle.to_tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])

out, argmax = scatter_max(src, index, dim=-1)

print(out)
Tensor(shape=[2, 6], dtype=int64, place=Place(gpu:0), stop_gradient=True,
       [[0, 0, 4, 3, 2, 0],
        [2, 4, 3, 0, 0, 0]])

print(argmax)
Tensor(shape=[2, 6], dtype=int64, place=Place(gpu:0), stop_gradient=True,
       [[5, 5, 3, 4, 0, 1],
        [1, 4, 3, 5, 5, 5]])
```
