# paddle_harmonics(Paddle Backend)

> [!IMPORTANT]
> This branch(paddle) experimentally supports [Paddle backend](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html)
> as almost all the core code has been completely rewritten using the Paddle API.
>
> It is recommended to install **nightly-build(develop)** Paddle before running any code in this branch.

Install:

``` shell
# paddlepaddle develop
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/
pip install -r requirements.txt

pip install .

# test
pytest tests/

# example
mkdir examples/checkpoints
mkdir examples/figures
mkdir examples/output_data
python examples/train_sfno.py

# notebooks
mkdir notebooks/data
mkdir notebooks/plots
chmod a+rwx notebooks/data
chmod a+rwx notebooks/plots
```

# example code

``` python
import math
import unittest

import numpy as np
import paddle

class TestLegendrePolynomials(unittest.TestCase):
    def setUp(self):
        self.cml = lambda m, l: np.sqrt((2 * l + 1) / 4 / np.pi) * np.sqrt(
            math.factorial(l - m) / math.factorial(l + m)
        )
        self.pml = dict()

        # preparing associated Legendre Polynomials (These include the Condon-Shortley phase)
        # for reference see e.g. https://en.wikipedia.org/wiki/Associated_Legendre_polynomials
        self.pml[(0, 0)] = lambda x: np.ones_like(x)
        self.pml[(0, 1)] = lambda x: x
        self.pml[(1, 1)] = lambda x: -np.sqrt(1.0 - x**2)
        self.pml[(0, 2)] = lambda x: 0.5 * (3 * x**2 - 1)
        self.pml[(1, 2)] = lambda x: -3 * x * np.sqrt(1.0 - x**2)
        self.pml[(2, 2)] = lambda x: 3 * (1 - x**2)
        self.pml[(0, 3)] = lambda x: 0.5 * (5 * x**3 - 3 * x)
        self.pml[(1, 3)] = lambda x: 1.5 * (1 - 5 * x**2) * np.sqrt(1.0 - x**2)
        self.pml[(2, 3)] = lambda x: 15 * x * (1 - x**2)
        self.pml[(3, 3)] = lambda x: -15 * np.sqrt(1.0 - x**2) ** 3

        self.lmax = self.mmax = 4

        self.tol = 1e-9

    def test_legendre(self):
        print("Testing computation of associated Legendre polynomials")
        from paddle_harmonics.legendre import legpoly

        t = np.linspace(0, 1, 100)
        vdm = legpoly(self.mmax, self.lmax, t)

        for l in range(self.lmax):
            for m in range(l + 1):
                diff = vdm[m, l] / self.cml(m, l) - self.pml[(m, l)](t)
                self.assertTrue(diff.max() <= self.tol)
```
