"""
Ref:
    - https://github.com/mir-group/nequip
    - https://www.nature.com/articles/s41467-022-29939-5
"""
import math

import paddle
from mattersim.utils.paddle_utils import dim2perm


class BesselBasis(paddle.nn.Layer):
    def __init__(self, r_max, num_basis=8, trainable=True):
        """Radial Bessel Basis, as proposed in
            DimeNet: https://arxiv.org/abs/2003.03123

        Parameters
        ----------
        r_max : float
            Cutoff radius

        num_basis : int
            Number of Bessel Basis functions

        trainable : bool
            Train the :math:`n \\pi` part or not.
        """
        super(BesselBasis, self).__init__()
        self.trainable = trainable
        self.num_basis = num_basis
        self.r_max = float(r_max)
        self.prefactor = 2.0 / self.r_max
        bessel_weights = (
            paddle.linspace(start=1.0, stop=num_basis, num=num_basis) * math.pi
        )
        if self.trainable:
            self.bessel_weights = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=bessel_weights
            )
        else:
            self.register_buffer(name="bessel_weights", tensor=bessel_weights)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : paddle.Tensor
            Input
        """
        numerator = paddle.sin(
            x=self.bessel_weights * x.unsqueeze(axis=-1) / self.r_max
        )
        return self.prefactor * (numerator / x.unsqueeze(axis=-1))


class SmoothBesselBasis(paddle.nn.Layer):
    def __init__(self, r_max, max_n=10):
        """Smooth Radial Bessel Basis, as proposed
            in DimeNet: https://arxiv.org/abs/2003.03123
            This is an orthogonal basis with first
            and second derivative at the cutoff
            equals to zero. The function was derived from
            the order 0 spherical Bessel function,
            and was expanded by the different zero roots
        Ref:
            https://arxiv.org/pdf/1907.02374.pdf
        Args:
            r_max: paddle.Tensor distance tensor
            max_n: int, max number of basis, expanded by the zero roots
        Returns: expanded spherical harmonics with
                 derivatives smooth at boundary
        Parameters
        ----------
        """
        super(SmoothBesselBasis, self).__init__()
        self.max_n = max_n
        n = paddle.arange(start=0, end=max_n).astype(dtype="float32")[None, :]
        PI = 3.1415926535897
        SQRT2 = 1.41421356237
        fnr = (
            (-1) ** n
            * SQRT2
            * PI
            / r_max**1.5
            * (n + 1)
            * (n + 2)
            / paddle.sqrt(x=2 * n**2 + 6 * n + 5)
        )
        en = n**2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
        dn = [paddle.to_tensor(data=1.0).astype(dtype="float32")]
        for i in range(1, max_n):
            dn.append(1 - en[0, i] / dn[-1])
        dn = paddle.stack(x=dn)
        self.register_buffer(name="dn", tensor=dn)
        self.register_buffer(name="en", tensor=en)
        self.register_buffer(name="fnr_weights", tensor=fnr)
        self.register_buffer(
            name="n_1_pi_cutoff",
            tensor=(
                (paddle.arange(start=0, end=max_n).astype(dtype="float32") + 1)
                * PI
                / r_max
            ).reshape(1, -1),
        )
        self.register_buffer(
            name="n_2_pi_cutoff",
            tensor=(
                (paddle.arange(start=0, end=max_n).astype(dtype="float32") + 2)
                * PI
                / r_max
            ).reshape(1, -1),
        )
        self.register_buffer(name="r_max", tensor=paddle.to_tensor(data=r_max))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Evaluate Smooth Bessel Basis for input x.

        Parameters
        ----------
        x : paddle.Tensor
            Input
        """
        x_1 = x.unsqueeze(axis=-1) * self.n_1_pi_cutoff
        x_2 = x.unsqueeze(axis=-1) * self.n_2_pi_cutoff
        fnr = self.fnr_weights * (paddle.sin(x=x_1) / x_1 + paddle.sin(x=x_2) / x_2)
        gn = [fnr[:, 0]]
        for i in range(1, self.max_n):
            gn.append(
                1
                / paddle.sqrt(x=self.dn[i])
                * (fnr[:, i] + paddle.sqrt(x=self.en[0, i] / self.dn[i - 1]) * gn[-1])
            )
        return paddle.transpose(
            x=paddle.stack(x=gn), perm=dim2perm(paddle.stack(x=gn).ndim, 1, 0)
        )
