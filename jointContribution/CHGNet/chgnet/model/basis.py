from __future__ import annotations

import numpy as np
import paddle


class Fourier(paddle.nn.Layer):
    """Fourier Expansion for angle features."""

    def __init__(self, *, order: int = 5, learnable: bool = False) -> None:
        """Initialize the Fourier expansion.

        Args:
            order (int): the maximum order, refer to the N in eq 1 in CHGNet paper
                Default = 5
            learnable (bool): whether to set the frequencies as learnable parameters
                Default = False
        """
        super().__init__()
        self.order = order
        if learnable:
            self.frequencies = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.arange(start=1, end=order + 1, dtype="float32"),
                trainable=True,
            )
        else:
            self.register_buffer(
                name="frequencies",
                tensor=paddle.arange(start=1, end=order + 1, dtype="float32"),
            )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Apply Fourier expansion to a feature Tensor."""

        # result = paddle.zeros(shape=[tuple(x.shape)[0], 1 + 2 * self.order],
        #     dtype=x.dtype)
        result = paddle.ones(shape=[tuple(x.shape)[0], 1], dtype=x.dtype)
        result = result / paddle.sqrt(x=paddle.to_tensor(data=[2.0]))

        tmp = paddle.outer(x=x, y=self.frequencies)
        # result[:, 1:self.order + 1] = paddle.sin(x=tmp)
        # result[:, self.order + 1:] = paddle.cos(x=tmp)

        result = paddle.concat([result, paddle.sin(tmp), paddle.cos(tmp)], axis=1)

        return result / np.sqrt(np.pi)


class RadialBessel(paddle.nn.Layer):
    """1D Bessel Basis
    from: https://github.com/TUM-DAML/gemnet_pytorch/.
    """

    def __init__(
        self,
        *,
        num_radial: int = 9,
        cutoff: float = 5,
        learnable: bool = False,
        smooth_cutoff: int = 5,
    ) -> None:
        """Initialize the SmoothRBF function.

        Args:
            num_radial (int): Controls maximum frequency
                Default = 9
            cutoff (float):  Cutoff distance in Angstrom.
                Default = 5
            learnable (bool): whether to set the frequencies learnable
                Default = False
            smooth_cutoff (int): smooth cutoff strength
                Default = 5
        """
        super().__init__()
        self.num_radial = num_radial
        self.inv_cutoff = 1 / cutoff
        self.norm_const = (2 * self.inv_cutoff) ** 0.5
        if learnable:
            self.frequencies = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.to_tensor(
                    data=np.pi * np.arange(1, self.num_radial + 1, dtype=np.float32),
                    dtype="float32",
                ),
                trainable=True,
            )
        else:
            self.register_buffer(
                name="frequencies",
                tensor=np.pi
                * paddle.arange(start=1, end=self.num_radial + 1, dtype="float32"),
            )
        if smooth_cutoff is not None:
            self.smooth_cutoff = CutoffPolynomial(
                cutoff=cutoff, cutoff_coeff=smooth_cutoff
            )
        else:
            self.smooth_cutoff = None

    def forward(
        self, dist: paddle.Tensor, *, return_smooth_factor: bool = False
    ) -> (paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor]):
        """Apply Bessel expansion to a feature Tensor.

        Args:
            dist (Tensor): tensor of distances [n, 1]
            return_smooth_factor (bool): whether to return the smooth factor
                Default = False

        Returns:
            out (Tensor): tensor of Bessel distances [n, dim]
            where the expanded dimension will be num_radial
            smooth_factor (Tensor): tensor of smooth factors [n, 1]
        """
        dist = dist[:, None]
        d_scaled = dist * self.inv_cutoff
        out = self.norm_const * paddle.sin(x=self.frequencies * d_scaled) / dist
        if self.smooth_cutoff is not None:
            smooth_factor = self.smooth_cutoff(dist)
            out = smooth_factor * out
            if return_smooth_factor:
                return out, smooth_factor
        return out


class GaussianExpansion(paddle.nn.Layer):
    """Expands the distance by Gaussian basis.
    Unit: angstrom.
    """

    def __init__(
        self,
        min: float = 0,
        max: float = 5,
        step: float = 0.5,
        var: (float | None) = None,
    ) -> None:
        """Gaussian Expansion
        expand a scalar feature to a soft-one-hot feature vector.

        Args:
            min (float): minimum Gaussian center value
            max (float): maximum Gaussian center value
            step (float): Step size between the Gaussian centers
            var (float): variance in gaussian filter, default to step
        """
        super().__init__()
        if min >= max:
            raise ValueError(f"min={min!r} must be less than max={max!r}")
        if max - min <= step:
            raise ValueError(
                f"max - min={max - min!r} must be greater than step={step!r}"
            )
        self.register_buffer(
            name="gaussian_centers",
            tensor=paddle.arange(start=min, end=max + step, step=step),
        )
        self.var = var or step
        if self.var <= 0:
            raise ValueError(f"var={var!r} must be positive")

    def expand(self, features: paddle.Tensor) -> paddle.Tensor:
        """Apply Gaussian filter to a feature Tensor.

        Args:
            features (Tensor): tensor of features [n]

        Returns:
            expanded features (Tensor): tensor of Gaussian distances [n, dim]
            where the expanded dimension will be (dmax - dmin) / step + 1
        """
        return paddle.exp(
            x=-((features.reshape(-1, 1) - self.gaussian_centers) ** 2) / self.var**2
        )


class CutoffPolynomial(paddle.nn.Layer):
    """Polynomial soft-cutoff function for atom graph
    ref: https://github.com/TUM-DAML/gemnet_pytorch/blob/-/gemnet/model/layers/envelope.py.
    """

    def __init__(self, cutoff: float = 5, cutoff_coeff: float = 5) -> None:
        """Initialize the polynomial cutoff function.

        Args:
            cutoff (float): cutoff radius (A) in atom graph construction
            Default = 5
            cutoff_coeff (float): the strength of soft-Cutoff
            0 will disable the cutoff, returning 1 at every r
            for positive numbers > 0, the smaller cutoff_coeff is, the faster this
                function decays. Default = 5.
        """
        super().__init__()
        self.cutoff = cutoff
        self.p = cutoff_coeff
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, r: paddle.Tensor) -> paddle.Tensor:
        """Polynomial cutoff function.

        Args:
            r (Tensor): radius distance tensor

        Returns:
            polynomial cutoff functions: decaying from 1 at r=0 to 0 at r=cutoff
        """
        if self.p != 0:
            r_scaled = r / self.cutoff
            env_val = (
                1
                + self.a * r_scaled**self.p
                + self.b * r_scaled ** (self.p + 1)
                + self.c * r_scaled ** (self.p + 2)
            )
            return paddle.where(
                condition=r_scaled < 1, x=env_val, y=paddle.zeros_like(x=r_scaled)
            )
        return paddle.ones(shape=tuple(r.shape), dtype=r.dtype)
