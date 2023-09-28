import math

import paddle


class GaussianRF(object):
    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic"):
        self.dim = dim

        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))

        k_max = size // 2

        if dim == 1:
            k = paddle.concat(
                (
                    paddle.arange(start=0, end=k_max, step=1),
                    paddle.arange(start=-k_max, end=0, step=1),
                ),
                0,
            )

            self.sqrt_eig = (
                size
                * math.sqrt(2.0)
                * sigma
                * ((4 * (math.pi**2) * (k**2) + tau**2) ** (-alpha / 2.0))
            )
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = paddle.concat(
                (
                    paddle.arange(start=0, end=k_max, step=1),
                    paddle.arange(start=-k_max, end=0, step=1),
                ),
                0,
            ).tile((size, 1))

            perm = list(range(wavenumers.ndim))
            perm[1] = 0
            perm[0] = 1
            k_x = wavenumers.transpose(perm=perm)
            k_y = wavenumers

            self.sqrt_eig = (
                (size**2)
                * math.sqrt(2.0)
                * sigma
                * (
                    (4 * (math.pi**2) * (k_x**2 + k_y**2) + tau**2)
                    ** (-alpha / 2.0)
                )
            )
            self.sqrt_eig[0, 0] = 0.0

        elif dim == 3:
            wavenumers = paddle.concat(
                (
                    paddle.arange(start=0, end=k_max, step=1),
                    paddle.arange(start=-k_max, end=0, step=1),
                ),
                0,
            ).tile((size, size, 1))

            perm = list(range(wavenumers.ndim))
            perm[1] = 2
            perm[2] = 1
            k_x = wavenumers.transpose(perm=perm)
            k_y = wavenumers

            perm = list(range(wavenumers.ndim))
            perm[0] = 2
            perm[2] = 0
            k_z = wavenumers.transpose(perm=perm)

            self.sqrt_eig = (
                (size**3)
                * math.sqrt(2.0)
                * sigma
                * (
                    (4 * (math.pi**2) * (k_x**2 + k_y**2 + k_z**2) + tau**2)
                    ** (-alpha / 2.0)
                )
            )
            self.sqrt_eig[0, 0, 0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = paddle.randn((N, *self.size, 2))

        coeff[..., 0] = self.sqrt_eig * coeff[..., 0]
        coeff[..., 1] = self.sqrt_eig * coeff[..., 1]

        if self.dim == 2:
            u = paddle.as_real(paddle.fft.ifft2(paddle.as_complex(coeff)))
        else:
            raise f"self.dim not in (2): {self.dim}"

        u = u[..., 0]

        return u
