import math
from typing import Dict

import paddle


def central_diff_1d(x, h, fix_x_bnd=False):
    dx = (paddle.roll(x, shifts=-1, axis=-1) - paddle.roll(x, shifts=1, axis=-1)) / (
        2.0 * h
    )

    if fix_x_bnd:
        dx[..., 0] = (x[..., 1] - x[..., 0]) / h
        dx[..., -1] = (x[..., -1] - x[..., -2]) / h

    return dx


def central_diff_2d(x, h, fix_x_bnd=False, fix_y_bnd=False):
    if isinstance(h, float):
        h = [h, h]
    dx = (paddle.roll(x, shifts=-1, axis=-2) - paddle.roll(x, shifts=1, axis=-2)) / (
        2.0 * h[0]
    )
    dy = (paddle.roll(x, shifts=-1, axis=-1) - paddle.roll(x, shifts=1, axis=-1)) / (
        2.0 * h[1]
    )

    if fix_x_bnd:
        dx[..., 0, :] = (x[..., 1, :] - x[..., 0, :]) / h[0]
        dx[..., -1, :] = (x[..., -1, :] - x[..., -2, :]) / h[0]

    if fix_y_bnd:
        dy[..., :, 0] = (x[..., :, 1] - x[..., :, 0]) / h[1]
        dy[..., :, -1] = (x[..., :, -1] - x[..., :, -2]) / h[1]

    return dx, dy


def central_diff_3d(x, h, fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
    if isinstance(h, float):
        h = [h, h, h]

    dx = (paddle.roll(x, shifts=-1, axis=-3) - paddle.roll(x, shifts=1, axis=-3)) / (
        2.0 * h[0]
    )
    dy = (paddle.roll(x, shifts=-1, axis=-2) - paddle.roll(x, shifts=1, axis=-2)) / (
        2.0 * h[1]
    )
    dz = (paddle.roll(x, shifts=-1, axis=-1) - paddle.roll(x, shifts=1, axis=-1)) / (
        2.0 * h[2]
    )

    if fix_x_bnd:
        dx[..., 0, :, :] = (x[..., 1, :, :] - x[..., 0, :, :]) / h[0]
        dx[..., -1, :, :] = (x[..., -1, :, :] - x[..., -2, :, :]) / h[0]

    if fix_y_bnd:
        dy[..., :, 0, :] = (x[..., :, 1, :] - x[..., :, 0, :]) / h[1]
        dy[..., :, -1, :] = (x[..., :, -1, :] - x[..., :, -2, :]) / h[1]

    if fix_z_bnd:
        dz[..., :, :, 0] = (x[..., :, :, 1] - x[..., :, :, 0]) / h[2]
        dz[..., :, :, -1] = (x[..., :, :, -1] - x[..., :, :, -2]) / h[2]

    return dx, dy, dz


class LpLoss(object):
    """loss function with rel/abs Lp loss

    Args:
        d (int, optional): The scaling factor of loss. Defaults to 1.
        p (int, optional): The scaling factor of diff. Defaults to 2.
        L (math, optional): The founction of loss. Defaults to 2*math.pi.
        reduce_dims (int, optional): The dims of reduction. Defaults to 0.
        reductions (str, optional): The type of reduction. Defaults to 'sum'.
    """

    def __init__(self, d=1, p=2, L=2 * math.pi, reduce_dims=0, reductions="sum"):
        super().__init__()

        self.d = d
        self.p = p

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == "sum" or reductions == "mean"
                self.reductions = [reductions] * len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == "sum" or reductions[j] == "mean"
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L] * self.d
        else:
            self.L = L

    def uniform_h(self, x):
        h = [0.0] * self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j] / x.size(-j)

        return h

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == "sum":
                x = paddle.sum(x, axis=self.reduce_dims[j], keepdim=True)
            else:
                x = paddle.mean(x, axis=self.reduce_dims[j], keepdim=True)
        return x

    def abs(self, x, y, h=None):
        # Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h] * self.d

        const = math.prod(h) ** (1.0 / self.p)
        diff = const * paddle.norm(
            paddle.flatten(x, start_axis=-self.d)
            - paddle.flatten(y, start_axis=-self.d),
            p=self.p,
            axis=-1,
            keepdim=False,
        )

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff

    def rel(self, x, y):
        diff = paddle.norm(
            paddle.flatten(x, start_axis=-self.d)
            - paddle.flatten(y, start_axis=-self.d),
            p=self.p,
            axis=-1,
            keepdim=False,
        )
        ynorm = paddle.norm(
            paddle.flatten(y, start_axis=-self.d), p=self.p, axis=-1, keepdim=False
        )
        diff = diff / ynorm

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
        return diff

    def __call__(
        self,
        output_dict: Dict[str, "paddle.Tensor"],
        label_dict: Dict[str, "paddle.Tensor"],
    ):
        x = output_dict["y"]
        y = label_dict["y"]
        return {"l2": self.rel(x, y) / x.shape[0]}


class LpLoss_train(LpLoss):
    def __init__(self, d=1, p=2, L=2 * math.pi, reduce_dims=0, reductions="sum"):
        super().__init__(d=d, p=p, L=L, reduce_dims=reduce_dims, reductions=reductions)

    def __call__(
        self,
        output_dict: Dict[str, "paddle.Tensor"],
        label_dict: Dict[str, "paddle.Tensor"],
        weight_dict=None,
    ):
        x = output_dict["y"]
        y = label_dict["y"]
        return {"y": self.rel(x, y)}


class H1Loss(object):
    """loss function with rel/abs H1 loss

    Args:
        d (int, optional): The scaling factor of loss. Defaults to 1.
        L (math, optional): The founction of loss. Defaults to 2*math.pi.
        reduce_dims (int, optional): The dims of reduction. Defaults to 0.
        reductions (str, optional): The type of reduction. Defaults to 'sum'.
        fix_x_bnd (bool, optional): Whether to fix the x boundaries. Defaults to False.
        fix_y_bnd (bool, optional): Whether to fix the y boundaries. Defaults to False.
        fix_z_bnd (bool, optional): Whether to fix the z boundaries. Defaults to False.
    """

    def __init__(
        self,
        d=1,
        L=2 * math.pi,
        reduce_dims=0,
        reductions="sum",
        fix_x_bnd=False,
        fix_y_bnd=False,
        fix_z_bnd=False,
    ):

        super().__init__()

        assert d > 0 and d < 4, "Currently only implemented for 1, 2, and 3-D."

        self.d = d
        self.fix_x_bnd = fix_x_bnd
        self.fix_y_bnd = fix_y_bnd
        self.fix_z_bnd = fix_z_bnd

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == "sum" or reductions == "mean"
                self.reductions = [reductions] * len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == "sum" or reductions[j] == "mean"
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L] * self.d
        else:
            self.L = L

    def compute_terms(self, x, y, h):
        dict_x = {}
        dict_y = {}

        if self.d == 1:
            dict_x[0] = x
            dict_y[0] = y

            x_x = central_diff_1d(x, h[0], fix_x_bnd=self.fix_x_bnd)
            y_x = central_diff_1d(y, h[0], fix_x_bnd=self.fix_x_bnd)

            dict_x[1] = x_x
            dict_y[1] = y_x

        elif self.d == 2:
            dict_x[0] = paddle.flatten(x, start_axis=-2)
            dict_y[0] = paddle.flatten(y, start_axis=-2)

            x_x, x_y = central_diff_2d(
                x, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd
            )
            y_x, y_y = central_diff_2d(
                y, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd
            )

            dict_x[1] = paddle.flatten(x_x, start_axis=-2)
            dict_x[2] = paddle.flatten(x_y, start_axis=-2)

            dict_y[1] = paddle.flatten(y_x, start_axis=-2)
            dict_y[2] = paddle.flatten(y_y, start_axis=-2)

        else:
            dict_x[0] = paddle.flatten(x, start_axis=-3)
            dict_y[0] = paddle.flatten(y, start_axis=-3)

            x_x, x_y, x_z = central_diff_3d(
                x,
                h,
                fix_x_bnd=self.fix_x_bnd,
                fix_y_bnd=self.fix_y_bnd,
                fix_z_bnd=self.fix_z_bnd,
            )
            y_x, y_y, y_z = central_diff_3d(
                y,
                h,
                fix_x_bnd=self.fix_x_bnd,
                fix_y_bnd=self.fix_y_bnd,
                fix_z_bnd=self.fix_z_bnd,
            )

            dict_x[1] = paddle.flatten(x_x, start_axis=-3)
            dict_x[2] = paddle.flatten(x_y, start_axis=-3)
            dict_x[3] = paddle.flatten(x_z, start_axis=-3)

            dict_y[1] = paddle.flatten(y_x, start_axis=-3)
            dict_y[2] = paddle.flatten(y_y, start_axis=-3)
            dict_y[3] = paddle.flatten(y_z, start_axis=-3)

        return dict_x, dict_y

    def uniform_h(self, x):
        h = [0.0] * self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j] / x.shape[-j]

        return h

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == "sum":
                x = paddle.sum(x, axis=self.reduce_dims[j], keepdim=True)
            else:
                x = paddle.mean(x, axis=self.reduce_dims[j], keepdim=True)

        return x

    def abs(self, x, y, h=None):
        # Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h] * self.d

        dict_x, dict_y = self.compute_terms(x, y, h)

        const = math.prod(h)
        diff = (
            const * paddle.norm(dict_x[0] - dict_y[0], p=2, axis=-1, keepdim=False) ** 2
        )

        for j in range(1, self.d + 1):
            diff += (
                const
                * paddle.norm(dict_x[j] - dict_y[j], p=2, axis=-1, keepdim=False) ** 2
            )

        diff = diff**0.5

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff

    def rel(self, x, y, h=None):
        # Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h] * self.d

        dict_x, dict_y = self.compute_terms(x, y, h)

        diff = paddle.norm(dict_x[0] - dict_y[0], p=2, axis=-1, keepdim=False) ** 2
        ynorm = paddle.norm(dict_y[0], p=2, axis=-1, keepdim=False) ** 2

        for j in range(1, self.d + 1):
            diff += paddle.norm(dict_x[j] - dict_y[j], p=2, axis=-1, keepdim=False) ** 2
            ynorm += paddle.norm(dict_y[j], p=2, axis=-1, keepdim=False) ** 2

        diff = (diff**0.5) / (ynorm**0.5)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff

    def __call__(
        self,
        output_dict: Dict[str, "paddle.Tensor"],
        label_dict: Dict[str, "paddle.Tensor"],
        weight_dict: Dict[str, "paddle.Tensor"] = None,
        h=None,
    ):
        x = output_dict["y"]
        y = label_dict["y"]
        return {"h1": self.rel(x, y, h=h) / x.shape[0]}


class H1Loss_train(H1Loss):
    def __init__(
        self,
        d=1,
        L=2 * math.pi,
        reduce_dims=0,
        reductions="sum",
        fix_x_bnd=False,
        fix_y_bnd=False,
        fix_z_bnd=False,
    ):
        super().__init__(
            d=d,
            L=L,
            reduce_dims=reduce_dims,
            reductions=reductions,
            fix_x_bnd=fix_x_bnd,
            fix_y_bnd=fix_y_bnd,
            fix_z_bnd=fix_z_bnd,
        )

    def __call__(
        self,
        output_dict: Dict[str, "paddle.Tensor"],
        label_dict: Dict[str, "paddle.Tensor"],
        weight_dict: Dict[str, "paddle.Tensor"] = None,
        h=None,
    ):
        x = output_dict["y"]
        y = label_dict["y"]

        return {"y": self.rel(x, y, h=h)}
