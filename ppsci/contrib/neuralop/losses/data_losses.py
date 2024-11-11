"""
losses.py contains code to compute standard data objective
functions for training Neural Operators.

By default, losses expect arguments y_pred (model predictions) and y (ground y.)
"""

import math
from typing import List

import paddle


# Set fix{x,y,z}_bnd if function is non-periodic in {x,y,z} direction
# x: (*, s)
# y: (*, s)
def central_diff_1d(x, h, fix_x_bnd=False):
    dx = (paddle.roll(x, -1, axis=-1) - paddle.roll(x, 1, axis=-1))/(2.0*h)

    if fix_x_bnd:
        dx[..., 0] = (x[..., 1] - x[..., 0])/h
        dx[..., -1] = (x[..., -1] - x[..., -2])/h

    return dx


# x: (*, s1, s2)
# y: (*, s1, s2)
def central_diff_2d(x, h, fix_x_bnd=False, fix_y_bnd=False):
    if isinstance(h, float):
        h = [h, h]

    dx = (paddle.roll(x, -1, axis=-2) - paddle.roll(x, 1, axis=-2))/(2.0*h[0])
    dy = (paddle.roll(x, -1, axis=-1) - paddle.roll(x, 1, axis=-1))/(2.0*h[1])

    if fix_x_bnd:
        dx[..., 0, :] = (x[..., 1, :] - x[..., 0, :])/h[0]
        dx[..., -1, :] = (x[..., -1, :] - x[..., -2, :])/h[0]

    if fix_y_bnd:
        dy[..., :, 0] = (x[..., :, 1] - x[..., :, 0])/h[1]
        dy[..., :, -1] = (x[..., :, -1] - x[..., :, -2])/h[1]

    return dx, dy


# x: (*, s1, s2, s3)
# y: (*, s1, s2, s3)
def central_diff_3d(x, h, fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
    if isinstance(h, float):
        h = [h, h, h]

    dx = (paddle.roll(x, -1, axis=-3) - paddle.roll(x, 1, axis=-3))/(2.0*h[0])
    dy = (paddle.roll(x, -1, axis=-2) - paddle.roll(x, 1, axis=-2))/(2.0*h[1])
    dz = (paddle.roll(x, -1, axis=-1) - paddle.roll(x, 1, axis=-1))/(2.0*h[2])

    if fix_x_bnd:
        dx[..., 0, :, :] = (x[..., 1, :, :] - x[..., 0, :, :])/h[0]
        dx[..., -1, :, :] = (x[..., -1, :, :] - x[..., -2, :, :])/h[0]

    if fix_y_bnd:
        dy[..., :, 0, :] = (x[..., :, 1, :] - x[..., :, 0, :])/h[1]
        dy[..., :, -1, :] = (x[..., :, -1, :] - x[..., :, -2, :])/h[1]

    if fix_z_bnd:
        dz[..., :, :, 0] = (x[..., :, :, 1] - x[..., :, :, 0])/h[2]
        dz[..., :, :, -1] = (x[..., :, :, -1] - x[..., :, :, -2])/h[2]

    return dx, dy, dz


# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=1, p=2, L=2*math.pi, reduce_dims=0, reductions='sum'):
        super().__init__()

        self.d = d
        self.p = p

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean'
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean'
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L]*self.d
        else:
            self.L = L

    def uniform_h(self, x):
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)

        return h

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
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
                h = [h]*self.d

        const = math.prod(h)**(1.0/self.p)
        diff = const*paddle.norm(paddle.flatten(x, start_axis=-self.d) - paddle.flatten(y, start_axis=-self.d), p=self.p, axis=-1, keepdim=False)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff

    def rel(self, x, y):

        diff = paddle.norm(paddle.flatten(x, start_axis=-self.d) - paddle.flatten(y, start_axis=-self.d), p=self.p, axis=-1, keepdim=False)
        ynorm = paddle.norm(paddle.flatten(y, start_axis=-self.d), p=self.p, axis=-1, keepdim=False)

        diff = diff/ynorm

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff

    def __call__(self, y_pred, y, **kwargs):
        return self.rel(y_pred, y)


class H1Loss(object):
    def __init__(self, d=1, L=2*math.pi, reduce_dims=0, reductions='sum', fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
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
                assert reductions == 'sum' or reductions == 'mean'
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean'
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L]*self.d
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

            x_x, x_y = central_diff_2d(x, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)
            y_x, y_y = central_diff_2d(y, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)

            dict_x[1] = paddle.flatten(x_x, start_axis=-2)
            dict_x[2] = paddle.flatten(x_y, start_axis=-2)

            dict_y[1] = paddle.flatten(y_x, start_axis=-2)
            dict_y[2] = paddle.flatten(y_y, start_axis=-2)

        else:
            dict_x[0] = paddle.flatten(x, start_axis=-3)
            dict_y[0] = paddle.flatten(y, start_axis=-3)

            x_x, x_y, x_z = central_diff_3d(x, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)
            y_x, y_y, y_z = central_diff_3d(y, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)

            dict_x[1] = paddle.flatten(x_x, start_axis=-3)
            dict_x[2] = paddle.flatten(x_y, start_axis=-3)
            dict_x[3] = paddle.flatten(x_z, start_axis=-3)

            dict_y[1] = paddle.flatten(y_x, start_axis=-3)
            dict_y[2] = paddle.flatten(y_y, start_axis=-3)
            dict_y[3] = paddle.flatten(y_z, start_axis=-3)

        return dict_x, dict_y

    def uniform_h(self, x):
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.shape[-j]

        return h

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
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
                h = [h]*self.d

        dict_x, dict_y = self.compute_terms(x, y, h)

        const = math.prod(h)
        diff = const*paddle.norm(dict_x[0] - dict_y[0], p=2, axis=-1, keepdim=False)**2

        for j in range(1, self.d + 1):
            diff += const*paddle.norm(dict_x[j] - dict_y[j], p=2, axis=-1, keepdim=False)**2

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
                h = [h]*self.d

        dict_x, dict_y = self.compute_terms(x, y, h)

        diff = paddle.norm(dict_x[0] - dict_y[0], p=2, axis=-1, keepdim=False)**2
        ynorm = paddle.norm(dict_y[0], p=2, axis=-1, keepdim=False)**2

        for j in range(1, self.d + 1):
            diff += paddle.norm(dict_x[j] - dict_y[j], p=2, axis=-1, keepdim=False)**2
            ynorm += paddle.norm(dict_y[j], p=2, axis=-1, keepdim=False)**2

        diff = (diff**0.5)/(ynorm**0.5)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff

    def __call__(self, y_pred, y, h=None, **kwargs):
        return self.rel(y_pred, y, h=h)


class IregularLpqLoss(paddle.nn.Layer):
    def __init__(self, p=2.0, q=2.0):
        super().__init__()

        self.p = 2.0
        self.q = 2.0

    # x, y are (n, c) or (n,)
    # vol_elm is (n,)

    def norm(self, x, vol_elm):
        if len(x.shape) > 1:
            s = paddle.sum(paddle.abs(x)**self.q, axis=1, keepdim=False)**(self.p/self.q)
        else:
            s = paddle.abs(x)**self.p

        return paddle.sum(s*vol_elm)**(1.0/self.p)

    def abs(self, x, y, vol_elm):
        return self.norm(x - y, vol_elm)

    # y is assumed y
    def rel(self, x, y, vol_elm):
        return self.abs(x, y, vol_elm)/self.norm(y, vol_elm)

    def forward(self, y_pred, y, vol_elm, **kwargs):
        return self.rel(y_pred, y, vol_elm)


def pressure_drag(pressure, vol_elm, inward_surface_normal,
                  flow_direction_normal, flow_speed,
                  reference_area, mass_density=1.0):

    const = 2.0/(mass_density*(flow_speed**2)*reference_area)
    direction = paddle.sum(inward_surface_normal*flow_direction_normal, axis=1, keepdim=False)

    return const*paddle.sum(pressure*direction*vol_elm)


def friction_drag(wall_shear_stress, vol_elm, 
                  flow_direction_normal, flow_speed, 
                  reference_area, mass_density=1.0):

    const = 2.0/(mass_density*(flow_speed**2)*reference_area)
    direction = paddle.sum(wall_shear_stress*flow_direction_normal, axis=1, keepdim=False)

    x = paddle.sum(direction*vol_elm)

    return const*paddle.sum(direction*vol_elm)


def total_drag(pressure, wall_shear_stress, vol_elm,
               inward_surface_normal, flow_direction_normal,
               flow_speed, reference_area, mass_density=1.0):

    cp = pressure_drag(pressure, vol_elm, inward_surface_normal,
                       flow_direction_normal, flow_speed,
                       reference_area, mass_density)

    cf = friction_drag(wall_shear_stress, vol_elm,
                       flow_direction_normal, flow_speed,
                       reference_area, mass_density)

    return cp + cf


class WeightedL2DragLoss(object):

    def __init__(self, mappings: dict, device: str = 'cuda'):
        """WeightedL2DragPlusLPQLoss calculates the l2 drag loss
            over the shear stress and pressure outputs of a model.

        Parameters
        ----------
        mappings: dict[tuple(Slice)]
            indices of an input tensor corresponding to above fields
        device : str, optional
            device on which to do tensor calculations, by default 'cuda'
        """
        # take in a dictionary of drag functions to be calculated on model output over output fields
        super().__init__()
        self.mappings = mappings
        self.device = device

    def __call__(self, y_pred, y, vol_elm, inward_normals, flow_normals, flow_speed, reference_area, **kwargs):
        c_pred = None
        c_truth = None
        loss = 0.

        stress_indices = self.mappings['wall_shear_stress']
        pred_stress = y_pred[stress_indices].view(-1, 1)
        truth_stress = y[stress_indices]

        # friction drag takes padded input
        pred_stress_pad = paddle.zeros((pred_stress.shape[0], 3))
        pred_stress_pad[:, 0] = pred_stress.view(-1,)

        truth_stress_pad = paddle.zeros((truth_stress.shape[0], 3))
        truth_stress_pad[:, 0] = truth_stress.view(-1,)

        pressure_indices = self.mappings['pressure']
        pred_pressure = y_pred[pressure_indices].view(-1, 1)
        truth_pressure = y[pressure_indices]

        c_pred = total_drag(pressure=pred_pressure,
                            wall_shear_stress=pred_stress_pad,
                            vol_elm=vol_elm,
                            inward_surface_normal=inward_normals,
                            flow_direction_normal=flow_normals,
                            flow_speed=flow_speed,
                            reference_area=reference_area
                            )
        c_truth = total_drag(
            pressure=truth_pressure,
            wall_shear_stress=truth_stress_pad,
            vol_elm=vol_elm,
            inward_surface_normal=inward_normals,
            flow_direction_normal=flow_normals,
            flow_speed=flow_speed,
            reference_area=reference_area
        )

        loss += paddle.abs(c_pred - c_truth) / paddle.abs(c_truth)

        loss = (1.0/len(self.mappings) + 1)*loss

        return loss
