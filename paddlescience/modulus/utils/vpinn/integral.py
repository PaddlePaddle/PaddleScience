""" Helper functions and classes for integration
"""

import torch
import quadpy as qd
import numpy as np


def tensor_int(w, v, u=None):
    # u is a N*1 tensor
    # v is a N*M tensor
    # w is a N*1 tensor, quadrature (cubature) weights
    # N is the number of points
    # M is the number of (test) functions
    # return: 1*M tensor: integrals of u*v[i] if u is not None
    # return: 1*M tensor: integrals of v[i] if u is None
    if u is None:
        return torch.einsum("ik,ij->jk", v, w)
    else:
        return torch.einsum("ij,ik,ij->jk", u, v, w)


# class template for the quadrature
class Quadrature:
    def __init__(self, scheme, trans, jac):
        # scheme is the quadpy scheme
        # trans is the transform from reference domain to target domain
        # jac is the jacobian of trans, SHOULD BE 1D NUMPY ARRAY!
        # points_ref and weights_ref are on the reference domain
        self.scheme = scheme
        self.trans = trans
        self.jac = jac

        self.points_ref = scheme.points
        self.weights_ref = scheme.weights
        self.make_numpy()
        # self.make_tensor()
        self.N_points = self.points_numpy.shape[0]

    def make_numpy(self):
        # points_numpy and weights_numpy are N*d numpy arrays, where N is the # of points and d is the dimension
        # The approximated integral value is given by np.dot(f(p[:,0],p[:,1],p[:,2]),w)
        self.points_numpy = self.trans(self.points_ref)
        self.weights_numpy = (
            self.weights_ref * self.jac
        )  # check here, should be 1D*1D numpy array, or 1D*constant

    def make_tensor(self):
        # points_tensor and weights_tensor are N*d tf tensors, where N is the # of points and d is the dimension
        self.points_tensor = torch.tensor(self.points_numpy, dtype=torch.float32)
        self.weights_tensor = torch.tensor(
            self.weights_numpy.reshape((-1, 1)), dtype=torch.float32
        )


class Quadrature_Data:
    def __init__(self, points_numpy, weights_numpy):
        self.points_numpy = points_numpy
        self.weights_numpy = weights_numpy
        # self.make_tensor()

    def make_tensor(self):
        # points_tensor and weights_tensor are N*d tf tensors, where N is the # of points and d is the dimension
        self.points_tensor = torch.tensor(self.points_numpy, dtype=torch.float32)
        self.weights_tensor = torch.tensor(
            self.weights_numpy.reshape((-1, 1)), dtype=torch.float32
        )


def Quad_Collection(quad_class, paras):
    points_tmp = []
    weights_tmp = []
    for para in paras:
        quad_tmp = quad_class(*para)
        points_tmp.append(quad_tmp.points_numpy)
        weights_tmp.append(quad_tmp.weights_numpy)
    return Quadrature_Data(np.vstack(points_tmp), np.hstack(weights_tmp))


# 1D classes. (Quad_Line can be used in nD)
class Quad_Line(Quadrature):
    def __init__(self, p0, p1, n, scheme_fcn=qd.c1.gauss_legendre):
        self.p0 = np.reshape(np.array(p0), (1, -1))
        self.p1 = np.reshape(np.array(p1), (1, -1))
        super().__init__(
            scheme=scheme_fcn(n),
            trans=lambda t: 0.5 * (self.p0 + self.p1)
            + 0.5 * (self.p1 - self.p0) * np.reshape(t, (-1, 1)),
            jac=np.linalg.norm(self.p1 - self.p0) / 2,
        )


# 2D curves
class Quad_Circle(Quadrature):
    def __init__(self, r, c, n, scheme_fcn=qd.u2.get_good_scheme):
        self.r = np.array(r)
        self.c = np.array(c)

        def my_trans(x):
            rr = np.multiply.outer(self.r, x)
            rr = np.swapaxes(rr, 0, -2)
            return rr + self.c

        super().__init__(scheme=scheme_fcn(n), trans=my_trans, jac=2 * np.pi * self.r)


# 2D domains
class Quad_Tri(Quadrature):
    def __init__(self, v, n, scheme_fcn=qd.t2.get_good_scheme):
        from quadpy.tn._helpers import get_vol

        self.v = np.array(v)  # 3x2 numpy array
        if self.v.shape != (3, 2):
            self.v = self.v.T
        assert self.v.shape == (3, 2), "Vertices must be a 3 by 2 list or numpy array!"
        self.vol = get_vol(self.v)
        super().__init__(
            scheme=scheme_fcn(n), trans=lambda x: x.T @ self.v, jac=self.vol
        )


class Quad_Disk(Quadrature):
    def __init__(self, r, c, n, scheme_fcn=qd.s2.get_good_scheme):
        self.r = np.array(r)
        self.c = np.array(c)

        def my_trans(x):
            rr = np.multiply.outer(self.r, x.T)
            rr = np.swapaxes(rr, 0, -2)
            return rr + self.c

        super().__init__(scheme=scheme_fcn(n), trans=my_trans, jac=np.pi * self.r ** 2)


class Quad_Rect(Quadrature):
    """
    The points are specified in an array of shape (2, 2, ...) such that arr[0][0] is the lower left corner, arr[1][1] the upper right, and set region_type=False.
    If your c2 has its sides aligned with the coordinate axes, you can use v=[[x0, x1], [y0, y1]], and set region_type=True (default).
    """

    def __init__(self, v, n, region_type=True, scheme_fcn=qd.c2.get_good_scheme):
        from quadpy.cn._helpers import transform, get_detJ

        if region_type:
            from quadpy.c2 import rectangle_points

            self.v = rectangle_points(*v)
        else:
            self.v = v
        super().__init__(
            scheme=scheme_fcn(n),
            trans=lambda x: transform(x, self.v),
            jac=lambda x: np.abs(get_detJ(x, self.v))
            * 2 ** np.prod(len(self.v.shape) - 1),
        )

    def make_numpy(self):
        self.points_numpy = self.trans(self.points_ref)
        self.weights_numpy = self.weights_ref * self.jac(
            self.points_ref
        )  # check here, should be 1D*1D numpy array, or 1D*constant


# 3D surfaces
class Quad_Sphere(Quadrature):
    def __init__(self, r, c, n, scheme_fcn=qd.u3.get_good_scheme):
        self.r = np.array(r)
        self.c = np.array(c)
        super().__init__(
            scheme=scheme_fcn(n),
            trans=lambda x: x.T * self.r + self.c,
            jac=4 * np.pi * self.r ** 2,
        )


# 3D domain
class Quad_Ball(Quadrature):
    def __init__(self, r, c, n, scheme_fcn=qd.s3.get_good_scheme):
        assert (
            n <= 7
        ), "The degree of the cubature is not more than 7. Otherwise use nD ball scheme!"
        self.r = np.array(r)
        self.c = np.array(c)

        def my_trans(x):
            rr = np.multiply.outer(self.r, x.T)
            rr = np.swapaxes(rr, 0, -2)
            return rr + self.c

        super().__init__(
            scheme=scheme_fcn(n), trans=my_trans, jac=4 / 3 * np.pi * self.r ** 3
        )


class Quad_Tet(Quadrature):
    def __init__(self, v, n, scheme_fcn=qd.t3.get_good_scheme):
        assert (
            n <= 14
        ), "The degree of the cubature is not more than 14. Otherwise use nD simplex scheme!"
        self.v = np.array(v)
        if self.v.shape != (4, 3):
            self.v = self.v.T
        assert self.v.shape == (4, 3), "Vertices must be a 4 by 3 list or numpy array!"
        from quadpy.tn._helpers import transform, get_vol

        self.vol = get_vol(self.v)
        super().__init__(
            scheme=scheme_fcn(n), trans=lambda x: transform(x, self.v.T).T, jac=self.vol
        )


class Quad_Cube(Quadrature):
    def __init__(self, v, n, region_type=True, scheme_fcn=qd.c3.get_good_scheme):
        from quadpy.cn._helpers import transform, get_detJ

        assert (
            n <= 11
        ), "The degree of the cubature is not more than 11. Otherwise use nD cube scheme!"
        if region_type:
            from quadpy.c3 import cube_points

            self.v = cube_points(*v)
        else:
            self.v = v
        super().__init__(
            scheme=scheme_fcn(n),
            trans=lambda x: transform(x, self.v),
            jac=lambda x: np.abs(get_detJ(x, self.v))
            * 2 ** np.prod(len(self.v.shape) - 1),
        )

    def make_numpy(self):
        self.points_numpy = self.trans(self.points_ref)
        self.weights_numpy = self.weights_ref * self.jac(
            self.points_ref
        )  # check here, should be 1D*1D numpy array, or 1D*constant


class Quad_Pyramid(Quadrature):
    def __init__(self, v, scheme_fcn=qd.p3.felippa_5):
        from quadpy.p3._helpers import _transform, _get_det_J

        self.v = v
        super().__init__(
            scheme=scheme_fcn(),
            trans=lambda x: _transform(x.T, self.v).T,
            jac=lambda x: np.abs(_get_det_J(self.v, x.T)),
        )

    def make_numpy(self):
        self.points_numpy = self.trans(self.points_ref)
        self.weights_numpy = self.weights_ref * self.jac(
            self.points_ref
        )  # check here, should be 1D*1D numpy array, or 1D*constant


class Quad_Wedge(Quadrature):
    def __init__(self, v, scheme_fcn=qd.w3.felippa_6):
        from quadpy.w3._helpers import _transform, _get_detJ

        self.v = np.array(v)
        super().__init__(
            scheme=scheme_fcn(),
            trans=lambda x: _transform(x.T, self.v).T,
            jac=lambda x: np.abs(_get_detJ(x.T, self.v)),
        )

    def make_numpy(self):
        self.points_numpy = self.trans(self.points_ref)
        self.weights_numpy = self.weights_ref * self.jac(
            self.points_ref
        )  # check here, should be 1D*1D numpy array, or 1D*constant


# nD manifold
class Quad_nD_Sphere(Quadrature):
    def __init__(self, r, c, dim, scheme_fcn=qd.un.dobrodeev_1978):
        import ndim

        self.r = np.array(r)
        self.c = np.array(c)
        self.dim = dim

        def my_trans(x):
            rr = np.multiply.outer(self.r, x)
            rr = np.swapaxes(rr, 0, -2)
            return rr + self.c

        self.vol = ndim.nsphere.volume(self.dim, r=self.r)
        super().__init__(scheme=scheme_fcn(self.dim), trans=my_trans, jac=self.vol)


class Quad_nD_Simplex(Quadrature):
    def __init__(self, v, dim, n, scheme_fcn=qd.tn.grundmann_moeller):
        from quadpy.tn._helpers import transform, get_vol

        self.v = np.array(v)
        self.dim = dim
        self.vol = get_vol(self.v)
        super().__init__(
            scheme=scheme_fcn(self.dim, n),
            trans=lambda x: transform(x, self.v.T).T,
            jac=self.vol,
        )


class Quad_nD_Ball(Quadrature):
    def __init__(self, r, c, dim, scheme_fcn=qd.sn.dobrodeev_1970):
        import ndim

        self.r = np.array(r)
        self.c = np.array(c)
        self.dim = dim
        self.vol = ndim.nball.volume(self.dim, r=self.r, symbolic=False)

        def my_trans(x):
            rr = np.multiply.outer(self.r, x.T)
            rr = np.swapaxes(rr, 0, -2)
            return rr + self.c

        super().__init__(scheme=scheme_fcn(self.dim), trans=my_trans, jac=self.vol)


class Quad_nD_Cube(Quadrature):
    def __init__(self, v, dim, region_type=True, scheme_fcn=qd.cn.stroud_cn_3_3):
        from quadpy.cn._helpers import transform, get_detJ

        self.dim = dim
        if region_type:
            from quadpy.cn._helpers import ncube_points

            self.v = ncube_points(*v)
        else:
            self.v = v
        super().__init__(
            scheme=scheme_fcn(self.dim),
            trans=lambda x: transform(x, self.v),
            jac=lambda x: 2 ** np.prod(len(self.v.shape) - 1)
            * np.abs(get_detJ(x, self.v)),
        )

    def make_numpy(self):
        self.points_numpy = self.trans(self.points_ref)
        self.weights_numpy = self.weights_ref * self.jac(
            self.points_ref
        )  # check here, should be 1D*1D numpy array, or 1D*constant


# 2D cubature based on mesh
def domain_weights_and_points_2D(P, T, n=5, scheme=None):
    # P is the point info
    # T is the triangle info
    # n is the cubature order, if applicable
    T = T.astype(np.int)
    Nt = T.shape[0]
    if scheme is None:
        scheme = qd.t2._lether.lether(n)
    p_ref = scheme.points
    w_ref = scheme.weights
    xy_tmp = []
    w_tmp = []
    for i in range(1, Nt):
        idp = T[i, :]
        tri = np.vstack((P[idp[0], :], P[idp[1], :], P[idp[2], :]))
        S = 0.5 * np.abs(np.linalg.det(np.hstack((tri, np.ones((3, 1))))))
        xy_tmp.append(p_ref.T @ tri)
        w_tmp.append(S * w_ref)
    xy = np.vstack(xy_tmp)
    w = np.hstack(w_tmp)
    return w.astype(np.float32), xy.astype(np.float32)


# 3D cubature based on mesh
def domain_weights_and_points_3D(P, T, n=5, scheme=None):
    # P is the point info
    # T is the triangle info
    # n is the cubature order, if applicable
    T = T.astype(np.int)
    Nt = T.shape[0]
    if scheme is None:
        scheme = qd.t3.get_good_scheme(n)
    p_ref = scheme.points
    w_ref = scheme.weights
    xyz_tmp = []
    w_tmp = []
    for i in range(0, Nt):
        idp = T[i, :]
        tet = np.vstack((P[idp[0], :], P[idp[1], :], P[idp[2], :], P[idp[3], :]))
        V = np.abs(np.linalg.det(np.hstack((tet, np.ones((4, 1)))))) / 6
        xyz_tmp.append(p_ref.T @ tet)
        w_tmp.append(V * w_ref)
    xyz = np.vstack(xyz_tmp)
    w = np.hstack(w_tmp)
    return w.astype(np.float32), xyz.astype(np.float32)


# Householder reflector
def Householder_reflector(u0, v0):
    # u and v are unit vectors
    # Hu=v, Hv=u
    u = u0.reshape((-1, 1)) / np.linalg.norm(u0)
    v = v0.reshape((-1, 1)) / np.linalg.norm(v0)
    return np.eye(3) + (u @ v.T + v @ u.T - u @ u.T - v @ v.T) / (1 - u.T @ v)
