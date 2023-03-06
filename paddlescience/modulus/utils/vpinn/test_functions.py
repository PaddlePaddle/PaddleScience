""" Helper functions for and classes for making test functions used in VPINNs
"""

import torch
import numpy as np
import sympy as sp
from sympy import I
import random, itertools
from modulus.utils.sympy.torch_printer import torch_lambdify

x, y, z = sp.symbols("x, y ,z", real=True)


class meta_test_function:
    def __init__(self, name, interval1d, sympy_fcn, is_real=True):
        self.name = name
        self.interval1d = interval1d
        self.sympy_fcn = sympy_fcn
        self.is_real = is_real


def my_trig(n, x):
    return sp.exp(I * sp.pi * (n + 1) * x)


Legendre_test = meta_test_function("Legendre", [-1, 1], sp.legendre)
Chebyshev_T_test = meta_test_function("Chebyshev_T", [-1, 1], sp.chebyshevt)
Chebyshev_U_test = meta_test_function("Chebyshev_U", [-1, 1], sp.chebyshevu)
Trig_test = meta_test_function("Trig", [-1, 1], my_trig, False)


class Degree_nk:
    def __init__(self, dim):
        self.dim = dim
        self.L = 0
        self.last_degrees = [None, None]

    def __iter__(self):
        return self

    def __next__(self):
        dim = self.dim

        if self.L == 0:
            degrees = np.array([np.zeros(dim, dtype=int)])
        else:

            degrees = []

            mask0 = np.ones(len(self.last_degrees[0]), dtype=bool)
            if self.L > 1:
                mask1 = np.ones(len(self.last_degrees[1]), dtype=bool)

            for i in range(dim):
                deg = self.last_degrees[0][mask0]
                deg[:, i] += 1
                degrees.append(deg)
                mask0 &= self.last_degrees[0][:, i] == 0
                if self.L > 1:
                    mask1 &= self.last_degrees[1][:, i] == 0

            degrees = np.concatenate(degrees)

        self.last_degrees[1] = self.last_degrees[0]
        self.last_degrees[0] = degrees
        self.L += 1

        return degrees


class Test_Function:
    def __init__(
        self,
        name_ord_dict=None,
        box=None,
        diff_list=None,
        weight_fcn=None,
        simplify=None,
    ):
        # name_ord_dict: list of name and order of test functions. E.G. {Legendre_test:[1,2,3], sin_test:[1,5]}
        # 0 order Legendre is recommended, as it is constant 1, which is very helpful in most problems
        # box: the lower and upper limit of the domain. It also gives the dimension of the domain and functions.
        # diff_list: the list of derivatives of test functions need to return, E.G. [[1,0,0],[0,2,0],'grad','Delta']
        if diff_list is None:
            diff_list = ["grad", "Delta"]
        if box is None:
            box = [[0, 0], [1, 1]]
        if name_ord_dict is None:
            name_ord_dict = {Legendre_test: [0, 1], Trig_test: [0, 1, 2, 3]}
        if weight_fcn is None:
            weight_fcn = 1.0
        if simplify is None:
            simplify = False
        self.name_ord_dict = name_ord_dict
        self.lb = box[0]
        self.ub = box[1]
        self.diff_list = diff_list
        self.weight_fcn = weight_fcn
        self.simplify = simplify
        if self.simplify:
            self.simplify_fcn = sp.simplify
        else:
            self.simplify_fcn = lambda x: x
        self.dim = len(self.lb)
        self.initialize()
        self.make_fcn_list()
        self.lambdify_fcn_list()

    def initialize(self):
        self.test_sympy_dict = {"v": []}
        for k in self.diff_list:
            if k == "grad":
                self.test_sympy_dict["vx"] = []
                if self.dim >= 2:
                    self.test_sympy_dict["vy"] = []
                if self.dim == 3:
                    self.test_sympy_dict["vz"] = []
            elif k == "Delta":
                self.test_sympy_dict["dv"] = []
            else:
                my_str = "v" + "x" * k[0]
                if self.dim >= 2:
                    my_str += "y" * k[1]
                if self.dim == 3:
                    my_str += "z" * k[2]
                self.test_sympy_dict[my_str] = []

    def generator(self, test_class):
        ord_list = self.name_ord_dict[test_class]
        if self.dim == 1:
            x_trans = test_class.interval1d[0] + (
                test_class.interval1d[1] - test_class.interval1d[0]
            ) / (self.ub[0] - self.lb[0]) * (x - self.lb[0])
            for k in ord_list:
                if test_class.is_real:
                    yield self.simplify_fcn(
                        self.weight_fcn * test_class.sympy_fcn(k, x_trans)
                    )
                else:
                    for f in test_class.sympy_fcn(k, x_trans).as_real_imag():
                        yield self.simplify_fcn(self.weight_fcn * f)
        elif self.dim == 2:
            x_trans = test_class.interval1d[0] + (
                test_class.interval1d[1] - test_class.interval1d[0]
            ) / (self.ub[0] - self.lb[0]) * (x - self.lb[0])
            y_trans = test_class.interval1d[0] + (
                test_class.interval1d[1] - test_class.interval1d[0]
            ) / (self.ub[1] - self.lb[1]) * (y - self.lb[1])
            ev = itertools.islice(Degree_nk(self.dim), ord_list[0], ord_list[-1] + 1)
            for _ in ord_list:
                ord = next(ev)
                for k in ord:
                    if test_class.is_real:
                        yield self.simplify_fcn(
                            self.weight_fcn
                            * test_class.sympy_fcn(k[0], x_trans)
                            * test_class.sympy_fcn(k[1], y_trans)
                        )
                    else:
                        for fx in test_class.sympy_fcn(k[0], x_trans).as_real_imag():
                            for fy in test_class.sympy_fcn(
                                k[1], y_trans
                            ).as_real_imag():
                                yield self.simplify_fcn(self.weight_fcn * fx * fy)
        else:
            x_trans = test_class.interval1d[0] + (
                test_class.interval1d[1] - test_class.interval1d[0]
            ) / (self.ub[0] - self.lb[0]) * (x - self.lb[0])
            y_trans = test_class.interval1d[0] + (
                test_class.interval1d[1] - test_class.interval1d[0]
            ) / (self.ub[1] - self.lb[1]) * (y - self.lb[1])
            z_trans = test_class.interval1d[0] + (
                test_class.interval1d[1] - test_class.interval1d[0]
            ) / (self.ub[2] - self.lb[2]) * (z - self.lb[2])
            ev = itertools.islice(Degree_nk(self.dim), ord_list[0], ord_list[-1] + 1)
            for _ in ord_list:
                ord = next(ev)
                for k in ord:
                    if test_class.is_real:
                        yield self.simplify_fcn(
                            self.weight_fcn
                            * test_class.sympy_fcn(k[0], x_trans)
                            * test_class.sympy_fcn(k[1], y_trans)
                            * test_class.sympy_fcn(k[2], z_trans)
                        )
                    else:
                        for fx in test_class.sympy_fcn(k[0], x_trans).as_real_imag():
                            for fy in test_class.sympy_fcn(
                                k[1], y_trans
                            ).as_real_imag():
                                for fz in test_class.sympy_fcn(
                                    k[2], z_trans
                                ).as_real_imag():
                                    yield self.simplify_fcn(
                                        self.weight_fcn * fx * fy * fz
                                    )
        return

    def make_fcn_list(self):
        if self.dim == 1:
            for name in self.name_ord_dict.keys():
                for fcn in self.generator(name):
                    self.test_sympy_dict["v"].append(fcn)
                    for k in self.diff_list:
                        if k == "grad":
                            self.test_sympy_dict["vx"].append(
                                self.simplify_fcn(sp.diff(fcn, x))
                            )
                        elif k == "Delta":
                            self.test_sympy_dict["dv"].append(
                                self.simplify_fcn(sp.diff(fcn, x, 2))
                            )
                        else:
                            self.test_sympy_dict["v" + "x" * k[0]].append(
                                self.simplify_fcn(sp.diff(fcn, x, k[0]))
                            )
        elif self.dim == 2:
            for name in self.name_ord_dict.keys():
                for fcn in self.generator(name):
                    self.test_sympy_dict["v"].append(fcn)
                    for k in self.diff_list:
                        if k == "grad":
                            self.test_sympy_dict["vx"].append(
                                self.simplify_fcn(sp.diff(fcn, x))
                            )
                            self.test_sympy_dict["vy"].append(
                                self.simplify_fcn(sp.diff(fcn, y))
                            )
                        elif k == "Delta":
                            self.test_sympy_dict["dv"].append(
                                self.simplify_fcn(
                                    sp.diff(fcn, x, 2) + sp.diff(fcn, y, 2)
                                )
                            )
                        else:
                            self.test_sympy_dict["v" + "x" * k[0] + "y" * k[1]].append(
                                self.simplify_fcn(sp.diff(fcn, x, k[0], y, k[1]))
                            )
        elif self.dim == 3:
            for name in self.name_ord_dict.keys():
                for fcn in self.generator(name):
                    self.test_sympy_dict["v"].append(fcn)
                    for k in self.diff_list:
                        if k == "grad":
                            self.test_sympy_dict["vx"].append(
                                self.simplify_fcn(sp.diff(fcn, x))
                            )
                            self.test_sympy_dict["vy"].append(
                                self.simplify_fcn(sp.diff(fcn, y))
                            )
                            self.test_sympy_dict["vz"].append(
                                self.simplify_fcn(sp.diff(fcn, z))
                            )
                        elif k == "Delta":
                            self.test_sympy_dict["dv"].append(
                                self.simplify_fcn(
                                    sp.diff(fcn, x, 2)
                                    + sp.diff(fcn, y, 2)
                                    + sp.diff(fcn, z, 2)
                                )
                            )
                        else:
                            self.test_sympy_dict[
                                "v" + "x" * k[0] + "y" * k[1] + "z" * k[2]
                            ].append(
                                self.simplify_fcn(
                                    sp.diff(fcn, x, k[0], y, k[1], z, k[2])
                                )
                            )
        self.num_fcn = len(self.test_sympy_dict["v"])

    @staticmethod
    def lambdify(f_sympy, var_list):
        dim = len(var_list)
        if f_sympy.is_number:
            if dim == 1:
                return lambda x0, f_sympy=f_sympy: torch.zeros_like(x0) + float(f_sympy)
            elif dim == 2:
                return lambda x0, y0, f_sympy=f_sympy: torch.zeros_like(x0) + float(
                    f_sympy
                )
            elif dim == 3:
                return lambda x0, y0, z0, f_sympy=f_sympy: torch.zeros_like(x0) + float(
                    f_sympy
                )
        else:
            return torch_lambdify(f_sympy, var_list, separable=True)

    def lambdify_fcn_list(self):
        self.test_lambda_dict = {}
        if self.dim == 1:
            var_list = [x]
        elif self.dim == 2:
            var_list = [x, y]
        elif self.dim == 3:
            var_list = [x, y, z]

        for k in self.test_sympy_dict.keys():
            self.test_lambda_dict[k] = []
            for f_sympy in self.test_sympy_dict[k]:
                self.test_lambda_dict[k].append(
                    Test_Function.lambdify(f_sympy, var_list)
                )  ### use torch_lambdify

    def eval_test(self, ind, x, y=None, z=None):
        # return N*M tensor
        # N is the number of points
        # M is the number of test functions
        tmp_list = []
        for f in self.test_lambda_dict[ind]:
            if self.dim == 1:
                tmp_list.append(f(x))
            elif self.dim == 2:
                assert y is not None, "please provide tensor y"
                tmp_list.append(f(x, y))
            elif self.dim == 3:
                assert (y is not None) and (
                    z is not None
                ), "please provide tensor y and z"
                tmp_list.append(f(x, y, z))

        return torch.cat(tmp_list, 1)  ### tf.concat -> torch.cat


class Vector_Test:
    def __init__(self, v1, v2, v3=None, mix=None):
        # 0<mix<1 is the percentage of how many test functions to generate.
        # mix>=1 is the number of test functions to generate.
        # self.dim: dimension of functions
        # self.num: number of total functions at hand
        # self.num_output: number of output functions
        self.test_lambda_dict = {}
        self.dim = v1.dim
        self.v1 = v1
        self.v2 = v2
        if v3 is None:
            self.num = 2
            self.num_fcn = self.v1.num_fcn * self.v2.num_fcn
        else:
            self.num = 3
            self.v3 = v3
            self.num_fcn = self.v1.num_fcn * self.v2.num_fcn * self.v3.num_fcn
        self.mix = mix
        self.sample_vector_test()

    def sample_vector_test(self):
        mix = self.mix
        if (mix is None) or (mix == "all") or (mix == 1):
            self.mix = "all"
            self.num_output = self.num_fcn
            if self.num == 2:
                self.output_ind = [
                    k
                    for k in itertools.product(
                        range(self.v1.num_fcn), range(self.v2.num_fcn)
                    )
                ]
            else:
                self.output_ind = [
                    k
                    for k in itertools.product(
                        range(self.v1.num_fcn),
                        range(self.v2.num_fcn),
                        range(self.v3.num_fcn),
                    )
                ]
        elif 0 < mix < 1:
            self.mix = mix
            self.num_output = int(self.mix * self.num_fcn) if self.mix > 0 else 1
            if self.num == 2:
                self.output_ind = random.sample(
                    set(
                        itertools.product(
                            range(self.v1.num_fcn), range(self.v2.num_fcn)
                        )
                    ),
                    self.num_output,
                )
            else:
                self.output_ind = random.sample(
                    set(
                        itertools.product(
                            range(self.v1.num_fcn),
                            range(self.v2.num_fcn),
                            range(self.v3.num_fcn),
                        )
                    ),
                    self.num_output,
                )
        elif mix >= 1:
            self.mix = int(mix)
            self.num_output = self.mix
            if self.num == 2:
                self.output_ind = random.sample(
                    set(
                        itertools.product(
                            range(self.v1.num_fcn), range(self.v2.num_fcn)
                        )
                    ),
                    self.num_output,
                )
            else:
                self.output_ind = random.sample(
                    set(
                        itertools.product(
                            range(self.v1.num_fcn),
                            range(self.v2.num_fcn),
                            range(self.v3.num_fcn),
                        )
                    ),
                    self.num_output,
                )

    def eval_test(self, ind, x, y=None, z=None):
        # return a list of N*M tensor
        # N is the number of points
        # M is the number of test functions
        # Usage:
        # v = Vector_Test(v1, v2)
        # v_x, v_y = v.eval_test('v', x_tensor, y_tensor)
        if self.dim == 1:
            var_list = [x]
        elif self.dim == 2:
            var_list = [x, y]
        else:
            var_list = [x, y, z]
        v1_val = self.v1.eval_test(ind, *var_list)
        v2_val = self.v2.eval_test(ind, *var_list)

        if self.num == 2:
            # Cannot use cuda graphs because of this
            x_ind = torch.tensor([k[0] for k in self.output_ind], device=x.device)
            y_ind = torch.tensor([k[1] for k in self.output_ind], device=x.device)
            return v1_val.index_select(1, x_ind), v2_val.index_select(1, y_ind)

        else:
            # Cannot use cuda graphs because of this
            v3_val = self.v3.eval_test(ind, *var_list)
            x_ind = torch.tensor([k[0] for k in self.output_ind], device=x.device)
            y_ind = torch.tensor([k[1] for k in self.output_ind], device=x.device)
            z_ind = torch.tensor([k[2] for k in self.output_ind], device=x.device)
            return (
                v1_val.index_select(1, x_ind),
                v2_val.index_select(1, y_ind),
                v3_val.index_select(1, z_ind),
            )


class RBF_Function:
    def __init__(
        self, dim=2, RBF_name=None, diff_list=None, weight_fcn=None, simplify=None
    ):
        # center is N*d array, d is dimension.
        # eps is 1D array with length N.
        if RBF_name is None:
            self.RBF_name = "Gaussian"
        else:
            self.RBF_name = RBF_name
        if diff_list is None:
            diff_list = ["grad", "Delta"]
        if weight_fcn is None:
            weight_fcn = 1.0
        if simplify is None:
            simplify = False
        self.simplify = simplify
        if self.simplify:
            self.simplify_fcn = sp.simplify
        else:
            self.simplify_fcn = lambda x: x
        self.dim = dim
        self.diff_list = diff_list
        self.weight_fcn = weight_fcn
        if self.dim == 1:
            self.r_sympy = sp.Abs(x)
        elif self.dim == 2:
            self.r_sympy = sp.sqrt(x ** 2 + y ** 2)
        else:
            self.r_sympy = sp.sqrt(x ** 2 + y ** 2 + z ** 2)
        if self.RBF_name == "Inverse quadratic":
            self.RBF_prototype = 1 / (1 + self.r_sympy ** 2)
        elif self.RBF_name == "Inverse multiquadric":
            self.RBF_prototype = 1 / sp.sqrt(1 + self.r_sympy ** 2)
        else:
            self.RBF_prototype = sp.exp(-self.r_sympy ** 2)
        self.initialize()
        self.make_fcn_list()
        self.lambdify_fcn_list()

    def initialize(self):
        self.test_sympy_dict = {"v": []}
        self.pow_dict = {"v": 0}
        for k in self.diff_list:
            if k == "grad":
                self.test_sympy_dict["vx"] = []
                self.pow_dict["vx"] = 1
                if self.dim >= 2:
                    self.test_sympy_dict["vy"] = []
                    self.pow_dict["vy"] = 1
                if self.dim == 3:
                    self.test_sympy_dict["vz"] = []
                    self.pow_dict["vz"] = 1
            elif k == "Delta":
                self.test_sympy_dict["dv"] = []
                self.pow_dict["dv"] = 2
            else:
                my_str = "v" + "x" * k[0]
                if self.dim >= 2:
                    my_str += "y" * k[1]
                if self.dim == 3:
                    my_str += "z" * k[2]
                self.test_sympy_dict[my_str] = []
                self.pow_dict[my_str] = sum(k)

    def make_fcn_list(self):
        self.test_sympy_dict["v"] = self.RBF_prototype
        if self.dim == 1:
            for k in self.diff_list:
                if k == "grad":
                    self.test_sympy_dict["vx"] = self.simplify_fcn(
                        sp.diff(self.RBF_prototype, x)
                    )
                elif k == "Delta":
                    self.test_sympy_dict["dv"] = self.simplify_fcn(
                        sp.diff(self.RBF_prototype, x, 2)
                    )
                else:
                    self.test_sympy_dict["v" + "x" * k[0]] = self.simplify_fcn(
                        sp.diff(self.RBF_prototype, x, k[0])
                    )
        elif self.dim == 2:
            for k in self.diff_list:
                if k == "grad":
                    self.test_sympy_dict["vx"] = self.simplify_fcn(
                        sp.diff(self.RBF_prototype, x)
                    )
                    self.test_sympy_dict["vy"] = self.simplify_fcn(
                        sp.diff(self.RBF_prototype, y)
                    )
                elif k == "Delta":
                    self.test_sympy_dict["dv"] = self.simplify_fcn(
                        sp.diff(self.RBF_prototype, x, 2)
                        + sp.diff(self.RBF_prototype, y, 2)
                    )
                else:
                    self.test_sympy_dict[
                        "v" + "x" * k[0] + "y" * k[1]
                    ] = self.simplify_fcn(sp.diff(self.RBF_prototype, x, k[0], y, k[1]))
        else:
            for k in self.diff_list:
                if k == "grad":
                    self.test_sympy_dict["vx"] = self.simplify_fcn(
                        sp.diff(self.RBF_prototype, x)
                    )
                    self.test_sympy_dict["vy"] = self.simplify_fcn(
                        sp.diff(self.RBF_prototype, y)
                    )
                    self.test_sympy_dict["vz"] = self.simplify_fcn(
                        sp.diff(self.RBF_prototype, z)
                    )
                elif k == "Delta":
                    self.test_sympy_dict["dv"] = self.simplify_fcn(
                        sp.diff(self.RBF_prototype, x, 2)
                        + sp.diff(self.RBF_prototype, y, 2)
                        + sp.diff(self.RBF_prototype, z, 2)
                    )
                else:
                    self.test_sympy_dict[
                        "v" + "x" * k[0] + "y" * k[1] + "z" * k[2]
                    ] = self.simplify_fcn(
                        sp.diff(self.RBF_prototype, x, k[0], y, k[1], z, k[2])
                    )

    def lambdify_fcn_list(self):
        self.test_lambda_dict = {}
        if self.dim == 1:
            var_list = x
        elif self.dim == 2:
            var_list = [x, y]
        elif self.dim == 3:
            var_list = [x, y, z]

        for k in self.test_sympy_dict.keys():
            f_sympy = self.test_sympy_dict[k]
            self.test_lambda_dict[k] = torch_lambdify(f_sympy, var_list, separable=True)

    def eval_test(
        self,
        ind,
        x,
        y=None,
        z=None,
        x_center=None,
        y_center=None,
        z_center=None,
        eps=None,
    ):
        # return N*M tensor
        # N is the number of points
        # M is the number of test functions
        # eps is a real number or tensor
        # all input tensors are column vectors
        assert x_center is not None, "please provide x_center"
        if eps is None:
            eps = torch.full(
                [1, x_center.shape[0]], 10.0, device=x.device
            )  ### tf.fill -> torch.full
        elif isinstance(eps, int) or isinstance(eps, float):
            eps = torch.full([1, x_center.shape[0]], np.float32(eps), device=x.device)
        elif isinstance(eps, torch.Tensor):
            eps = torch.reshape(eps, [1, -1])
        x_center_t = torch.transpose(
            x_center, 0, 1
        )  ### tf.transpose -> torch.transpose
        if self.dim == 1:
            x_new = eps * (x - x_center_t)
        elif self.dim == 2:
            y_center_t = torch.transpose(
                y_center, 0, 1
            )  ### tf.transpose -> torch.transpose
            x_new = eps * (x - x_center_t)
            y_new = eps * (y - y_center_t)
        else:
            y_center_t = torch.transpose(
                y_center, 0, 1
            )  ### tf.transpose -> torch.transpose
            z_center_t = torch.transpose(
                z_center, 0, 1
            )  ### tf.transpose -> torch.transpose
            x_new = eps * (x - x_center_t)
            y_new = eps * (y - y_center_t)
            z_new = eps * (z - z_center_t)

        fcn = self.test_lambda_dict[ind]
        p = self.pow_dict[ind]
        if self.dim == 1:
            return fcn(x_new) * torch.pow(eps, p)  ### tf.pow -> torch.pow
        elif self.dim == 2:
            return fcn(x_new, y_new) * torch.pow(eps, p)  ### tf.pow -> torch.pow
        else:
            return fcn(x_new, y_new, z_new) * torch.pow(eps, p)  ### tf.pow -> torch.pow
