"""
Helper functions for converting sympy equations to pytorch
"""

from sympy import lambdify, Symbol, Derivative, Function, Basic, Max, Min
from sympy.printing.str import StrPrinter
#import torch
import numpy as np
import functools
from typing import List, Dict

from paddlescience.modulus.constants import diff_str, paddle_dt


def paddle_lambdify(f, r, separable=False):
    """
    generates a PyTorch function from a sympy equation

    Parameters
    ----------
    f : Sympy Exp, float, int, bool
      the equation to convert to torch.
      If float, int, or bool this gets converted
      to a constant function of value `f`.
    r : list, dict
      A list of the arguments for `f`. If dict then
      the keys of the dict are used.

    Returns
    -------
    torch_f : PyTorch function
    """

    try:
        f = float(f)
    except:
        pass
    if isinstance(f, (float, int, bool)):  # constant function

        def loop_lambda(constant):
            return lambda **x: torch.zeros_like(next(iter(x.items()))[1]) + constant

        lambdify_f = loop_lambda(f)
    else:
        vars = [k for k in r] if separable else [[k for k in r]]
        try:  # NOTE this fixes a very odd bug in SymPy TODO add issue to SymPy
            lambdify_f = lambdify(vars, f, [TORCH_SYMPY_PRINTER])
        except:
            lambdify_f = lambdify(vars, f, [TORCH_SYMPY_PRINTER])
    return lambdify_f


def _where_torch(conditions, x, y):
    if isinstance(x, (int, float)):
        x = float(x) * torch.ones(*conditions.get_shape())
    if isinstance(y, (int, float)):
        y = float(y) * torch.ones(*conditions.get_shape())
    return torch.where(conditions, x, y)


def _heaviside_torch(x):
    return torch.maximum(torch.sign(x), torch.zeros(1, device=x.device))


def _sqrt_torch(x):
    return torch.sqrt((x - 1e-6) * _heaviside_torch(x - 1e-6) + 1e-6)


# TODO: Add jit version here
def _or_torch(*x):
    return_value = x[0]
    for value in x:
        return_value = torch.logical_or(return_value, value)
    return return_value


# TODO: Add jit version here
def _and_torch(*x):
    return_value = x[0]
    for value in x:
        return_value = torch.logical_and(return_value, value)
    return return_value


@torch.jit.script
def _min_jit(x: List[torch.Tensor]):
    assert len(x) > 0
    min_tensor = x[0]
    for i in range(1, len(x)):
        min_tensor = torch.minimum(min_tensor, x[i])
    return min_tensor


def _min_torch(*x):
    # get tensor shape
    for value in x:
        if not isinstance(value, (int, float)):
            tensor_shape = list(map(int, value.shape))
            device = value.device

    # convert all floats and ints to tensor
    x_only_tensors = []
    for value in x:
        if isinstance(value, (int, float)):
            value = torch.zeros(tensor_shape, device=device) + value
        x_only_tensors.append(value)

    # reduce min
    min_tensor, _ = torch.min(torch.stack(x_only_tensors, -1), -1)
    return min_tensor

    # jit option
    # return _min_jit(x_only_tensors)

    # TODO: benchmark this other option that avoids stacking and extra memory movement
    # Update: cannot jit this because TorchScript doesn't support functools.reduce
    # return functools.reduce(torch.minimum, x)


@torch.jit.script
def _max_jit(x: List[torch.Tensor]):
    assert len(x) > 0
    max_tensor = x[0]
    for i in range(1, len(x)):
        max_tensor = torch.maximum(max_tensor, x[i])
    return max_tensor


def _max_torch(*x):
    # get tensor shape
    for value in x:
        if not isinstance(value, (int, float)):
            tensor_shape = list(map(int, value.shape))
            device = value.device

    # convert all floats and ints to tensor
    x_only_tensors = []
    for value in x:
        if isinstance(value, (int, float)):
            value = (torch.zeros(tensor_shape) + value).to(device)
        x_only_tensors.append(value)

    # reduce max
    max_tensor, _ = torch.max(torch.stack(x_only_tensors, -1), -1)
    return max_tensor

    # jit option
    # return _max_jit(x_only_tensors)


def _dirac_delta_torch(x):
    return torch.eq(x, 0.0).to(paddle_dt)


TORCH_SYMPY_PRINTER = {
    "abs": torch.abs,
    "Abs": torch.abs,
    "sign": torch.sign,
    "ceiling": torch.ceil,
    "floor": torch.floor,
    "log": torch.log,
    "exp": torch.exp,
    "sqrt": _sqrt_torch,
    "cos": torch.cos,
    "acos": torch.acos,
    "sin": torch.sin,
    "asin": torch.asin,
    "tan": torch.tan,
    "atan": torch.atan,
    "atan2": torch.atan2,
    "cosh": torch.cosh,
    "acosh": torch.acosh,
    "sinh": torch.sinh,
    "asinh": torch.asinh,
    "tanh": torch.tanh,
    "atanh": torch.atanh,
    "erf": torch.erf,
    "loggamma": torch.lgamma,
    "Min": _min_torch,
    "Max": _max_torch,
    "Heaviside": _heaviside_torch,
    "DiracDelta": _dirac_delta_torch,
    "logical_or": _or_torch,
    "logical_and": _and_torch,
    "where": _where_torch,
    "pi": np.pi,
    "conjugate": torch.conj,
}


class CustomDerivativePrinter(StrPrinter):
    def _print_Function(self, expr):
        """
        Custom printing of the SymPy Derivative class.
        Instead of:
        D(x(t), t)
        We will print:
        x__t
        """
        return expr.func.__name__

    def _print_Derivative(self, expr):
        """
        Custom printing of the SymPy Derivative class.
        Instead of:
        D(x(t), t)
        We will print:
        x__t
        """
        prefix = str(expr.args[0].func)
        for expr in expr.args[1:]:
            prefix += expr[1] * (diff_str + str(expr[0]))
        return prefix


def _subs_derivatives(expr):
    while True:
        try:
            deriv = expr.atoms(Derivative).pop()
            new_fn_name = str(deriv)
            expr = expr.subs(deriv, Function(new_fn_name)(*deriv.free_symbols))
        except:
            break
    while True:
        try:
            fn = {
                fn for fn in expr.atoms(Function) if fn.class_key()[1] == 0
            }.pop()  # check if standard Sympy Eq (TODO better check)
            new_symbol_name = str(fn)
            expr = expr.subs(fn, Symbol(new_symbol_name))
        except:
            break
    return expr


# Override the __str__ method of to use CustromStrPrinter
Basic.__str__ = lambda self: CustomDerivativePrinter().doprint(self)


# Class to compile and evaluate a sympy expression in PyTorch
# Cannot currently script this module because self.torch_expr is unknown
class SympyToTorch(torch.nn.Module):
    def __init__(self, sympy_expr, name: str, detach_names: List[str] = []):
        super().__init__()
        # Sort keys to guarantee ordering
        self.keys = sorted([k.name for k in sympy_expr.free_symbols])
        self.torch_expr = torch_lambdify(sympy_expr, self.keys)
        self.name = name
        self.detach_names = detach_names

    def forward(self, var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        args = [
            var[k].detach() if k in self.detach_names else var[k] for k in self.keys
        ]
        return {self.name: self.torch_expr(args)}
