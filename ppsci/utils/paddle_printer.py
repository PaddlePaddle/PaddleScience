# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import copy
import sympy
from sympy import lambdify, Symbol, Derivative, Function
from typing import List, Dict
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from typing import List, Dict, Union

def paddle_lambdify(
    f: Union[sympy.core.basic.Basic, float, int, bool], 
    r: List[str]):
    """Generates a Paddle function from a sympy expression

    Args:
        f (Tuple[sympy.core.basic.Basic, float, int, bool]): 
            The equation to convert to torch. If float, int, or bool,
            this gets converted to a constant function of value `f`.
        r (List[str]): List, dict A list of the arguments for `f`. 
        
    Returns:
        lambdify_f : Paddle function
    """

    try:
        f = float(f)
    except:
        pass

    if isinstance(f, (float, int, bool)):  # constant function
        def loop_lambda(constant):
            return lambda **x: paddle.zeros_like(next(iter(x.items()))[1]) + constant
        lambdify_f = loop_lambda(f)
    else:
        vars = [[k for k in r]]
        try:  # NOTE Bug in SymPy 
            lambdify_f = lambdify(vars, f, [PADDLE_SYMPY_PRINTER])
        except:
            lambdify_f = lambdify(vars, f, [PADDLE_SYMPY_PRINTER])
    return lambdify_f

def _derivative_to_str(deriv: (sympy.core.basic.Basic)) -> str:    
    """Converts a sympy expression representing a derivative to a string for display purposes.

    Args:
        deriv (sympy.core.basic.Basic): A sympy expression representing a derivative.

    Returns:
        deriv_str (str): A string representing the derivative for display purposes.
    """
    m = len(deriv.args)
    deriv_str = deriv.args[0].name
    for i in range(1, m):
        n = int(deriv.args[i][1])
        denominator =  deriv.args[i][0].name
        for i in range(n):
            deriv_str += "__"+ denominator
    return deriv_str

def _subs_derivatives(expr_old: (sympy.core.basic.Basic)) -> sympy.core.basic.Basic:
    """Replaces derivatives in an expression with function symbols.

    Args:
        expr_old: (sympy.core.basic.Basic) The expression to be processed.

    Returns:
        (sympy.core.basic.Basic): The expression with replaced derivatives.
    """
    expr = copy.deepcopy(expr_old)
    while True:
        try:
            deriv = expr.atoms(Derivative).pop()
            new_fn_name = _derivative_to_str(deriv)
            expr = expr.subs(deriv, Function(new_fn_name)(*deriv.free_symbols))
        except:
            break
    while True:
        try:
            fn = {
                fn for fn in expr.atoms(Function) if fn.class_key()[1] == 0
            }.pop()  # check if standard Sympy Eq (TODO better check)
            new_symbol_name = fn.name
            expr = expr.subs(fn, Symbol(new_symbol_name))
        except:
            break
    return expr

def _min_paddle(x, y):
    tensor = x if isinstance(x, paddle.Tensor) else y
    scalar = x if isinstance(x, (int, float)) else y
    return paddle.clip(tensor, max=scalar)

def _heaviside_paddle(x, y):
    return paddle.heaviside(x ,paddle.full_like(x, y))

PADDLE_SYMPY_PRINTER = {
    "abs": paddle.abs,
    "Min": _min_paddle,
    "Heaviside": _heaviside_paddle,
    "sqrt": paddle.sqrt,}


class SympyToPaddle(paddle.nn.Layer):
    """Initialize the object to convert sympy expression to paddle expression.

    Args:
        sympy_expr_old (sympy.core.basic.Basic): The sympy expression to be converted.
        name (str): The name of the expression.
        detach_names (List[str], optional): The list of variable names that should be detached 
                                            from the expression and computed separately. The default is an empty list.

    Examples:
        >>> import ppsci
        >>> SympyToPaddle(sympy_expr, "nu_symbol")
    """
    def __init__(
        self,
        sympy_expr_old: sympy.core.basic.Basic,
        name: str,
        detach_names: List[str] = [],
    ):
        super().__init__()
        sympy_expr = _subs_derivatives(sympy_expr_old)
        # Sort keys to guarantee ordering
        self.keys = sorted([k.name for k in sympy_expr.free_symbols])
        self.paddle_expr = paddle_lambdify(sympy_expr, self.keys)
        self.name = name
        self.detach_names = detach_names

    def forward(self, out: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        x, y = out["x"], out["y"]
        u, v, p = out["u"], out["v"], out["p"]
        sympy_to_paddle = {
                "u__x" : jacobian(u, x),
                "v__x" : jacobian(v, x),
                "p__x" : jacobian(p, x),
                "u__y" : jacobian(u, y),
                "v__y" : jacobian(v, y),
                "p__y" : jacobian(p, y),
                "u__x__x" : hessian(u, x),
                "u__y__y" : hessian(u, y),
                "v__x__x" : hessian(v, x),
                "v__y__y" : hessian(v, y),
        } #TODO external developers task
        sympy_to_paddle.update({
                "u__x__y" : jacobian(sympy_to_paddle["u__x"], y),
                "u__y__x" : jacobian(sympy_to_paddle["u__y"], x),
                "v__x__y" : jacobian(sympy_to_paddle["v__x"], y),
                "v__y__x" : jacobian(sympy_to_paddle["v__y"], x),
        }) #TODO external developers task
        args = [
            out[k] if k in out else sympy_to_paddle[k] for k in self.keys
        ]
        output = self.paddle_expr(args)
        return output
