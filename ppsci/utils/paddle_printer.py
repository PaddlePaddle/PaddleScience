from sympy import lambdify, Symbol, Derivative, Function, Basic, Add, Max, Min
from typing import List, Dict


from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
import paddle
import copy

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
            return lambda **x: paddle.zeros_like(next(iter(x.items()))[1]) + constant

        lambdify_f = loop_lambda(f)
    else:
        vars = [k for k in r] if separable else [[k for k in r]]
        try:  # NOTE this fixes a very odd bug in SymPy TODO add issue to SymPy
            lambdify_f = lambdify(vars, f, [PADDLE_SYMPY_PRINTER])
        except:
            lambdify_f = lambdify(vars, f, [PADDLE_SYMPY_PRINTER])
    return lambdify_f

def _derivative_to_str(deriv):
    n = int(deriv.args[1][1])
    deriv_simple = deriv.args[0].name
    for i in range(n):
        deriv_simple += "__"+ str(deriv.args[1][0])
    return deriv_simple


def _subs_derivatives(expr_old):
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
    return paddle.clip(tensor, max = scalar)


def _heaviside_paddle(x, y):
    # return paddle.heaviside(x ,paddle.zeros_like(x))
    return paddle.heaviside(x ,paddle.full_like(x, y))

                

PADDLE_SYMPY_PRINTER = {
    "abs": paddle.abs,
    "Min": _min_paddle,
    "Heaviside": _heaviside_paddle,}


class SympyToPaddle(paddle.nn.Layer):
    def __init__(
        self,
        sympy_expr_old,
        name: str,
        freeze_terms: List[int] = [],
        detach_names: List[str] = [],
    ):
        super().__init__()
        if name == 'momentum_y':
            print("sympy-torch momentum y")
        sympy_expr = _subs_derivatives(sympy_expr_old)
        # Sort keys to guarantee ordering
        self.keys = sorted([k.name for k in sympy_expr.free_symbols])
        self.freeze_terms = freeze_terms
        if not self.freeze_terms:
            self.paddle_expr = paddle_lambdify(sympy_expr, self.keys)
        else:
            assert all(
                x < len(Add.make_args(sympy_expr)) for x in freeze_terms
            ), "The freeze term index cannot be larger than the total terms in the expression"
            self.paddle_expr = []
            for i in range(len(Add.make_args(sympy_expr))):
                self.paddle_expr.append(
                    paddle_lambdify(Add.make_args(sympy_expr)[i], self.keys)
                )
            self.freeze_list = list(self.paddle_expr[i] for i in freeze_terms)
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
                "v__x__x" : hessian(v, x),
                "u__y__y" : hessian(u, y),
                "v__y__y" : hessian(v, y),
        }
        args = [
            out[k] if k in out else sympy_to_paddle[k] for k in self.keys
        ]
        if not self.freeze_terms:
            output = self.paddle_expr(args)
        else:
            output = paddle.zeros_like(out[self.keys[0]])
            for _, expr in enumerate(self.paddle_expr):
                if expr in self.freeze_list:
                    pass
                    # output += expr(args).detach() #FIXME why not just output+=expr(args)
                else:
                    output += expr(args)

        return output