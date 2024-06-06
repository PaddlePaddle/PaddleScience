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

from __future__ import annotations

from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import paddle
import sympy as sp
from paddle import nn

DETACH_FUNC_NAME = "detach"


class PDE:
    """Base class for Partial Differential Equation"""

    def __init__(self):
        super().__init__()
        self.equations: Dict[str, Union[Callable, sp.Basic]] = {}
        # for PDE which has learnable parameter(s)
        self.learnable_parameters = nn.ParameterList()

        self.detach_keys: Optional[Tuple[str, ...]] = None

    @staticmethod
    def create_symbols(
        symbol_str: str,
    ) -> Union[sp.Symbol, Tuple[sp.Symbol, ...]]:
        """create symbolic variables.

        Args:
            symbol_str (str): String contains symbols, such as "x", "x y z".

        Returns:
            Union[sympy.Symbol, Tuple[sympy.Symbol, ...]]: Created symbol(s).

        Examples:
            >>> import ppsci
            >>> pde = ppsci.equation.PDE()
            >>> symbol_x = pde.create_symbols('x')
            >>> symbols_xyz = pde.create_symbols('x y z')
            >>> print(symbol_x)
            x
            >>> print(symbols_xyz)
            (x, y, z)
        """
        return sp.symbols(symbol_str)

    def create_function(self, name: str, invars: Tuple[sp.Symbol, ...]) -> sp.Function:
        """Create named function depending on given invars.

        Args:
            name (str): Function name. such as "u", "v", and "f".
            invars (Tuple[sympy.Symbol, ...]): List of independent variable of function.

        Returns:
            sympy.Function: Named sympy function.

        Examples:
            >>> import ppsci
            >>> pde = ppsci.equation.PDE()
            >>> x, y, z = pde.create_symbols('x y z')
            >>> u = pde.create_function('u', (x, y))
            >>> f = pde.create_function('f', (x, y, z))
            >>> print(u)
            u(x, y)
            >>> print(f)
            f(x, y, z)
        """
        expr = sp.Function(name)(*invars)

        return expr

    def _apply_detach(self):
        """
        Wrap detached sub_expr into detach(sub_expr) to prevent gradient
        back-propagation, only for those items speicified in self.detach_keys.

        NOTE: This function is expected to be called after self.equations is ready in PDE.__init__.

        Examples:
            >>> import ppsci
            >>> ns = ppsci.equation.NavierStokes(1.0, 1.0, 2, False)
            >>> print(ns)
            NavierStokes
                continuity: Derivative(u(x, y), x) + Derivative(v(x, y), y)
                momentum_x: u(x, y)*Derivative(u(x, y), x) + v(x, y)*Derivative(u(x, y), y) + 1.0*Derivative(p(x, y), x) - 1.0*Derivative(u(x, y), (x, 2)) - 1.0*Derivative(u(x, y), (y, 2))
                momentum_y: u(x, y)*Derivative(v(x, y), x) + v(x, y)*Derivative(v(x, y), y) + 1.0*Derivative(p(x, y), y) - 1.0*Derivative(v(x, y), (x, 2)) - 1.0*Derivative(v(x, y), (y, 2))
            >>> detach_keys = ("u", "v__y")
            >>> ns = ppsci.equation.NavierStokes(1.0, 1.0, 2, False, detach_keys=detach_keys)
            >>> print(ns)
            NavierStokes
                continuity: detach(Derivative(v(x, y), y)) + Derivative(u(x, y), x)
                momentum_x: detach(u(x, y))*Derivative(u(x, y), x) + v(x, y)*Derivative(u(x, y), y) + 1.0*Derivative(p(x, y), x) - 1.0*Derivative(u(x, y), (x, 2)) - 1.0*Derivative(u(x, y), (y, 2))
                momentum_y: detach(u(x, y))*Derivative(v(x, y), x) + detach(Derivative(v(x, y), y))*v(x, y) + 1.0*Derivative(p(x, y), y) - 1.0*Derivative(v(x, y), (x, 2)) - 1.0*Derivative(v(x, y), (y, 2))
        """
        if self.detach_keys is None:
            return

        from copy import deepcopy

        from sympy.core.traversal import postorder_traversal

        from ppsci.utils.symbolic import _cvt_to_key

        for name, expr in self.equations.items():
            if not isinstance(expr, sp.Basic):
                continue
            # only process sympy expression
            expr_ = deepcopy(expr)
            for item in postorder_traversal(expr):
                if _cvt_to_key(item) in self.detach_keys:
                    # inplace all related sub_expr into detach(sub_expr)
                    expr_ = expr_.replace(item, sp.Function(DETACH_FUNC_NAME)(item))

                    # remove all detach wrapper for more-than-once wrapped items to prevent duplicated wrapping
                    expr_ = expr_.replace(
                        sp.Function(DETACH_FUNC_NAME)(
                            sp.Function(DETACH_FUNC_NAME)(item)
                        ),
                        sp.Function(DETACH_FUNC_NAME)(item),
                    )

                    # remove unccessary detach wrapping for the first arg of Derivative
                    for item_ in list(postorder_traversal(expr_)):
                        if isinstance(item_, sp.Derivative):
                            if item_.args[0].name == DETACH_FUNC_NAME:
                                expr_ = expr_.replace(
                                    item_,
                                    sp.Derivative(
                                        item_.args[0].args[0], *item_.args[1:]
                                    ),
                                )

            self.equations[name] = expr_

    def add_equation(self, name: str, equation: Callable):
        """Add an equation.

        Args:
            name (str): Name of equation
            equation (Callable): Computation function for equation.

        Examples:
            >>> import ppsci
            >>> import sympy
            >>> pde = ppsci.equation.PDE()
            >>> x, y = pde.create_symbols('x y')
            >>> u = x**2 + y**2
            >>> equation = sympy.diff(u, x) + sympy.diff(u, y)
            >>> pde.add_equation('linear_pde', equation)
            >>> print(pde)
            PDE
                linear_pde: 2*x + 2*y
        """
        self.equations.update({name: equation})

    def parameters(self) -> List[paddle.Tensor]:
        """Return learnable parameters contained in PDE.

        Returns:
            List[Tensor]: A list of learnable parameters.

        Examples:
            >>> import ppsci
            >>> pde = ppsci.equation.Vibration(2, -4, 0)
            >>> print(pde.parameters())
            [Parameter containing:
            Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                   -4.), Parameter containing:
            Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                   0.)]
        """
        return self.learnable_parameters.parameters()

    def state_dict(self) -> Dict[str, paddle.Tensor]:
        """Return named learnable parameters in dict.

        Returns:
            Dict[str, Tensor]: A dict of states(str) and learnable parameters(Tensor).

        Examples:
            >>> import ppsci
            >>> pde = ppsci.equation.Vibration(2, -4, 0)
            >>> print(pde.state_dict())
            OrderedDict([('0', Parameter containing:
            Tensor(shape=[], dtype=float64, place=Place(gpu:0), stop_gradient=False,
                   -4.)), ('1', Parameter containing:
            Tensor(shape=[], dtype=float64, place=Place(gpu:0), stop_gradient=False,
                   0.))])
        """

        return self.learnable_parameters.state_dict()

    def set_state_dict(
        self, state_dict: Dict[str, paddle.Tensor]
    ) -> Tuple[List[str], List[str]]:
        """Set state dict from dict.

        Args:
            state_dict (Dict[str, paddle.Tensor]): The state dict to be set.

        Returns:
            Tuple[List[str], List[str]]: List of missing_keys and unexpected_keys.
                Expected to be two empty tuples mostly.

        Examples:
            >>> import paddle
            >>> import ppsci
            >>> paddle.set_default_dtype("float64")
            >>> pde = ppsci.equation.Vibration(2, -4, 0)
            >>> state = pde.state_dict()
            >>> state['0'] = paddle.to_tensor(-3.1)
            >>> pde.set_state_dict(state)
            ([], [])
            >>> print(state)
            OrderedDict([('0', Tensor(shape=[], dtype=float64, place=Place(gpu:0), stop_gradient=True,
                   -3.10000000)), ('1', Parameter containing:
            Tensor(shape=[], dtype=float64, place=Place(gpu:0), stop_gradient=False,
                   0.))])
        """
        return self.learnable_parameters.set_state_dict(state_dict)

    def __str__(self):
        return "\n".join(
            [self.__class__.__name__]
            + [f"    {name}: {eq}" for name, eq in self.equations.items()]
        )
