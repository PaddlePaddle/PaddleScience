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
import sympy
from paddle import nn

DETACH_FUNC_NAME = "detach"


class PDE:
    """Base class for Partial Differential Equation"""

    def __init__(self):
        super().__init__()
        self.equations = {}
        # for PDE which has learnable parameter(s)
        self.learnable_parameters = nn.ParameterList()

        self.detach_keys: Optional[Tuple[str, ...]] = None

    @staticmethod
    def create_symbols(
        symbol_str: str,
    ) -> Union[sympy.Symbol, Tuple[sympy.Symbol, ...]]:
        """Create symbols

        Args:
            symbol_str (str): String contains symbols, such as "x", "x y z".

        Returns:
            Union[sympy.Symbol, Tuple[sympy.Symbol, ...]]: Created symbol(s).
        """
        return sympy.symbols(symbol_str)

    def create_function(
        self, name: str, invars: Tuple[sympy.Symbol, ...]
    ) -> sympy.Function:
        """Create named function depending on given invars.

        Args:
            name (str): Function name. such as "u", "v", and "f".
            invars (Tuple[sympy.Symbol, ...]): List of independent variable of function.

        Returns:
            sympy.Function: Named sympy function.
        """
        expr = sympy.Function(name)(*invars)

        # wrap `expression(...)` to `detach(expression(...))`
        # if name of expression is in given detach_keys
        if self.detach_keys and name in self.detach_keys:
            expr = sympy.Function(DETACH_FUNC_NAME)(expr)
        return expr

    def add_equation(self, name: str, equation: Callable):
        """Add an equation.

        Args:
            name (str): Name of equation
            equation (Callable): Computation function for equation.
        """
        self.equations.update({name: equation})

    def parameters(self) -> List[paddle.Tensor]:
        """Return learnable parameters contained in PDE.

        Args:
            None

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

        Args:
            None

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

    def set_state_dict(self, state_dict: Dict[str, paddle.Tensor]):
        """Set state dict from dict.

        Args:
            state_dict (Dict[str, paddle.Tensor]): The state dict to be set.

        Returns:
            None

        Examples:
            >>> import paddle
            >>> import ppsci
            >>> paddle.set_default_dtype("float64")
            >>> pde = ppsci.equation.Vibration(2, -4, 0)
            >>> state = pde.state_dict()
            >>> state['0'] = paddle.to_tensor(-3.1)
            >>> pde.set_state_dict(state)
            >>> print(state)
            OrderedDict([('0', Tensor(shape=[], dtype=float64, place=Place(gpu:0), stop_gradient=True,
                   -3.10000000)), ('1', Parameter containing:
            Tensor(shape=[], dtype=float64, place=Place(gpu:0), stop_gradient=False,
                   0.))])
        """
        self.learnable_parameters.set_state_dict(state_dict)

    def __str__(self):
        return ", ".join(
            [self.__class__.__name__]
            + [f"{name}: {eq}" for name, eq in self.equations.items()]
        )
