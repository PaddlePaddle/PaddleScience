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

from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import paddle
import sympy
from paddle import nn

from ppsci.utils import sym_to_func

if TYPE_CHECKING:
    from ppsci import arch


class PDE:
    """Base class for Partial Differential Equation"""

    def __init__(self):
        super().__init__()
        self.equations = {}
        self.detach_keys = []

        # for PDE which has learnable parameter(s)
        self.learnable_parameters = nn.ParameterList()

    def create_symbols(self, symbol_str: str) -> Tuple[sympy.Symbol, ...]:
        """Create symbols

        Args:
            symbol_str (str): String contains symbols, such as "x", "x y z".

        Returns:
            Tuple[sympy.Symbol, ...]: Created symbol(s).
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
        return sympy.Function(name)(*invars)

    def add_equation(self, name: str, equation: Callable):
        """Add an equation.

        Args:
            name (str): Name of equation
            equation (Callable): Computation function for equation.
        """
        self.equations.update({name: equation})

    def parameters(self) -> List[paddle.Tensor]:
        """Return parameters contained in PDE.

        Returns:
            List[Tensor]: A list of parameters.
        """
        return self.learnable_parameters.parameters()

    def state_dict(self) -> Dict[str, paddle.Tensor]:
        """Return named parameters in dict."""
        return self.learnable_parameters.state_dict()

    def set_state_dict(self, state_dict):
        """Set state dict from dict."""
        self.learnable_parameters.set_state_dict(state_dict)

    def cvt_sympy_to_function(
        self, models: Optional[Union[arch.Arch, Tuple[arch.Arch, ...]]]
    ) -> None:
        """Convert equation(s) to callable function"""
        for name, expr in self.equations.items():
            if isinstance(expr, sympy.Basic):
                self.equations[name] = sym_to_func.sympy_to_function(
                    expr,
                    models,
                    self.detach_keys,
                    self.learnable_parameters,
                )

    def __str__(self):
        return ", ".join(
            [self.__class__.__name__]
            + [f"{name}: {eq}" for name, eq in self.equations.items()]
        )
