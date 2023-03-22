"""Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sympy

# from sympy import printing


class PDE(object):
    """Base class for Partial Differential Equation"""

    def __init__(self):
        super().__init__()
        self.equations = {}

    def create_symbols(self, symbol_str):
        """Create symbols

        Args:
            symbol_str (str): String contains symbols, such as "x", "x y z".

        Returns:
            List[symbol.Symbol]: Created symbol(s).
        """
        return sympy.symbols(symbol_str)

    def create_function(self, name, invars):
        """Create named function depending on given invars.

        Args:
            name (str): Function name. such as "u", "v", and "f".
            invars (List[Symbols]): List of independent variable of function.

        Returns:
            Function: Named sympy function.
        """
        return sympy.Function(name)(*invars)

    def add_equation(self, name, equation):
        self.equations.update({name: equation})

    def __str__(self):
        return ", ".join(
            [self.__class__.__name__]
            + [f"{name}: {eq}" for name, eq in self.equations.items()]
        )
