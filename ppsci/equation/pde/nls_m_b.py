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

from typing import Optional
from typing import Tuple
from typing import Union

from ppsci.equation.pde import base


class NLSMB(base.PDE):
    r"""Class for nonlinear Schrodinger-Maxwell-Bloch equation.

    $$
    \begin{cases}
        \dfrac{\partial E}{\partial x} = i \alpha_1 \dfrac{\partial^2 E}{\partial t ^2} - i \alpha_2 |E|^2 E+2 p \\
        \dfrac{\partial p}{\partial t} = 2 i \omega_0 p+2 E \eta \\
        \dfrac{\partial \eta}{\partial t} = -(E p^* + E^* p)
    \end{cases}
    $$

    Args:
        alpha_1 (Union[float, str]): Group velocity dispersion.
        alpha_2 (Union[float, str]): Kerr nonlinearity.
        omega_0 (Union[float, str]): The offset of resonance frequency.
        time (bool): Whether the equation is time-dependent.
        detach_keys (Optional[Tuple[str, ...]]): Keys used for detach during computing.
            Defaults to None.

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.NLSMB(0.5, -1.0, 0.5, True)
    """

    def __init__(
        self,
        alpha_1: Union[float, str],
        alpha_2: Union[float, str],
        omega_0: Union[float, str],
        time: bool,
        detach_keys: Optional[Tuple[str, ...]] = None,
    ):
        super().__init__()
        self.detach_keys = detach_keys
        self.time = time

        t, x = self.create_symbols("t x")
        invars = (x,)
        if time:
            invars = (t,) + invars

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.omega_0 = omega_0

        Eu = self.create_function("Eu", invars)
        Ev = self.create_function("Ev", invars)
        pu = self.create_function("pu", invars)
        pv = self.create_function("pv", invars)
        eta = self.create_function("eta", invars)

        pu_t = pu.diff(t)
        pv_t = pv.diff(t)
        eta_t = eta.diff(t)

        Eu_x = Eu.diff(x)
        Ev_x = Ev.diff(x)

        Eu_tt = Eu.diff(t).diff(t)
        Ev_tt = Ev.diff(t).diff(t)

        Schrodinger_1 = (
            alpha_1 * Eu_tt - alpha_2 * Eu * (Eu**2 + Ev**2) + 2 * pv - Ev_x
        )
        Schrodinger_2 = (
            alpha_1 * Ev_tt - alpha_2 * Ev * (Eu**2 + Ev**2) - 2 * pu + Eu_x
        )
        Maxwell_1 = 2 * Ev * eta - pv_t + 2 * pu * omega_0
        Maxwell_2 = -2 * Eu * eta + pu_t + 2 * pv * omega_0
        Bloch = 2 * pv * Ev + 2 * pu * Eu + eta_t

        self.add_equation("Schrodinger_1", Schrodinger_1)
        self.add_equation("Schrodinger_2", Schrodinger_2)
        self.add_equation("Maxwell_1", Maxwell_1)
        self.add_equation("Maxwell_2", Maxwell_2)
        self.add_equation("Bloch", Bloch)
