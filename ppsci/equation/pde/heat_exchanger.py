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

from typing import Union

from ppsci.equation.pde import base


class HeatExchanger(base.PDE):
    r"""Class for heat exchanger equation.

    $$
    \begin{aligned}
    & L\left(\frac{q_m c_p}{v}\right)_{\mathrm{c}} \frac{\partial T_{\mathrm{c}}}{\partial \tau}-L\left(q_m c_p\right)_{\mathrm{c}} \frac{\partial T_{\mathrm{c}}}{\partial x}=\left(\eta_{\mathrm{o}} \alpha A\right)_{\mathrm{c}}\left(T_{\mathrm{w}}-T_{\mathrm{c}}\right), \\
    & L\left(\frac{q_m c_p}{v}\right)_{\mathrm{h}} \frac{\partial T_{\mathrm{h}}}{\partial \tau}+L\left(q_m c_p\right)_{\mathrm{h}} \frac{\partial T_{\mathrm{h}}}{\partial x}=\left(\eta_{\mathrm{o}} \alpha A\right)_{\mathrm{h}}\left(T_{\mathrm{w}}-T_{\mathrm{h}}\right), \\
    & \left(M c_p\right)_{\mathrm{w}} \frac{\partial T_{\mathrm{w}}}{\partial \tau}=\left(\eta_{\mathrm{o}} \alpha A\right)_{\mathrm{h}}\left(T_{\mathrm{h}}-T_{\mathrm{w}}\right)+\left(\eta_{\mathrm{o}} \alpha A\right)_{\mathrm{c}}\left(T_{\mathrm{c}}-T_{\mathrm{w}}\right).
    \end{aligned}
    $$

    where:

    - $T$ is temperature,
    - $q_m$ is mass flow rate,
    - $c_p$ represents specific heat capacity,
    - $v$ denotes flow velocity,
    - $L$ stands for flow length,
    - $\eta_{\mathrm{o}}$ signifies fin surface efficiency,
    - $\alpha$ stands for heat transfer coefficient,
    - $A$ indicates heat transfer area,
    - $M$ represents the mass of the heat transfer structure,
    - $\tau$ correspond to time,
    - $x$ correspond flow direction,
    - Subscripts $\mathrm{h}$, $\mathrm{c}$, and $\mathrm{w}$ denote the hot fluid side, cold fluid side, and heat transfer wall, respectively.

    Args:
        alpha_h: $\frac{(\eta_o\alpha A)_h}{L(c_p)_h}$
        alpha_c: $\frac{(\eta_o\alpha A)_c}{L(c_p)_c}$
        v_h: $v_h$
        v_c: $v_c$
        w_h: $\frac{(\eta_o\alpha A)_h}{M(c_p)_w}$
        w_c: $\frac{(\eta_o\alpha A)_c}{M(c_p)_w}$

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.HeatExchanger(1.0,1.0,1.0,1.0,1.0,1.0)
    """

    def __init__(
        self,
        alpha_h: Union[float, str],
        alpha_c: Union[float, str],
        v_h: Union[float, str],
        v_c: Union[float, str],
        w_h: Union[float, str],
        w_c: Union[float, str],
    ):
        super().__init__()
        x, t, qm_h, qm_c = self.create_symbols("x t qm_h qm_c")

        T_h = self.create_function("T_h", (x, t, qm_h))
        T_c = self.create_function("T_c", (x, t, qm_c))
        T_w = self.create_function("T_w", (x, t))

        T_h_x = T_h.diff(x)
        T_h_t = T_h.diff(t)
        T_c_x = T_c.diff(x)
        T_c_t = T_c.diff(t)
        T_w_t = T_w.diff(t)

        beta_h = (alpha_h * v_h) / qm_h
        beta_c = (alpha_c * v_c) / qm_c

        heat_boundary = T_h_t + v_h * T_h_x - beta_h * (T_w - T_h)
        cold_boundary = T_c_t - v_c * T_c_x - beta_c * (T_w - T_c)
        wall = T_w_t - w_h * (T_h - T_w) - w_c * (T_c - T_w)

        self.add_equation("heat_boundary", heat_boundary)
        self.add_equation("cold_boundary", cold_boundary)
        self.add_equation("wall", wall)
