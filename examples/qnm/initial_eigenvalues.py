#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 12:26:13 2025

Get initial values for omega_lm and A_lm of Schwarzchild and Kerr black hole.

This work makes use of the Black Hole Perturbation Toolkit.
https://github.com/BlackHolePerturbationToolkit/QuasiNormalModes

author: Hanlin Song (PKU)
e-mail: hanlin@stu.pku.edu.cn
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.special import ellipe

# Here we consider the G = c = 2M =1 & a \in [0,0.5],
# while the Black Hole Perturbation Toolkit consider G = c = M = 1 & a \in [0,1].
# Thus, considering the M = 1/2 will make sure the consistence for the two conventions.
M = 1 / 2


def Kerrfinit(l, m, a, n=0, s=-2):
    if a == 0:
        a = a + 1e-6
    # Useful parameter
    mu = m / (l + 0.5)

    # Delta(r)
    def Delta(r):
        return r**2 - 2 * M * r + a**2

    # Eikonal function
    def Eikonal(rp):
        x = rp / M
        term1 = 2 * x**4 * (x - 3) ** 2
        term2 = (
            4
            * x**2
            * ((1 - mu**2) * x**2 - 2 * x - 3 * (1 - mu**2))
            * (a / M) ** 2
        )
        term3 = (
            (1 - mu**2)
            * ((2 - mu**2) * x**2 + 2 * (2 + mu**2) * x + (2 - mu**2))
            * (a / M) ** 4
        )
        return term1 + term2 + term3

    # Find the root Rp of the Eikonal equation
    Rp_guess = 3.0
    Rp = fsolve(Eikonal, Rp_guess)[0]

    # Omega_r
    if mu == 0:
        numerator = np.pi / 2 * np.sqrt(Delta(Rp))
        denominator = (Rp**2 + a**2) * ellipe(
            (a**2 * Delta(Rp)) / (Rp**2 + a**2) ** 2
        )
        Omega_r = -numerator / denominator
    else:
        numerator = (M - Rp) * mu * a
        denominator = (Rp - 3 * M) * Rp**2 + (Rp + M) * a**2
        Omega_r = -numerator / denominator

    # Omega_i
    Delta_Rp = Delta(Rp)
    sqrt_term = np.sqrt(
        4 * (6 * Rp**2 * Omega_r**2 - 1) + 2 * a**2 * Omega_r**2 * (3 - mu**2)
    )

    denom = (
        2 * Rp**4 * Omega_r
        - 4 * a * M * Rp * mu
        + a**2 * Rp * Omega_r * (Rp * (3 - mu**2) + 2 * M * (1 + mu**2))
        + a**4 * Omega_r * (1 - mu**2)
    )

    Omega_i = Delta_Rp * sqrt_term / denom

    # Complex frequency finit
    finit_temp = (l + 0.5) * Omega_r - 1j * (n + 0.5) * Omega_i
    # Here we consider all initial frequency in the 4-th quadrant
    finit = np.abs(np.real(finit_temp)) - 1j * np.abs(np.imag(finit_temp))
    return finit


def KerrAinit(l, m, a, n=0, s=-2):
    finit = Kerrfinit(l, m, a, n=0, s=-2)
    mu = m / (l + 0.5)
    Ainit_temp = (l + 0.5) ** 2 - ((a * finit) ** 2 / 2) * (1 - mu**2)
    # Here we consider all initial A_lm in the 1-st quadrant
    Ainit = np.abs(np.real(Ainit_temp)) + 1j * np.abs(np.imag(Ainit_temp))
    return Ainit
