import numpy as np


def analyticalPossion(xcg, Tc, Tb=0):
    Ue = Tc * (1 - xcg[0, :] ** 2 - xcg[1, :] ** 2) / 4 + Tb
    return Ue.flatten()


def analyticalConeInterpolation(xcg, Tc, Tb=0):
    Ue = Tc * (1 - np.sqrt(xcg[0, :] ** 2 + xcg[1, :] ** 2)) / 4 + Tb
    return Ue.flatten()
