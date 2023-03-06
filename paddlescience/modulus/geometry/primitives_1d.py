from sympy import Symbol, Abs
import numpy as np
from .geometry import Geometry, csg_curve_naming
from .curve import SympyCurve
from .parameterization import Parameterization, Parameter, Bounds
from .helper import _sympy_sdf_to_sdf


class Point1D(Geometry):
    """
    1D Point along x-axis

    Parameters
    ----------
    point : int or float
        x coordinate of the point
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, point, parameterization=Parameterization()):
        # make sympy symbols to use
        x = Symbol("x")

        # curves for each side
        curve_parameterization = Parameterization({Symbol(csg_curve_naming(0)): (0, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        pt_1 = SympyCurve(
            functions={"x": point, "normal_x": 1.0},
            area=1.0,
            parameterization=curve_parameterization,
        )
        curves = [pt_1]

        # calculate SDF
        sdf = x - point

        # calculate bounds
        bounds = Bounds(
            {Parameter("x"): (point, point)}, parameterization=parameterization
        )

        # initialize
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=1,
            bounds=bounds,
            parameterization=parameterization,
        )


class Line1D(Geometry):
    """
    1D Line along x-axis

    Parameters
    ----------
    point_1 : int or float
      lower bound point of line
    point_2 : int or float
      upper bound point of line
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, point_1, point_2, parameterization=Parameterization()):
        # make sympy symbols to use
        x = Symbol("x")

        # curves for each side
        curve_parameterization = Parameterization({Symbol(csg_curve_naming(0)): (0, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        pt1 = SympyCurve(
            functions={"x": point_1, "normal_x": -1},
            area=1.0,
            parameterization=curve_parameterization,
        )
        pt2 = SympyCurve(
            functions={"x": point_2, "normal_x": 1},
            area=1.0,
            parameterization=curve_parameterization,
        )
        curves = [pt1, pt2]

        # calculate SDF
        dist = point_2 - point_1
        center_x = point_1 + dist / 2
        sdf = dist / 2 - Abs(x - center_x)

        # calculate bounds
        bounds = Bounds(
            {Parameter("x"): (point_1, point_2)}, parameterization=parameterization
        )

        # initialize
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=1,
            bounds=bounds,
            parameterization=parameterization,
        )
