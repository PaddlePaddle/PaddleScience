"""
Primitives for 2D geometries
see https://www.iquilezles.org/www/articles/distfunctions/distfunctions.html
"""

import sys
from operator import mul
from sympy import Symbol, Abs, Max, Min, sqrt, sin, cos, acos, atan2, pi, Heaviside
from functools import reduce

pi = float(pi)
from sympy.vector import CoordSys3D
from .curve import SympyCurve
from .helper import _sympy_sdf_to_sdf
from .geometry import Geometry, csg_curve_naming
from .parameterization import Parameterization, Parameter, Bounds


class Line(Geometry):
    """
    2D Line parallel to y-axis

    Parameters
    ----------
    point_1 : tuple with 2 ints or floats
        lower bound point of line segment
    point_2 : tuple with 2 ints or floats
        upper bound point of line segment
    normal : int or float
        normal direction of line (+1 or -1)
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, point_1, point_2, normal=1, parameterization=Parameterization()):
        assert point_1[0] == point_2[0], "Points must have same x-coordinate"

        # make sympy symbols to use
        l = Symbol(csg_curve_naming(0))
        x = Symbol("x")

        # curves for each side
        curve_parameterization = Parameterization({l: (0, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        dist_y = point_2[1] - point_1[1]
        line_1 = SympyCurve(
            functions={
                "x": point_1[0],
                "y": point_1[1] + l * dist_y,
                "normal_x": 1e-10 + normal,  # TODO rm 1e-10
                "normal_y": 0,
            },
            parameterization=curve_parameterization,
            area=dist_y,
        )
        curves = [line_1]

        # calculate SDF
        sdf = normal * (point_1[0] - x)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (point_1[0], point_2[0]),
                Parameter("y"): (point_1[1], point_2[1]),
            },
            parameterization=parameterization,
        )

        # initialize Line
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=2,
            bounds=bounds,
            parameterization=parameterization,
        )


class Channel2D(Geometry):
    """
    2D Channel (no bounding curves in x-direction)

    Parameters
    ----------
    point_1 : tuple with 2 ints or floats
        lower bound point of channel
    point_2 : tuple with 2 ints or floats
        upper bound point of channel
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, point_1, point_2, parameterization=Parameterization()):
        # make sympy symbols to use
        l = Symbol(csg_curve_naming(0))
        y = Symbol("y")

        # curves for each side
        curve_parameterization = Parameterization({l: (0, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        dist_x = point_2[0] - point_1[0]
        dist_y = point_2[1] - point_1[1]
        line_1 = SympyCurve(
            functions={
                "x": l * dist_x + point_1[0],
                "y": point_1[1],
                "normal_x": 0,
                "normal_y": -1,
            },
            parameterization=curve_parameterization,
            area=dist_x,
        )
        line_2 = SympyCurve(
            functions={
                "x": l * dist_x + point_1[0],
                "y": point_2[1],
                "normal_x": 0,
                "normal_y": 1,
            },
            parameterization=curve_parameterization,
            area=dist_x,
        )
        curves = [line_1, line_2]

        # calculate SDF
        center_y = point_1[1] + (dist_y) / 2
        y_diff = Abs(y - center_y) - (point_2[1] - center_y)
        outside_distance = sqrt(Max(y_diff, 0) ** 2)
        inside_distance = Min(y_diff, 0)
        sdf = -(outside_distance + inside_distance)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (point_1[0], point_2[0]),
                Parameter("y"): (point_1[1], point_2[1]),
            },
            parameterization=parameterization,
        )

        # initialize Channel2D
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=2,
            bounds=bounds,
            parameterization=parameterization,
        )


class Rectangle(Geometry):
    """
    2D Rectangle

    Parameters
    ----------
    point_1 : tuple with 2 ints or floats
        lower bound point of rectangle
    point_2 : tuple with 2 ints or floats
        upper bound point of rectangle
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, point_1, point_2, parameterization=Parameterization()):
        # make sympy symbols to use
        l = Symbol(csg_curve_naming(0))
        x, y = Symbol("x"), Symbol("y")

        # curves for each side
        curve_parameterization = Parameterization({l: (0, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        dist_x = point_2[0] - point_1[0]
        dist_y = point_2[1] - point_1[1]
        line_1 = SympyCurve(
            functions={
                "x": l * dist_x + point_1[0],
                "y": point_1[1],
                "normal_x": 0,
                "normal_y": -1,
            },
            parameterization=curve_parameterization,
            area=dist_x,
        )
        line_2 = SympyCurve(
            functions={
                "x": point_2[0],
                "y": l * dist_y + point_1[1],
                "normal_x": 1,
                "normal_y": 0,
            },
            parameterization=curve_parameterization,
            area=dist_y,
        )
        line_3 = SympyCurve(
            functions={
                "x": l * dist_x + point_1[0],
                "y": point_2[1],
                "normal_x": 0,
                "normal_y": 1,
            },
            parameterization=curve_parameterization,
            area=dist_x,
        )
        line_4 = SympyCurve(
            functions={
                "x": point_1[0],
                "y": -l * dist_y + point_2[1],
                "normal_x": -1,
                "normal_y": 0,
            },
            parameterization=curve_parameterization,
            area=dist_y,
        )
        curves = [line_1, line_2, line_3, line_4]

        # calculate SDF
        center_x = point_1[0] + (dist_x) / 2
        center_y = point_1[1] + (dist_y) / 2
        x_diff = Abs(x - center_x) - (point_2[0] - center_x)
        y_diff = Abs(y - center_y) - (point_2[1] - center_y)
        outside_distance = sqrt(Max(x_diff, 0) ** 2 + Max(y_diff, 0) ** 2)
        inside_distance = Min(Max(x_diff, y_diff), 0)
        sdf = -(outside_distance + inside_distance)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (point_1[0], point_2[0]),
                Parameter("y"): (point_1[1], point_2[1]),
            },
            parameterization=parameterization,
        )

        # initialize Rectangle
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=2,
            bounds=bounds,
            parameterization=parameterization,
        )


class Circle(Geometry):
    """
    2D Circle

    Parameters
    ----------
    center : tuple with 2 ints or floats
        center point of circle
    radius : int or float
        radius of circle
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, center, radius, parameterization=Parameterization()):
        # make sympy symbols to use
        theta = Symbol(csg_curve_naming(0))
        x, y = Symbol("x"), Symbol("y")

        # curve for perimeter of the circle
        curve_parameterization = Parameterization({theta: (0, 2 * pi)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        curve = SympyCurve(
            functions={
                "x": center[0] + radius * cos(theta),
                "y": center[1] + radius * sin(theta),
                "normal_x": 1 * cos(theta),
                "normal_y": 1 * sin(theta),
            },
            parameterization=curve_parameterization,
            area=2 * pi * radius,
        )
        curves = [curve]

        # calculate SDF
        sdf = radius - sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - radius, center[0] + radius),
                Parameter("y"): (center[1] - radius, center[1] + radius),
            },
            parameterization=parameterization,
        )

        # initialize Circle
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=2,
            bounds=bounds,
            parameterization=parameterization,
        )


class Triangle(Geometry):
    """
    2D Isosceles Triangle
    Symmetrical axis parallel to y-axis

    Parameters
    ----------
    center : tuple with 2 ints or floats
        center of base of triangle
    base : int or float
        base of triangle
    height : int or float
        height of triangle
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, center, base, height, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y = Symbol("x"), Symbol("y")
        t, h = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1))

        N = CoordSys3D("N")
        P = x * N.i + y * N.j
        O = center[0] * N.i + center[1] * N.j
        H = center[0] * N.i + (center[1] + height) * N.j
        B = (center[0] + base / 2) * N.i + center[1] * N.j
        OP = P - O
        OH = H - O
        PH = OH - OP
        angle = acos(PH.dot(OH) / sqrt(PH.dot(PH)) / sqrt(OH.dot(OH)))
        apex_angle = atan2(base / 2, height)
        hypo_sin = sqrt(height ** 2 + (base / 2) ** 2) * sin(apex_angle)
        hypo_cos = sqrt(height ** 2 + (base / 2) ** 2) * cos(apex_angle)
        dist = sqrt(PH.dot(PH)) * sin(Min(angle - apex_angle, pi / 2))

        # curve for each side
        curve_parameterization = Parameterization({t: (-1, 1), h: (0, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        curve_1 = SympyCurve(
            functions={
                "x": center[0] + t * base / 2,
                "y": center[1] + t * 0,
                "normal_x": 0,
                "normal_y": -1,
            },
            parameterization=curve_parameterization,
            area=base,
        )
        curve_2 = SympyCurve(
            functions={
                "x": center[0] + h * hypo_sin,
                "y": center[1] + height - h * hypo_cos,
                "normal_x": 1 * cos(apex_angle),
                "normal_y": 1 * sin(apex_angle),
            },
            parameterization=curve_parameterization,
            area=sqrt(height ** 2 + (base / 2) ** 2),
        )
        curve_3 = SympyCurve(
            functions={
                "x": center[0] - h * hypo_sin,
                "y": center[1] + height - h * hypo_cos,
                "normal_x": -1 * cos(apex_angle),
                "normal_y": 1 * sin(apex_angle),
            },
            parameterization=curve_parameterization,
            area=sqrt(height ** 2 + (base / 2) ** 2),
        )
        curves = [curve_1, curve_2, curve_3]

        # calculate SDF
        outside_distance = 1 * sqrt(Max(0, dist) ** 2 + Max(0, center[1] - y) ** 2)
        inside_distance = -1 * Min(Abs(Min(0, dist)), Abs(Min(0, center[1] - y)))
        sdf = -(outside_distance + inside_distance)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - base / 2, center[0] + base / 2),
                Parameter("y"): (center[1], center[1] + height),
            },
            parameterization=parameterization,
        )

        # initialize Triangle
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=2,
            bounds=bounds,
            parameterization=parameterization,
        )


class Ellipse(Geometry):
    """
    2D Ellipse

    Parameters
    ----------
    center : tuple with 2 ints or floats
        center point of circle
    radius : int or float
        radius of circle
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, center, major, minor, parameterization=Parameterization()):
        # make sympy symbols to use
        theta = Symbol(csg_curve_naming(0))
        x, y = Symbol("x"), Symbol("y")
        mag = sqrt((minor * cos(theta)) ** 2 + (major * sin(theta)) ** 2)
        area = pi * (
            3 * (major + minor) - sqrt((3 * minor + major) * (3 * major + minor))
        )
        try:
            area = float(area)
        except:
            pass

        # curve for perimeter of the circle
        curve_parameterization = Parameterization({theta: (0, 2 * pi)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        curve = SympyCurve(
            functions={
                "x": center[0] + major * cos(theta),
                "y": center[1] + minor * sin(theta),
                "normal_x": minor * cos(theta) / mag,
                "normal_y": major * sin(theta) / mag,
            },
            parameterization=curve_parameterization,
            area=area,
        )
        curves = [curve]

        # calculate SDF
        sdf = 1 - (((x - center[0]) / major) ** 2 + ((y - center[1]) / minor) ** 2)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - major, center[0] + major),
                Parameter("y"): (center[1] - minor, center[1] + minor),
            },
            parameterization=parameterization,
        )

        # initialize Ellipse
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=2,
            bounds=bounds,
            parameterization=parameterization,
        )


class Polygon(Geometry):
    """
    2D Polygon

    Parameters
    ----------
    points : list of tuple with 2 ints or floats
        lower bound point of line segment
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, points, parameterization=Parameterization()):
        # make sympy symbols to use
        s = Symbol(csg_curve_naming(0))
        x = Symbol("x")
        y = Symbol("y")

        # wrap points
        wrapted_points = points + [points[0]]

        # curves for each side
        curve_parameterization = Parameterization({s: (0, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        curves = []
        for v1, v2 in zip(wrapted_points[:-1], wrapted_points[1:]):
            # area
            dx = v2[0] - v1[0]
            dy = v2[1] - v1[1]
            area = (dx ** 2 + dy ** 2) ** 0.5

            # generate normals
            normal_x = dy / area
            normal_y = -dx / area
            line = SympyCurve(
                functions={
                    "x": dx * s + v1[0],
                    "y": dy * s + v1[1],
                    "normal_x": dy / area,
                    "normal_y": -dx / area,
                },
                parameterization=curve_parameterization,
                area=area,
            )
            curves.append(line)

        # calculate SDF
        sdfs = [(x - wrapted_points[0][0]) ** 2 + (y - wrapted_points[0][1]) ** 2]
        conds = []
        for v1, v2 in zip(wrapted_points[:-1], wrapted_points[1:]):
            # sdf calculation
            dx = v1[0] - v2[0]
            dy = v1[1] - v2[1]
            px = x - v2[0]
            py = y - v2[1]
            d_dot_d = dx ** 2 + dy ** 2
            p_dot_d = px * dx + py * dy
            max_min = Max(Min(p_dot_d / d_dot_d, 1.0), 0.0)
            vx = px - dx * max_min
            vy = py - dy * max_min
            sdf = vx ** 2 + vy ** 2
            sdfs.append(sdf)

            # winding calculation
            cond_1 = Heaviside(y - v2[1])
            cond_2 = Heaviside(v1[1] - y)
            cond_3 = Heaviside((dx * py) - (dy * px))
            all_cond = cond_1 * cond_2 * cond_3
            none_cond = (1.0 - cond_1) * (1.0 - cond_2) * (1.0 - cond_3)
            cond = 1.0 - 2.0 * Min(all_cond + none_cond, 1.0)
            conds.append(cond)

        # set inside outside
        sdf = Min(*sdfs)
        cond = reduce(mul, conds)
        sdf = sqrt(sdf) * -cond

        # calculate bounds
        min_x = Min(*[p[0] for p in points])
        if min_x.is_number:
            min_x = float(min_x)
        max_x = Max(*[p[0] for p in points])
        if max_x.is_number:
            max_x = float(max_x)
        min_y = Min(*[p[1] for p in points])
        if min_y.is_number:
            min_y = float(min_y)
        max_y = Max(*[p[1] for p in points])
        if max_y.is_number:
            max_y = float(max_y)
        bounds = Bounds(
            {
                Parameter("x"): (min_x, max_x),
                Parameter("y"): (min_y, max_y),
            },
            parameterization=parameterization,
        )

        # initialize Polygon
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=2,
            bounds=bounds,
            parameterization=parameterization,
        )
