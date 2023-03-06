"""
Primitives for 3D geometries
see https://www.iquilezles.org/www/articles/distfunctions/distfunctions.html
"""

from sympy import (
    Symbol,
    Function,
    Abs,
    Max,
    Min,
    sqrt,
    pi,
    sin,
    cos,
    atan,
    atan2,
    acos,
    asin,
    sign,
)

from sympy.vector import CoordSys3D
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from .geometry import Geometry, csg_curve_naming
from .helper import _sympy_sdf_to_sdf
from .curve import SympyCurve, Curve
from .parameterization import Parameterization, Parameter, Bounds
from ..constants import diff_str


class Plane(Geometry):
    """
    3D Plane perpendicular to x-axis

    Parameters
    ----------
    point_1 : tuple with 3 ints or floats
        lower bound point of plane
    point_2 : tuple with 3 ints or floats
        upper bound point of plane
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, point_1, point_2, normal=1, parameterization=Parameterization()):
        assert (
            point_1[0] == point_2[0]
        ), "Points must have same coordinate on normal dim"

        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        s_1, s_2 = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1))
        center = (
            point_1[0] + (point_2[0] - point_1[0]) / 2,
            point_1[1] + (point_2[1] - point_1[1]) / 2,
            point_1[2] + (point_2[2] - point_1[2]) / 2,
        )
        side_y = point_2[1] - point_1[1]
        side_z = point_2[2] - point_1[2]

        # surface of the plane
        curve_parameterization = Parameterization({s_1: (-1, 1), s_2: (-1, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        curve_1 = SympyCurve(
            functions={
                "x": center[0],
                "y": center[1] + 0.5 * s_1 * side_y,
                "z": center[2] + 0.5 * s_2 * side_z,
                "normal_x": 1e-10 + normal,  # TODO rm 1e-10
                "normal_y": 0,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=side_y * side_z,
        )
        curves = [curve_1]

        # calculate SDF
        sdf = normal * (center[0] - x)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (point_1[0], point_2[0]),
                Parameter("y"): (point_1[1], point_2[1]),
                Parameter("z"): (point_1[2], point_2[2]),
            },
            parameterization=parameterization,
        )

        # initialize Plane
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )


class Channel(Geometry):
    """
    3D Channel (no bounding surfaces in x-direction)

    Parameters
    ----------
    point_1 : tuple with 3 ints or floats
        lower bound point of channel
    point_2 : tuple with 3 ints or floats
        upper bound point of channel
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, point_1, point_2, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        s_1, s_2 = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1))
        center = (
            point_1[0] + (point_2[0] - point_1[0]) / 2,
            point_1[1] + (point_2[1] - point_1[1]) / 2,
            point_1[2] + (point_2[2] - point_1[2]) / 2,
        )
        side_x = point_2[0] - point_1[0]
        side_y = point_2[1] - point_1[1]
        side_z = point_2[2] - point_1[2]

        # surface of the channel
        curve_parameterization = Parameterization({s_1: (-1, 1), s_2: (-1, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        curve_1 = SympyCurve(
            functions={
                "x": center[0] + 0.5 * s_1 * side_x,
                "y": center[1] + 0.5 * s_2 * side_y,
                "z": center[2] + 0.5 * side_z,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": 1,
            },
            parameterization=curve_parameterization,
            area=side_x * side_y,
        )
        curve_2 = SympyCurve(
            functions={
                "x": center[0] + 0.5 * s_1 * side_x,
                "y": center[1] + 0.5 * s_2 * side_y,
                "z": center[2] - 0.5 * side_z,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": -1,
            },
            parameterization=curve_parameterization,
            area=side_x * side_y,
        )
        curve_3 = SympyCurve(
            functions={
                "x": center[0] + 0.5 * s_1 * side_x,
                "y": center[1] + 0.5 * side_y,
                "z": center[2] + 0.5 * s_2 * side_z,
                "normal_x": 0,
                "normal_y": 1,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=side_x * side_z,
        )
        curve_4 = SympyCurve(
            functions={
                "x": center[0] + 0.5 * s_1 * side_x,
                "y": center[1] - 0.5 * side_y,
                "z": center[2] + 0.5 * s_2 * side_z,
                "normal_x": 0,
                "normal_y": -1,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=side_x * side_z,
        )
        curves = [curve_1, curve_2, curve_3, curve_4]

        # calculate SDF
        y_dist = Abs(y - center[1]) - 0.5 * side_y
        z_dist = Abs(z - center[2]) - 0.5 * side_z
        outside_distance = sqrt(Max(y_dist, 0) ** 2 + Max(z_dist, 0) ** 2)
        inside_distance = Min(Max(y_dist, z_dist), 0)
        sdf = -(outside_distance + inside_distance)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (point_1[0], point_2[0]),
                Parameter("y"): (point_1[1], point_2[1]),
                Parameter("z"): (point_1[2], point_2[2]),
            },
            parameterization=parameterization,
        )

        # initialize Channel
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )


class Box(Geometry):
    """
    3D Box/Cuboid

    Parameters
    ----------
    point_1 : tuple with 3 ints or floats
        lower bound point of box
    point_2 : tuple with 3 ints or floats
        upper bound point of box
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, point_1, point_2, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        s_1, s_2 = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1))
        center = (
            point_1[0] + (point_2[0] - point_1[0]) / 2,
            point_1[1] + (point_2[1] - point_1[1]) / 2,
            point_1[2] + (point_2[2] - point_1[2]) / 2,
        )
        side_x = point_2[0] - point_1[0]
        side_y = point_2[1] - point_1[1]
        side_z = point_2[2] - point_1[2]

        # surface of the box
        curve_parameterization = Parameterization({s_1: (-1, 1), s_2: (-1, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )

        # Top
        curve_1 = SympyCurve(
            functions={
                "x": center[0] + 0.5 * s_1 * side_x,
                "y": center[1] + 0.5 * s_2 * side_y,
                "z": center[2] + 0.5 * side_z,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": 1,
            },
            parameterization=curve_parameterization,
            area=side_x * side_y,
        )

        # Bottom
        curve_2 = SympyCurve(
            functions={
                "x": center[0] + 0.5 * s_1 * side_x,
                "y": center[1] + 0.5 * s_2 * side_y,
                "z": center[2] - 0.5 * side_z,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": -1,
            },
            parameterization=curve_parameterization,
            area=side_x * side_y,
        )

        # Back
        curve_3 = SympyCurve(
            functions={
                "x": center[0] + 0.5 * s_1 * side_x,
                "y": center[1] + 0.5 * side_y,
                "z": center[2] + 0.5 * s_2 * side_z,
                "normal_x": 0,
                "normal_y": 1,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=side_x * side_z,
        )

        # Front
        curve_4 = SympyCurve(
            functions={
                "x": center[0] + 0.5 * s_1 * side_x,
                "y": center[1] - 0.5 * side_y,
                "z": center[2] + 0.5 * s_2 * side_z,
                "normal_x": 0,
                "normal_y": -1,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=side_x * side_z,
        )

        # Outlet
        curve_5 = SympyCurve(
            functions={
                "x": center[0] + 0.5 * side_x,
                "y": center[1] + 0.5 * s_1 * side_y,
                "z": center[2] + 0.5 * s_2 * side_z,
                "normal_x": 1,
                "normal_y": 0,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=side_y * side_z,
        )

        # Inlet
        curve_6 = SympyCurve(
            functions={
                "x": center[0] - 0.5 * side_x,
                "y": center[1] + 0.5 * s_1 * side_y,
                "z": center[2] + 0.5 * s_2 * side_z,
                "normal_x": -1,
                "normal_y": 0,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=side_y * side_z,
        )
        curves = [curve_1, curve_2, curve_3, curve_4, curve_5, curve_6]

        # calculate SDF
        x_dist = Abs(x - center[0]) - 0.5 * side_x
        y_dist = Abs(y - center[1]) - 0.5 * side_y
        z_dist = Abs(z - center[2]) - 0.5 * side_z
        outside_distance = sqrt(
            Max(x_dist, 0) ** 2 + Max(y_dist, 0) ** 2 + Max(z_dist, 0) ** 2
        )
        inside_distance = Min(Max(x_dist, y_dist, z_dist), 0)
        sdf = -(outside_distance + inside_distance)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (point_1[0], point_2[0]),
                Parameter("y"): (point_1[1], point_2[1]),
                Parameter("z"): (point_1[2], point_2[2]),
            },
            parameterization=parameterization,
        )

        # initialize Box
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )


class VectorizedBoxes(Geometry):
    """
    Vectorized 3D Box/Cuboid for faster surface and interior sampling.
    This primitive can be used if many boxes are required and is
    significantly faster then combining many boxes together with Boolean
    operations.

    Parameters
    ----------
    box_bounds : np.ndarray
        An array specifying the bounds of boxes. Shape of array is
        `[nr_boxes, 3, 2]` where the last dim stores the lower and
        upper bounds respectively.
    dx : float
        delta x used for SDF derivative calculations.
    """

    def __init__(self, box_bounds, dx=0.0001):
        # compute box centers and sides once for optimization
        box_centers = (
            box_bounds[:, :, 0] + (box_bounds[:, :, 1] - box_bounds[:, :, 0]) / 2
        )
        side = box_bounds[:, :, 1] - box_bounds[:, :, 0]

        # create curves
        def _sample(box_bounds, box_centers, side):
            def sample(nr_points, parameterization, quasirandom):
                # area of all faces
                face_area = np.concatenate(
                    2
                    * [
                        side[:, 0] * side[:, 1],
                        side[:, 0] * side[:, 2],
                        side[:, 1] * side[:, 2],
                    ]
                )  # [6 * nr_boxes]

                # calculate number or points per face
                face_probabilities = face_area / np.linalg.norm(face_area, ord=1)
                face_index = np.arange(face_area.shape[0])
                points_per_face = np.random.choice(
                    face_index, nr_points, p=face_probabilities
                )
                points_per_face, _ = np.histogram(
                    points_per_face, np.arange(face_area.shape[0] + 1) - 0.5
                )

                # generate random values to use when sampling faces
                s_1 = 2.0 * (np.random.rand(nr_points) - 0.5)
                s_2 = 2.0 * (np.random.rand(nr_points) - 0.5)

                # repeat side and center for each point
                repeat_side = np.repeat(
                    np.concatenate(6 * [side], axis=0), points_per_face, axis=0
                )
                repeat_centers = np.repeat(
                    np.concatenate(6 * [box_centers], axis=0), points_per_face, axis=0
                )
                repeat_face_area = np.repeat(
                    face_area / points_per_face, points_per_face, axis=0
                )

                # sample face 1
                nr_face_1 = np.sum(points_per_face[0 : box_bounds.shape[0]])
                face_1_x = (
                    repeat_centers[:nr_face_1, 0]
                    + 0.5 * s_1[:nr_face_1] * repeat_side[:nr_face_1, 0]
                )
                face_1_y = (
                    repeat_centers[:nr_face_1, 1]
                    + 0.5 * s_2[:nr_face_1] * repeat_side[:nr_face_1, 1]
                )
                face_1_z = (
                    repeat_centers[:nr_face_1, 2] + 0.5 * repeat_side[:nr_face_1, 2]
                )
                face_1_normal_x = np.zeros_like(face_1_x)
                face_1_normal_y = np.zeros_like(face_1_x)
                face_1_normal_z = np.ones_like(face_1_x)
                area_1 = repeat_face_area[:nr_face_1]

                # sample face 2
                nr_face_2 = (
                    np.sum(
                        points_per_face[box_bounds.shape[0] : 2 * box_bounds.shape[0]]
                    )
                    + nr_face_1
                )
                face_2_x = (
                    repeat_centers[nr_face_1:nr_face_2, 0]
                    + 0.5
                    * s_1[nr_face_1:nr_face_2]
                    * repeat_side[nr_face_1:nr_face_2, 0]
                )
                face_2_y = (
                    repeat_centers[nr_face_1:nr_face_2, 1]
                    + 0.5 * repeat_side[nr_face_1:nr_face_2, 1]
                )
                face_2_z = (
                    repeat_centers[nr_face_1:nr_face_2, 2]
                    + 0.5
                    * s_2[nr_face_1:nr_face_2]
                    * repeat_side[nr_face_1:nr_face_2, 2]
                )
                face_2_normal_x = np.zeros_like(face_2_x)
                face_2_normal_y = np.ones_like(face_2_x)
                face_2_normal_z = np.zeros_like(face_2_x)
                area_2 = repeat_face_area[nr_face_1:nr_face_2]

                # sample face 3
                nr_face_3 = (
                    np.sum(
                        points_per_face[
                            2 * box_bounds.shape[0] : 3 * box_bounds.shape[0]
                        ]
                    )
                    + nr_face_2
                )
                face_3_x = (
                    repeat_centers[nr_face_2:nr_face_3, 0]
                    + 0.5 * repeat_side[nr_face_2:nr_face_3, 0]
                )
                face_3_y = (
                    repeat_centers[nr_face_2:nr_face_3, 1]
                    + 0.5
                    * s_1[nr_face_2:nr_face_3]
                    * repeat_side[nr_face_2:nr_face_3, 1]
                )
                face_3_z = (
                    repeat_centers[nr_face_2:nr_face_3, 2]
                    + 0.5
                    * s_2[nr_face_2:nr_face_3]
                    * repeat_side[nr_face_2:nr_face_3, 2]
                )
                face_3_normal_x = np.ones_like(face_3_x)
                face_3_normal_y = np.zeros_like(face_3_x)
                face_3_normal_z = np.zeros_like(face_3_x)
                area_3 = repeat_face_area[nr_face_2:nr_face_3]

                # sample face 4
                nr_face_4 = (
                    np.sum(
                        points_per_face[
                            3 * box_bounds.shape[0] : 4 * box_bounds.shape[0]
                        ]
                    )
                    + nr_face_3
                )
                face_4_x = (
                    repeat_centers[nr_face_3:nr_face_4, 0]
                    + 0.5
                    * s_1[nr_face_3:nr_face_4]
                    * repeat_side[nr_face_3:nr_face_4, 0]
                )
                face_4_y = (
                    repeat_centers[nr_face_3:nr_face_4, 1]
                    + 0.5
                    * s_2[nr_face_3:nr_face_4]
                    * repeat_side[nr_face_3:nr_face_4, 1]
                )
                face_4_z = (
                    repeat_centers[nr_face_3:nr_face_4, 2]
                    - 0.5 * repeat_side[nr_face_3:nr_face_4, 2]
                )
                face_4_normal_x = np.zeros_like(face_4_x)
                face_4_normal_y = np.zeros_like(face_4_x)
                face_4_normal_z = -np.ones_like(face_4_x)
                area_4 = repeat_face_area[nr_face_3:nr_face_4]

                # sample face 5
                nr_face_5 = (
                    np.sum(
                        points_per_face[
                            4 * box_bounds.shape[0] : 5 * box_bounds.shape[0]
                        ]
                    )
                    + nr_face_4
                )
                face_5_x = (
                    repeat_centers[nr_face_4:nr_face_5, 0]
                    + 0.5
                    * s_1[nr_face_4:nr_face_5]
                    * repeat_side[nr_face_4:nr_face_5, 0]
                )
                face_5_y = (
                    repeat_centers[nr_face_4:nr_face_5, 1]
                    - 0.5 * repeat_side[nr_face_4:nr_face_5, 1]
                )
                face_5_z = (
                    repeat_centers[nr_face_4:nr_face_5, 2]
                    + 0.5
                    * s_2[nr_face_4:nr_face_5]
                    * repeat_side[nr_face_4:nr_face_5, 2]
                )
                face_5_normal_x = np.zeros_like(face_5_x)
                face_5_normal_y = -np.ones_like(face_5_x)
                face_5_normal_z = np.zeros_like(face_5_x)
                area_5 = repeat_face_area[nr_face_4:nr_face_5]

                # sample face 6
                nr_face_6 = (
                    np.sum(points_per_face[5 * box_bounds.shape[0] :]) + nr_face_5
                )
                face_6_x = (
                    repeat_centers[nr_face_5:nr_face_6, 0]
                    - 0.5 * repeat_side[nr_face_5:nr_face_6, 0]
                )
                face_6_y = (
                    repeat_centers[nr_face_5:nr_face_6, 1]
                    + 0.5
                    * s_1[nr_face_5:nr_face_6]
                    * repeat_side[nr_face_5:nr_face_6, 1]
                )
                face_6_z = (
                    repeat_centers[nr_face_5:nr_face_6, 2]
                    + 0.5
                    * s_2[nr_face_5:nr_face_6]
                    * repeat_side[nr_face_5:nr_face_6, 2]
                )
                face_6_normal_x = -np.ones_like(face_6_x)
                face_6_normal_y = np.zeros_like(face_6_x)
                face_6_normal_z = np.zeros_like(face_6_x)
                area_6 = repeat_face_area[nr_face_5:nr_face_6]

                # gather for invar
                invar = {
                    "x": np.concatenate(
                        [face_1_x, face_2_x, face_3_x, face_4_x, face_5_x, face_6_x],
                        axis=0,
                    )[:, None],
                    "y": np.concatenate(
                        [face_1_y, face_2_y, face_3_y, face_4_y, face_5_y, face_6_y],
                        axis=0,
                    )[:, None],
                    "z": np.concatenate(
                        [face_1_z, face_2_z, face_3_z, face_4_z, face_5_z, face_6_z],
                        axis=0,
                    )[:, None],
                    "normal_x": np.concatenate(
                        [
                            face_1_normal_x,
                            face_2_normal_x,
                            face_3_normal_x,
                            face_4_normal_x,
                            face_5_normal_x,
                            face_6_normal_x,
                        ],
                        axis=0,
                    )[:, None],
                    "normal_y": np.concatenate(
                        [
                            face_1_normal_y,
                            face_2_normal_y,
                            face_3_normal_y,
                            face_4_normal_y,
                            face_5_normal_y,
                            face_6_normal_y,
                        ],
                        axis=0,
                    )[:, None],
                    "normal_z": np.concatenate(
                        [
                            face_1_normal_z,
                            face_2_normal_z,
                            face_3_normal_z,
                            face_4_normal_z,
                            face_5_normal_z,
                            face_6_normal_z,
                        ],
                        axis=0,
                    )[:, None],
                    "area": np.concatenate(
                        [area_1, area_2, area_3, area_4, area_5, area_6], axis=0
                    )[:, None],
                }
                return invar, {}

            return sample

        curves = [Curve(_sample(box_bounds, box_centers, side), dims=3)]

        # create closure for SDF function
        def _sdf(box_bounds, box_centers, side, dx):
            def sdf(invar, param_ranges={}, compute_sdf_derivatives=False):
                # get input and tile for each box
                xyz = np.stack([invar["x"], invar["y"], invar["z"]], axis=-1)
                xyz = np.tile(np.expand_dims(xyz, 1), (1, box_bounds.shape[0], 1))

                # compute distance
                outputs = {"sdf": VectorizedBoxes._sdf_box(xyz, box_centers, side)}

                # compute distance derivatives if needed
                if compute_sdf_derivatives:
                    for i, d in enumerate(["x", "y", "z"]):
                        # compute sdf plus dx/2
                        plus_xyz = np.copy(xyz)
                        plus_xyz[..., i] += dx / 2
                        computed_sdf_plus = VectorizedBoxes._sdf_box(
                            plus_xyz, box_centers, side
                        )

                        # compute sdf minus dx/2
                        minus_xyz = np.copy(xyz)
                        minus_xyz[..., i] -= dx / 2
                        computed_sdf_minus = VectorizedBoxes._sdf_box(
                            minus_xyz, box_centers, side
                        )

                        # store sdf derivative
                        outputs["sdf" + diff_str + d] = (
                            computed_sdf_plus - computed_sdf_minus
                        ) / dx
                return outputs

            return sdf

        # create bounds
        bounds = Bounds(
            {
                "x": (np.min(box_bounds[:, 0, 0]), np.max(box_bounds[:, 0, 1])),
                "y": (np.min(box_bounds[:, 1, 0]), np.max(box_bounds[:, 1, 1])),
                "z": (np.min(box_bounds[:, 2, 0]), np.max(box_bounds[:, 2, 1])),
            }
        )

        # initialize geometry
        Geometry.__init__(
            self, curves, _sdf(box_bounds, box_centers, side, dx), bounds=bounds, dims=3
        )

    @staticmethod
    def _sdf_box(xyz, box_centers, side):
        xyz_dist = np.abs(xyz - np.expand_dims(box_centers, 0)) - 0.5 * np.expand_dims(
            side, 0
        )
        outside_distance = np.sqrt(np.sum(np.maximum(xyz_dist, 0) ** 2, axis=-1))
        inside_distance = np.minimum(np.max(xyz_dist, axis=-1), 0)
        return np.max(-(outside_distance + inside_distance), axis=-1)


class Sphere(Geometry):
    """
    3D Sphere

    Parameters
    ----------
    center : tuple with 3 ints or floats
        center of sphere
    radius : int or float
        radius of sphere
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, center, radius, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        r_1, r_2, r_3 = (
            Symbol(csg_curve_naming(0)),
            Symbol(csg_curve_naming(1)),
            Symbol(csg_curve_naming(2)),
        )

        # surface of the sphere
        curve_parameterization = Parameterization(
            {r_1: (-1, 1), r_2: (-1, 1), r_3: (-1, 1)}
        )
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        norm = sqrt(r_1 ** 2 + r_2 ** 2 + r_3 ** 2)
        curve_1 = SympyCurve(
            functions={
                "x": center[0] + radius * r_1 / norm,  # TODO GAUSSIAN DIST
                "y": center[1] + radius * r_2 / norm,
                "z": center[2] + radius * r_3 / norm,
                "normal_x": r_1 / norm,
                "normal_y": r_2 / norm,
                "normal_z": r_3 / norm,
            },
            parameterization=curve_parameterization,
            area=4 * pi * radius ** 2,
        )
        curves = [curve_1]

        # calculate SDF
        sdf = radius - sqrt(
            (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
        )

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - radius, center[0] + radius),
                Parameter("y"): (center[1] - radius, center[1] + radius),
                Parameter("z"): (center[2] - radius, center[2] + radius),
            },
            parameterization=parameterization,
        )

        # initialize Sphere
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )


class Cylinder(Geometry):
    """
    3D Cylinder
    Axis parallel to z-axis

    Parameters
    ----------
    center : tuple with 3 ints or floats
        center of cylinder
    radius : int or float
        radius of cylinder
    height : int or float
        height of cylinder
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, center, radius, height, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        h, r = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1))
        theta = Symbol(csg_curve_naming(2))

        # surface of the cylinder
        curve_parameterization = Parameterization(
            {h: (-1, 1), r: (0, 1), theta: (0, 2 * pi)}
        )
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )

        # column
        curve_1 = SympyCurve(
            functions={
                "x": center[0] + radius * cos(theta),
                "y": center[1] + radius * sin(theta),
                "z": center[2] + 0.5 * h * height,
                "normal_x": 1 * cos(theta),
                "normal_y": 1 * sin(theta),
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=height * 2 * pi * radius,
        )

        # top cover 
        curve_2 = SympyCurve(
            functions={
                "x": center[0] + sqrt(r) * radius * cos(theta),
                "y": center[1] + sqrt(r) * radius * sin(theta),
                "z": center[2] + 0.5 * height,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": 1,
            },
            parameterization=curve_parameterization,
            area=pi * radius ** 2,
        )

        # bottom cover
        curve_3 = SympyCurve(
            functions={
                "x": center[0] + sqrt(r) * radius * cos(theta),
                "y": center[1] + sqrt(r) * radius * sin(theta),
                "z": center[2] - 0.5 * height,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": -1,
            },
            parameterization=curve_parameterization,
            area=pi * radius ** 2,
        )
        curves = [curve_1, curve_2, curve_3]

        # calculate SDF
        r_dist = sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        z_dist = Abs(z - center[2])
        outside_distance = sqrt(
            Min(0, radius - r_dist) ** 2 + Min(0, 0.5 * height - z_dist) ** 2
        )
        inside_distance = -1 * Min(
            Abs(Min(0, r_dist - radius)), Abs(Min(0, z_dist - 0.5 * height))
        )
        sdf = -(outside_distance + inside_distance)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - radius, center[0] + radius),
                Parameter("y"): (center[1] - radius, center[1] + radius),
                Parameter("z"): (center[2] - height / 2, center[2] + height / 2),
            },
            parameterization=parameterization,
        )

        # initialize Cylinder
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )


class Torus(Geometry):
    """
    3D Torus

    Parameters
    ----------
    center : tuple with 3 ints or floats
        center of torus
    radius : int or float
        distance from center to center of tube (major radius)
    radius_tube : int or float
        radius of tube (minor radius)
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(
        self, center, radius, radius_tube, parameterization=Parameterization()
    ):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        r_1, r_2, r_3 = (
            Symbol(csg_curve_naming(0)),
            Symbol(csg_curve_naming(1)),
            Symbol(csg_curve_naming(2)),
        )

        N = CoordSys3D("N")
        P = x * N.i + y * N.j + z * N.k
        O = center[0] * N.i + center[1] * N.j + center[2] * N.k
        OP_xy = (x - center[0]) * N.i + (y - center[1]) * N.j + (0) * N.k
        OR = radius * OP_xy / sqrt(OP_xy.dot(OP_xy))
        OP = P - O
        RP = OP - OR
        dist = sqrt(RP.dot(RP))

        # surface of the torus
        curve_parameterization = Parameterization(
            {r_1: (0, 1), r_2: (0, 1), r_3: (0, 1)}
        )
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        theta = 2 * pi * r_1
        phi = 2 * pi * r_2
        curve_1 = SympyCurve(
            functions={
                "x": center[0] + (radius + radius_tube * cos(theta)) * cos(phi),
                "y": center[1] + (radius + radius_tube * cos(theta)) * sin(phi),
                "z": center[2] + radius_tube * sin(theta),
                "normal_x": 1 * cos(theta) * cos(phi),
                "normal_y": 1 * cos(theta) * sin(phi),
                "normal_z": 1 * sin(theta),
            },
            parameterization=curve_parameterization,
            area=4 * pi * pi * radius * radius_tube,
            criteria=radius_tube * Abs(radius + radius_tube * cos(theta))
            >= r_3 * radius_tube * (radius + radius_tube),
        )
        curves = [curve_1]

        # calculate SDF
        sdf = radius_tube - dist

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (
                    center[0] - radius - radius_tube,
                    center[0] + radius + radius_tube,
                ),
                Parameter("y"): (
                    center[1] - radius - radius_tube,
                    center[1] + radius + radius_tube,
                ),
                Parameter("z"): (center[2] - radius_tube, center[2] + radius_tube),
            },
            parameterization=parameterization,
        )

        # initialize Torus
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )


class Cone(Geometry):
    """
    3D Cone
    Axis parallel to z-axis

    Parameters
    ----------
    center : tuple with 3 ints or floats
        base center of cone
    radius : int or float
        base radius of cone
    height : int or float
        height of cone
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, center, radius, height, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        r, t = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1))
        theta = Symbol(csg_curve_naming(2))

        N = CoordSys3D("N")
        P = x * N.i + y * N.j + z * N.k
        O = center[0] * N.i + center[1] * N.j + center[2] * N.k
        H = center[0] * N.i + center[1] * N.j + (center[2] + height) * N.k
        R = (
            (center[0] + radius * cos(atan2(y, x))) * N.i
            + (center[1] + radius * sin(atan2(y, x))) * N.j
            + (center[2]) * N.k
        )
        OP_xy = (x - center[0]) * N.i + (y - center[1]) * N.j + (0) * N.k
        OR = radius * OP_xy / sqrt(OP_xy.dot(OP_xy))
        OP = P - O
        OH = H - O
        RP = OP - OR
        RH = OH - OR
        PH = OH - OP
        cone_angle = atan2(radius, height)
        angle = acos(PH.dot(OH) / sqrt(PH.dot(PH)) / sqrt(OH.dot(OH)))
        dist = sqrt(PH.dot(PH)) * sin(angle - cone_angle)

        # surface of the cone
        curve_parameterization = Parameterization(
            {r: (0, 1), t: (0, 1), theta: (0, 2 * pi)}
        )
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        curve_1 = SympyCurve(
            functions={
                "x": center[0] + (sqrt(t)) * radius * cos(theta),
                "y": center[1] + (sqrt(t)) * radius * sin(theta),
                "z": center[2] + (1 - sqrt(t)) * height,
                "normal_x": 1 * cos(cone_angle) * cos(theta),
                "normal_y": 1 * cos(cone_angle) * sin(theta),
                "normal_z": 1 * sin(cone_angle),
            },
            parameterization=curve_parameterization,
            area=pi * radius * (sqrt(height ** 2 + radius ** 2)),
        )
        curve_2 = SympyCurve(
            functions={
                "x": center[0] + sqrt(r) * radius * cos(theta),
                "y": center[1] + sqrt(r) * radius * sin(theta),
                "z": center[2],
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": -1,
            },
            parameterization=curve_parameterization,
            area=pi * radius ** 2,
        )
        curves = [curve_1, curve_2]

        # calculate SDF
        outside_distance = 1 * sqrt(Max(0, dist) ** 2 + Max(0, center[2] - z) ** 2)
        inside_distance = -1 * Min(Abs(Min(0, dist)), Abs(Min(0, center[2] - z)))
        sdf = -(outside_distance + inside_distance)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - radius, center[0] + radius),
                Parameter("y"): (center[1] - radius, center[1] + radius),
                Parameter("z"): (center[2], center[2] + height),
            },
            parameterization=parameterization,
        )

        # initialize Cone
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )


class TriangularPrism(Geometry):
    """
    3D Uniform Triangular Prism
    Axis parallel to z-axis

    Parameters
    ----------
    center : tuple with 3 ints or floats
        center of prism
    side : int or float
        side of equilateral base
    height : int or float
        height of prism
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, center, side, height, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        s_1, s_2, s_3 = (
            Symbol(csg_curve_naming(0)),
            Symbol(csg_curve_naming(1)),
            Symbol(csg_curve_naming(2)),
        )

        N = CoordSys3D("N")
        P = x * N.i + y * N.j + z * N.k
        O = center[0] * N.i + center[1] * N.j + center[2] * N.k
        OP = P - O
        OP_xy = OP - OP.dot(1 * N.k)
        normal_1 = -1 * N.j
        normal_2 = -sqrt(3) / 2 * N.i + 1 / 2 * N.j
        normal_3 = sqrt(3) / 2 * N.i + 1 / 2 * N.j
        r_ins = side / 2 / sqrt(3)
        distance_side = Min(
            Abs(r_ins - OP_xy.dot(normal_1)),
            Abs(r_ins - OP_xy.dot(normal_2)),
            Abs(r_ins - OP_xy.dot(normal_3)),
        )
        distance_top = Abs(z - center[2]) - 0.5 * height

        v1 = O + (
            -0.5 * side * N.i - 0.5 * sqrt(1 / 3) * side * N.j - height / 2 * side * N.k
        )
        v2 = O + (
            0.5 * side * N.i - 0.5 * sqrt(1 / 3) * side * N.j - height / 2 * side * N.k
        )
        v3 = O + (1 * sqrt(1 / 3) * side * N.j - height / 2 * side * N.k)

        # surface of the prism
        curve_parameterization = Parameterization({s_1: (0, 1), s_2: (0, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        curve_1 = SympyCurve(
            functions={
                "x": v1.dot(1 * N.i) + (v2 - v1).dot(1 * N.i) * s_1,
                "y": v1.dot(1 * N.j) + (v2 - v1).dot(1 * N.j) * s_1,
                "z": v1.dot(1 * N.k) + height * s_2,
                "normal_x": 0,
                "normal_y": -1,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=side * height,
        )
        curve_2 = SympyCurve(
            functions={
                "x": v1.dot(1 * N.i) + (v3 - v1).dot(1 * N.i) * s_1,
                "y": v1.dot(1 * N.j) + (v3 - v1).dot(1 * N.j) * s_1,
                "z": v1.dot(1 * N.k) + height * s_2,
                "normal_x": -sqrt(3) / 2,
                "normal_y": 1 / 2,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=side * height,
        )
        curve_3 = SympyCurve(
            functions={
                "x": v2.dot(1 * N.i) + (v3 - v2).dot(1 * N.i) * s_1,
                "y": v2.dot(1 * N.j) + (v3 - v2).dot(1 * N.j) * s_1,
                "z": v2.dot(1 * N.k) + height * s_2,
                "normal_x": sqrt(3) / 2,
                "normal_y": 1 / 2,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=side * height,
        )
        curve_4 = SympyCurve(
            functions={
                "x": (
                    (
                        (1 - sqrt(s_1)) * v1
                        + (sqrt(s_1) * (1 - s_2)) * v2
                        + s_2 * sqrt(s_1) * v3
                    ).dot(1 * N.i)
                ),
                "y": (
                    (
                        (1 - sqrt(s_1)) * v1
                        + (sqrt(s_1) * (1 - s_2)) * v2
                        + s_2 * sqrt(s_1) * v3
                    ).dot(1 * N.j)
                ),
                "z": (
                    (
                        (1 - sqrt(s_1)) * v1
                        + (sqrt(s_1) * (1 - s_2)) * v2
                        + s_2 * sqrt(s_1) * v3
                    ).dot(1 * N.k)
                ),
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": -1,
            },
            parameterization=curve_parameterization,
            area=sqrt(3) * side * side / 4,
        )
        curve_5 = SympyCurve(
            functions={
                "x": (
                    (
                        (1 - sqrt(s_1)) * v1
                        + (sqrt(s_1) * (1 - s_2)) * v2
                        + s_2 * sqrt(s_1) * v3
                    ).dot(1 * N.i)
                ),
                "y": (
                    (
                        (1 - sqrt(s_1)) * v1
                        + (sqrt(s_1) * (1 - s_2)) * v2
                        + s_2 * sqrt(s_1) * v3
                    ).dot(1 * N.j)
                ),
                "z": (
                    (
                        (1 - sqrt(s_1)) * v1
                        + (sqrt(s_1) * (1 - s_2)) * v2
                        + s_2 * sqrt(s_1) * v3
                    ).dot(1 * N.k)
                    + height
                ),
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": 1,
            },
            parameterization=curve_parameterization,
            area=sqrt(3) * side * side / 4,
        )
        curves = [curve_1, curve_2, curve_3, curve_4, curve_5]

        # calculate SDF
        inside_distance = Max(
            Min(
                Max(OP_xy.dot(normal_1), OP_xy.dot(normal_2), OP_xy.dot(normal_3))
                - r_ins,
                0,
            ),
            Min(Abs(z - center[2]) - 0.5 * height, 0),
        )
        outside_distance = sqrt(
            Min(
                r_ins
                - Max(OP_xy.dot(normal_1), OP_xy.dot(normal_2), OP_xy.dot(normal_3)),
                0,
            )
            ** 2
            + Min(0.5 * height - Abs(z - center[2]), 0) ** 2
        )
        sdf = -(outside_distance + inside_distance)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - side / 2, center[0] + side / 2),
                Parameter("y"): (center[1] - side / 2, center[1] + side / 2),
                Parameter("z"): (center[2], center[2] + height),
            },
            parameterization=parameterization,
        )

        # initialize TriangularPrism
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )


class Tetrahedron(Geometry):
    """
    3D Tetrahedron
    The 4 symmetrically placed points are on a unit sphere.
    Centroid of the tetrahedron is at origin and lower face is parallel to
    x-y plane
    Reference: https://en.wikipedia.org/wiki/Tetrahedron

    Parameters
    ----------
    center : tuple with 3 ints or floats
        centroid of tetrahedron
    radius : int or float
        radius of circumscribed sphere
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, center, radius, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        r_1, r_2 = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1))

        N = CoordSys3D("N")
        P = x * N.i + y * N.j + z * N.k
        O = center[0] * N.i + center[1] * N.j + center[2] * N.k

        side = sqrt(8 / 3) * radius

        # vertices of the tetrahedron
        v1 = (
            center[0] + radius * sqrt(8 / 9),
            center[1] + radius * 0,
            center[2] + radius * (-1 / 3),
        )
        v2 = (
            center[0] - radius * sqrt(2 / 9),
            center[1] + radius * sqrt(2 / 3),
            center[2] + radius * (-1 / 3),
        )
        v3 = (
            center[0] - radius * sqrt(2 / 9),
            center[1] - radius * sqrt(2 / 3),
            center[2] + radius * (-1 / 3),
        )
        v4 = (
            center[0] + radius * 0,
            center[1] + radius * 0,
            center[2] + radius * 1,
        )  # apex vector

        vv1 = v1[0] * N.i + v1[1] * N.j + v1[2] * N.k
        vv2 = v2[0] * N.i + v2[1] * N.j + v2[2] * N.k
        vv3 = v3[0] * N.i + v3[1] * N.j + v2[2] * N.k
        vv4 = v4[0] * N.i + v4[1] * N.j + v4[2] * N.k

        v4P = P - vv4

        # surface of the tetrahedron
        curve_parameterization = Parameterization({r_1: (-1, 1), r_2: (-1, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )

        # face between v1, v2, v3
        normal_1 = ((vv3 - vv1).cross(vv2 - vv1)).normalize()
        curve_1 = SympyCurve(
            functions={
                "x": (
                    center[0]
                    + (1 - sqrt(r_1)) * v1[0]
                    + (sqrt(r_1) * (1 - r_2)) * v2[0]
                    + r_2 * sqrt(r_1) * v3[0]
                ),
                "y": (
                    center[1]
                    + (1 - sqrt(r_1)) * v1[1]
                    + (sqrt(r_1) * (1 - r_2)) * v2[1]
                    + r_2 * sqrt(r_1) * v3[1]
                ),
                "z": (
                    center[2]
                    + (1 - sqrt(r_1)) * v1[2]
                    + (sqrt(r_1) * (1 - r_2)) * v2[2]
                    + r_2 * sqrt(r_1) * v3[2]
                ),
                "normal_x": normal_1.to_matrix(N)[0],
                "normal_y": normal_1.to_matrix(N)[1],
                "normal_z": normal_1.to_matrix(N)[2],
            },
            parameterization=curve_parameterization,
            area=sqrt(3) * side * side / 4,
        )
        # face between v1, v2, v4
        normal_2 = ((vv2 - vv1).cross(vv4 - vv1)).normalize()
        curve_2 = SympyCurve(
            functions={
                "x": (
                    center[0]
                    + (1 - sqrt(r_1)) * v1[0]
                    + (sqrt(r_1) * (1 - r_2)) * v2[0]
                    + r_2 * sqrt(r_1) * v4[0]
                ),
                "y": (
                    center[1]
                    + (1 - sqrt(r_1)) * v1[1]
                    + (sqrt(r_1) * (1 - r_2)) * v2[1]
                    + r_2 * sqrt(r_1) * v4[1]
                ),
                "z": (
                    center[2]
                    + (1 - sqrt(r_1)) * v1[2]
                    + (sqrt(r_1) * (1 - r_2)) * v2[2]
                    + r_2 * sqrt(r_1) * v4[2]
                ),
                "normal_x": normal_2.to_matrix(N)[0],
                "normal_y": normal_2.to_matrix(N)[1],
                "normal_z": normal_2.to_matrix(N)[2],
            },
            parameterization=curve_parameterization,
            area=sqrt(3) * side * side / 4,
        )
        # face between v1, v4, v3
        normal_3 = ((vv4 - vv1).cross(vv3 - vv1)).normalize()
        curve_3 = SympyCurve(
            functions={
                "x": (
                    center[0]
                    + (1 - sqrt(r_1)) * v1[0]
                    + (sqrt(r_1) * (1 - r_2)) * v4[0]
                    + r_2 * sqrt(r_1) * v3[0]
                ),
                "y": (
                    center[1]
                    + (1 - sqrt(r_1)) * v1[1]
                    + (sqrt(r_1) * (1 - r_2)) * v4[1]
                    + r_2 * sqrt(r_1) * v3[1]
                ),
                "z": (
                    center[2]
                    + (1 - sqrt(r_1)) * v1[2]
                    + (sqrt(r_1) * (1 - r_2)) * v4[2]
                    + r_2 * sqrt(r_1) * v3[2]
                ),
                "normal_x": normal_3.to_matrix(N)[0],
                "normal_y": normal_3.to_matrix(N)[1],
                "normal_z": normal_3.to_matrix(N)[2],
            },
            parameterization=curve_parameterization,
            area=sqrt(3) * side * side / 4,
        )
        # face between v4, v2, v3
        normal_4 = ((vv2 - vv4).cross(vv3 - vv4)).normalize()
        curve_4 = SympyCurve(
            functions={
                "x": (
                    center[0]
                    + (1 - sqrt(r_1)) * v4[0]
                    + (sqrt(r_1) * (1 - r_2)) * v2[0]
                    + r_2 * sqrt(r_1) * v3[0]
                ),
                "y": (
                    center[1]
                    + (1 - sqrt(r_1)) * v4[1]
                    + (sqrt(r_1) * (1 - r_2)) * v2[1]
                    + r_2 * sqrt(r_1) * v3[1]
                ),
                "z": (
                    center[2]
                    + (1 - sqrt(r_1)) * v4[2]
                    + (sqrt(r_1) * (1 - r_2)) * v2[2]
                    + r_2 * sqrt(r_1) * v3[2]
                ),
                "normal_x": normal_4.to_matrix(N)[0],
                "normal_y": normal_4.to_matrix(N)[1],
                "normal_z": normal_4.to_matrix(N)[2],
            },
            parameterization=curve_parameterization,
            area=sqrt(3) * side * side / 4,
        )
        curves = [curve_1, curve_2, curve_3, curve_4]

        dist = Max(
            v4P.dot(normal_2) / normal_2.magnitude(),
            v4P.dot(normal_3) / normal_3.magnitude(),
            v4P.dot(normal_4) / normal_4.magnitude(),
        )

        # calculate SDF
        outside_distance = -1 * sqrt(Max(0, dist) ** 2 + Max(0, v1[2] - z) ** 2)
        inside_distance = Min(Abs(Min(0, dist)), Abs(Min(0, v1[2] - z)))
        sdf = outside_distance + inside_distance

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - radius, center[0] + radius),
                Parameter("y"): (center[1] - radius, center[1] + radius),
                Parameter("z"): (center[2] - radius, center[2] + radius),
            },
            parameterization=parameterization,
        )

        # initialize Tetrahedron
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )


class IsoTriangularPrism(Geometry):
    """
    2D Isosceles Triangular Prism
    Symmetrical axis parallel to y-axis

    Parameters
    ----------
    center : tuple with 3 ints or floats
        center of base of triangle
    base : int or float
        base of triangle
    height : int or float
        height of triangle
    height_prism : int or float
        height of triangular prism
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(
        self, center, base, height, height_prism, parameterization=Parameterization()
    ):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        t, h, hz = (
            Symbol(csg_curve_naming(0)),
            Symbol(csg_curve_naming(1)),
            Symbol(csg_curve_naming(2)),
        )

        N = CoordSys3D("N")
        P = (x) * N.i + y * N.j + center[2] * N.k
        Q = x * N.i + y * N.j + center[2] * N.k
        O = center[0] * N.i + center[1] * N.j + center[2] * N.k
        H = center[0] * N.i + (center[1] + height) * N.j + center[2] * N.k
        B = (center[0] + base / 2) * N.i + center[1] * N.j + center[2] * N.k
        B_p = (center[0] - base / 2) * N.i + center[1] * N.j + center[2] * N.k

        OP = P - O
        OH = H - O
        PH = OH - OP
        OQ = Q - O
        QH = OH - OQ
        HP = OP - OH
        HB = B - H
        HB_p = B_p - H

        norm = ((HB_p).cross(HB)).normalize()
        norm_HB = (norm.cross(HB)).normalize()
        hypo = sqrt(height ** 2 + (base / 2) ** 2)
        angle = acos(PH.dot(OH) / sqrt(PH.dot(PH)) / sqrt(OH.dot(OH)))
        apex_angle = asin(base / 2 / hypo)
        hypo_sin = sqrt(height ** 2 + (base / 2) ** 2) * sin(apex_angle)
        hypo_cos = sqrt(height ** 2 + (base / 2) ** 2) * cos(apex_angle)
        dist = sqrt(PH.dot(PH)) * sin(Min(angle - apex_angle, pi / 2))

        a = (center[0] - base / 2) * N.i + center[1] * N.j + center[2] * N.k
        b = (center[0] + base / 2) * N.i + center[1] * N.j + center[2] * N.k
        c = center[0] * N.i + (center[1] + height) * N.j + center[2] * N.k
        s_1, s_2 = Symbol(csg_curve_naming(3)), Symbol(csg_curve_naming(4))

        # curve for each side
        ranges = {t: (-1, 1), h: (0, 1), hz: (-1, 1), s_1: (0, 1), s_2: (0, 1)}
        curve_parameterization = Parameterization(ranges)
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        curve_1 = SympyCurve(
            functions={
                "x": center[0] + t * base / 2,
                "y": center[1] + t * 0,
                "z": center[2] + 0.5 * hz * height_prism,
                "normal_x": 0,
                "normal_y": -1,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=base * height_prism,
        )
        curve_2 = SympyCurve(
            functions={
                "x": center[0] + h * hypo_sin,
                "y": center[1] + height - h * hypo_cos,
                "z": center[2] + 0.5 * hz * height_prism,
                "normal_x": 1 * cos(apex_angle),
                "normal_y": 1 * sin(apex_angle),
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=sqrt(height ** 2 + (base / 2) ** 2) * height_prism,
        )
        curve_3 = SympyCurve(
            functions={
                "x": center[0] - h * hypo_sin,
                "y": center[1] + height - h * hypo_cos,
                "z": center[2] + 0.5 * hz * height_prism,
                "normal_x": -1 * cos(apex_angle),
                "normal_y": 1 * sin(apex_angle),
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=sqrt(height ** 2 + (base / 2) ** 2) * height_prism,
        )
        curve_4 = SympyCurve(
            functions={
                "x": (
                    (
                        (1 - sqrt(s_1)) * a
                        + (sqrt(s_1) * (1 - s_2)) * b
                        + s_2 * sqrt(s_1) * c
                    ).dot(1 * N.i)
                ),
                "y": (
                    (
                        (1 - sqrt(s_1)) * a
                        + (sqrt(s_1) * (1 - s_2)) * b
                        + s_2 * sqrt(s_1) * c
                    ).dot(1 * N.j)
                ),
                "z": (
                    (
                        (1 - sqrt(s_1)) * a
                        + (sqrt(s_1) * (1 - s_2)) * b
                        + s_2 * sqrt(s_1) * c
                    ).dot(1 * N.k)
                )
                - height_prism / 2,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": -1,
            },
            parameterization=curve_parameterization,
            area=0.5 * base * height,
        )
        curve_5 = SympyCurve(
            functions={
                "x": (
                    (
                        (1 - sqrt(s_1)) * a
                        + (sqrt(s_1) * (1 - s_2)) * b
                        + s_2 * sqrt(s_1) * c
                    ).dot(1 * N.i)
                ),
                "y": (
                    (
                        (1 - sqrt(s_1)) * a
                        + (sqrt(s_1) * (1 - s_2)) * b
                        + s_2 * sqrt(s_1) * c
                    ).dot(1 * N.j)
                ),
                "z": (
                    (
                        (1 - sqrt(s_1)) * a
                        + (sqrt(s_1) * (1 - s_2)) * b
                        + s_2 * sqrt(s_1) * c
                    ).dot(1 * N.k)
                    + height_prism / 2
                ),
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": 1,
            },
            parameterization=curve_parameterization,
            area=0.5 * base * height,
        )

        curves = [curve_1, curve_2, curve_3, curve_4, curve_5]

        # calculate SDF
        z_dist = Abs(z - center[2])
        outside_distance = 1 * sqrt(
            sqrt(Max(0, dist) ** 2 + Max(0, center[1] - y) ** 2) ** 2
            + Min(0.5 * height_prism - Abs(z - center[2]), 0) ** 2
        )
        inside_distance = -1 * Min(
            Abs(Min(0, dist)),
            Abs(Min(0, center[1] - y)),
            Abs(Min(Abs(z - center[2]) - 0.5 * height_prism, 0)),
        )
        sdf = -(outside_distance + inside_distance)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - base / 2, center[0] + base / 2),
                Parameter("y"): (center[1], center[1] + height),
                Parameter("z"): (center[2], center[2] + height_prism),
            },
            parameterization=parameterization,
        )

        # initialize IsoTriangularPrism
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )


class ElliCylinder(Geometry):
    """
    3D Elliptical Cylinder
    Axis parallel to z-axis

    Approximation based on 4-arc ellipse construction
    https://www.researchgate.net/publication/241719740_Approximating_an_ellipse_with_four_circular_arcs

    Please manually ensure a>b

    Parameters
    ----------
    center : tuple with 3 ints or floats
        center of base of ellipse
    a : int or float
        semi-major axis of ellipse
    b : int or float
        semi-minor axis of ellipse
    height : int or float
        height of elliptical cylinder
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, center, a, b, height, parameterization=Parameterization()):
        # TODO Assertion creates issues while parameterization
        # assert a > b, "a must be greater than b. To have a ellipse with larger b create a ellipse with flipped a and b and then rotate by pi/2"

        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        h = Symbol(csg_curve_naming(0))
        r_1, r_2 = Symbol(csg_curve_naming(1)), Symbol(csg_curve_naming(2))
        angle = Symbol(csg_curve_naming(3))

        phi = asin(b / sqrt(a ** 2 + b ** 2))
        # phi = atan2(b, a)
        theta = pi / 2 - phi

        r1 = (a * sin(theta) + b * cos(theta) - a) / (sin(theta) + cos(theta) - 1)
        r2 = (a * sin(theta) + b * cos(theta) - b) / (sin(theta) + cos(theta) - 1)

        # surface of the cylinder
        ranges = {h: (-1, 1), r_1: (0, 1), r_2: (0, 1), angle: (-1, 1)}
        curve_parameterization = Parameterization(ranges)
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        curve_1 = SympyCurve(
            functions={
                "x": center[0] + a - r1 + r1 * cos(angle * theta),
                "y": center[1] + r1 * sin(angle * theta),
                "z": center[2] + 0.5 * h * height,
                "normal_x": 1 * cos(angle * theta),
                "normal_y": 1 * sin(angle * theta),
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=height * 2 * theta * r1,
        )
        curve_2 = SympyCurve(
            functions={
                "x": center[0] + r2 * cos(pi / 2 + angle * phi),
                "y": center[1] - r2 + b + r2 * sin(pi / 2 + angle * phi),
                "z": center[2] + 0.5 * h * height,
                "normal_x": 1 * cos(pi / 2 + angle * phi),
                "normal_y": 1 * sin(pi / 2 + angle * phi),
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=height * 2 * phi * r2,
        )
        curve_3 = SympyCurve(
            functions={
                "x": center[0] - a + r1 + r1 * cos(pi + angle * theta),
                "y": center[1] + r1 * sin(pi + angle * theta),
                "z": center[2] + 0.5 * h * height,
                "normal_x": 1 * cos(pi + angle * theta),
                "normal_y": 1 * sin(pi + angle * theta),
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=height * 2 * theta * r1,
        )
        curve_4 = SympyCurve(
            functions={
                "x": center[0] + r2 * cos(3 * pi / 2 + angle * phi),
                "y": center[1] + r2 - b + r2 * sin(3 * pi / 2 + angle * phi),
                "z": center[2] + 0.5 * h * height,
                "normal_x": 1 * cos(3 * pi / 2 + angle * phi),
                "normal_y": 1 * sin(3 * pi / 2 + angle * phi),
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=height * 2 * phi * r2,
        )
        # Flat surfaces top
        curve_5 = SympyCurve(
            functions={
                "x": center[0] + a - r1 + sqrt(r_1) * r1 * cos(angle * theta),
                "y": center[1] + sqrt(r_1) * r1 * sin(angle * theta),
                "z": center[2] + 0.5 * height,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": 1,
            },
            parameterization=curve_parameterization,
            area=theta * r1 ** 2,
        )
        curve_6 = SympyCurve(
            functions={
                "x": center[0] + sqrt(r_2) * r2 * cos(pi / 2 + angle * phi),
                "y": center[1] - r2 + b + sqrt(r_2) * r2 * sin(pi / 2 + angle * phi),
                "z": center[2] + 0.5 * height,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": 1,
            },
            parameterization=curve_parameterization,
            area=phi * r2 ** 2 - 0.5 * (r2 - b) * (a - r1) * 2,
            criteria=center[1] - r2 + b + sqrt(r_2) * r2 * sin(pi / 2 + angle * phi)
            > center[1],
        )
        # criteria=(((x-(center[0]+r2-b))**2+y**2)<r2**2))
        curve_7 = SympyCurve(
            functions={
                "x": center[0] - a + r1 + sqrt(r_1) * r1 * cos(pi + angle * theta),
                "y": center[1] + sqrt(r_1) * r1 * sin(pi + angle * theta),
                "z": center[2] + 0.5 * height,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": 1,
            },
            parameterization=curve_parameterization,
            area=theta * r1 ** 2,
        )
        curve_8 = SympyCurve(
            functions={
                "x": center[0] + sqrt(r_2) * r2 * cos(3 * pi / 2 + angle * phi),
                "y": center[1]
                + r2
                - b
                + sqrt(r_2) * r2 * sin(3 * pi / 2 + angle * phi),
                "z": center[2] + 0.5 * height,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": 1,
            },
            parameterization=curve_parameterization,
            area=phi * r2 ** 2 - 0.5 * (r2 - b) * (a - r1) * 2,
            criteria=center[1] + r2 - b + sqrt(r_2) * r2 * sin(3 * pi / 2 + angle * phi)
            < center[1],
        )
        # criteria=(((x-(center[0]-r2+b))**2+y**2)<r2**2))
        # Flat surfaces bottom
        curve_9 = SympyCurve(
            functions={
                "x": center[0] + a - r1 + sqrt(r_1) * r1 * cos(angle * theta),
                "y": center[1] + sqrt(r_1) * r1 * sin(angle * theta),
                "z": center[2] - 0.5 * height,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": -1,
            },
            parameterization=curve_parameterization,
            area=theta * r1 ** 2,
        )
        curve_10 = SympyCurve(
            functions={
                "x": center[0] + sqrt(r_2) * r2 * cos(pi / 2 + angle * phi),
                "y": center[1] - r2 + b + sqrt(r_2) * r2 * sin(pi / 2 + angle * phi),
                "z": center[2] - 0.5 * height,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": -1,
            },
            parameterization=curve_parameterization,
            area=phi * r2 ** 2 - 0.5 * (r2 - b) * (a - r1) * 2,
            criteria=center[1] - r2 + b + sqrt(r_2) * r2 * sin(pi / 2 + angle * phi)
            > center[1],
        )
        # criteria=(((x-(center[0]+r2-b))**2+y**2)<r2**2))
        curve_11 = SympyCurve(
            functions={
                "x": center[0] - a + r1 + sqrt(r_1) * r1 * cos(pi + angle * theta),
                "y": center[1] + sqrt(r_1) * r1 * sin(pi + angle * theta),
                "z": center[2] - 0.5 * height,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": -1,
            },
            parameterization=curve_parameterization,
            area=theta * r1 ** 2,
        )
        curve_12 = SympyCurve(
            functions={
                "x": center[0] + sqrt(r_2) * r2 * cos(3 * pi / 2 + angle * phi),
                "y": center[1]
                + r2
                - b
                + sqrt(r_2) * r2 * sin(3 * pi / 2 + angle * phi),
                "z": center[2] - 0.5 * height,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": -1,
            },
            parameterization=curve_parameterization,
            area=phi * r2 ** 2 - 0.5 * (r2 - b) * (a - r1) * 2,
            criteria=center[1] + r2 - b + sqrt(r_2) * r2 * sin(3 * pi / 2 + angle * phi)
            < center[1],
        )
        # criteria=(((x-(center[0]-r2+b))**2+y**2)<r2**2))

        curves = [
            curve_1,
            curve_2,
            curve_3,
            curve_4,
            curve_5,
            curve_6,
            curve_7,
            curve_8,
            curve_9,
            curve_10,
            curve_11,
            curve_12,
        ]

        # calculate SDF
        c1 = (center[0] + (a - r1), center[1], center[2])
        c2 = (center[0], center[1] - (r2 - b), center[2])
        c3 = (center[0] - (a - r1), center[1], center[2])
        c4 = (center[0], center[1] + (r2 - b), center[2])

        l1_m = (c1[1] - c2[1]) / (c1[0] - c2[0])
        l1_c = c1[1] - l1_m * c1[0]

        l2_m = (c1[1] - c4[1]) / (c1[0] - c4[0])
        l2_c = c1[1] - l2_m * c1[0]

        l3_m = (c3[1] - c4[1]) / (c3[0] - c4[0])
        l3_c = c3[1] - l3_m * c3[0]

        l4_m = (c3[1] - c2[1]) / (c3[0] - c2[0])
        l4_c = c3[1] - l4_m * c3[0]

        # (sign((x-min)*(max-x))+1)/2       # gives 0 if outside range, 0.5 if on min/max, 1 if inside range
        # if negative is desired (1-sign(x))/2
        # if positive is desired (sign(x)+1)/2

        outside_distance_1 = (
            Max((sqrt(((x) - c1[0]) ** 2 + ((y) - c1[1]) ** 2) - r1), 0)
            * ((1 - sign((y) - l1_m * (x) - l1_c)) / 2)
            * ((sign((y) - l2_m * (x) - l2_c) + 1) / 2)
        )
        outside_distance_2 = (
            Max((sqrt(((x) - c2[0]) ** 2 + ((y) - c2[1]) ** 2) - r2), 0)
            * ((sign((y) - l1_m * (x) - l1_c) + 1) / 2)
            * ((sign((y) - l4_m * (x) - l4_c) + 1) / 2)
        )
        outside_distance_3 = (
            Max((sqrt(((x) - c3[0]) ** 2 + ((y) - c3[1]) ** 2) - r1), 0)
            * ((sign((y) - l3_m * (x) - l3_c) + 1) / 2)
            * ((1 - sign((y) - l4_m * (x) - l4_c)) / 2)
        )
        outside_distance_4 = (
            Max((sqrt(((x) - c4[0]) ** 2 + ((y) - c4[1]) ** 2) - r2), 0)
            * ((1 - sign((y) - l2_m * (x) - l2_c)) / 2)
            * ((1 - sign((y) - l3_m * (x) - l3_c)) / 2)
        )

        curved_outside_distance = (
            outside_distance_1
            + outside_distance_2
            + outside_distance_3
            + outside_distance_4
        )
        flat_outside_distance = Max(Abs(z - center[2]) - 0.5 * height, 0)

        outside_distance = sqrt(
            curved_outside_distance ** 2 + flat_outside_distance ** 2
        )

        # (sign((x-min)*(max-x))+1)/2       # gives 0 if outside range, 0.5 if on min/max, 1 if inside range
        inside_distance_1 = (
            Max((r1 - sqrt(((x) - c1[0]) ** 2 + ((y) - c1[1]) ** 2)), 0)
            * ((1 - sign((y) - l1_m * (x) - l1_c)) / 2)
            * ((sign((y) - l2_m * (x) - l2_c) + 1) / 2)
        )
        inside_distance_2 = (
            Max((r2 - sqrt(((x) - c2[0]) ** 2 + ((y) - c2[1]) ** 2)), 0)
            * ((sign((y) - l1_m * (x) - l1_c) + 1) / 2)
            * ((sign((y) - l4_m * (x) - l4_c) + 1) / 2)
            * ((sign(y - center[1]) + 1) / 2)
        )
        inside_distance_3 = (
            Max((r1 - sqrt(((x) - c3[0]) ** 2 + ((y) - c3[1]) ** 2)), 0)
            * ((sign((y) - l3_m * (x) - l3_c) + 1) / 2)
            * ((1 - sign((y) - l4_m * (x) - l4_c)) / 2)
        )
        inside_distance_4 = (
            Max((r2 - sqrt(((x) - c4[0]) ** 2 + ((y) - c4[1]) ** 2)), 0)
            * ((1 - sign((y) - l2_m * (x) - l2_c)) / 2)
            * ((1 - sign((y) - l3_m * (x) - l3_c)) / 2)
            * ((sign(center[1] - y) + 1) / 2)
        )

        curved_inside_distance = (
            inside_distance_1
            + inside_distance_2
            + inside_distance_3
            + inside_distance_4
        )
        flat_inside_distance = Max(0.5 * height - Abs(z - center[2]), 0)

        inside_distance = Min(curved_inside_distance, flat_inside_distance)

        sdf = -outside_distance + inside_distance

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - a, center[0] + a),
                Parameter("y"): (center[0] - b, center[0] + b),
                Parameter("y"): (center[0] - height / 2, center[0] + height / 2),
            },
            parameterization=parameterization,
        )

        # initialize Cylinder
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )
