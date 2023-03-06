"""
Defines a Discrete geometry
"""

import numpy as np
import csv
from stl import mesh as np_mesh
from sympy import Symbol

from .geometry import Geometry
from .parameterization import Parameterization, Bounds, Parameter
from modulus.constants import diff_str


class DiscreteGeometry(Geometry):
    """
    Constructs a geometry for a discrete list of geometries
    """

    def __init__(
        self, geometries, parameterization=Parameterization(), interior_epsilon=1e-6
    ):

        # make sdf function
        def _sdf(list_sdf, discrete_parameterization, dims):
            def sdf(invar, params, compute_sdf_derivatives=False):
                # make output array to gather sdf values
                outputs = {"sdf": np.full_like(next(iter(invar.values())), np.nan)}
                if compute_sdf_derivatives:
                    for d in dims:
                        outputs["sdf" + diff_str + d] = np.full_like(
                            next(iter(invar.values())), -1000
                        )

                # compute sdf values for given parameterizations
                for i, f in enumerate(list_sdf):
                    # get sdf index for each point evaluating on
                    sdf_index = np.full_like(
                        next(iter(invar.values())), True
                    )  # TODO this could be simplified
                    for key in discrete_parameterization.parameters:
                        expanded_d = np.tile(
                            discrete_parameterization.param_ranges[Parameter(key)][
                                i : i + 1
                            ],
                            (params[key].shape[0], 1),
                        )
                        sdf_index = np.logical_and(
                            sdf_index, (params[key] == expanded_d)
                        )

                    # compute sdf values on indexed sdf function
                    sdf_indexed_invar = {
                        key: value[sdf_index[:, 0], :] for key, value in invar.items()
                    }
                    sdf_indexed_params = {
                        key: value[sdf_index[:, 0], :] for key, value in params.items()
                    }
                    computed_sdf = f(
                        sdf_indexed_invar, sdf_indexed_params, compute_sdf_derivatives
                    )

                    # update output values
                    for key, value in computed_sdf.items():
                        outputs[key][sdf_index[:, 0], :] = value
                return outputs

            return sdf

        new_sdf = _sdf(
            [g.sdf for g in geometries], parameterization, geometries[0].dims
        )

        # compute bounds
        bounds = geometries[0].bounds
        for g in geometries[1:]:
            bounds = bounds.union(g.bounds)

        # make curves
        new_curves = []
        for g in geometries:
            new_curves += g.curves

        # initialize geometry
        super().__init__(
            new_curves,
            new_sdf,
            dims=len(geometries[0].dims),
            bounds=bounds,
            parameterization=parameterization,
        )


class DiscreteCurve:
    def __init__(self, curves, discrete_parameterization=Parameterization()):
        # store attributes
        self.curves = curves
        self._dims = len(curves[0].dims)
        self.discrete_parameterization = discrete_parameterization

    def sample(
        self, nr_points, criteria=None, parameterization=None, quasirandom=False
    ):
        # use internal parameterization if not given
        if parameterization is None:
            parameterization = self.parameterization

        # continually sample points throwing out points that don't satisfy criteria
        invar = {
            key: np.empty((0, 1))
            for key in self.dims + ["normal_" + x for x in self.dims] + ["area"]
        }
        params = {key: np.empty((0, 1)) for key in parameterization.parameters}
        total_sampled = 0
        total_tried = 0
        nr_try = 0
        while True:
            # sample curve
            local_invar, local_params = self._sample(
                nr_points, parameterization, quasirandom
            )

            # compute given criteria and remove points
            if criteria is not None:
                computed_criteria = criteria(local_invar, local_params)
                local_invar = {
                    key: value[computed_criteria[:, 0], :]
                    for key, value in local_invar.items()
                }
                local_params = {
                    key: value[computed_criteria[:, 0], :]
                    for key, value in local_params.items()
                }

            # store invar
            for key in local_invar.keys():
                invar[key] = np.concatenate([invar[key], local_invar[key]], axis=0)

            # store params
            for key in local_params.keys():
                params[key] = np.concatenate([params[key], local_params[key]], axis=0)

            # keep track of sampling
            total_sampled = next(iter(invar.values())).shape[0]
            total_tried += nr_points
            nr_try += 1

            # break when finished sampling
            if total_sampled >= nr_points:
                for key, value in invar.items():
                    invar[key] = value[:nr_points]
                for key, value in params.items():
                    params[key] = value[:nr_points]
                break

            # check if couldn't sample
            if nr_try > 1000 and total_sampled < 1:
                raise Exception("Unable to sample curve")

        return invar, params
