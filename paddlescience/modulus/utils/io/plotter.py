"""
Defines helper Plotter class for adding plots to tensorboard summaries
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

from typing import Dict


class _Plotter:
    def __call__(self, *args):
        raise NotImplementedError

    def _add_figures(self, group, name, results_dir, writer, step, *args):
        "Try to make plots and write them to tensorboard summary"

        # catch exceptions on (possibly user-defined) __call__
        try:
            fs = self(*args)
        except Exception as e:
            print(f"error: {self}.__call__ raised an exception:", str(e))
        else:
            for f, tag in fs:
                f.savefig(
                    results_dir + name + "_" + tag + ".png",
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
                writer.add_figure(group + "/" + name + "/" + tag, f, step, close=True)
            plt.close("all")

    def _interpolate_2D(self, size, invar, *outvars):
        "Interpolate 2D outvar solutions onto a regular mesh"

        assert len(invar) == 2

        # define regular mesh to interpolate onto
        xs = [invar[k][:, 0] for k in invar]
        extent = (xs[0].min(), xs[0].max(), xs[1].min(), xs[1].max())
        xyi = np.meshgrid(
            np.linspace(extent[0], extent[1], size),
            np.linspace(extent[2], extent[3], size),
            indexing="ij",
        )

        # interpolate outvars onto mesh
        outvars_interp = []
        for outvar in outvars:
            outvar_interp = {}
            for k in outvar:
                outvar_interp[k] = scipy.interpolate.griddata(
                    (xs[0], xs[1]), outvar[k][:, 0], tuple(xyi)
                )
            outvars_interp.append(outvar_interp)

        return [extent] + outvars_interp


class ValidatorPlotter(_Plotter):
    "Default plotter class for validator"

    def __call__(self, invar, true_outvar, pred_outvar):
        "Default function for plotting validator data"

        ndim = len(invar)
        if ndim > 2:
            print("Default plotter can only handle <=2 input dimensions, passing")
            return []

        # interpolate 2D data onto grid
        if ndim == 2:
            extent, true_outvar, pred_outvar = self._interpolate_2D(
                100, invar, true_outvar, pred_outvar
            )

        # make plots
        dims = list(invar.keys())
        fs = []
        for k in pred_outvar:
            f = plt.figure(figsize=(3 * 5, 4), dpi=100)
            for i, (o, tag) in enumerate(
                zip(
                    [true_outvar[k], pred_outvar[k], true_outvar[k] - pred_outvar[k]],
                    ["true", "pred", "diff"],
                )
            ):
                plt.subplot(1, 3, 1 + i)
                if ndim == 1:
                    plt.plot(invar[dims[0]][:, 0], o[:, 0])
                    plt.xlabel(dims[0])
                elif ndim == 2:
                    plt.imshow(o.T, origin="lower", extent=extent)
                    plt.xlabel(dims[0])
                    plt.ylabel(dims[1])
                    plt.colorbar()
                plt.title(f"{k}_{tag}")
            plt.tight_layout()
            fs.append((f, k))

        return fs


class InferencerPlotter(_Plotter):
    "Default plotter class for inferencer"

    def __call__(self, invar, outvar):
        "Default function for plotting inferencer data"

        ndim = len(invar)
        if ndim > 2:
            print("Default plotter can only handle <=2 input dimensions, passing")
            return []

        # interpolate 2D data onto grid
        if ndim == 2:
            extent, outvar = self._interpolate_2D(100, invar, outvar)

        # make plots
        dims = list(invar.keys())
        fs = []
        for k in outvar:
            f = plt.figure(figsize=(5, 4), dpi=100)
            if ndim == 1:
                plt.plot(invar[dims[0]][:, 0], outvar[:, 0])
                plt.xlabel(dims[0])
            elif ndim == 2:
                plt.imshow(outvar[k].T, origin="lower", extent=extent)
                plt.xlabel(dims[0])
                plt.ylabel(dims[1])
                plt.colorbar()
            plt.title(k)
            plt.tight_layout()
            fs.append((f, k))

        return fs


class GridValidatorPlotter(_Plotter):
    """Grid validation plotter for structured data"""

    def __init__(self, n_examples: int = 1):
        self.n_examples = n_examples

    def __call__(
        self,
        invar: Dict[str, np.array],
        true_outvar: Dict[str, np.array],
        pred_outvar: Dict[str, np.array],
    ):

        ndim = next(iter(invar.values())).ndim - 2
        if ndim > 3:
            print("Default plotter can only handle <=3 input dimensions, passing")
            return []

        # get difference
        diff_outvar = {}
        for k, v in true_outvar.items():
            diff_outvar[k] = true_outvar[k] - pred_outvar[k]

        fs = []
        for ie in range(self.n_examples):
            f = self._make_plot(ndim, ie, invar, true_outvar, pred_outvar, diff_outvar)
            fs.append((f, f"prediction_{ie}"))
        return fs

    def _make_plot(self, ndim, ie, invar, true_outvar, pred_outvar, diff_outvar):

        # make plot
        nrows = max(len(invar), len(true_outvar))
        f = plt.figure(figsize=(4 * 5, nrows * 4), dpi=100)
        for ic, (d, tag) in enumerate(
            zip(
                [invar, true_outvar, pred_outvar, diff_outvar],
                ["in", "true", "pred", "diff"],
            )
        ):
            for ir, k in enumerate(d):
                plt.subplot2grid((nrows, 4), (ir, ic))
                if ndim == 1:
                    plt.plot(d[k][ie, 0, :])
                elif ndim == 2:
                    plt.imshow(d[k][ie, 0, :, :].T, origin="lower")
                else:
                    z = d[k].shape[-1] // 2  # Z slice
                    plt.imshow(d[k][ie, 0, :, :, z].T, origin="lower")
                plt.title(f"{k}_{tag}")
                plt.colorbar()
        plt.tight_layout()
        return f


class DeepONetValidatorPlotter(_Plotter):
    """DeepONet validation plotter for structured data"""

    def __init__(self, n_examples: int = 1):
        self.n_examples = n_examples

    def __call__(
        self,
        invar: Dict[str, np.array],
        true_outvar: Dict[str, np.array],
        pred_outvar: Dict[str, np.array],
    ):

        ndim = next(iter(invar.values())).shape[-1]
        if ndim > 3:
            print("Default plotter can only handle <=2 input dimensions, passing")
            return []

        # get difference
        diff_outvar = {}
        for k, v in true_outvar.items():
            diff_outvar[k] = true_outvar[k] - pred_outvar[k]

        fs = []
        for ie in range(self.n_examples):
            f = self._make_plot(ndim, ie, invar, true_outvar, pred_outvar, diff_outvar)
            fs.append((f, f"prediction_{ie}"))
        return fs

    def _make_plot(self, ndim, ie, invar, true_outvar, pred_outvar, diff_outvar):
        # make plot
        # invar: input of trunk net. Dim: N*P*ndim
        # outvar: output of DeepONet. Dim: N*P

        nrows = max(len(invar), len(true_outvar))
        f = plt.figure(figsize=(4 * 5, nrows * 4), dpi=100)
        invar_data = next(iter(invar.values()))

        for ic, (d, tag) in enumerate(
            zip(
                [true_outvar, pred_outvar, diff_outvar],
                ["true", "pred", "diff"],
            )
        ):
            for ir, k in enumerate(d):
                plt.subplot2grid((nrows, 4), (ir, ic))
                if ndim == 1:
                    plt.plot(invar_data[ie, :].flatten(), d[k][ie, :])
                elif ndim == 2:
                    plt.scatter(
                        x=invar_data[ie, :, 0],
                        y=invar_data[ie, :, 1],
                        c=d[k][ie, :],
                        s=0.5,
                        origin="lower",
                        cmap="jet",
                    )
                    plt.colorbar()
                plt.title(f"{k}_{tag}")
        plt.tight_layout()
        return f
