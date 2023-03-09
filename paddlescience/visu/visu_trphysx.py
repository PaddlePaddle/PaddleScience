# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refer from: 
https://github.com/zabaras/transformer-physx/tree/main/trphysx/viz
"""

import os
import numpy as np
from abc import abstractmethod
from typing import Optional
import matplotlib
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase

import paddle

Tensor = paddle.Tensor


class Viz(object):
    """Parent class for visualization

    Args:
        plot_dir (str, optional): Directory to save visualizations in. Defaults to None.
    """

    def __init__(self, plot_dir: str=None) -> None:
        """Constructor method
        """
        super().__init__()
        self.plot_dir = plot_dir

    @abstractmethod
    def plotPrediction(self,
                       y_pred: Tensor,
                       y_target: Tensor,
                       plot_dir: str=None,
                       **kwargs) -> None:
        """Plots model prediction and target values

        Args:
            y_pred (Tensor): prediction tensor
            y_target (Tensor): target tensor
            plot_dir (str, optional): Directory to save plot at. Defaults to None.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: If function has not been overridden by a child dataset class.
        """
        raise NotImplementedError(
            "plotPrediction not initialized by child class.")

    @abstractmethod
    def plotEmbeddingPrediction(self,
                                y_pred: Tensor,
                                y_target: Tensor,
                                plot_dir: str=None,
                                **kwargs) -> None:
        """Plots model prediction and target values during the embedding training

        Args:
            y_pred (Tensor): mini-batch of prediction tensor
            y_target (Tensor): mini-batch target tensor
            plot_dir (str, optional): Directory to save plot at. Defaults to None.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: If function has not been overridden by a child dataset class.
        """
        raise NotImplementedError(
            "plotEmbeddingPrediction not initialized by child class.")

    def saveFigure(self,
                   plot_dir: str=None,
                   file_name: str='plot',
                   savepng: bool=True,
                   savepdf: bool=False) -> None:
        """Saves active matplotlib figure to file

        Args:
            plot_dir (str, optional): Directory to save plot at, will use class plot_dir if none provided. Defaults to None.
            file_name (str, optional): File name of the saved figure. Defaults to 'plot'.
            savepng (bool, optional): Save figure in png format. Defaults to True.
            savepdf (bool, optional): Save figure in pdf format. Defaults to False.
        """
        if plot_dir is None:
            plot_dir = self.plot_dir

        assert os.path.isdir(
            plot_dir
        ), 'Provided directory string is not a valid directory: {:s}'.format(
            plot_dir)
        # Create plotting path if it does not exist
        os.makedirs(plot_dir, exist_ok=True)

        if savepng:
            plt.savefig(
                os.path.join(plot_dir, file_name) + ".png",
                bbox_inches='tight')
        if savepdf:
            plt.savefig(
                os.path.join(plot_dir, file_name) + ".pdf",
                bbox_inches='tight')


# Interface to LineCollection:
def _colorline3d(x,
                 y,
                 z,
                 t=None,
                 cmap=plt.get_cmap('viridis'),
                 linewidth=1,
                 alpha=1.0,
                 ax=None):
    """
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    https://stackoverflow.com/questions/52884221/how-to-plot-a-matplotlib-line-plot-using-colormap
    """
    # Default colors equally spaced on [0,1]:
    if t is None:
        t = np.linspace(0.25, 1.0, len(x))
    if ax is None:
        ax = plt.gca()

    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    colors = np.array([cmap(i) for i in t])
    lc = Line3DCollection(
        segments, colors=colors, linewidth=linewidth, alpha=alpha)
    ax.add_collection(lc)
    ax.scatter(
        x, y, z, c=colors, marker='*', alpha=alpha)  #Adding line markers


class HandlerColormap(HandlerBase):
    """Class for creating colormap legend rectangles

    Args:
        cmap (matplotlib.cm): Matplotlib colormap
        num_stripes (int): Number of countour levels (strips) in rectangle
    """

    def __init__(self, cmap: matplotlib.cm, num_stripes: int=8, **kw):
        HandlerBase.__init__(self, **kw)
        self.cmap = cmap
        self.num_stripes = num_stripes

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width,
                       height, fontsize, trans):
        stripes = []
        for i in range(self.num_stripes):
            s = Rectangle(
                [xdescent + i * width / self.num_stripes, ydescent],
                width / self.num_stripes,
                height,
                fc=self.cmap((2 * i + 1) / (2 * self.num_stripes)),
                transform=trans)
            stripes.append(s)
        return stripes


class LorenzViz(Viz):
    """Visualization class for Lorenz ODE

    Args:
        plot_dir (str, optional): Directory to save visualizations in. Defaults to None.
    """

    def __init__(self, plot_dir: str=None) -> None:
        super().__init__(plot_dir=plot_dir)

    def plotPrediction(self,
                       y_pred: Tensor,
                       y_target: Tensor,
                       plot_dir: str=None,
                       epoch: int=None,
                       pid: int=0) -> None:
        """Plots a 3D line of a single Lorenz prediction

        Args:
            y_pred (Tensor): [T, 3] Prediction tensor.
            y_target (Tensor): [T, 3] Target tensor.
            plot_dir (str, optional): Directory to save figure, overrides plot_dir one if provided. Defaults to None.
            epoch (int, optional): Current epoch, used for file name. Defaults to None.
            pid (int, optional): Optional plotting id for indexing file name manually. Defaults to 0.
        """
        # Convert to numpy array
        y_pred = y_pred.detach().cpu().numpy()
        y_target = y_target.detach().cpu().numpy()

        plt.close('all')
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rcParams['figure.dpi'] = 300
        # rc('text', usetex=True)
        # Set up figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        cmaps = [plt.get_cmap("Reds"), plt.get_cmap("Blues")]
        _colorline3d(
            y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], cmap=cmaps[0], ax=ax)
        _colorline3d(
            y_target[:, 0],
            y_target[:, 1],
            y_target[:, 2],
            cmap=cmaps[1],
            ax=ax)

        ax.set_xlim([-20, 20])
        ax.set_ylim([-20, 20])
        ax.set_zlim([10, 50])

        cmap_handles = [Rectangle((0, 0), 1, 1) for _ in cmaps]
        handler_map = dict(
            zip(cmap_handles,
                [HandlerColormap(
                    cm, num_stripes=8) for cm in cmaps]))
        # Create custom legend with color map rectangels
        ax.legend(
            handles=cmap_handles,
            labels=['Prediction', 'Target'],
            handler_map=handler_map,
            loc='upper right',
            framealpha=0.95)

        if (not epoch is None):
            file_name = 'lorenzPred{:d}_{:d}'.format(pid, epoch)
        else:
            file_name = 'lorenzPred{:d}'.format(pid)

        self.saveFigure(plot_dir, file_name)

    def plotMultiPrediction(
            self,
            y_pred: Tensor,
            y_target: Tensor,
            plot_dir: str=None,
            epoch: int=None,
            pid: int=0,
            nplots: int=2, ) -> None:
        """Plots the 3D lines of multiple Lorenz predictions

        Args:
            y_pred (Tensor): [B, T, 3] Prediction tensor.
            y_target (Tensor): [B, T, 3] Target tensor.
            plot_dir (str, optional): Directory to save figure, overrides plot_dir one if provided. Defaults to None.
            epoch (int, optional): Current epoch, used for file name. Defaults to None.
            pid (int, optional): Optional plotting id for indexing file name manually. Defaults to 0.
            nplots (int, optional): Number of cases to plot. Defaults to 2.
        """
        assert y_pred.size(
            0
        ) >= nplots, 'Number of provided predictions is less than the requested number of subplots'
        assert y_target.size(
            0
        ) >= nplots, 'Number of provided targets is less than the requested number of subplots'
        # Convert to numpy array
        y_pred = y_pred.detach().cpu().numpy()
        y_target = y_target.detach().cpu().numpy()

        plt.close('all')
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rcParams['figure.dpi'] = 300
        # rc('text', usetex=True)
        # Set up figure
        fig, ax = plt.subplots(
            1,
            nplots,
            figsize=(6 * nplots, 6),
            subplot_kw={'projection': '3d'})
        plt.subplots_adjust(wspace=0.025)

        cmaps = [plt.get_cmap("Reds"), plt.get_cmap("Blues")]
        for i in range(nplots):
            _colorline3d(
                y_pred[i, :, 0],
                y_pred[i, :, 1],
                y_pred[i, :, 2],
                cmap=cmaps[0],
                ax=ax[i],
                alpha=0.6)
            _colorline3d(
                y_target[i, :, 0],
                y_target[i, :, 1],
                y_target[i, :, 2],
                cmap=cmaps[1],
                ax=ax[i],
                alpha=0.6)

            ax[i].set_xlim([-20, 20])
            ax[i].set_ylim([-20, 20])
            ax[i].set_zlim([10, 50])

            ax[i].set_xlabel('x', fontsize=14)
            ax[i].set_ylabel('y', fontsize=14)
            ax[i].set_zlabel('z', fontsize=14)
        cmap_handles = [Rectangle((0, 0), 1, 1) for _ in cmaps]
        handler_map = dict(
            zip(cmap_handles,
                [HandlerColormap(
                    cm, num_stripes=10) for cm in cmaps]))
        # Create custom legend with color map rectangels
        ax[-1].legend(
            handles=cmap_handles,
            labels=['Prediction', 'Target'],
            handler_map=handler_map,
            loc='upper right',
            framealpha=0.95)

        if epoch is not None:
            file_name = 'lorenzMultiPred{:d}_{:d}'.format(pid, epoch)
        else:
            file_name = 'lorenzMultiPred{:d}'.format(pid)

        self.saveFigure(plot_dir, file_name)

    def plotPredictionScatter(self,
                              y_pred: Tensor,
                              plot_dir: str=None,
                              epoch: int=None,
                              pid: int=0) -> None:
        """Plots scatter plots of a Lorenz prediction contoured based on distance from the basins.
        This will only contour correctly for the parameters s=10, r=28, b=2.667

        Args:
            y_pred (Tensor): [T, 3] Prediction tensor.
            plot_dir (str, optional): Directory to save figure, overrides plot_dir one if provided. Defaults to None.
            epoch (int, optional): Current epoch, used for file name. Defaults to None.
            pid (int, optional): Optional plotting id for indexing file name manually. Defaults to 0.
        """
        # Convert to numpy array
        y_pred = y_pred.detach().cpu().numpy()

        plt.close('all')
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rcParams['figure.dpi'] = 300
        # rc('text', usetex=True)
        # Set up figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        cmap = plt.get_cmap("plasma")
        # Lorenz attraction centers
        s = 10
        r = 28
        b = 2.667
        cp0 = np.array([np.sqrt(b * (r - 1)), np.sqrt(b * (r - 1)), r - 1])
        cp1 = np.array([-np.sqrt(b * (r - 1)), -np.sqrt(b * (r - 1)), r - 1])

        c = np.minimum(
            np.sqrt((y_pred[:, 0] - cp0[0])**2 + (y_pred[:, 1] - cp0[1])**2 + (
                y_pred[:, 2] - cp0[2])**2),
            np.sqrt((y_pred[:, 0] - cp1[0])**2 + (y_pred[:, 1] - cp1[1])**2 + (
                y_pred[:, 2] - cp1[2])**2))
        c = np.maximum(0, 1 - c / 25)

        ax.set_xlim([-20, 20])
        ax.set_ylim([-20, 20])
        ax.set_zlim([10, 50])

        ax.scatter(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], c=c)

        if (not epoch is None):
            file_name = 'lorenzScatter{:d}_{:d}'.format(pid, epoch)
        else:
            file_name = 'lorenzScatter{:d}'.format(pid)

        self.saveFigure(plot_dir, file_name)


class CylinderViz(Viz):
    """Visualization class for flow around a cylinder

    Args:
        plot_dir (str, optional): Directory to save visualizations in. Defaults to None.
    """

    def __init__(self, plot_dir: str=None) -> None:
        """Constructor method
        """
        super().__init__(plot_dir=plot_dir)

    def _createColorBarVertical(self,
                                fig,
                                ax0,
                                c_min,
                                c_max,
                                label_format="{:02.2f}",
                                cmap='viridis'):
        """Util method for plotting a colorbar next to a subplot
        """
        p0 = ax0[0, -1].get_position().get_points().flatten()
        p1 = ax0[1, -1].get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p1[2] + 0.005, p1[1], 0.0075, p0[3] - p1[1]])
        # ax_cbar = fig.add_axes([p0[0], p0[1]-0.075, p0[2]-p0[0], 0.02])
        ticks = np.linspace(0, 1, 5)
        tickLabels = np.linspace(c_min, c_max, 5)
        tickLabels = [label_format.format(t0) for t0 in tickLabels]
        cbar = mpl.colorbar.ColorbarBase(
            ax_cbar,
            cmap=plt.get_cmap(cmap),
            orientation='vertical',
            ticks=ticks)
        cbar.set_ticklabels(tickLabels)

    def plotPrediction(self,
                       y_pred: Tensor,
                       y_target: Tensor,
                       plot_dir: str=None,
                       epoch: int=None,
                       pid: int=0,
                       nsteps: int=10,
                       stride: int=20) -> None:
        """Plots the predicted x-velocity, y-velocity and pressure field contours

        Args:
            y_pred (Tensor): [T, 3, H, W] Prediction tensor.
            y_target (Tensor): [T, 3, H, W] Target tensor.
            plot_dir (str, optional): Directory to save figure, overrides plot_dir one if provided. Defaults to None.
            epoch (int, optional): Current epoch, used for file name. Defaults to None.
            pid (int, optional): Optional plotting id for indexing file name manually. Defaults to 0.
            nsteps (int, optional): Number of timesteps to plot. Defaults to 10.
            stride (int, optional): Number of timesteps in between plots. Defaults to 10.
        """
        if plot_dir is None:
            plot_dir = self.plot_dir
        # Convert to numpy array
        y_pred = y_pred.detach().cpu().numpy()
        y_target = y_target.detach().cpu().numpy()

        plt.close('all')
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rcParams['figure.dpi'] = 300
        mpl.rcParams['xtick.labelsize'] = 2
        mpl.rcParams['ytick.labelsize'] = 2
        # rc('text', usetex=True)

        # Set up figure
        cmap0 = 'inferno'
        for i, field in enumerate(['ux', 'uy', 'p']):
            plt.close("all")

            # fig, ax = plt.subplots(2+yPred0.size(0), yPred0.size(2), figsize=(2*yPred0.size(1), 3+3*yPred0.size(0)))
            fig, ax = plt.subplots(2, nsteps, figsize=(2.1 * nsteps, 2.25))
            fig.subplots_adjust(wspace=0.25)

            c_max = max([np.amax(y_target[:, i, :, :])])
            c_min = min([np.amin(y_target[:, i, :, :])])
            for t0 in range(nsteps):
                # Plot target
                ax[0, t0].imshow(
                    y_target[t0 * stride, i, :, :],
                    extent=[-2, 14, -4, 4],
                    cmap=cmap0,
                    origin='lower',
                    vmax=c_max,
                    vmin=c_min)
                # Plot sampled predictions
                pcm = ax[1, t0].imshow(
                    y_pred[t0 * stride, i, :, :],
                    extent=[-2, 14, -4, 4],
                    cmap=cmap0,
                    origin='lower',
                    vmax=c_max,
                    vmin=c_min)
                # fig.colorbar(pcm, ax=ax[1, t0], shrink=0.6)

                ax[0, t0].set_xticks(np.linspace(-2, 14, 9))
                ax[0, t0].set_yticks(np.linspace(-4, 4, 5))
                ax[1, t0].set_xticks(np.linspace(-2, 14, 9))
                ax[1, t0].set_yticks(np.linspace(-4, 4, 5))

                for tick in ax[0, t0].xaxis.get_major_ticks():
                    tick.label.set_fontsize(5)
                for tick in ax[0, t0].yaxis.get_major_ticks():
                    tick.label.set_fontsize(5)
                for tick in ax[1, t0].xaxis.get_major_ticks():
                    tick.label.set_fontsize(5)
                for tick in ax[1, t0].yaxis.get_major_ticks():
                    tick.label.set_fontsize(5)

                ax[0, t0].set_title('t={:d}'.format(t0 * stride), fontsize=8)

            self._createColorBarVertical(fig, ax, c_min, c_max, cmap=cmap0)
            ax[0, 0].set_ylabel('Target', fontsize=8)
            ax[1, 0].set_ylabel('Prediction', fontsize=8)

            if (not epoch is None):
                file_name = 'cylinder{:s}Pred{:d}_{:d}'.format(field, pid,
                                                               epoch)
            else:
                file_name = 'cylinder{:s}Pred{:d}'.format(field, pid)
            self.saveFigure(plot_dir, file_name)

    def plotPredictionVorticity(self,
                                y_pred: paddle.Tensor,
                                y_target: paddle.Tensor,
                                plot_dir: str=None,
                                epoch: int=None,
                                pid: int=0,
                                nsteps: int=10,
                                stride: int=10) -> None:
        """Plots vorticity contours of flow around a cylinder at several time-steps. Vorticity gradients
        are calculated using standard smoothed central finite difference.

        Args:
            y_pred (Tensor): [T, 3, H, W] Prediction tensor.
            y_target (Tensor): [T, 3, H, W] Target tensor.
            plot_dir (str, optional): Directory to save figure, overrides class plot_dir if provided. Defaults to None.
            epoch (int, optional): Current epoch, used for file name. Defaults to None.
            pid (int, optional): Optional plotting id for indexing file name manually. Defaults to 0.
            nsteps (int, optional): Number of timesteps to plot. Defaults to 10.
            stride (int, optional): Number of timesteps in between plots. Defaults to 5.
        """

        @paddle.no_grad()
        def xGrad(u, dx=1, padding=(1, 1, 1, 1)):
            WEIGHT_H = paddle.to_tensor(
                [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                dtype=paddle.float32) / 8.
            ux = F.conv2d(
                F.pad(u, padding, mode='replicate'),
                WEIGHT_H,
                stride=1,
                padding=0,
                bias=None) / (dx)
            return ux

        @paddle.no_grad()
        def yGrad(u, dy=1, padding=(1, 1, 1, 1)):
            WEIGHT_V = paddle.to_tensor(
                [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                dtype=paddle.float32).transpose([0, 1, 3, 2]) / 8.
            uy = F.conv2d(
                F.pad(u, padding, mode='replicate'),
                WEIGHT_V,
                stride=1,
                padding=0,
                bias=None) / (dy)
            return uy

        if plot_dir is None:
            plot_dir = self.plot_dir
        # Convert to numpy array
        dx = 6. / 64
        dy = 6. / 64

        vortPred = xGrad(
            y_pred[:, 1].unsqueeze(1), dx=dx) - yGrad(
                y_pred[:, 0].unsqueeze(1), dy=dy)
        vortTarget = xGrad(
            y_target[:, 1].unsqueeze(1), dx=dx) - yGrad(
                y_target[:, 0].unsqueeze(1), dy=dy)

        vortPred = vortPred.reshape(y_pred[:, 0].shape).detach().cpu().numpy()
        vortTarget = vortTarget.reshape(y_target[:, 0].shape).detach().cpu(
        ).numpy()

        plt.close('all')
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rcParams['figure.dpi'] = 200
        mpl.rcParams['xtick.labelsize'] = 2
        mpl.rcParams['ytick.labelsize'] = 2
        # rc('text', usetex=True)

        # Set up figure
        cmap0 = 'seismic'
        # fig, ax = plt.subplots(2+yPred0.size(0), yPred0.size(2), figsize=(2*yPred0.size(1), 3+3*yPred0.size(0)))
        fig, ax = plt.subplots(2, nsteps, figsize=(2 * nsteps, 2.25))
        fig.subplots_adjust(wspace=0.25)

        c_max = max([np.amax(vortTarget[:, :, :]) - 4])
        c_min = min([np.amin(vortTarget[:, :, :]) + 4])
        c_max = 7
        c_min = -7

        for t0 in range(nsteps):
            # Plot target
            ax[0, t0].imshow(
                vortTarget[t0 * stride, :, :],
                extent=[-2, 14, -4, 4],
                cmap=cmap0,
                origin='lower',
                vmax=c_max,
                vmin=c_min)
            # Plot sampled predictions
            pcm = ax[1, t0].imshow(
                vortPred[t0 * stride, :, :],
                extent=[-2, 14, -4, 4],
                cmap=cmap0,
                origin='lower',
                vmax=c_max,
                vmin=c_min)
            # fig.colorbar(pcm, ax=ax[1, t0], shrink=0.6)
            ax[0, t0].set_xticks(np.linspace(-2, 14, 9))
            ax[0, t0].set_yticks(np.linspace(-4, 4, 5))
            ax[1, t0].set_xticks(np.linspace(-2, 14, 9))
            ax[1, t0].set_yticks(np.linspace(-4, 4, 5))

            for tick in ax[0, t0].xaxis.get_major_ticks():
                tick.label.set_fontsize(5)
            for tick in ax[0, t0].yaxis.get_major_ticks():
                tick.label.set_fontsize(5)
            for tick in ax[1, t0].xaxis.get_major_ticks():
                tick.label.set_fontsize(5)
            for tick in ax[1, t0].yaxis.get_major_ticks():
                tick.label.set_fontsize(5)

            ax[0, t0].set_title('t={:d}'.format(t0 * stride), fontsize=8)

        ax[0, 0].set_ylabel('Target', fontsize=8)
        ax[1, 0].set_ylabel('Prediction', fontsize=8)

        if not epoch is None:
            file_name = 'cylinderVortPred{:d}_{:d}'.format(pid, epoch)
        else:
            file_name = 'cylinderVortPred{:d}'.format(pid)

        self.saveFigure(plot_dir, file_name)

    def plotEmbeddingPrediction(self,
                                y_pred: Tensor,
                                y_target: Tensor,
                                plot_dir: str=None,
                                epoch: int=None,
                                bidx: int=None,
                                tidx: int=None,
                                pid: int=0) -> None:
        """Plots the predicted x-velocity, y-velocity and pressure field contours

        Args:
            y_pred (Tensor): [B, T, 3, H, W] Prediction tensor.
            y_target (Tensor): [B, T, 3, H, W] Target tensor.
            plot_dir (str, optional): Directory to save figure, overrides plot_dir one if provided. Defaults to None.
            epoch (int, optional): Current epoch, used for file name. Defaults to None.
            bidx (int, optional): Batch index to plot. Defaults to None (plot random example in batch).
            tidx (int, optional): Timestep index to plot. Defaults to None (plot random time-step).
            pid (int, optional): Optional plotting id for indexing file name manually. Defaults to 0.
        """

        if plot_dir is None:
            plot_dir = self.plot_dir
        # Convert to numpy array
        if bidx is None:
            bidx = np.random.randint(0, y_pred.shape[0])
        if tidx is None:
            tidx = np.random.randint(0, y_pred.shape[1])
        y_pred = y_pred[bidx, tidx].detach().cpu().numpy()
        y_target = y_target[bidx, tidx].detach().cpu().numpy()
        y_error = np.power(y_pred - y_target, 2)

        plt.close('all')
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rcParams['figure.dpi'] = 300
        mpl.rcParams['xtick.labelsize'] = 2
        mpl.rcParams['ytick.labelsize'] = 2
        # rc('text', usetex=True)

        # Set up figure
        cmap0 = 'viridis'
        cmap1 = 'inferno'
        # fig, ax = plt.subplots(2+yPred0.size(0), yPred0.size(2), figsize=(2*yPred0.size(1), 3+3*yPred0.size(0)))
        fig, ax = plt.subplots(3, 3, figsize=(2.1 * 3, 2.25))
        fig.subplots_adjust(wspace=0.1)

        for i, field in enumerate(['ux', 'uy', 'p']):
            c_max = max([np.amax(y_target[i, :, :])])
            c_min = min([np.amin(y_target[i, :, :])])

            ax[0, i].imshow(
                y_target[i, :, :],
                extent=[-2, 14, -4, 4],
                cmap=cmap0,
                origin='lower',
                vmax=c_max,
                vmin=c_min)

            ax[1, i].imshow(
                y_pred[i, :, :],
                extent=[-2, 14, -4, 4],
                cmap=cmap0,
                origin='lower',
                vmax=c_max,
                vmin=c_min)

            ax[2, i].imshow(
                y_error[i, :, :],
                extent=[-2, 14, -4, 4],
                cmap=cmap1,
                origin='lower')

            for j in range(3):
                ax[j, i].set_yticks(np.linspace(-4, 4, 5))

                for tick in ax[j, i].yaxis.get_major_ticks():
                    tick.label.set_fontsize(5)

            ax[2, i].set_xticks(np.linspace(-2, 14, 9))
            for tick in ax[2, i].xaxis.get_major_ticks():
                tick.label.set_fontsize(5)

            ax[0, i].set_title(f'{field}', fontsize=8)

        ax[0, 0].set_ylabel('Target', fontsize=8)
        ax[1, 0].set_ylabel('Prediction', fontsize=8)
        ax[2, 0].set_ylabel('Error', fontsize=8)

        if (not epoch is None):
            file_name = 'embeddingPred{:d}_{:d}'.format(pid, epoch)
        else:
            file_name = 'embeddingPred{:d}'.format(pid)
        self.saveFigure(plot_dir, file_name)


class RosslerViz(Viz):
    """Visualization class for Rosler ODE

    Args:
        plot_dir (str, optional): Directory to save visualizations in. Defaults to None.
    """

    def __init__(self, plot_dir: str=None) -> None:
        super().__init__(plot_dir=plot_dir)

    def plotPrediction(self,
                       y_pred: Tensor,
                       y_target: Tensor,
                       plot_dir: str=None,
                       epoch: int=None,
                       pid: int=0,
                       nsteps: int=256) -> None:
        """Plots a 3D line of a single Rossler prediction

        Args:
            y_pred (Tensor): [T, 3] Prediction tensor.
            y_target (Tensor): [T, 3] Target tensor.
            plot_dir (str, optional): Directory to save figure, overrides plot_dir one if provided. Defaults to None.
            epoch (int, optional): Current epoch, used for file name. Defaults to None.
            pid (int, optional): Optional plotting id for indexing file name manually. Defaults to 0.
            nsteps (int, optional): Number of steps to plot. Defaults to 256.
        """
        # Convert to numpy array
        y_pred = y_pred[:nsteps].detach().cpu().numpy()
        y_target = y_target[:nsteps].detach().cpu().numpy()

        plt.close('all')
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rcParams['figure.dpi'] = 300
        # rc('text', usetex=True)
        # Set up figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        cmaps = [plt.get_cmap("Reds"), plt.get_cmap("Blues")]
        _colorline3d(
            y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], cmap=cmaps[0], ax=ax)
        _colorline3d(
            y_target[:, 0],
            y_target[:, 1],
            y_target[:, 2],
            cmap=cmaps[1],
            ax=ax)

        ax.set_xlim([np.amin(y_target[:, 0]) - 5, np.amax(y_target[:, 0]) + 5])
        ax.set_ylim([np.amin(y_target[:, 1]) - 5, np.amax(y_target[:, 1]) + 5])
        ax.set_zlim([np.amin(y_target[:, 2]) - 5, np.amax(y_target[:, 2]) + 5])

        cmap_handles = [Rectangle((0, 0), 1, 1) for _ in cmaps]
        handler_map = dict(
            zip(cmap_handles,
                [HandlerColormap(
                    cm, num_stripes=8) for cm in cmaps]))
        # Create custom legend with color map rectangels
        ax.legend(
            handles=cmap_handles,
            labels=['Prediction', 'Target'],
            handler_map=handler_map,
            loc='upper right',
            framealpha=0.95)

        if (not epoch is None):
            file_name = 'rosslerPred{:d}_{:d}'.format(pid, epoch)
        else:
            file_name = 'rosslerPred{:d}'.format(pid)

        self.saveFigure(plot_dir, file_name)

    def plotMultiPrediction(self,
                            y_pred: Tensor,
                            y_target: Tensor,
                            plot_dir: str=None,
                            epoch: int=None,
                            pid: int=0,
                            nplots: int=2) -> None:
        """Plots the 3D lines of multiple Lorenz predictions

        Args:
            y_pred (Tensor): [T, 3] Prediction tensor.
            y_target (Tensor): [T, 3] Target tensor.
            plot_dir (str, optional): Directory to save figure, overrides plot_dir one if provided. Defaults to None.
            epoch (int, optional): Current epoch, used for file name. Defaults to None.
            pid (int, optional): Optional plotting id for indexing file name manually, Defaults to 0.
            nplots (int, optional): Number of cases to plot, Defaults to 2.
        """
        assert y_pred.size(
            0
        ) >= nplots, 'Number of provided predictions is less than the requested number of subplots'
        assert y_target.size(
            0
        ) >= nplots, 'Number of provided targets is less than the requested number of subplots'
        # Convert to numpy array
        y_pred = y_pred.detach().cpu().numpy()
        y_target = y_target.detach().cpu().numpy()

        plt.close('all')
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rcParams['figure.dpi'] = 300
        # rc('text', usetex=True)
        # Set up figure
        fig, ax = plt.subplots(
            1,
            nplots,
            figsize=(6 * nplots, 6),
            subplot_kw={'projection': '3d'})
        plt.subplots_adjust(wspace=0.025)

        cmaps = [plt.get_cmap("Reds"), plt.get_cmap("Blues")]
        for i in range(nplots):
            _colorline3d(
                y_pred[i, :, 0],
                y_pred[i, :, 1],
                y_pred[i, :, 2],
                cmap=cmaps[0],
                ax=ax[i],
                alpha=0.6)
            _colorline3d(
                y_target[i, :, 0],
                y_target[i, :, 1],
                y_target[i, :, 2],
                cmap=cmaps[1],
                ax=ax[i],
                alpha=0.6)

            ax[i].set_xlim(
                [np.amin(y_target[:, 0]) - 5, np.amax(y_target[:, 0]) + 5])
            ax[i].set_ylim(
                [np.amin(y_target[:, 1]) - 5, np.amax(y_target[:, 1]) + 5])
            ax[i].set_zlim(
                [np.amin(y_target[:, 2]) - 5, np.amax(y_target[:, 2]) + 5])

            ax[i].set_xlabel('x', fontsize=14)
            ax[i].set_ylabel('y', fontsize=14)
            ax[i].set_zlabel('z', fontsize=14)
        cmap_handles = [Rectangle((0, 0), 1, 1) for _ in cmaps]
        handler_map = dict(
            zip(cmap_handles,
                [HandlerColormap(
                    cm, num_stripes=10) for cm in cmaps]))

        # Create custom legend with color map rectangels
        ax[-1].legend(
            handles=cmap_handles,
            labels=['Prediction', 'Target'],
            handler_map=handler_map,
            loc='upper right',
            framealpha=0.95)

        if epoch is not None:
            file_name = 'rosslerMultiPred{:d}_{:d}'.format(pid, epoch)
        else:
            file_name = 'rosslerMultiPred{:d}'.format(pid)

        self.saveFigure(plot_dir, file_name)
