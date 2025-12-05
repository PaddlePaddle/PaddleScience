# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib as mpl
import numpy as np
import paddle
from celluloid import Camera
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm


def make_transparent_color(ntimes, fraction):
    """
    Create transparent colors for trajectory visualization.

    This function generates a gradient from white to a specified color,
    with transparency increasing from 0 to 1. Used for drawing particle
    trajectories where old positions are transparent and new positions are opaque.

    Args:
        ntimes (int): Number of time steps, i.e., number of colors to generate.
        fraction (float): Color fraction in range [0,1] for selecting color from spectrum.

    Returns:
        numpy.ndarray: RGBA color array with shape (ntimes, 4).
                      Each row represents the color [R, G, B, Alpha] at a time step.
    """
    rgba = np.ones((ntimes, 4))

    alpha = np.linspace(0, 1, ntimes)[:, np.newaxis]

    color = np.array(mpl.colors.to_rgba(mpl.cm.gist_ncar(fraction)))[np.newaxis, :]

    rgba[:, :] = 1 * (1 - alpha) + color * alpha

    rgba[:, 3] = alpha[:, 0]

    return rgba


def get_potential(sim, sim_obj):
    """
    Get potential energy function based on simulation type.

    This is a factory function that returns the corresponding potential energy
    calculation function based on different physical scenarios.
    Supported scenarios include: gravity, springs, charges, damped systems, etc.

    Args:
        sim (str): Simulation type, options:
                  'r2' - Inverse square law (e.g., universal gravitation)
                  'r1' - Logarithmic potential
                  'spring' - Spring potential
                  'damped' - Damped spring
                  'string' - String potential
                  'string_ball' - String with ball obstacle
                  'charge' - Charge interaction
                  'superposition' - Superposition of gravity and charge
                  'discontinuous' - Piecewise discontinuous potential
        sim_obj (SimulationDataset): Simulation dataset object for obtaining parameters like dimension.

    Returns:
        function: Potential energy function that takes two particle states and returns their potential energy.
    """
    dim = sim_obj._dim

    def potential(x1, x2):
        """
        Calculate potential energy between two particles.

        Particle state format:
            2D: [x, y, vx, vy, charge, mass]
            3D: [x, y, z, vx, vy, vz, charge, mass]

        Args:
            x1 (numpy.ndarray): State vector of the first particle.
            x2 (numpy.ndarray): State vector of the second particle.

        Returns:
            float: Potential energy value between the two particles.
        """
        dist = np.sqrt(np.sum(np.square(x1[:dim] - x2[:dim])))

        min_dist = 1e-2
        bounded_dist = dist + min_dist

        if sim == "r2":
            return -x1[-1] * x2[-1] / bounded_dist

        elif sim == "r1":
            return x1[-1] * x2[-1] * np.log(bounded_dist)

        elif sim in ["spring", "damped"]:
            potential_val = (bounded_dist - 1) ** 2

            if sim == "damped":
                damping = 1
                potential_val += damping * x1[1] * x1[1 + sim_obj._dim] / sim_obj._n
                potential_val += damping * x1[0] * x1[0 + sim_obj._dim] / sim_obj._n
                if sim_obj._dim == 3:
                    potential_val += damping * x1[2] * x1[2 + sim_obj._dim] / sim_obj._n

            return potential_val

        elif sim == "string":

            return (bounded_dist - 1) ** 2 + x1[1] * x1[-1]

        elif sim == "string_ball":
            potential_val = (bounded_dist - 1) ** 2 + x1[1] * x1[-1]

            r = np.sqrt((x1[1] + 15) ** 2 + (x1[0] - 5) ** 2)
            radius = 4.0
            k_repel = 100.0
            potential_val += k_repel / np.maximum(r - radius + 0.5, 0.01)
            return potential_val

        elif sim in ["charge", "superposition"]:

            charge1 = x1[-2]
            charge2 = x2[-2]
            potential_val = charge1 * charge2 / bounded_dist

            if sim in ["superposition"]:

                m1 = x1[-1]
                m2 = x2[-1]
                potential_val += -m1 * m2 / bounded_dist

            return potential_val

        elif sim in ["discontinuous"]:

            m1 = x1[-1]
            m2 = x2[-1]

            pot_a = 0.0
            pot_b = 0.0
            pot_c = (bounded_dist - 1) ** 2
            potential_val = pot_a * (bounded_dist < 1) + (bounded_dist >= 1) * (
                pot_b * (bounded_dist < 2) + pot_c * (bounded_dist >= 2)
            )
            return potential_val

        else:
            raise NotImplementedError("No such simulation " + str(sim))

    return potential


class SimulationDataset(object):
    """
    Physical system simulation dataset class (PaddlePaddle version).

    This class is used to generate and manage simulation data for many-body physical systems.
    It simulates the time evolution of particle systems by numerically solving Newton's
    equations of motion.

    Main features:
        - Generate simulation data for multiple samples
        - Compute system acceleration (using PaddlePaddle automatic differentiation)
        - Visualize particle trajectories
        - Support custom potential energy functions

    Attributes:
        _sim (str): Simulation type.
        _n (int): Number of particles.
        _dim (int): Spatial dimension.
        dt (float): Time step size.
        nt (int): Number of time steps.
        data (ndarray): Simulation data with shape (num_samples, num_timesteps, num_particles, state_dim).
        times (ndarray): Time series.
        G (float): Gravitational constant.
        pairwise (function): Pairwise potential energy function.
    """

    def __init__(
        self, sim="r2", n=5, dim=2, dt=0.01, nt=100, extra_potential=None, **kwargs
    ):
        """
        Initialize simulation dataset.

        Args:
            sim (str, optional): Simulation type. Defaults to 'r2' (inverse square law).
            n (int, optional): Number of particles. Defaults to 5.
            dim (int, optional): Spatial dimension (2 or 3). Defaults to 2.
            dt (float, optional): Time step size. Defaults to 0.01.
            nt (int, optional): Number of time steps to return. Defaults to 100.
            extra_potential (function, optional): Additional potential energy function. Defaults to None.
            **kwargs: Other parameters reserved for extension.
        """
        self._sim = sim
        self._n = n
        self._dim = dim
        self._kwargs = kwargs
        self.dt = dt
        self.nt = nt
        self.data = None

        self.times = np.linspace(0, self.dt * self.nt, num=self.nt)

        self.G = 1
        self.extra_potential = extra_potential

        self.pairwise = get_potential(sim=sim, sim_obj=self)

    def simulate(self, ns, seed=0):
        """
        Generate simulation data.

        This method is the core function of the simulator. For each sample:
        1. Randomly initialize particle positions, velocities, masses, and charges
        2. Solve Newton's equations of motion using Scipy ODE solver
        3. Record the entire time evolution process

        Physical equations:
            d²x/dt² = F/m = -∇U/m
            where U is potential energy and F is force

        Args:
            ns (int): Number of samples to generate.
            seed (int, optional): Seed for numpy random number generator. Defaults to 0.

        Returns:
            None: Results are stored in self.data.
        """

        np.random.seed(seed)

        n = self._n
        dim = self._dim
        sim = self._sim
        params = 2
        total_dim = dim * 2 + params
        times = self.times
        G = self.G

        def compute_pairwise_potential(x1, x2_list):
            """Vectorized calculation of potential energy between one particle and multiple particles"""
            potentials = []
            for x2 in x2_list:
                potentials.append(self.pairwise(x1, x2))
            return np.array(potentials)

        def total_potential(xt):
            """
            Calculate total potential energy of the system (using Numpy).

            Args:
                xt (numpy.ndarray): System state with shape (n, total_dim).

            Returns:
                float: Total potential energy of the system.
            """
            sum_potential = 0.0

            for i in range(n - 1):
                if sim in ["string", "string_ball"]:

                    sum_potential += G * self.pairwise(xt[i], xt[i + 1])
                else:

                    for j in range(i + 1, n):
                        sum_potential += G * self.pairwise(xt[i], xt[j])

            if self.extra_potential is not None:
                for i in range(n):
                    sum_potential += self.extra_potential(xt[i])

            return sum_potential

        def force_paddle(xt):
            """
            Calculate force using PaddlePaddle automatic differentiation.

            This is a key part of the migration:
            JAX: force = -grad(total_potential)(xt)
            Paddle: Use paddle.grad to calculate gradient

            Complete implementation for all potential scenarios.

            Args:
                xt (numpy.ndarray): System state with shape (n, total_dim).

            Returns:
                numpy.ndarray: Force on each particle with shape (n, dim).
            """

            xt_tensor = paddle.to_tensor(xt, dtype="float64", stop_gradient=False)

            positions = xt_tensor[:, :dim]
            positions.stop_gradient = False

            xt_for_potential = paddle.concat([positions, xt_tensor[:, dim:]], axis=1)

            sum_potential = paddle.to_tensor(0.0, dtype="float64")

            for i in range(n - 1):
                if sim in ["string", "string_ball"]:

                    x1 = xt_for_potential[i]
                    x2 = xt_for_potential[i + 1]
                    dist = paddle.sqrt(paddle.sum(paddle.square(x1[:dim] - x2[:dim])))
                    bounded_dist = dist + 1e-2

                    if sim == "string":

                        pot = (bounded_dist - 1) ** 2 + x1[1] * x1[-1]

                    elif sim == "string_ball":

                        pot = (bounded_dist - 1) ** 2 + x1[1] * x1[-1]

                        r = paddle.sqrt((x1[1] + 15) ** 2 + (x1[0] - 5) ** 2)
                        radius = paddle.to_tensor(4.0, dtype="float64")

                        k_repel = 100.0
                        pot = pot + k_repel / paddle.maximum(
                            r - radius + 0.5, paddle.to_tensor(0.01, dtype="float64")
                        )

                    sum_potential = sum_potential + G * pot

                else:

                    for j in range(i + 1, n):
                        x1 = xt_for_potential[i]
                        x2 = xt_for_potential[j]
                        dist = paddle.sqrt(
                            paddle.sum(paddle.square(x1[:dim] - x2[:dim]))
                        )
                        bounded_dist = dist + 1e-2

                        if sim == "r2":

                            pot = -x1[-1] * x2[-1] / bounded_dist

                        elif sim == "r1":

                            pot = x1[-1] * x2[-1] * paddle.log(bounded_dist)

                        elif sim in ["spring", "damped"]:

                            pot = (bounded_dist - 1) ** 2

                            if sim == "damped":
                                damping = paddle.to_tensor(1.0, dtype="float64")

                                pot = pot + damping * x1[1] * x1[1 + dim] / n
                                pot = pot + damping * x1[0] * x1[0 + dim] / n
                                if dim == 3:
                                    pot = pot + damping * x1[2] * x1[2 + dim] / n

                        elif sim in ["charge", "superposition"]:

                            charge1 = x1[-2]
                            charge2 = x2[-2]
                            pot = charge1 * charge2 / bounded_dist

                            if sim == "superposition":

                                m1 = x1[-1]
                                m2 = x2[-1]
                                pot = pot + (-m1 * m2 / bounded_dist)

                        elif sim == "discontinuous":

                            pot_a = paddle.to_tensor(0.0, dtype="float64")
                            pot_b = paddle.to_tensor(0.0, dtype="float64")
                            pot_c = (bounded_dist - 1) ** 2

                            cond1 = paddle.cast(bounded_dist < 1, "float64")
                            cond2 = paddle.cast(
                                bounded_dist >= 1, "float64"
                            ) * paddle.cast(bounded_dist < 2, "float64")
                            cond3 = paddle.cast(bounded_dist >= 2, "float64")

                            pot = pot_a * cond1 + pot_b * cond2 + pot_c * cond3

                        else:

                            pot = (bounded_dist - 1) ** 2

                        sum_potential = sum_potential + G * pot

            if self.extra_potential is not None:

                for i in range(n):

                    extra_pot = self.extra_potential(xt_for_potential[i])
                    if not isinstance(extra_pot, paddle.Tensor):
                        extra_pot = paddle.to_tensor(extra_pot, dtype="float64")
                    sum_potential = sum_potential + extra_pot

            grads = paddle.grad(
                outputs=sum_potential,
                inputs=positions,
                create_graph=False,
                retain_graph=False,
            )[0]

            force = -grads.numpy()

            return force

        def acceleration(xt):
            """
            Calculate acceleration: a = F/m.

            Args:
                xt (numpy.ndarray): System state.

            Returns:
                numpy.ndarray: Acceleration.
            """
            force = force_paddle(xt)
            masses = xt[:, -1:]
            return force / masses

        def odefunc(y, t):
            """
            Ordinary differential equation function (for scipy.odeint).

            Args:
                y (numpy.ndarray): Flattened system state.
                t (float): Time.

            Returns:
                numpy.ndarray: Time derivative of the state.
            """

            y = y.reshape((n, total_dim))

            a = acceleration(y)

            dydt = np.concatenate(
                [y[:, dim : 2 * dim], a, np.zeros((n, params))], axis=1
            )

            return dydt.flatten()

        def make_sim(sample_idx):
            """
            Generate single simulation sample.

            Args:
                sample_idx (int): Sample index (for progress display).

            Returns:
                numpy.ndarray: Simulation trajectory with shape (nt, n, total_dim).
            """
            if sim in ["string", "string_ball"]:

                x0 = np.random.randn(n, total_dim)
                x0[:, -1] = 1.0
                x0[:, 0] = np.arange(n) + x0[:, 0] * 0.5
                x0[:, 2:3] = 0.0
            else:

                x0 = np.random.randn(n, total_dim)
                x0[:, -1] = np.exp(x0[:, -1])

                if sim in ["charge", "superposition"]:
                    x0[:, -2] = np.sign(x0[:, -2])

            x_times = odeint(odefunc, x0.flatten(), times, mxstep=2000).reshape(
                -1, n, total_dim
            )

            return x_times

        data = []
        print(f"Start generating {ns} samples...")
        for i in tqdm(range(ns)):
            data.append(make_sim(i))

        self.data = np.array(data)
        print(f"Generation complete! Data shape: {self.data.shape}")

    def get_acceleration(self):
        """
        Calculate acceleration for all simulation data.

        This method uses PaddlePaddle automatic differentiation to calculate acceleration.
        Uses the same complete potential implementation as in simulate().

        Returns:
            numpy.ndarray: Acceleration data with shape (ns, nt, n, dim).
        """
        if self.data is None:
            raise ValueError("Please call simulate() first to generate data")

        ns, nt, n, total_dim = self.data.shape
        dim = self._dim
        sim = self._sim
        G = self.G

        accelerations = np.zeros((ns, nt, n, dim))

        print("Calculating acceleration...")
        for sample_idx in tqdm(range(ns)):
            for time_idx in range(nt):
                xt = self.data[sample_idx, time_idx]

                xt_tensor = paddle.to_tensor(xt, dtype="float64", stop_gradient=False)
                positions = xt_tensor[:, :dim]
                positions.stop_gradient = False

                xt_for_potential = paddle.concat(
                    [positions, xt_tensor[:, dim:]], axis=1
                )

                sum_potential = paddle.to_tensor(0.0, dtype="float64")

                for i in range(n - 1):
                    if sim in ["string", "string_ball"]:

                        x1 = xt_for_potential[i]
                        x2 = xt_for_potential[i + 1]
                        dist = paddle.sqrt(
                            paddle.sum(paddle.square(x1[:dim] - x2[:dim]))
                        )
                        bounded_dist = dist + 1e-2

                        if sim == "string":
                            pot = (bounded_dist - 1) ** 2 + x1[1] * x1[-1]
                        elif sim == "string_ball":

                            pot = (bounded_dist - 1) ** 2 + x1[1] * x1[-1]
                            r = paddle.sqrt((x1[1] + 15) ** 2 + (x1[0] - 5) ** 2)
                            radius = paddle.to_tensor(4.0, dtype="float64")

                            k_repel = 100.0
                            pot = pot + k_repel / paddle.maximum(
                                r - radius + 0.5,
                                paddle.to_tensor(0.01, dtype="float64"),
                            )

                        sum_potential = sum_potential + G * pot
                    else:

                        for j in range(i + 1, n):
                            x1 = xt_for_potential[i]
                            x2 = xt_for_potential[j]
                            dist = paddle.sqrt(
                                paddle.sum(paddle.square(x1[:dim] - x2[:dim]))
                            )
                            bounded_dist = dist + 1e-2

                            if sim == "r2":
                                pot = -x1[-1] * x2[-1] / bounded_dist
                            elif sim == "r1":
                                pot = x1[-1] * x2[-1] * paddle.log(bounded_dist)
                            elif sim in ["spring", "damped"]:
                                pot = (bounded_dist - 1) ** 2

                                if sim == "damped":
                                    damping = paddle.to_tensor(1.0, dtype="float64")
                                    pot = pot + damping * x1[1] * x1[1 + dim] / n
                                    pot = pot + damping * x1[0] * x1[0 + dim] / n
                                    if dim == 3:
                                        pot = pot + damping * x1[2] * x1[2 + dim] / n

                            elif sim in ["charge", "superposition"]:
                                charge1 = x1[-2]
                                charge2 = x2[-2]
                                pot = charge1 * charge2 / bounded_dist
                                if sim == "superposition":
                                    m1 = x1[-1]
                                    m2 = x2[-1]
                                    pot = pot + (-m1 * m2 / bounded_dist)

                            elif sim == "discontinuous":

                                pot_a = paddle.to_tensor(0.0, dtype="float64")
                                pot_b = paddle.to_tensor(0.0, dtype="float64")
                                pot_c = (bounded_dist - 1) ** 2

                                cond1 = paddle.cast(bounded_dist < 1, "float64")
                                cond2 = paddle.cast(
                                    bounded_dist >= 1, "float64"
                                ) * paddle.cast(bounded_dist < 2, "float64")
                                cond3 = paddle.cast(bounded_dist >= 2, "float64")

                                pot = pot_a * cond1 + pot_b * cond2 + pot_c * cond3

                            else:
                                pot = (bounded_dist - 1) ** 2

                            sum_potential = sum_potential + G * pot

                if self.extra_potential is not None:
                    for i in range(n):
                        extra_pot = self.extra_potential(xt_for_potential[i])
                        if not isinstance(extra_pot, paddle.Tensor):
                            extra_pot = paddle.to_tensor(extra_pot, dtype="float64")
                        sum_potential = sum_potential + extra_pot

                grads = paddle.grad(
                    outputs=sum_potential,
                    inputs=positions,
                    create_graph=False,
                    retain_graph=False,
                )[0]

                force = -grads.numpy()
                masses = xt[:, -1:]
                accel = force / masses

                accelerations[sample_idx, time_idx] = accel

        return accelerations

    def plot(self, i, animate=False, plot_size=True, s_size=1):
        """
        Visualize simulation results.

        Args:
            i (int): Sample index.
            animate (bool, optional): Whether to generate animation. Defaults to False.
            plot_size (bool, optional): Whether to adjust point size based on mass. Defaults to True.
            s_size (float, optional): Point size scaling factor. Defaults to 1.

        Returns:
            None (static plot) or HTML (animation).
        """
        if self.data is None:
            raise ValueError("Please call simulate() first to generate data")

        n = self._n
        times = self.times
        x_times = self.data[i]
        sim = self._sim
        masses = x_times[:, :, -1]

        if not animate:

            if sim in ["string", "string_ball"]:
                rgba = make_transparent_color(len(times), 0)
                for idx in range(0, len(times), len(times) // 10):
                    ctimes = x_times[idx]
                    plt.plot(ctimes[:, 0], ctimes[:, 1], color=rgba[idx])
                plt.xlim(-5, 20)
                plt.ylim(-20, 5)
            else:
                for j in range(n):
                    rgba = make_transparent_color(len(times), j / n)
                    if plot_size:
                        plt.scatter(
                            x_times[:, j, 0],
                            x_times[:, j, 1],
                            color=rgba,
                            s=3 * masses[:, j] * s_size,
                        )
                    else:
                        plt.scatter(
                            x_times[:, j, 0], x_times[:, j, 1], color=rgba, s=s_size
                        )
        else:

            if sim in ["string", "string_ball"]:
                raise NotImplementedError(
                    "Animation mode not yet supported for string type"
                )

            fig = plt.figure()
            camera = Camera(fig)
            d_idx = 20

            for t_idx in range(d_idx, len(times), d_idx):
                start = max([0, t_idx - 300])
                ctimes = times[start:t_idx]
                cx_times = x_times[start:t_idx]

                for j in range(n):
                    rgba = make_transparent_color(len(ctimes), j / n)
                    if plot_size:
                        plt.scatter(
                            cx_times[:, j, 0],
                            cx_times[:, j, 1],
                            color=rgba,
                            s=3 * masses[:, j],
                        )
                    else:
                        plt.scatter(
                            cx_times[:, j, 0], cx_times[:, j, 1], color=rgba, s=s_size
                        )

                camera.snap()

            from IPython.display import HTML

            return HTML(camera.animate().to_jshtml())
