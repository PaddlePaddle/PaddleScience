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

from typing import Callable
from typing import Dict

import paddle
from paddle import distribution

from ppsci import utils


class EnableGradient:
    """
    This class is for enabling a dict of tensor for autodiff
    """

    def __init__(self, tensor_dict: Dict[str, paddle.Tensor]):
        self.tensor_dict = tensor_dict

    def __enter__(self):
        for t in self.tensor_dict.values():
            t.stop_gradient = False
            t.clear_grad()

    def __exit__(self, exec_type, exec_val, exec_tb):
        for t in self.tensor_dict.values():
            t.stop_gradient = True


class HamiltonianMonteCarlo:
    """
    Using the HamiltonianMonteCarlo(HMC) to sample from the desired probability distribution. The HMC combine the Hamiltonian Dynamics and Markov Chain Monte Carlo sampling algorithm which is a more efficient way compared to the Metropolis Hasting (MH) method.

    Args:
        distribution_fn (paddle.distribution.Distribution): The Log (Posterior) Distribution function that of the parameters needed to be sampled.
        path_len (float): The total path length.
        step_size (float): Every step size.
        num_warmup_steps (int): The number of warm-up steps for the MCMC run.
        random_seed (int): Random seed number.

    Examples:
        >>> import paddle
        >>> from ppsci.probability.hmc import HamiltonianMonteCarlo
        >>> def log_posterior(**kwargs):
        ...     dist = paddle.distribution.Normal(loc=0, scale=1)
        ...     return dist.log_prob(kwargs['x'])
        >>> HMC = HamiltonianMonteCarlo(log_posterior, path_len=1.5, step_size=0.25)
        >>> trial = HMC.run_chain(1000, {'x': paddle.to_tensor(0.0)})
    """

    def __init__(
        self,
        distribution_fn: Callable,
        path_len: float = 1.0,
        step_size: float = 0.25,
        num_warmup_steps: int = 0,
        random_seed: int = 1024,
    ):
        self.distribution_fn = distribution_fn
        self.steps = int(path_len / step_size)
        self.step_size = step_size
        self.path_len = path_len
        self.num_warmup_steps = num_warmup_steps
        utils.set_random_seed(random_seed)
        self._rv_unif = distribution.Uniform(0, 1)

    def sample(
        self, last_position: Dict[str, paddle.Tensor]
    ) -> Dict[str, paddle.Tensor]:
        """
        Single step for sample
        """
        q0 = q1 = last_position
        p0 = p1 = self._sample_r(q0)

        for s in range(self.steps):
            grad = self._potential_energy_gradient(q1)
            for site_name in p1.keys():
                p1[site_name] -= self.step_size * grad[site_name] / 2
            for site_name in q1.keys():
                q1[site_name] += self.step_size * p1[site_name]

            grad = self._potential_energy_gradient(q1)
            for site_name in p1.keys():
                p1[site_name] -= self.step_size * grad[site_name] / 2

        # set the next state in the Markov chain
        return q1 if self._check_acceptance(q0, q1, p0, p1) else q0

    def run_chain(
        self, epochs: int, initial_position: Dict[str, paddle.Tensor]
    ) -> Dict[str, paddle.Tensor]:
        sampling_result: Dict[str, paddle.Tensor] = {}
        for k in initial_position.keys():
            sampling_result[k] = []
        pos = initial_position

        # warmup
        for _ in range(self.num_warmup_steps):
            pos = self.sample(pos)

        # begin collecting sampling result
        for e in range(epochs):
            pos = self.sample(pos)
            for k in pos.keys():
                sampling_result[k].append(pos[k].numpy())

        for k in initial_position.keys():
            sampling_result[k] = paddle.to_tensor(sampling_result[k])

        return sampling_result

    def _potential_energy_gradient(
        self, pos: Dict[str, paddle.Tensor]
    ) -> Dict[str, paddle.Tensor]:
        """
        Calculate the gradient of potential energy
        """
        grads = {}
        with EnableGradient(pos):
            (-self.distribution_fn(**pos)).backward()
            for k, v in pos.items():
                grads[k] = v.grad.detach()
        return grads

    def _k_energy_fn(self, r: Dict[str, paddle.Tensor]) -> paddle.Tensor:
        energy = 0.0
        for v in r.values():
            energy = energy + v.dot(v)
        return 0.5 * energy

    def _sample_r(
        self, params_dict: Dict[str, paddle.Tensor]
    ) -> Dict[str, paddle.Tensor]:
        # sample r for params
        r = {}
        for k, v in params_dict.items():
            rv_r = distribution.Normal(paddle.zeros_like(v), paddle.ones_like(v))
            r[k] = rv_r.sample([1])
            if not (v.shape == [] or v.shape == 1):
                r[k] = r[k].squeeze()
        return r

    def _check_acceptance(
        self,
        q0: Dict[str, paddle.Tensor],
        q1: Dict[str, paddle.Tensor],
        p0: Dict[str, paddle.Tensor],
        p1: Dict[str, paddle.Tensor],
    ) -> bool:
        # calculate the Metropolis acceptance probability
        energy_current = -self.distribution_fn(**q0) + self._k_energy_fn(p0)
        energy_proposed = -self.distribution_fn(**q1) + self._k_energy_fn(p1)

        acceptance = paddle.minimum(
            paddle.to_tensor(1.0), paddle.exp(energy_current - energy_proposed)
        )

        # whether accept the proposed state position
        event = self._rv_unif.sample([])
        return event <= acceptance
