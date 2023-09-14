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

from collections import OrderedDict
from typing import Callable

import paddle
from paddle.distribution import Normal
from paddle.distribution import Uniform

from ppsci.utils import set_random_seed


class EnableGradient:
    """
    This class is for enabling a dict of tensor for autodiff
    """

    def __init__(self, tensor_dict: OrderedDict):
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

    Examples:
        >>> import paddle
        >>> from ppsci.probability.hmc import HamiltonianMonteCarlo
        >>> def log_posterior(**kwargs):
        >>>    dist = paddle.distribution.Normal(true_mean, true_std)
        >>>    return dist.log_prob(kwargs['x'])
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
        self.dist = distribution_fn
        self.steps = int(path_len / step_size)
        self.step_size = step_size
        self.path_len = path_len
        self.num_warmup_steps = num_warmup_steps
        set_random_seed(random_seed)
        self._rv_unif = Uniform(0, 1)

    def sample(self, last_position):
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

    def run_chain(self, epochs, initial_position):
        sampling_result = {}
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

    def _potential_energy_gradient(self, pos):
        """
        Calculate the gradient of potential energy
        """
        grads = {}
        with EnableGradient(pos):
            (-self.dist(**pos)).backward()
            for k, v in pos.items():
                grads[k] = v.grad.detach()
        return grads

    def _k_energy_fn(self, r):
        energy = 0.0
        for v in r.values():
            energy = energy + v.dot(v)
        return 0.5 * energy

    def _sample_r(self, params_dict):  # sample r for params
        r = {}
        for k, v in params_dict.items():
            rv_r = Normal(paddle.zeros_like(v), paddle.ones_like(v))
            r[k] = rv_r.sample([1])
            if not (v.shape == [] or v.shape == 1):
                r[k] = r[k].squeeze()
        return r

    def _check_acceptance(self, q0, q1, p0, p1):
        # calculate the Metropolis acceptance probability
        q0_nlp = -self.dist(**q0)
        q1_nlp = -self.dist(**q1)

        p0_nlp = self._k_energy_fn(p0)
        p1_nlp = self._k_energy_fn(p1)

        acceptance = paddle.minimum(
            paddle.to_tensor(1.0), paddle.exp((q0_nlp + p0_nlp) - (p1_nlp + q1_nlp))
        )

        # whether accept the proposed state position
        event = self._rv_unif.sample([1]).squeeze()
        return event <= acceptance
