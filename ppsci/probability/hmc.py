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

import paddle


class EnableGradient:
    """
    This class is for enabling tensor for autodiff
    """

    def __init__(self, tensor: paddle.Tensor):
        self.tensor = tensor

    def __enter__(self):
        self.tensor.stop_gradient = False

    def __exit__(self, exec_type, exec_val, exec_tb):
        self.tensor.stop_gradient = True
        self.tensor.clear_grad()


class HamiltonianMonteCarlo:
    """
    Using the HamiltonianMonteCarlo(HMC) to sample from the desired probability distribution. The HMC combine the Hamiltonian Dynamics and Markov Chain Monte Carlo sampling algorithm which is a more efficient way compared to the Metropolis Hasting (MH) method.

    Args:
        distribution (paddle.distribution.Distribution): The Distribution that need to be sampled.
        path_len (float): The total path length.
        step_size (float): Every step size.

    Examples:
        >>> import paddle
        >>> from ppsci.probability.hmc import HamiltonianMonteCarlo
        >>> dist = paddle.distribution.Normal(loc=0, scale=1)
        >>> HMC = HamiltonianMonteCarlo(dist, path_len=1.5, step_size=0.25)
        >>> trial = HMC.run_chain(1000, paddle.to_tensor(0.))
    """

    def __init__(self, distribution, path_len=1.0, step_size=0.25):
        self.dist = distribution
        self.steps = int(path_len / step_size)
        self.step_size = step_size
        self.path_len = path_len

    def sample(self, last_position):
        """
        Single step for sample
        """
        q0 = q1 = last_position
        p0 = p1 = paddle.randn([])

        for s in range(self.steps):
            p1 -= self.step_size * self._potential_energy_gradient(q1) / 2
            q1 += self.step_size * p1
            p1 -= self.step_size * self._potential_energy_gradient(q1) / 2

        # set the next state in the Markov chain
        return q1 if self._check_acceptance(q0, q1, p0, p1) else q0

    def run_chain(self, epochs, initial_position):
        samples = []
        pos = initial_position
        for e in range(epochs):
            pos = self.sample(pos)
            samples.append(pos)
        return paddle.to_tensor(samples)

    def _potential_energy_gradient(self, pos):
        """
        Calculate the gradient of potential energy
        """
        with EnableGradient(pos):
            (-self.dist.log_prob(pos)).backward()
            dVdQ = pos.grad.detach()
        return dVdQ

    def _kinetic_energy(self, p):
        return p * p / 2

    def _check_acceptance(self, q0, q1, p0, p1):
        # calculate the Metropolis acceptance probability
        q0_nlp = -self.dist.log_prob(q0)
        q1_nlp = -self.dist.log_prob(q1)

        p0_nlp = self._kinetic_energy(p0)
        p1_nlp = self._kinetic_energy(p1)

        acceptance = paddle.minimum(
            paddle.to_tensor(1.0), paddle.exp((q0_nlp + p0_nlp) - (p1_nlp + q1_nlp))
        )

        # whether accept the proposed state position
        event = paddle.uniform(shape=[], min=0, max=1)
        return event <= acceptance
