# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
https://github.com/zabaras/transformer-physx/blob/main/trphysx/embedding/embedding_lorenz.py
https://github.com/zabaras/transformer-physx/blob/main/trphysx/embedding/embedding_cylinder.py
https://github.com/zabaras/transformer-physx/blob/main/examples/rossler/rossler_module/embedding_rossler.py
"""

import os
import numpy as np
from typing import List, Tuple

import paddle
import paddle.nn as nn
from paddle.nn.initializer import Constant, Normal, Uniform

from paddlescience.visu import CylinderViz

Tensor = paddle.Tensor
TensorTuple = Tuple[paddle.Tensor]
FloatTuple = Tuple[float]

normal_ = Normal
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


class LorenzEmbedding(nn.Layer):
    """ Embedding Koopman model for the Lorenz ODE system """

    def __init__(self, state_dims, n_embd, norm_eps=1e-5, embd_pdrop=0.0):
        super().__init__()
        self.state_dims = state_dims
        self.n_embd = n_embd

        hidden_states = int(abs(state_dims[0] - n_embd) / 2) + 1
        hidden_states = 500

        self.observableNet = nn.Sequential(
            nn.Linear(state_dims[0], hidden_states),
            nn.ReLU(),
            nn.Linear(hidden_states, n_embd),
            nn.LayerNorm(
                n_embd, epsilon=norm_eps),
            nn.Dropout(embd_pdrop))

        self.recoveryNet = nn.Sequential(
            nn.Linear(n_embd, hidden_states),
            nn.ReLU(), nn.Linear(hidden_states, state_dims[0]))
        # Learned Koopman operator
        self.obsdim = n_embd
        data = paddle.linspace(1, 0, n_embd)
        self.kMatrixDiag = paddle.create_parameter(
            shape=data.shape,
            dtype=data.dtype,
            default_initializer=nn.initializer.Assign(data))

        # Off-diagonal indices
        xidx = []
        yidx = []
        for i in range(1, 3):
            yidx.append(np.arange(i, n_embd))
            xidx.append(np.arange(0, n_embd - i))

        self.xidx = paddle.to_tensor(np.concatenate(xidx), dtype='int64')
        self.yidx = paddle.to_tensor(np.concatenate(yidx), dtype='int64')

        data = 0.1 * paddle.rand([self.xidx.shape[0]])
        self.kMatrixUT = paddle.create_parameter(
            shape=data.shape,
            dtype=data.dtype,
            default_initializer=nn.initializer.Assign(data))

        self.register_buffer('mu',
                             paddle.to_tensor([0., 0., 0.]).reshape([1, 3]))
        self.register_buffer('std',
                             paddle.to_tensor([1., 1., 1.]).reshape([1, 3]))
        self.apply(self._init_weights)

        self.loss_weight = 1e4
        self.koopman_weight = 1e-1
        self.loss = nn.MSELoss()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            k = 1 / m.weight.shape[0]
            uniform = Uniform(-k**0.5, k**0.5)
            uniform(m.weight)
            if m.bias is not None:
                uniform(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward(self, x: Tensor) -> TensorTuple:
        # Encode
        x = self._normalize(x)
        g = self.observableNet(x)
        # Decode
        out = self.recoveryNet(g)
        xhat = self._unnormalize(out)

        return g, xhat

    def embed(self, x: Tensor) -> Tensor:
        x = self._normalize(x)
        g = self.observableNet(x)
        return g

    def recover(self, g: Tensor) -> Tensor:
        out = self.recoveryNet(g)
        x = self._unnormalize(out)
        return x

    def koopmanOperation(self, g: Tensor) -> Tensor:
        # # Koopman operator
        kMatrixUT_tensor = self.kMatrixUT * 1
        kMatrixUT_tensor = paddle.diag(
            kMatrixUT_tensor[0:self.obsdim - 1], offset=1) + paddle.diag(
                kMatrixUT_tensor[self.obsdim - 1:], offset=2)
        kMatrix = kMatrixUT_tensor + (-1) * kMatrixUT_tensor.t()
        kMatrix = kMatrix + paddle.diag(self.kMatrixDiag)

        # Apply Koopman operation
        gnext = paddle.bmm(
            kMatrix.expand([g.shape[0], kMatrix.shape[0], kMatrix.shape[0]]),
            g.unsqueeze(-1))
        self.kMatrix = kMatrix
        return gnext.squeeze(-1)

    @property
    def koopmanOperator(self, requires_grad: bool=True) -> Tensor:
        if not requires_grad:
            return self.kMatrix.detach()
        else:
            return self.kMatrix

    def _normalize(self, x):
        return (x - self.mu) / self.std

    def _unnormalize(self, x):
        return self.std * x + self.mu

    def compute_loss(self, inputs, **kwargs):
        self.train()
        loss_reconstruct = 0

        xin0 = inputs[:, 0]

        # Model forward for initial time-step
        g0, xRec0 = self.forward(xin0)
        loss = self.loss_weight * self.loss(xin0, xRec0)
        loss_reconstruct = loss_reconstruct + self.loss(xin0, xRec0).detach()

        g1_old = g0
        # Loop through time-series
        for t0 in range(1, inputs.shape[1]):
            xin0 = inputs[:, t0, :]
            _, xRec1 = self.forward(xin0)
            g1Pred = self.koopmanOperation(g1_old)
            xgRec1 = self.recover(g1Pred)

            loss = loss + self.loss(xgRec1, xin0) + self.loss_weight * self.loss(xRec1, xin0) \
                + self.koopman_weight * paddle.sum(paddle.pow(self.koopmanOperator, 2))

            loss_reconstruct = loss_reconstruct + self.loss(xRec1,
                                                            xin0).detach()
            g1_old = g1Pred
        return dict(loss=loss, loss_reconstruct=loss_reconstruct)

    def evaluate(self, inputs, **kwargs):
        self.eval()
        # Pull out targets from prediction dataset
        yTarget = inputs[:, 1:]
        xInput = inputs[:, :-1]
        yPred = paddle.zeros(yTarget.shape)

        # Test accuracy of one time-step
        for i in range(xInput.shape[1]):
            xInput0 = xInput[:, i]
            g0 = self.embed(xInput0)
            yPred0 = self.recover(g0)
            yPred[:, i] = yPred0.squeeze().detach()

        test_loss = self.loss(yTarget, yPred)
        return dict(loss=test_loss, pred=yPred, target=yTarget)


class CylinderEmbedding(nn.Layer):
    """ Embedding Koopman model for the Lorenz ODE system """

    def __init__(self, state_dims, n_embd, norm_eps=1e-5, embd_pdrop=0.0):
        super().__init__()
        self.state_dims = state_dims

        X, Y = np.meshgrid(np.linspace(-2, 14, 128), np.linspace(-4, 4, 64))
        self.mask = paddle.to_tensor(np.sqrt(X**2 + Y**2))

        # Encoder conv. net
        self.observableNet = nn.Sequential(
            nn.Conv2D(
                4,
                16,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2D(
                16,
                32,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                padding_mode='replicate'),
            nn.ReLU(),
            # 16, 16, 32
            nn.Conv2D(
                32,
                64,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                padding_mode='replicate'),
            nn.ReLU(),
            # 16, 8, 16
            nn.Conv2D(
                64,
                128,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                padding_mode='replicate'),
            nn.ReLU(),
            # 16, 4, 8
            nn.Conv2D(
                128,
                n_embd // 32,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                padding_mode='replicate'), )

        self.observableNetFC = nn.Sequential(
            nn.LayerNorm(
                n_embd, epsilon=norm_eps), nn.Dropout(embd_pdrop))

        # Decoder conv. net
        self.recoveryNet = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2D(
                n_embd // 32,
                128,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                padding_mode='replicate'),
            nn.ReLU(),
            # 16, 8, 16
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2D(
                128,
                64,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                padding_mode='replicate'),
            nn.ReLU(),
            # 16, 16, 32
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2D(
                64,
                32,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                padding_mode='replicate'),
            nn.ReLU(),
            # 8, 32, 64
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2D(
                32,
                16,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                padding_mode='replicate'),
            nn.ReLU(),
            # 16, 64, 128
            nn.Conv2D(
                16,
                3,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                padding_mode='replicate'), )
        # Learned Koopman operator parameters
        self.obsdim = n_embd
        # We parameterize the Koopman operator as a function of the viscosity
        self.kMatrixDiagNet = nn.Sequential(
            nn.Linear(1, 50), nn.ReLU(), nn.Linear(50, self.obsdim))

        # Off-diagonal indices
        xidx = []
        yidx = []
        for i in range(1, 5):
            yidx.append(np.arange(i, self.obsdim))
            xidx.append(np.arange(0, self.obsdim - i))
        self.xidx = paddle.to_tensor(np.concatenate(xidx), dtype='int64')
        self.yidx = paddle.to_tensor(np.concatenate(yidx), dtype='int64')

        # The matrix here is a small NN since we need to make it dependent on the viscosity
        self.kMatrixUT = nn.Sequential(
            nn.Linear(1, 50), nn.ReLU(), nn.Linear(50, self.xidx.shape[0]))
        self.kMatrixLT = nn.Sequential(
            nn.Linear(1, 50), nn.ReLU(), nn.Linear(50, self.xidx.shape[0]))

        # Normalization occurs inside the model
        self.register_buffer(
            'mu', paddle.to_tensor([0., 0., 0., 0.]).reshape([1, 4, 1, 1]))
        self.register_buffer(
            'std', paddle.to_tensor([1., 1., 1., 1.]).reshape([1, 4, 1, 1]))
        self.apply(self._init_weights)

        self.loss_weight = 1e1
        self.koopman_weight = 1e-2
        self.loss = nn.MSELoss()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            k = 1 / m.weight.shape[0]
            uniform = Uniform(-k**0.5, k**0.5)
            uniform(m.weight)
            if m.bias is not None:
                uniform(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            k = 1 / (m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3])
            uniform = Uniform(-k**0.5, k**0.5)
            uniform(m.weight)
            if isinstance(m, nn.Conv2D) and m.bias is not None:
                uniform(m.bias)

    def forward(self, x: Tensor, viscosity: Tensor) -> TensorTuple:
        x = paddle.concat(
            [
                x, viscosity.unsqueeze(-1).unsqueeze(-1) *
                paddle.ones_like(x[:, :1])
            ],
            axis=1)
        x = self._normalize(x)
        g0 = self.observableNet(x)
        g = self.observableNetFC(g0.reshape([g0.shape[0], -1]))
        # Decode
        out = self.recoveryNet(g.reshape(g0.shape))
        xhat = self._unnormalize(out)
        # Apply cylinder mask
        # mask0 = self.mask.unsqueeze(0).unsqueeze(0).repeat_interleave(xhat.shape[1], axis=1).repeat_interleave(xhat.shape[0], axis=0) < 1
        # mask0 = self.mask.repeat(xhat.shape[0], xhat.shape[1], 1, 1) is True
        # xhat[mask0] = 0
        return g, xhat

    def embed(self, x: Tensor, viscosity: Tensor) -> Tensor:
        # Concat viscosities as a feature map
        if len(x.shape) == 5:
            g = []
            for i in range(x.shape[1]):
                x_i = x[:, i]
                x_i = paddle.concat(
                    [
                        x[:, i], viscosity.unsqueeze(-1).unsqueeze(-1) *
                        paddle.ones_like(x_i[:, :1])
                    ],
                    axis=1)
                x_i = self._normalize(x_i)
                g_i = self.observableNet(x_i)
                g_i = self.observableNetFC(g_i.reshape([x_i.shape[0], -1]))
                g.append(g_i)
            g = paddle.to_tensor(g).transpose([1, 0, 2])
        else:
            x = paddle.concat(
                [
                    x, viscosity.unsqueeze(-1).unsqueeze(-1) *
                    paddle.ones_like(x[:, :1])
                ],
                axis=1)
            x = self._normalize(x)

            g = self.observableNet(x)
            g = self.observableNetFC(g.reshape([x.shape[0], -1]))
        return g

    def recover(self, g: Tensor) -> Tensor:
        x = self.recoveryNet(g.reshape([-1, self.obsdim // 32, 4, 8]))
        x = self._unnormalize(x)
        # Apply cylinder mask
        mask0 = self.mask.unsqueeze(0).unsqueeze(0).repeat_interleave(
            x.shape[1], axis=1).repeat_interleave(
                x.shape[0], axis=0) < 1
        x[mask0] = 0
        return x

    def koopmanOperation(self, g: Tensor, visc: Tensor) -> Tensor:
        """Applies the learned Koopman operator on the given observables

        Args:
            g (Tensor): [B, config.n_embd] Koopman observables
            visc (Tensor): [B] Viscosities of the fluid in the mini-batch

        Returns:
            Tensor: [B, config.n_embd] Koopman observables at the next time-step
        """
        kMatrix = paddle.zeros([g.shape[0], self.obsdim, self.obsdim])
        kMatrix.stop_gradient = False
        # Populate the off diagonal terms
        kMatrixUT_data = self.kMatrixUT(100 * visc)
        kMatrixLT_data = self.kMatrixLT(100 * visc)

        kMatrix = kMatrix.transpose([1, 2, 0])
        kMatrixUT_data_t = kMatrixUT_data.transpose([1, 0])
        kMatrixLT_data_t = kMatrixLT_data.transpose([1, 0])
        kMatrix[self.xidx, self.yidx] = kMatrixUT_data_t
        kMatrix[self.yidx, self.xidx] = kMatrixLT_data_t

        # Populate the diagonal
        ind = np.diag_indices(kMatrix.shape[1])
        ind = paddle.to_tensor(ind, dtype='int64')

        self.kMatrixDiag = self.kMatrixDiagNet(100 * visc)
        kMatrixDiag_t = self.kMatrixDiag.transpose([1, 0])
        kMatrix[ind[0], ind[1]] = kMatrixDiag_t
        self.kMatrix = kMatrix.transpose([2, 0, 1])

        # Apply Koopman operation
        gnext = paddle.bmm(self.kMatrix, g.unsqueeze(-1))
        return gnext.squeeze(-1)  # Squeeze empty dim from bmm

    @property
    def koopmanOperator(self, requires_grad: bool=True) -> Tensor:
        if not requires_grad:
            return self.kMatrix.detach()
        else:
            return self.kMatrix

    def _normalize(self, x: Tensor) -> Tensor:
        x = (x - self.mu) / self.std
        return x

    def _unnormalize(self, x: Tensor) -> Tensor:
        return self.std[:, :3] * x + self.mu[:, :3]

    def compute_loss(self, inputs, viscosity, **kwargs):
        self.train()
        assert inputs.shape[0] == viscosity.shape[
            0], 'State variable and viscosity tensor should have the same batch dimensions.'

        loss_reconstruct = 0

        xin0 = inputs[:, 0]

        # Model forward for initial time-step
        g0, xRec0 = self.forward(xin0, viscosity)
        loss = self.loss_weight * self.loss(xin0, xRec0)
        loss_reconstruct = loss_reconstruct + self.loss(xin0, xRec0).detach()

        g1_old = g0
        # Loop through time-series
        for t0 in range(1, inputs.shape[1]):
            xin0 = inputs[:, t0, :]
            _, xRec1 = self.forward(xin0, viscosity)
            # Apply Koopman transform
            g1Pred = self.koopmanOperation(g1_old, viscosity)
            xgRec1 = self.recover(g1Pred)

            # Loss function
            loss = loss + self.loss_weight * self.loss(xgRec1, xin0) + self.loss_weight * self.loss(xRec1, xin0) \
                + self.koopman_weight * paddle.sum(paddle.pow(self.koopmanOperator, 2))

            loss_reconstruct = loss_reconstruct + self.loss(xRec1,
                                                            xin0).detach()
            g1_old = g1Pred
        return dict(loss=loss, loss_reconstruct=loss_reconstruct)

    def evaluate(self, inputs, viscosity, visu_dir=None, **kwargs):
        self.eval()
        # Pull out targets from prediction dataset
        yTarget = inputs[:, 1:]
        xInput = inputs[:, :-1]
        yPred = paddle.zeros(yTarget.shape)
        # Test accuracy of one time-step
        for i in range(xInput.shape[1]):
            xInput0 = xInput[:, i]
            g0 = self.embed(xInput0, viscosity)
            yPred0 = self.recover(g0)
            yPred[:, i] = yPred0.squeeze().detach()

        test_loss = self.loss(yTarget, yPred)

        if visu_dir is not None:
            os.makedirs(visu_dir, exist_ok=True)
            viz = CylinderViz(visu_dir)

            viz.plotEmbeddingPrediction(yPred, yTarget, visu_dir)
        return dict(loss=test_loss, pred=yPred, target=yTarget)


class RosslerEmbedding(nn.Layer):
    """ Embedding Koopman model for the Lorenz ODE system """

    def __init__(self, state_dims, n_embd, norm_eps=1e-5, embd_pdrop=0.0):
        """Constructor method
        """
        super().__init__()
        self.state_dims = state_dims
        self.n_embd = n_embd

        hidden_states = int(abs(state_dims[0] - n_embd) / 2) + 1
        hidden_states = 500

        self.observableNet = nn.Sequential(
            nn.Linear(state_dims[0], hidden_states),
            nn.ReLU(),
            nn.Linear(hidden_states, n_embd),
            nn.LayerNorm(
                n_embd, epsilon=norm_eps),
            nn.Dropout(embd_pdrop))

        self.recoveryNet = nn.Sequential(
            nn.Linear(n_embd, hidden_states),
            nn.ReLU(), nn.Linear(hidden_states, state_dims[0]))
        # Learned Koopman operator
        self.obsdim = n_embd
        data = paddle.linspace(1, 0, n_embd)
        self.kMatrixDiag = paddle.create_parameter(
            shape=data.shape,
            dtype=data.dtype,
            default_initializer=nn.initializer.Assign(data))

        # Off-diagonal indices
        xidx = []
        yidx = []
        for i in range(1, 3):
            yidx.append(np.arange(i, n_embd))
            xidx.append(np.arange(0, n_embd - i))

        self.xidx = paddle.to_tensor(np.concatenate(xidx), dtype='int64')
        self.yidx = paddle.to_tensor(np.concatenate(yidx), dtype='int64')

        data = 0.1 * paddle.rand([self.xidx.shape[0]])
        self.kMatrixUT = paddle.create_parameter(
            shape=data.shape,
            dtype=data.dtype,
            default_initializer=nn.initializer.Assign(data))

        self.register_buffer('mu',
                             paddle.to_tensor([0., 0., 0.]).reshape([1, 3]))
        self.register_buffer('std',
                             paddle.to_tensor([1., 1., 1.]).reshape([1, 3]))
        self.apply(self._init_weights)

        self.loss_weight = 1e3
        self.koopman_weight = 1e-1
        self.loss = nn.MSELoss()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            k = 1 / m.weight.shape[0]
            uniform = Uniform(-k**0.5, k**0.5)
            uniform(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                uniform(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward(self, x: Tensor) -> TensorTuple:
        # Encode
        x = self._normalize(x)
        g = self.observableNet(x)
        # Decode
        out = self.recoveryNet(g)
        xhat = self._unnormalize(out)

        return g, xhat

    def embed(self, x: Tensor) -> Tensor:
        x = self._normalize(x)
        g = self.observableNet(x)
        return g

    def recover(self, g: Tensor) -> Tensor:
        out = self.recoveryNet(g)
        x = self._unnormalize(out)
        return x

    def koopmanOperation(self, g: Tensor) -> Tensor:
        # # Koopman operator
        kMatrixUT_tensor = self.kMatrixUT * 1
        kMatrixUT_tensor = paddle.diag(
            kMatrixUT_tensor[0:self.obsdim - 1], offset=1) + paddle.diag(
                kMatrixUT_tensor[self.obsdim - 1:], offset=2)
        kMatrix = kMatrixUT_tensor + (-1) * kMatrixUT_tensor.t()
        kMatrix = kMatrix + paddle.diag(self.kMatrixDiag)

        # Apply Koopman operation
        gnext = paddle.bmm(
            kMatrix.expand([g.shape[0], kMatrix.shape[0], kMatrix.shape[0]]),
            g.unsqueeze(-1))
        self.kMatrix = kMatrix
        return gnext.squeeze(-1)

    @property
    def koopmanOperator(self, requires_grad: bool=True) -> Tensor:
        if not requires_grad:
            return self.kMatrix.detach()
        else:
            return self.kMatrix

    def _normalize(self, x):
        return (x - self.mu) / self.std

    def _unnormalize(self, x):
        return self.std * x + self.mu

    def compute_loss(self, inputs, **kwargs):
        self.train()
        loss_reconstruct = 0

        xin0 = inputs[:, 0]

        # Model forward for initial time-step
        g0, xRec0 = self.forward(xin0)
        loss = self.loss_weight * self.loss(xin0, xRec0)
        loss_reconstruct = loss_reconstruct + self.loss(xin0, xRec0).detach()

        g1_old = g0
        # Loop through time-series
        for t0 in range(1, inputs.shape[1]):
            xin0 = inputs[:, t0, :]
            _, xRec1 = self.forward(xin0)
            g1Pred = self.koopmanOperation(g1_old)
            xgRec1 = self.recover(g1Pred)

            loss = loss + self.loss(xgRec1, xin0) + self.loss_weight * self.loss(xRec1, xin0) \
                + self.koopman_weight * paddle.sum(paddle.pow(self.koopmanOperator, 2))

            loss_reconstruct = loss_reconstruct + self.loss(xRec1,
                                                            xin0).detach()
            g1_old = g1Pred
        return dict(loss=loss, loss_reconstruct=loss_reconstruct)

    def evaluate(self, inputs, **kwargs):
        self.eval()
        # Pull out targets from prediction dataset
        yTarget = inputs[:, 1:]
        xInput = inputs[:, :-1]
        yPred = paddle.zeros(yTarget.shape)

        # Test accuracy of one time-step
        for i in range(xInput.shape[1]):
            xInput0 = xInput[:, i]
            g0 = self.embed(xInput0)
            yPred0 = self.recover(g0)
            yPred[:, i] = yPred0.squeeze().detach()

        test_loss = self.loss(yTarget, yPred)
        return dict(loss=test_loss, pred=yPred, target=yTarget)
