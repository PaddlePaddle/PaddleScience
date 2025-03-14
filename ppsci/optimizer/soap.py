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
#
# refs: https://github.com/nikhilvyas/SOAP

from collections import defaultdict
from itertools import chain

import paddle
import paddle.optimizer as optim


class SOAP(optim.Optimizer):
    """
    Improving and Stabilizing Shampoo using Adam. Implements SOAP algorithm (https://arxiv.org/abs/2409.11321).

    Parameters:
        parameters (list|tuple):
            List/Tuple of ``Tensor`` names to update to minimize ``loss``.
        learning_rate (float, optional):
            The learning rate to use. defaults to 0.003.
        beta1 (float optional):
            Adam's betas parameters b1. defaults to 0.95.
        beta2 (float optional):
            Adam's betas parameters b1. defaults to 0.95.
        shampoo_beta (float, optional):
            If >= 0, use this beta for the preconditioner (L and R in paper, state['GG'] below) moving average instead of betas[1].
            defaults to -1.
        epsilon (float, optional):
            Adam's epsilonilon for numerical stability. defaults to 1e-08.
        weight_decay (float, optional): weight decay coefficient. defaults to 0.01.
        precondition_frequency (int, optional):
            How often to update the preconditioner. defaults to 10.
        max_precond_dim (int, optional):
            Maximum dimension of the preconditioner.
            Set to 10000, so that we exclude most common vocab sizes while including layers. defaults to 10000.
        merge_dims (bool, optional):
            Whether or not to merge dimensions of the preconditioner. defaults to `False`.
        precondition_1d (bool, optional):
            Whether or not to precondition 1D gradients. defaults to `False`.
        normalize_grads (bool, optional):
            Whether or not to normalize gradients per layer.
            Helps at large precondition_frequency (~100 in our experiments),
            but hurts performance at small precondition_frequency (~10 in our experiments). defaults to `False`.
        data_format (str, optional):
            Data format of the input for convolutional layers.
            Should be "channels_last" for data_format of NHWC and "channels_first" for NCHW. defaults to `channels_first`.
        correct_bias (bool, optional):
            Whether or not to use bias correction in Adam. defaults to `True`.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    Return:
        loss (Tensor): the final loss of closure.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import ppsci
            >>> import numpy as np

            >>> np.random.seed(0)
            >>> np_w = np.random.rand(1).astype(np.float32)
            >>> np_x = np.random.rand(1).astype(np.float32)

            >>> inputs = [np.random.rand(1).astype(np.float32) for i in range(10)]
            >>> # y = 2x
            >>> targets = [2 * x for x in inputs]

            >>> class Net(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         w = paddle.to_tensor(np_w)
            ...         self.w = paddle.create_parameter(shape=w.shape, dtype=w.dtype, default_initializer=paddle.nn.initializer.Assign(w))
            ...
            ...     def forward(self, x):
            ...         return self.w * x
            ...
            >>> net = Net()
            >>> opt = ppsci.optimizer.soap.SOAP(parameters=net.parameters())
            >>> def train_step(inputs, targets):
            ...     def closure():
            ...         outputs = net(inputs)
            ...         loss = paddle.nn.functional.mse_loss(outputs, targets)
            ...         print('loss: ', loss.item())
            ...         opt.clear_grad()
            ...         loss.backward()
            ...         return loss
            ...     opt.step(closure)
            ...
            >>> for input, target in zip(inputs, targets):
            ...     input = paddle.to_tensor(input)
            ...     target = paddle.to_tensor(target)
            ...     train_step(input, target)
    """

    def __init__(
        self,
        parameters,
        learning_rate: float = 3e-3,
        beta1: float = 0.95,
        beta2: float = 0.95,
        shampoo_beta: float = -1,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,  #
        merge_dims: bool = False,  # Merge dimensions till the product of the dimensions is less than or equal to max_precond_dim.
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
        name: str = None,
    ):
        self._betas = paddle.to_tensor((beta1, beta2))
        self._shampoo_beta = shampoo_beta
        self._epsilon = epsilon
        self._weight_decay = weight_decay
        self._precondition_frequency = precondition_frequency
        self._max_precond_dim = max_precond_dim
        self._merge_dims = merge_dims
        self._precondition_1d = precondition_1d
        self._normalize_grads = normalize_grads
        self._correct_bias = correct_bias

        self.state = defaultdict(dict)

        super().__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            name=name,
        )

        if isinstance(self._parameter_list[0], dict):
            raise TypeError("The parameter groups is not supported on SOAP optimizer.")

        self._data_format = data_format

    def merge_dims(self, grad, max_precond_dim):
        """
        Merges dimensions of the gradient tensor till the product of the dimensions is less than or equal to max_precond_dim.
        """
        assert self._data_format in ["channels_first", "channels_last"]
        if self._data_format == "channels_last" and grad.ndim == 4:
            grad = grad.transpose(0, 3, 1, 2)
        shape = grad.shape
        new_shape = []

        curr_shape = 1
        for dim_size in shape:
            temp_shape = curr_shape * dim_size
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = dim_size
                else:
                    new_shape.append(dim_size)
                    curr_shape = 1
            else:
                curr_shape = temp_shape

        if curr_shape > 1 or len(new_shape) == 0:
            new_shape.append(curr_shape)

        new_grad = grad.reshape(new_shape)
        return new_grad

    @paddle.base.framework.non_static_only
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (Optional[Callable]): A closure that reevaluates the model and returns the loss.
        """
        with paddle.no_grad():
            if closure is None:
                loss = None
            else:
                closure = paddle.enable_grad()(closure)
                loss = closure()

            for p in self._parameter_list:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = paddle.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = paddle.zeros_like(grad)

                if "Q" not in state:
                    self.init_preconditioner(
                        grad,
                        state,
                        precondition_frequency=self._precondition_frequency,
                        precondition_1d=self._precondition_1d,
                        shampoo_beta=(
                            self._shampoo_beta
                            if self._shampoo_beta >= 0
                            else self._betas[1]
                        ),
                        max_precond_dim=self._max_precond_dim,
                        merge_dims=self._merge_dims,
                    )
                    self.update_preconditioner(
                        grad,
                        state,
                        max_precond_dim=self._max_precond_dim,
                        merge_dims=self._merge_dims,
                        precondition_1d=self._precondition_1d,
                    )
                    continue  # first step is skipped so that we never use the current gradients in the projection.

                # Projecting gradients to the eigenbases of Shampoo's preconditioner
                # i.e. projecting to the eigenbases of matrices in state['GG']
                grad_projected = self.project(
                    grad,
                    state,
                    merge_dims=self._merge_dims,
                    max_precond_dim=self._max_precond_dim,
                )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = self._betas

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.multiply_(beta1).add_((1.0 - beta1) * grad_projected)
                exp_avg_sq.multiply_(beta2).add_(
                    (1.0 - beta2) * grad_projected.square()
                )

                denom = exp_avg_sq.sqrt().add_(
                    paddle.full([], self._epsilon, dtype=exp_avg_sq.dtype)
                )

                # Projecting the exponential moving average of gradients to the eigenbases of Shampoo's preconditioner
                # i.e. projecting to the eigenbases of matrices in state['GG']
                # exp_avg_projected = self.project(exp_avg, state, merge_dims=self._merge_dims"],
                #                                  max_precond_dim=self._max_precond_dim'])
                exp_avg_projected = exp_avg

                step_size = self.get_lr()
                if self._correct_bias:
                    bias_correction1 = 1.0 - beta1 ** (state["step"])
                    bias_correction2 = 1.0 - beta2 ** (state["step"])
                    step_size = step_size * (bias_correction2**0.5) / bias_correction1

                # Projecting back the preconditioned (by Adam) exponential moving average of gradients
                # to the original space
                norm_grad = self.project_back(
                    exp_avg_projected / denom,
                    state,
                    merge_dims=self._merge_dims,
                    max_precond_dim=self._max_precond_dim,
                )

                if self._normalize_grads:
                    norm_grad = norm_grad / (1e-30 + paddle.mean(norm_grad**2) ** 0.5)

                p.add_(-step_size * norm_grad)

                # From AdamW code: Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if self._weight_decay > 0.0:
                    p.add_((-self.get_lr() * self._weight_decay) * p)

                # Update is done after the gradient step to avoid using current gradients in the projection.
                self.update_preconditioner(
                    grad,
                    state,
                    max_precond_dim=self._max_precond_dim,
                    merge_dims=self._merge_dims,
                    precondition_1d=self._precondition_1d,
                )

        return loss

    def init_preconditioner(
        self,
        grad,
        state,
        precondition_frequency=10,
        shampoo_beta=0.95,
        max_precond_dim=10000,
        precondition_1d=False,
        merge_dims=False,
    ):
        """
        Initializes the preconditioner matrices (L and R in the paper).
        """
        state[
            "GG"
        ] = []  # Will hold all the preconditioner matrices (L and R in the paper).
        if grad.ndim == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state["GG"].append([])
            else:
                state["GG"].append(paddle.zeros([grad.shape[0], grad.shape[0]]))
        else:
            if merge_dims:
                grad = self.merge_dims(grad, max_precond_dim)

            for dim_size in grad.shape:
                if dim_size > max_precond_dim:
                    state["GG"].append([])
                else:
                    state["GG"].append(paddle.zeros([dim_size, dim_size]))

        state["Q"] = None  # Will hold all the eigenbases of the preconditioner.
        state["precondition_frequency"] = precondition_frequency
        state["shampoo_beta"] = shampoo_beta

    def project(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """
        Projects the gradient to the eigenbases of the preconditioner.
        """
        original_shape = grad.shape
        if merge_dims:
            if grad.ndim == 4 and self._data_format == "channels_last":
                transposed_shape = grad.transpose(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)

        for mat in state["Q"]:
            if len(mat) > 0:
                grad = paddle.tensordot(
                    grad,
                    mat,
                    axes=[[0], [0]],
                )
            else:
                transpose_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.transpose(transpose_order)

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(transposed_shape).transpose(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def update_preconditioner(
        self,
        grad,
        state,
        max_precond_dim=10000,
        merge_dims=False,
        precondition_1d=False,
    ):
        """
        Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
        """
        if state["Q"] is not None:
            state["exp_avg"] = self.project_back(
                state["exp_avg"],
                state,
                merge_dims=merge_dims,
                max_precond_dim=max_precond_dim,
            )
        if grad.ndim == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state["GG"][0].lerp_(
                    grad.unsqueeze(1) @ grad.unsqueeze(0), 1 - state["shampoo_beta"]
                )
        else:
            if merge_dims:
                new_grad = self.merge_dims(grad, max_precond_dim)
                for idx, dim_size in enumerate(new_grad.shape):
                    if dim_size <= max_precond_dim:
                        outer_product = paddle.tensordot(
                            new_grad,
                            new_grad,
                            axes=[
                                [
                                    *chain(
                                        range(idx), range(idx + 1, len(new_grad.shape))
                                    )
                                ]
                            ]
                            * 2,
                        )
                        state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])
            else:
                for idx, dim_size in enumerate(grad.shape):
                    if dim_size <= max_precond_dim:
                        outer_product = paddle.tensordot(
                            grad,
                            grad,
                            # Contracts across all dimensions except for k.
                            axes=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]]
                            * 2,
                        )
                        state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])

        if state["Q"] is None:
            state["Q"] = self.get_orthogonal_matrix(state["GG"])
        if state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            state["Q"] = self.get_orthogonal_matrix_QR(
                state, max_precond_dim, merge_dims
            )
            # state['Q'] = self.get_fast_QR(state, max_precond_dim, merge_dims)

        if state["step"] > 0:
            state["exp_avg"] = self.project(
                state["exp_avg"],
                state,
                merge_dims=merge_dims,
                max_precond_dim=max_precond_dim,
            )

    def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """
        Projects the gradient back to the original space.
        """
        original_shape = grad.shape
        if merge_dims:
            if self._data_format == "channels_last" and grad.ndim == 4:
                transposed_shape = grad.transpose(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = paddle.tensordot(
                    grad,
                    mat,
                    axes=[[0], [1]],
                )
            else:
                transpose_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.transpose(transpose_order)

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(transposed_shape).transpose(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def get_orthogonal_matrix(self, mat):
        """
        Computes the eigenbases of the preconditioner using paddle.linalg.eigh decomposition.
        """
        matrix = []
        for m in mat:
            if len(m) == 0:
                matrix.append([])
                continue
            if m.dtype != paddle.float32:
                float_data = False
                original_type = m.dtype
                original_device = m.place
                matrix.append(m.to(paddle.float32))
            else:
                float_data = True
                matrix.append(m)

        final = []
        for m in matrix:
            if len(m) == 0:
                final.append([])
                continue
            _, Q = paddle.linalg.eigh(m + 1e-30 * paddle.eye(m.shape[0]))
            Q = paddle.flip(Q, [1])

            if not float_data:
                Q = Q.to(original_device, dtype=original_type)
            final.append(Q)
        return final

    def get_orthogonal_matrix_QR(self, state, max_precond_dim=10000, merge_dims=False):
        """
        Computes the eigenbases of the preconditioner using one round of power iteration
        followed by paddle.linalg.qr decomposition.
        """
        precond_list = state["GG"]
        orth_list = state["Q"]

        matrix = []
        orth_matrix = []
        for m, o in zip(precond_list, orth_list):
            if len(m) == 0:
                matrix.append([])
                orth_matrix.append([])
                continue
            if m.dtype != paddle.float32:
                float_data = False
                original_type = m.dtype
                original_device = m.place
                matrix.append(m.to(paddle.float32))
                orth_matrix.append(o.to(paddle.float32))
            else:
                float_data = True
                matrix.append(m.to(paddle.float32))
                orth_matrix.append(o.to(paddle.float32))

        orig_shape = state["exp_avg_sq"].shape
        if self._data_format == "channels_last" and len(orig_shape) == 4:
            transposed_shape = state["exp_avg_sq"].transpose(0, 3, 1, 2).shape
        if merge_dims:
            exp_avg_sq = self.merge_dims(state["exp_avg_sq"], max_precond_dim)
        else:
            exp_avg_sq = state["exp_avg_sq"]

        final = []
        for ind, (m, o) in enumerate(zip(matrix, orth_matrix)):
            if len(m) == 0:
                final.append([])
                continue
            est_eig = paddle.diag(o.T @ m @ o)
            sort_idx = paddle.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(sort_idx, ind)
            o = o[:, sort_idx]
            power_iter = m @ o
            Q, _ = paddle.linalg.qr(power_iter)

            if not float_data:
                Q = Q.to(original_device, dtype=original_type)
            final.append(Q)

        if merge_dims:
            if self._data_format == "channels_last" and len(orig_shape) == 4:
                exp_avg_sq = exp_avg_sq.reshape(transposed_shape).transpose(0, 2, 3, 1)
            else:
                exp_avg_sq = exp_avg_sq.reshape(orig_shape)

        state["exp_avg_sq"] = exp_avg_sq

        return final
