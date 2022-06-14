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

import paddle
from paddle.incubate.optimizer.functional.lbfgs import minimize_lbfgs


def Adam(**kargs):
    """
    The Adam optimizer uses an optimization described at the end of section 2 of Adam paper , it can dynamically adjusts the learning rate of each parameter using the 1st moment estimates and the 2nd moment estimates of the gradient.
    
    Parameters:
        learning_rate (float|LRScheduler, optional) – The learning rate used to update Parameter. It can be a float value or a LRScheduler. The default value is 0.001.
        beta1 (float|Tensor, optional) – The exponential decay rate for the 1st moment estimates. It should be a float number or a Tensor with shape [1] and data type as float32. The default value is 0.9.
        beta2 (float|Tensor, optional) – The exponential decay rate for the 2nd moment estimates. It should be a float number or a Tensor with shape [1] and data type as float32. The default value is 0.999.
        epsilon (float|Tensor, optional) – A small float value for numerical stability. It should be a float number or a Tensor with shape [1] and data type as float32. The default value is 1e-08.
        parameters (list|tuple, optional) – List/Tuple of Tensor to update to minimize loss. This parameter is required in dygraph mode. And you can specify different options for different parameter groups such as the learning rate, weight decay, etc, then the parameters are list of dict. Note that the learning_rate in paramter groups represents the scale of base learning_rate. The default value is None in static mode, at this time all parameters will be updated.
        weight_decay (float|WeightDecayRegularizer, optional) – The strategy of regularization. It canbe a float value as coeff of L2 regularization or api_fluid_regularizer_L1Decay, api_fluid_regularizer_L2Decay. If a parameter has set regularizer using api_fluid_ParamAttr already, the regularization setting here in optimizer will be ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect. Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional) – Gradient cliping strategy, it’s an instance of some derived class of GradientClipBase . There are three cliping strategies ( api_fluid_clip_GradientClipByGlobalNorm , api_fluid_clip_GradientClipByNorm , api_fluid_clip_GradientClipByValue ). Default None, meaning there is no gradient clipping.
        lazy_mode (bool, optional) – The official Adam algorithm has two moving-average accumulators. The accumulators are updated at every step. Every element of the two moving-average is updated in both dense mode and sparse mode. If the size of parameter is very large, then the update may be very slow. The lazy mode only update the element that has gradient in current mini-batch, so it will be much more faster. But this mode has different semantics with the original Adam algorithm and may lead to different result. The default value is False.
        multi_precision (bool, optional) – Whether to use multi-precision during weight updating. Default is false.
        name (str, optional) – Normally there is no need for user to set this property. For more information, please refer to Name. The default value is None.
 
    Example:
        >>> import paddlescience as psci
        >>> opt = psci.optimizer.Adam(learning_rate=0.1, parameters=net.parameters())
    """
    return paddle.optimizer.Adam(**kargs)


def Lbfgs(objective_func,
          initial_position,
          history_size=100,
          max_iters=50,
          tolerance_grad=1e-8,
          tolerance_change=1e-8,
          initial_inverse_hessian_estimate=None,
          line_search_fn='strong_wolfe',
          max_line_search_iters=50,
          initial_step_length=1.0,
          dtype='float32',
          name=None):
    """
    Minimizes a differentiable function `func` using the L-BFGS method.
    The L-BFGS is simalar as BFGS, the only difference is that L-BFGS use historical
    sk, yk, rhok rather than H_k-1 to compute Hk.
    Reference:
        Jorge Nocedal, Stephen J. Wright, Numerical Optimization, Second Edition, 2006.
        pp179: Algorithm 7.5 (L-BFGS).
    Following summarizes the the main logic of the program based on L-BFGS.Note: _k represents 
    value of k_th iteration, ^T represents the transposition of a vector or matrix.
    repeat
        compute p_k by two-loop recursion
        alpha = strong_wolfe(f, x_k, p_k)
        x_k+1 = x_k + alpha * p_k
        s_k = x_k+1 - x_k
        y_k = g_k+1 - g_k
        rho_k = 1 / (s_k^T * y_k)
        update sk_vec, yk_vec, rhok_vec
        check_converge
    end 
    Args:
        objective_func: the objective function to minimize. ``func`` accepts
            a multivariate input and returns a scalar.
        initial_position (Tensor): the starting point of the iterates. For methods like Newton and quasi-Newton 
        the initial trial step length should always be 1.0 .
        history_size (Scalar): the number of stored vector pairs {si,yi}.
        max_iters (Scalar): the maximum number of minimization iterations.
        tolerance_grad (Scalar): terminates if the gradient norm is smaller than
            this. Currently gradient norm uses inf norm.
        tolerance_change (Scalar): terminates if the change of function value/position/parameter between 
            two iterations is smaller than this value.
        initial_inverse_hessian_estimate (Tensor): the initial inverse hessian approximation.
        line_search_fn (str): indicate which line search method to use, only support 'strong wolfe' right now. May support 
            'Hager Zhang' in the futrue.
        max_line_search_iters (Scalar): the maximum number of line search iterations.
        initial_step_length: step length used in first iteration of line search. different initial_step_length 
        may cause different optimal result.
        dtype ('float' | 'float32' | 'float64' | 'double'): the data
            type to be used.
    
    Returns:
        is_converge (bool): Indicates whether found the minimum within tolerance.
        num_func_calls (int): number of objective function called.
        position (Tensor): the position of the last iteration. If the search converged, this value is the argmin of 
        the objective function regrading to the initial position.
        objective_value (Tensor): objective function value at the `position`.
        objective_gradient (Tensor): objective function gradient at the `position`.

    Example:
        >>> import paddlescience as psci
        >>> opt = psci.optimizer.Lbfgs()
    """

    return paddle.incubate.optimizer.functional.lbfgs.minimize_lbfgs(
        objective_func=objective_func,
        initial_position=initial_position,
        history_size=history_size,
        max_iters=max_iters,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        initial_inverse_hessian_estimate=initial_inverse_hessian_estimate,
        line_search_fn=line_search_fn,
        max_line_search_iters=max_line_search_iters,
        initial_step_length=initial_step_length,
        dtype=dtype,
        name=name)
