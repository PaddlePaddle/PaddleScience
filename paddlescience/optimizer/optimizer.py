# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
        >>> opt = psci.optimizer.Adam(learning_rate=0.1, parameters=linear.parameters())
    """
    return paddle.optimizer.Adam(**kargs)
