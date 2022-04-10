优化器
=========

.. automodule:: paddlescience.optimizer.optimizer
   Adam优化器出自 `Adam论文<https://arxiv.org/abs/1412.6980>_` 的第二节，能够利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。

   - 参数

   - **learning_rate** (*float|LRScheduler, optional*) - 用于更新的学习率，它可以是一个数据类型为float的值，也可以是LRScheduler的子类实例。默认值为0.001.
   - **beta1** (*float|Tensor,* *可选*) - 一阶矩估计的指数衰减率，是一个float类型或者一个shape为[1]，数据类型为float32的Tensor类型。默认值为0.9.
   - **beta2** (*float|Tensor,* *可选*) - 二阶矩估计的指数衰减率，是一个float类型或者一个shape为[1]，数据类型为float32的Tensor类型。默认值为0.999.
   - **epsilon** (*float|Tensor,* *可选*) - 保持数值稳定性的短浮点类型值，默认值为1e-08
   - **parameters** (*list|tuple,* *可选*) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数；在静态图模式下默认值为None，这时所有的参数都将被优化。
   - **weight_decay** (*float|WeightDecayRegularizer,* *可选*) - 正则化方法。可以是float类型的L2正则化系数或者正则化策略: cn_api_fluid_regularizer_L1Decay 、 cn_api_fluid_regularizer_L2Decay 。如果一个参数已经在 ParamAttr 中设置了正则化，这里的正则化设置将被忽略； 如果没有在 ParamAttr 中设置正则化，这里的设置才会生效。默认值为None，表示没有正则化。
   - **grad_clip** (*GradientClipBase,* *可选)* - 梯度裁剪的策略，支持三种裁剪策略： `paddle.nn.ClipGradByGlobalNorm<https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/ClipGradByGlobalNorm_cn.html#cn-api-fluid-clip-clipgradbyglobalnorm>_` 、 `paddle.nn.ClipGradByNorm<https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/ClipGradByNorm_cn.html#cn-api-fluid-clip-clipgradbynorm>_` 、 `paddle.nn.ClipGradByValue<https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/ClipGradByValue_cn.html#cn-api-fluid-clip-clipgradbyvalue>_` 。 默认值为None，此时将不进行梯度裁剪。
   - **lazy_mode** (*bool,* *可选*) - 设为True时，仅更新当前具有梯度的元素。官方Adam算法有两个移动平均累加器（moving-average accumulators）。累加器在每一步都会更新。在密集模式和稀疏模式下，两条移动平均线的每个元素都会更新。如果参数非常大，那么更新可能很慢。 lazy mode仅更新当前具有梯度的元素，所以它会更快。但是这种模式与原始的算法有不同的描述，可能会导致不同的结果，默认为False
   - **multi_precision** (*bool,可选*) - 权重更新时是否使用多精度，默认为False.
   - **name** (*str*, *可选*) - 通常情况下，用户不需要考虑这个参数。具体用法请参见 `Name<https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name>_`。默认为None.




样例

.. code-block::

    import paddlescience as psci
    opt = psci.optimizer.Adam(learning_rate=0.1, parameters=linear.parameters())