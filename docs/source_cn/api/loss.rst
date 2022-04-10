损失函数
===================================

.. automodule:: paddlescience.loss.loss_L2
   L2 loss由三个部分组成：方程损失，边界条件损失以及初始条件损失。


   参数

   - **pdes** (*PDE*) – 用于计算方程损失的偏微分方程。
   - **geo** (`GeometryDiscrete <https://paddlescience.paddlepaddle.org.cn/api/geometry.html#paddlescience.geometry.geometry_discrete.GeometryDiscrete>`_) – 计算损失的离散几何。
   - **aux_func** (*Callable|None*) – 可选，默认为None。如果被指定，它应该是一个返回包含Paddle Tensors的列表的函数。该列表被用作方程右边的值去计算损失。
   - **eq_weight** (*float|None*) – 可选，默认为None。如果被指定，与方程损失(the equation loss)相乘后再合成总loss。
   - **bc_weight** (*numpy.array|None*) –可选，默认为None。如果被指定，它应该是一个一维的numpy数组(array)，其元素数量与边界条件点相同。 其作为权重去计算边界条件损失。
   - **synthesis_method** (*string*) – 可选, 默认为'add'。在合成损失的三个部分时所使用的方法。如果是'add'，则直接将三部分相加；如果是'norm'，则通过计算2-norm计算最终损失。
   - **run_in_batch** (*bool*) – 可选, 默认为True。如果是True，在每一个batch上计算方程损失。如果是False，则在每一个点上计算方程损失。

样例

.. code-block::

   import paddlescience as psci
   net = psci.loss.L2(pdes=pdes, geo=geo)


