算法
=========

.. automodule:: paddlescience.algorithm.algorithm_pinns
    内嵌物理知识神经网络算法

      参数

      - **net** (*NetworkBase*) - 在PINNs算法中使用的神经网络
      - **loss** (*LossBase*) - 在PINNs算法中使用的损失函数


  样例

  .. code-block::

      import paddlescience as psci
      algo = psci.algorithm.PINNs(net=net, loss=loss)
