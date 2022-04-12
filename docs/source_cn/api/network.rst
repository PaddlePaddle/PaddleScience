神经网络
===================================

.. automodule:: paddlescience.network.network_fc

一个多层的全连接网络。除了最后一层，每层都包含矩阵乘法、矩阵逐元素加法和激活函数的操作。

   **参数:**

   - **num_ins** (int) - 网络输入的维度。
   - **num_outs** (int) - 网络输出的维度。
   - **num_layer** (int) - 网络的隐藏层数量。
   - **hidden_size** (int) - 网络隐藏层的神经元数量。
   - **dtype** (string) - 可选项，默认'float32'。网络权重和偏置的数据类型，目前只支持'float32'。
   - **activation** (string) - 可选项，默认'tanh'。网络每层激活函数的类型，可以是'tanh'或者'sigmoid'。

**样例**

.. code-block:: python

   import paddlescience as psci
   net = psci.network.FCNet(2, 3, 10, 50, dtype='float32', activiation='tanh')

