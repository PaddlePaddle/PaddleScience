求解器
========

.. automodule:: paddlescience.solver.solver

.. py:class:: Solver(algo,opt)


      **参数:**

         - **algo** (*AlgorithmBase*) - 算法，AlgorithmBase的子类实例
         - **opt** (*paddle.Optimizer*) - 优化器，paddle.Optimizer的子类实例

   **样例**

   .. code-block:: python

         import paddlescience as psci
         solver = psci.solver.Solver(algo=algo, opt=opt)


   .. py:function:: solve(num_epoch=1000, batch_size=None, checkpoint_freq=1000)

      根据设置的num_epoch进行网络训练。

      **参数:**

         - **num_epoch** (*int*) - 可选项, 默认值为1000. 表示训练的epoch数量。
         - **batch_size** (*int|None*) - Under develop. 可选项, 默认值为None. 表示在训练时多少样本点在一个batch中被使用。
         - **checkpoint_freq** (*int*) - Under develop. 可选项, 默认值为 1000. 表示经历多少个epoch后保存一次模型参数。

      **返回:**

         一个函数，其输入为一个Geometry Discrete实例，输出为numpy数组(array)类型。

      **返回类型:**

         对应问题的解(Callable)

   **样例**

   .. code-block::

         import paddlescience as psci
         solver = psci.solver.Solver(algo=algo, opt=opt)
         solution = solver.solve(num_epoch=10000)
         rslt = solution(geo)