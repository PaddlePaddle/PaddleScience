PDE (偏微分方程)
===================================

.. automodule:: paddlescience.pde.pde_laplace_2d

   二维拉普拉斯方程

   .. math::

        \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0

   **样例：**

   .. code-block:: 

      import paddlescience as psci
      pde = psci.pde.Laplace2D()


  *set_bc_value(bc_value, bc_check_dim=None)*

      为PDE设置边界值（狄里克雷边界条件）

      **参数：**
         - **bc_value** - 数组。

         - **bc_check_dim** (list) - 可选项，默认为None。如果不是None，该列表必须包含要设置边界条件值的维度。如果是None，则会在网络输出的所有维度设置边界条件值。

.. automodule:: paddlescience.pde.pde_navier_stokes

   二维时间无关纳维-斯托克斯方程

.. math::

      \begin{eqnarray*}
         \frac{\partial u}{\partial x} + \frac{\partial u}{\partial y} & = & 0,   \\
         u \frac{\partial u}{\partial x} +  v \frac{\partial u}{\partial y} - \frac{\nu}{\rho} \frac{\partial^2 u}{\partial x^2} - \frac{\nu}{\rho}  \frac{\partial^2 u}{\partial y^2} + dp/dx & = & 0,\\
         u \frac{\partial v}{\partial x} +  v \frac{\partial v}{\partial y} - \frac{\nu}{\rho} \frac{\partial^2 v}{\partial x^2} - \frac{\nu}{\rho}  \frac{\partial^2 v}{\partial y^2} + dp/dy & = & 0.
      \end{eqnarray*}


   参数：
       - **nu** (*float*）- 运动粘度。

       - **rho** (*float*）- 密度。

   **样例：**

   .. code-block:: python

      import paddlescience as psci
      pde = psci.pde.NavierStokes(0.01, 1.0)

   *set_bc_value(bc_value, bc_check_dim=None)*
   
      为PDE设置边界值（狄里克雷边界条件）

   **参数：**
       - **bc_value** - 数组。
       - **bc_check_dim** (list）- 可选项，默认为None。如果不是None，该列表必须包含需要设置边界条件值的维度。如果是None，则会在网络输出的所有维度上设置边界条件值。
