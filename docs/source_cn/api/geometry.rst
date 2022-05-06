几何图形
===================================

.. automodule:: paddlescience.geometry.rectangular

.. py:class:: Rectangular(space_origin=None, space_extent=None)

   二维矩形

      **参数:**

      - **space_origin** - 矩形左下角的坐标。
      - **space_extent** - 矩形右上角的坐标。

   **样例**

      .. code-block:: python

         import paddlescience as psci
         geo = psci.geometry.Rectangular(space_origin=(0.0,0.0), space_extent=(1.0,1.0))

.. automodule:: paddlescience.geometry.geometry_discrete

.. py:class:: GeometryDiscrete()

   离散几何

   .. py:function:: get_bc_index()
   
      获得边界索引

         **返回:** bc-index - 返回边界上点的索引。

         **返回类型:** numpy数组。
      
   .. py:function:: get_space_domain()

      获取空间域上的坐标

         **返回:** space_domain - 返回空间上的坐标

         **返回类型:** numpy数组。
