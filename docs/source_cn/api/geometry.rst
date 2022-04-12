几何图形
===================================

.. automodule:: paddlescience.geometry.rectangular

二维矩形

   **参数:**

   - **space_origin** - 矩形左下角的坐标。
   - **space_extent** - 矩形右上角的坐标。

**样例:**
   .. code-block:: python

      import paddlescience as psci
      geo = psci.geometry.Rectangular(space_origin=(0.0,0.0), space_extent=(1.0,1.0))

.. automodule:: paddlescience.geometry.geometry_discrete

离散几何

get_bc_index()
   获得边界索引
   **Returns**: bc-index - 返回边界上点的索引。
   **Return Type**: numpy数组。
      
get_space_domain()
   获取空间域上的坐标
   **Returns**: space_domain - 返回空间上的坐标
   **Return Type**: numpy数组。
