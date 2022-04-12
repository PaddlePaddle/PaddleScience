可视化
=============

.. automodule:: paddlescience.visu.visu_vtk

   将几何图形和数据保存为vtk文件以用于可视化

      **参数：**

      - **geo** - 几何图形，Geometry的实例
      - **data** - 要保存的数据

   **样例**

   .. code-block::

      import paddlescience as psci
      pde = psci.visu.save_vtk(geo, data, filename="output")
