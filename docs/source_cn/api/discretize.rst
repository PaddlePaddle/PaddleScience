离散化
===================================

.. autofunction:: paddlescience.discretize
    :no-undoc-member:

离散化PDE和Geometry

    **参数:**

    - **pde** (PDE) - 偏微分方程。
    - **geo** (Geometry) - 需要离散化的Geometry或其子类实例。

    **返回:**

    - **pde_disc** (PDE) - 保留参数。
    - **geo_disc** (DiscreteGeometry) - DiscreteGeometry的实例
