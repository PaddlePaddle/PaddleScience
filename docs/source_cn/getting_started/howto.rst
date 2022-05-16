运行样例
===============

这里包含一些简单的样例用来快速演示。您可以在 ``example`` 目录找到它们。想要运行样例，只需要进入该子目录并在命令行中执行相关代码即可。

- **拉普拉斯方程**

    在矩形域中求解二维稳态拉普拉斯方程。

    .. code-block::

        cd examples/laplace2d
        python3.7 laplace2d.py

- **达西渗流**

    求解二维环境下的达西渗流问题。

    .. code-block::
        
        cd examples/darcy2d
        python3.7 darcy2d.py

- **example**

    求解矩形域中二维顶盖驱动方腔流问题。

    .. code-block::

           cd examples/ldc2d
           python3.7 ldc2d.py