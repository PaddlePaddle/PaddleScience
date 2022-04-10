安装
=============

- **前置准备**

    硬件要求: NVIDIA GPU V100, NVIDIA GPU A100

    依赖包: paddle, numpy, matplotlib, vtk

- **安装**

    - 下载并且设置环境变量

        .. code-block::

            From GitHub: git clone https://github.com/PaddlePaddle/PaddleScience.git
            From Gitee:  git clone https://gitee.com/paddlepaddle/PaddleScience.git

            cd PaddleScience
            export PYTHONPATH=$PWD:$PYTHONPATH

    - 请参考`PaddlePaddle <https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html>`_来安装paddle， 并且选择"develop"版本。


    - 通过pip安装相关依赖

        .. code-block::

            pip3.7 install -r requirements.txt