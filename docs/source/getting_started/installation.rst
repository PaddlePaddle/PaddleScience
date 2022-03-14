Installation
=============

- **Prerequisites**

    Hardware requirements: NVIDIA GPU V100, NVIDIA GPU A100

    Package dependencies: paddle, numpy, matplotlib, vtk

- **Installation**

    - Download and setup environment variable

        .. code-block::

            From GitHub: git clone https://github.com/PaddlePaddle/PaddleScience.git
            From Gitee:  git clone https://gitee.com/paddlepaddle/PaddleScience.git

            cd PaddleScience
            export PYTHONPATH=$PWD:$PYTHONPATH

    - Please refer to `PaddlePaddle <https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html>`_ to install paddle. Please choose "develop" version.

    - Install dependencies with pip 

        .. code-block::

            pip3.7 install -r requirements.txt