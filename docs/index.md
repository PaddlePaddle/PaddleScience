# 欢迎使用 PaddleScience

## 安装

1. 安装 PaddlePaddle

    请在 [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html) 官网按照您的运行环境，安装对应版本的 **develop** 版的 PaddlePaddle

2. 下载 PaddleScience

    请按以下命令从 github 上克隆 PaddleScience 项目，进入到 PaddleScience 目录下，
    并将 PaddleScience 目录添加到系统环境变量 `PYTHONPATH` 中。

    ```shell linenums="1"
    git clone https://github.com/PaddlePaddle/PaddleScience.git
    cd PaddleScience

    export PYTHONPATH=$PWD:$PYTHONPAT
    ```

3. 安装必要的依赖包

    ```shell linenums="1"
    pip install -r requirements.txt

    # 安装较慢时可以加上-i选项，提升下载速度
    # pip install -r requirements.txt -i https://pypi.douban.com/simple/
    ```

## 使用

请参考 [快速开始](./zh/quickstart.md)
<!-- ## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files. -->

## 共同建设

PaddleScience 作为一个开源项目，欢迎来各行各业的伙伴携手共同建设基于飞桨的AI for Science领域顶尖开源项目, 打造活跃的前瞻性的AI for Science开源社区，建立产学研闭环，推动科研创新与产业赋能。

了解 [飞桨AI for Science共创计划](https://www.paddlepaddle.org.cn/science) 加入我们
