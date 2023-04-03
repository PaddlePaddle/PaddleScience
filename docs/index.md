# 欢迎使用 PaddleScience

## 安装

1. 安装 PaddlePaddle

    请在 [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html) 官网按照您的运行环境，安装对应版本的 **develop** 版的 PaddlePaddle

2. 下载 PaddleScience

    请按以下命令从 github 上克隆 PaddleScience 项目，进入到 PaddleScience 目录下，
    并将 PaddleScience 目录添加到系统环境变量 `PYTHONPATH` 中。

    ```shell linenums="1"
    git clone https://github.com/PaddlePaddle/PaddleScience.git
    ```

3. 安装必要的依赖包

    ```shell linenums="1"
    pip install -r requirements.txt

    # 安装较慢时可以加上-i选项，提升下载速度
    # pip install -r requirements.txt -i https://pypi.douban.com/simple/
    ```

    ???+ Info "安装注意事项"

        如需使用外部导入STL文件来构建几何，以及使用加密采样等功能，还需额外安装 [pymesh](https://pymesh.readthedocs.io/en/latest/installation.html#download-the-source)（推荐编译安装） 和 [open3d](https://github.com/isl-org/Open3D/tree/master#python-quick-start)（推荐pip安装）

## 使用

1. 进入 PaddleScience 目录，并添加该目录到系统环境变量 PYTHONPATH 中

    ``` shell linenums="1"
    cd PaddleScience
    export PYTHONPATH=$PWD:$PYTHONPATH
    ```

2. 运行案例

    - 运行内置的案例（以 **ldc2d_unsteady_Re10.py** 为例）

        ``` shell linenums="1"
        cd examples/ldc/
        python ./ldc2d_unsteady_Re10.py
        ```

    - 编写自己的案例（假设案例命为demo）

        推荐在 `examples/` 下新建 `demo` 文件夹，然后在 `demo` 文件夹下新建 `demo.py`，最后在 `demo.py` 文件下使用 PaddleScience 提供的 [API](./zh/api/arch.md) 编写代码

        ```py linenums="1" title="demo.py"
        import ppsci

        # write your code here...
        ```

        编写完毕后运行你的代码

        ```shell linenums="1"
        cd examples/demo
        python demo.py
        ```
