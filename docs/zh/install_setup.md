# 欢迎使用 PaddleScience

## 1. 开始安装

### 1.1 安装 PaddlePaddle

请在 [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html) 官网按照您的运行环境，安装 **develop** 版的 PaddlePaddle

### 1.2 安装 PaddleScience

从 [1.2.1 git 安装](#121-git) 和 [1.2.2 pip 安装](#122-pip) 任选一种安装方式即可

#### 1.2.1 git 安装

1. 执行以下命令，从 github 上克隆 PaddleScience 项目，进入 PaddleScience 目录，并将该目录添加到系统环境变量中

    ``` shell
    git clone https://github.com/PaddlePaddle/PaddleScience.git
    # 若 github clone 速度比较慢，可以使用 gitee clone
    # git clone https://gitee.com/paddlepaddle/PaddleScience.git

    cd PaddleScience
    git checkout develop
    export PYTHONPATH=$PWD:$PYTHONPATH
    ```

2. 安装必要的依赖包

    ``` shell
    pip install -r requirements.txt
    ```

#### 1.2.2 pip 安装

执行以下命令进行 pip 安装

``` shell
pip install paddlesci
```

???+ Info "安装注意事项"

    如需使用外部导入STL文件来构建几何，以及使用加密采样等功能，还需额外安装三个依赖库：
    [pymesh](https://pymesh.readthedocs.io/en/latest/installation.html#download-the-source)（推荐编译安装）,
    [open3d](https://github.com/isl-org/Open3D/tree/master#python-quick-start)（推荐pip安装）,
    [pysdf](https://github.com/sxyu/sdf)（推荐pip安装）

## 2. 验证安装

执行以下代码，验证安装的 PaddleScience 基础功能是否正常

``` shell
python -c "import ppsci; ppsci.utils.run_check()"
```

如果出现 `PaddleScience is installed successfully.✨ 🍰 ✨`，则说明安装验证成功。

## 3. 开始使用

- 运行内置的案例（以 **ldc2d_unsteady_Re10.py** 为例）

    ``` shell
    cd examples/ldc/
    python ./ldc2d_unsteady_Re10.py
    ```

- 编写自己的案例（假设案例名为demo）

    推荐在 `examples/` 下新建 `demo` 文件夹，然后在 `demo` 文件夹下新建 `demo.py`，最后在 `demo.py` 文件中使用 PaddleScience 提供的 [API](./api/arch.md) 编写代码

    ``` py linenums="1" title="examples/demo/demo.py"
    import ppsci

    # write your code here...
    ```

    编写完毕后运行你的代码

    ``` shell
    cd examples/demo
    python ./demo.py
    ```
