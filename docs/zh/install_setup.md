# 欢迎使用 PaddleScience

## 1. 开始安装

### 1.1 从 docker 镜像启动[可选]

如果你对 docker 有一定了解，则可以通过我们提供的 docker 文件，直接构建出能运行 PaddleScience 的环境。按照下列步骤构建 docker 并自动进入该环境，以运行 PaddleScience。

1. 下载 PyMesh 预编译文件压缩包 [pymesh.tar.xz](https://paddle-org.bj.bcebos.com/paddlescience/docker/pymesh.tar.xz)，并放置在 `PaddleScience/docker/` 目录下
2. 执行 `bash run.sh`，等待 docker build 完毕后自动进入环境。如果出现因网络问题导致的 apt 下载报错，则重复执行 `bash run.sh` 直至 build 完成即可。
3. 在 docker 环境中，执行 `ldconfig`

### 1.2 python 环境安装[可选]

如果你还没有 python 环境或者 python 版本小于 3.9，则推荐使用 Anaconda 安装并配置 python 环境，否则可以忽略本步骤。

1. 根据系统环境，从 [https://repo.anaconda.com/archive/](https://repo.anaconda.com/archive/) 中下载对应的 Anaconda3 安装包，并手动安装。
2. 创建 python 3.9 环境，并进入该环境。

    ``` sh
    # 使用 conda 创建 python 环境，并命名为 "ppsci_py39"
    conda create -n ppsci_py39 python=3.9

    # 进入创建好的 "ppsci_py39" 环境
    conda activate ppsci_py39
    ```

### 1.3 安装 PaddlePaddle

--8<--
./README.md:paddle_install
--8<--

### 1.4 安装 PaddleScience

从 [1.4.1 git 源码安装](#141-git) 和 [1.4.2 pip 安装](#142-pip) 任选一种安装方式即可。

#### 1.4.1 git 源码安装[**推荐**]

执行以下命令，从 github 上 clone PaddleScience 源代码，并以 editable 的方式安装 PaddleScience。

--8<--
./README.md:git_install
--8<--

#### 1.4.2 pip 安装

执行以下命令以 pip 的方式安装 PaddleScience。

``` shell
pip install paddlesci==1.2.1
```

#### 1.4.3 额外依赖安装[可选]

如需使用 `.obj`, `.ply`, `.off`, `.stl`, `.mesh`, `.node`, `.poly` and `.msh` 等复杂几何文件构建几何（计算域），以及使用加密采样等功能，则需按照下方给出的命令，安装 open3d、
pybind11、pysdf、PyMesh 四个依赖库。

否则无法使用 `ppsci.geometry.Mesh` 等基于复杂几何文件的 API，因此也无法运行如 [Aneurysm](./examples/aneurysm.md) 等依赖 `ppsci.geometry.Mesh` API 的复杂案例。

=== "open3d 安装命令"

    ``` sh
    pip install open3d -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

=== "pybind11 安装命令"

    ``` sh
    pip install pybind11 -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

=== "pysdf 安装命令"

    ``` sh
    pip install pysdf
    ```

=== "PyMesh 安装命令"

    在安装 PyMesh 之前，首先需通过 `cmake --version` 确认环境中是否已安装 cmake。
    如未安装，可按照下列命令下载、解压 cmake 包，并添加到 `PATH` 变量中即可完成安装。

    ``` sh
    wget -nc https://cmake.org/files/v3.23/cmake-3.23.0-linux-x86_64.tar.gz
    tar -zxvf cmake-3.23.0-linux-x86_64.tar.gz
    rm -f cmake-3.23.0-linux-x86_64.tar.gz
    PATH=$PWD/cmake-3.23.0-linux-x86_64/bin:$PATH

    # cmake --version
    # cmake version 3.23.0

    # CMake suite maintained and supported by Kitware (kitware.com/cmake).
    ```

    PyMesh 库需要以 setup 的方式进行安装，命令如下：

    ``` sh
    git clone https://github.com/PyMesh/PyMesh.git
    cd PyMesh

    git submodule update --init --recursive --progress
    export PYMESH_PATH=`pwd`

    apt-get install \
        libeigen3-dev \
        libgmp-dev \
        libgmpxx4ldbl \
        libmpfr-dev \
        libboost-dev \
        libboost-thread-dev \
        libtbb-dev \
        python3-dev

    python -m pip install -r $PYMESH_PATH/python/requirements.txt
    python setup.py build
    python setup.py install --user

    # test whether installed successfully
    python -c "import pymesh; pymesh.test()"

    # Ran 175 tests in 3.150s

    # OK (SKIP=2)
    ```

    !!! warning "安装注意事项"

        安装过程中可能会出现两个问题，可以按照以下方式解决：

        1. 由于网络问题，`git submodule update` 过程中可能某些 submodule 会 clone 失败，此时只需
        反复执行 `git submodule update --init --recursive --progress` 直到所有库都 clone 成功即可。

        2. 所有 submodule 都 clone 成功后，请检查 `PyMesh/third_party/` 下是否有空文件夹，若有则需
        手动找到并删除这些空文件夹，再执行 `git submodule update --init --recursive --progress` 命
        令即可恢复这些空文件夹至正常含有文件的状态，此时再继续执行剩余安装命令即可。

## 2. 验证安装

- 执行以下代码，验证安装的 PaddleScience 基础功能是否正常。

    ``` shell
    python -c "import ppsci; ppsci.run_check()"
    ```

    如果出现 `PaddleScience is installed successfully.✨ 🍰 ✨`，则说明安装验证成功。

- [可选]如果已按照 [1.4.3 额外依赖安装](#143) 正确安装了 4 个额外依赖库，则可以执行以下代码，
    验证 PaddleScience 的 `ppsci.geometry.Mesh` 模块是否能正常运行。

    ``` shell
    python -c "import ppsci; ppsci.run_check_mesh()"
    ```

    如果出现 `ppsci.geometry.Mesh module running successfully.✨ 🍰 ✨`，则说明该模块运行正常。

## 3. 开始使用

- 运行内置的案例（以 **ldc2d_unsteady_Re10.py** 为例）

    ``` shell
    cd examples/ldc/
    python ./ldc2d_unsteady_Re10.py
    ```

- 编写自己的案例（假设案例名为 demo）

    推荐在 `examples/` 下新建 `demo` 文件夹，然后在 `demo` 文件夹下新建 `demo.py`，最后在 `demo.py` 文件中使用 PaddleScience 提供的 [API](./api/arch.md) 编写代码。

    ``` py linenums="1" title="examples/demo/demo.py"
    import ppsci

    # write your code here...
    ```

    编写完毕后运行你的代码

    ``` shell
    cd examples/demo
    python ./demo.py
    ```

    如不了解接下来该如何基于 PaddleScience 编写代码，则推荐参考 [**快速开始**](./quickstart.md) 和其他案例的文档、代码，进一步了解如何使用 `ppsci` 下的模块来编写自己的案例。
