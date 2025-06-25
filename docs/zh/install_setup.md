# 欢迎使用 PaddleScience

## 1. 开始安装

### 1.1 从 docker 镜像启动[可选]

=== "从 DockerHub 拉取镜像"

    ``` sh
    # pull image
    docker pull hydrogensulfate/paddlescience

    # create a container named 'paddlescience_container' using the pulled image
    ## docker version < 19.03
    nvidia-docker run --name paddlescience_container --network=host -it --shm-size 64g hydrogensulfate/paddlescience:latest /bin/bash

    ## docker version >= 19.03
    # docker run --name paddlescience_container --gpus all --network=host -it shm-size 64g hydrogensulfate/paddlescience:latest /bin/bash
    ```

    !!! note

        Dockerhub 拉取的镜像**仅**预装了运行 PaddleScience 所需的依赖包，如 pymesh、open3d，**并不包含 PaddleScience**。
        因此请在镜像拉取和容器构建完成后，参考 [1.4 安装 PaddleScience](#14-paddlescience) 中的步骤，在容器中安装 PaddleScience。

=== "通过 Dockerfile 构建镜像"

    ``` sh
    git clone https://github.com/PaddlePaddle/PaddleScience.git
    cd PaddleScience/docker/
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/docker/pymesh.tar.xz
    bash run.sh
    ```

    如果出现因网络问题导致的 docker 构建时 apt 下载报错，则重复执行 `bash run.sh` 直至构建完成。

    更多关于 Paddle Docker 的安装和使用，请参考 [Docker 安装](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/docker/fromdocker.html)。

### 1.2 python 环境安装[可选]

如果你还没有 python 环境或者 python 版本小于 3.9，则推荐使用 Anaconda 安装并配置 python 环境，否则可以忽略本步骤。

1. 根据系统环境，从 [https://repo.anaconda.com/archive/](https://repo.anaconda.com/archive/) 中下载对应的 Anaconda3 安装包并手动安装。
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

#### 1.4.1 安装基础功能

从以下三种安装方式中**任选一种**。

=== "git 源码安装[**推荐**]"

    执行以下命令，从 github 上 clone PaddleScience 源代码，并以 editable 的方式安装 PaddleScience。

    --8<--
    ./README.md:git_install
    --8<--

=== "pip 安装"

    执行以下命令以 pip 的方式安装最新版本的 PaddleScience。

    --8<--
    ./README.md:pip_install
    --8<--

=== "conda 安装"

    执行以下命令以 conda 的方式安装 release / nightly build 版本的 PaddleScience。

    --8<--
    ./README.md:conda_install
    --8<--

=== "设置 PYTHONPATH"

    如果在您的环境中，上述两种方式都无法正常安装，则可以选择本方式，在终端内将环境变量 `PYTHONPATH` 临时设置为 `PaddleScience` 的**绝对路径**，如下所示。

    === "Linux"

        ``` sh
        cd PaddleScience/
        export PYTHONPATH=$PYTHONPATH:$PWD
        python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple # manually install requirements
        ```

    === "Windows"

        ``` sh
        cd PaddleScience/
        set PYTHONPATH=%PYTHONPATH%;%CD%
        python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple # manually install requirements
        ```

    上述方式的优点是步骤简单无需安装，缺点是当环境变量生效的终端被关闭后，需要重新执行上述命令设置 `PYTHONPATH` 才能再次使用 PaddleScience，较为繁琐。

#### 1.4.2 安装Mesh几何[可选]

PaddleScience 提供了两种复杂几何类型，如下所示：

| API 名称 | 支持的文件类型 | 安装方式 | 使用方式 |
| -- | -- | -- | -- |
|[`ppsci.geometry.Mesh`](./api/geometry.md#ppsci.geometry.Mesh) | `.obj`, `.ply`, `.off`, `.stl`, `.mesh`, `.node`, `.poly` and `.msh`| 参考下方的 "PyMesh 安装命令"| `ppsci.geometry.Mesh(mesh_path)`|
|[`ppsci.geometry.SDFMesh`](./api/geometry.md#ppsci.geometry.SDFMesh "实验性功能") | `.stl` | `pip install warp-lang 'numpy-stl>=2.16,<2.17'` | `ppsci.geometry.SDFMesh.from_stl(stl_path)` |

!!! warning "相关案例运行说明"

    [Bracket](./examples/aneurysm.md)、[Aneurysm](./examples/aneurysm.md) 等个别案例使用了 `ppsci.geometry.Mesh` 接口构建复杂几何，因此这些案例运行前需要按照下方给出的命令，安装 open3d、
    pybind11、pysdf、PyMesh 四个依赖库（上述**1.1 从 docker 镜像启动**中已安装上述依赖库）。如使用 `ppsci.geometry.SDFMesh` 接口构建复杂几何，则只需要安装 `warp-lang` 即可。

=== "open3d 安装命令"

    ``` sh
    python -m pip install open3d -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

=== "pybind11 安装命令"

    ``` sh
    python -m pip install pybind11 -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

=== "pysdf 安装命令"

    ``` sh
    python -m pip install pysdf
    ```

=== "PyMesh 安装命令"

    在安装 PyMesh 之前，首先需通过 `cmake --version` 确认环境中是否已安装 cmake。
    如果未安装，可以按照下列命令下载并解压 cmake 包，然后将其添加到 PATH 变量中以完成安装。

    ``` sh
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/cmake-3.23.0-linux-x86_64.tar.gz
    tar -zxvf cmake-3.23.0-linux-x86_64.tar.gz
    rm -f cmake-3.23.0-linux-x86_64.tar.gz
    PATH=$PWD/cmake-3.23.0-linux-x86_64/bin:$PATH

    # cmake --version
    # cmake version 3.23.0

    # CMake suite maintained and supported by Kitware (kitware.com/cmake).
    ```

    推荐以 setup 的方式安装 PyMesh 库，命令如下：

    ``` sh
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/PyMesh.tar.gz
    tar -zxvf PyMesh.tar.gz

    # 也可以使用 git 命令下载，速度可能会比较慢
    # git clone https://github.com/PyMesh/PyMesh.git
    # git submodule update --init --recursive --progress

    cd PyMesh
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

    python -m pip install --user -r $PYMESH_PATH/python/requirements.txt
    python setup.py build
    python setup.py install --user

    # test whether installed successfully
    python -c "import pymesh; pymesh.test()"

    # Ran 175 tests in 3.150s

    # OK (SKIP=2)
    ```

    !!! warning "安装注意事项"

        1. 由于网络问题，`git submodule update` 过程中可能某些 submodule 会 clone 失败，此时只需
        反复执行 `git submodule update --init --recursive --progress` 直到所有库都 clone 成功即可。

        2. 所有 submodule 都 clone 成功后，请检查 `PyMesh/third_party/` 下是否有空文件夹，若有则需
        手动找到并删除这些空文件夹，再执行 `git submodule update --init --recursive --progress` 命
        令即可恢复这些空文件夹至正常含有文件的状态，此时再继续执行剩余安装命令即可。

        3. 由于自测工具 nose 未适配 Python>=3.10，因此执行 `pymesh.test()` 会报错，**但这不影响 pymesh 正常使用**。

#### 1.4.3 安装第三方库[可选]

PaddleScience 提供了多种第三方库供用户在开发时使用，这些库位于 `ppsci/externals` 目录下，可以通过 `git submodule` 命令进行下载、安装，或者直接安装我们提供的 whl 包。以下是具体操作步骤：

=== "安装"

    === "deepali"

        ``` sh
        cd PaddleScience
        git submodule update --init ppsci/externals/deepali
        # install from source(recommended)
        python -m pip install -e ppsci/externals/deepali

        # install from whl
        python -m pip install https://paddle-qa.bj.bcebos.com/deepali/whl/latest/dist/hf_deepali-0.1.0-py3-none-any.whl
        ```

    === "neuraloperator"

        ``` sh
        cd PaddleScience
        git submodule update --init ppsci/externals/neuraloperator
        # install from source(recommended)
        python -m pip install -e ppsci/externals/neuraloperator

        # install from whl
        python -m pip install https://paddle-qa.bj.bcebos.com/neuraloperator/whl/cuda11.8/latest/dist/neuraloperator-0.3.0-py3-none-any.whl
        ```

    === "open3d"

        ``` sh
        cd PaddleScience
        git submodule update --init ppsci/externals/Open3D
        # install from whl(cuda 118)(recommended)
        python -m pip install https://paddle-qa.bj.bcebos.com/Open3D/whl/cuda11.8/latest/open3d-0.18.0-cp310-cp310-linux_x86_64.whl
        # install from whl(cuda 123)(recommended)
        python -m pip install https://paddle-qa.bj.bcebos.com/Open3D/whl/cuda12.3/latest/open3d-0.18.0-cp310-cp310-linux_x86_64.whl

        # install from source: https://github.com/PFCCLab/Open3D?tab=readme-ov-file#build-and-install
        ```

    === "paddle_harmonics"

        ``` sh
        cd PaddleScience
        git submodule update --init ppsci/externals/paddle_harmonics
        # install from source(recommended)
        python -m pip install -e ppsci/externals/paddle_harmonics

        # install from whl(cuda 118)
        python -m pip install https://paddle-qa.bj.bcebos.com/paddle_harmonics/whl/latest/dist/paddle_harmonics-0.1.0-py3-none-any.whl
        ```

    === "paddle_scatter"

        ``` sh
        cd PaddleScience
        git submodule update --init ppsci/externals/paddle_scatter
        # install from source(recommended)
        python -m pip install -e ppsci/externals/paddle_scatter
        ```

    === "tensorly"

        ``` sh
        cd PaddleScience
        git submodule update --init ppsci/externals/tensorly
        # install from source(recommended)
        python -m pip install -e ppsci/externals/tensorly

        # install from whl
        python -m pip install https://paddle-qa.bj.bcebos.com/tensorly/whl/latest/dist/tensorly-0.9.0-py3-none-any.whl
        ```

    === "warp"

        ``` sh
        cd PaddleScience
        git submodule update --init ppsci/externals/warp
        # install from source(recommended)
        python -m pip install -e ppsci/externals/warp

        # install from whl, see: https://nvidia.github.io/warp/installation.html#
        python -m pip install warp-lang
        ```

=== "使用"

    以 `tensorly` 为例，使用方法如下：

    ``` python
    >>> from ppsci import externals
    >>> print(externals.__all__)
    ['deepali', 'open3d', 'paddle_harmonics', 'paddle_scatter', 'tensorly', 'warp']

    >>> tl = externals.tensorly
    >>> tl.set_backend("paddle")

    >>> x = tl.tensor(np.ones((3, 3)))
    Tensor(shape=[3, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])
    ```

请根据以上步骤下载、安装和使用您所需的第三方库。

## 2. 验证安装

- 执行以下代码，验证安装的 PaddleScience 基础功能是否正常。

    ``` sh
    python -c "import ppsci; ppsci.run_check()"
    ```

    如果出现 `PaddleScience is installed successfully.✨ 🍰 ✨`，则说明安装验证成功。

- [可选]如果已按照 [1.4.2 安装Mesh几何](#142-mesh) 正确安装了 4 个依赖库，则可以执行以下代码，
    验证 PaddleScience 的 `ppsci.geometry.Mesh` 模块是否能正常运行。

    ``` sh
    python -c "import ppsci; ppsci.run_check_mesh()"
    ```

    如果出现 `ppsci.geometry.Mesh module running successfully.✨ 🍰 ✨`，则说明该模块运行正常。

## 3. 开始使用

- 运行内置的案例（以 **ldc2d_unsteady_Re10.py** 为例）

    ``` sh
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

    ``` sh
    cd examples/demo
    python ./demo.py
    ```

    如不了解接下来该如何基于 PaddleScience 编写代码，则推荐参考 [**快速开始**](./quickstart.md) 和其他案例的文档、代码，进一步了解如何使用 `ppsci` 下的模块来编写自己的案例。
