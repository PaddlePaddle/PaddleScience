# Welcome to PaddleScience

## 1. Installation

### 1.1 Start from Docker Image [Optional]

=== "Pull Image from DockerHub"

    ``` sh
    # pull image
    docker pull hydrogensulfate/paddlescience

    # create a container named 'paddlescience_container' using the pulled image
    ## docker version < 19.03
    nvidia-docker run --name paddlescience_container --network=host -it --shm-size 64g hydrogensulfate/paddlescience:latest /bin/bash

    ## docker version >= 19.03
    # docker run --name paddlescience_container --gpus all --network=host -it --shm-size 64g hydrogensulfate/paddlescience:latest /bin/bash
    ```

    !!! note

        The image pulled from Dockerhub **only** pre-installs the dependencies required to run PaddleScience, such as pymesh and open3d, and **does not contain PaddleScience**.
        Therefore, after the image pull and container construction are completed, please refer to the steps in [1.4 Install PaddleScience](#14-install-paddlescience) to install PaddleScience in the container.

=== "Build Image via Dockerfile"

    ``` sh
    git clone https://github.com/PaddlePaddle/PaddleScience.git
    cd PaddleScience/docker/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/docker/pymesh.tar.xz
    bash run.sh
    ```

    If `apt` download errors occur during the docker build due to network issues, repeat `bash run.sh` until the build completes.

    For more information on the installation and use of Paddle Docker, please refer to [Docker Installation](https://www.paddlepaddle.org.cn/documentation/docs/en/install/docker/fromdocker.html).

### 1.2 Python Environment Installation [Optional]

If you do not have a python environment or the python version is less than 3.9, it is recommended to use Anaconda to install and configure the python environment, otherwise you can ignore this step.

1. Depending on your system environment, download the corresponding Anaconda3 installation package from [https://repo.anaconda.com/archive/](https://repo.anaconda.com/archive/) and install it manually.
2. Create a python 3.10 environment and enter it.

    ``` sh
    # Use conda to create a python environment and name it "ppsci_py310"
    conda create -n ppsci_py310 python=3.10

    # Enter the created "ppsci_py310" environment
    conda activate ppsci_py310
    ```

### 1.3 Install PaddlePaddle

--8<--
./README.md:paddle_install
--8<--

### 1.4 Install PaddleScience

#### 1.4.1 Install Basic Functions

Choose **one** of the following three installation methods.

=== "Git Source Installation [**Recommended**]"

    Execute the following command to clone the PaddleScience source code from github and install PaddleScience in editable mode.

    --8<--
    ./README.md:git_install
    --8<--

=== "Pip Installation"

    Execute the following command to install the latest version of PaddleScience via pip.

    --8<--
    ./README.md:pip_install
    --8<--

=== "Conda Installation"

    Execute the following command to install the release / nightly build version of PaddleScience via conda.

    --8<--
    ./README.md:conda_install
    --8<--

=== "Set PYTHONPATH"

    If neither of the above two methods can be installed properly in your environment, you can choose this method to temporarily set the environment variable `PYTHONPATH` to the **absolute path** of `PaddleScience` in the terminal, as shown below.

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

    The advantage of the above method is that the steps are simple and no installation is required. The disadvantage is that after the terminal where the environment variable takes effect is closed, you need to re-execute the above command to set `PYTHONPATH` to use PaddleScience again, which is cumbersome.

#### 1.4.2 Install Mesh Geometry [Optional]

PaddleScience provides two complex geometry types, as shown below:

| API Name | Supported File Types | Installation Method | Usage Method |
| -- | -- | -- | -- |
|[`ppsci.geometry.Mesh`](./api/geometry.md#ppsci.geometry.Mesh) | `.obj`, `.ply`, `.off`, `.stl`, `.mesh`, `.node`, `.poly` and `.msh`| Refer to "PyMesh Installation Command" below| `ppsci.geometry.Mesh(mesh_path)`|
|[`ppsci.geometry.SDFMesh`](./api/geometry.md#ppsci.geometry.SDFMesh "Experimental Feature") | `.stl` | `pip install warp-lang 'numpy-stl>=2.16,<2.17'` | `ppsci.geometry.SDFMesh.from_stl(stl_path)` |

!!! warning "Relevant Example Running Instructions"

    Individual examples such as [Bracket](./examples/aneurysm.md) and [Aneurysm](./examples/aneurysm.md) use the `ppsci.geometry.Mesh` interface to construct complex geometries, so before running these examples, you need to follow the commands given below to install four dependent libraries: open3d, pybind11, pysdf, and PyMesh (the above dependent libraries have been installed in **1.1 Start from Docker Image**). If you use the `ppsci.geometry.SDFMesh` interface to construct complex geometries, you only need to install `warp-lang`.

=== "One-click Installation [Recommended]"

    ```sh
    bash PaddleScience/install_mesh.sh
    ```

=== "Manual Installation"

    ``` sh
    python -m pip install open3d pybind11 pysdf-i https://pypi.tuna.tsinghua.edu.cn/simple

    # Before installing PyMesh, first verify if cmake is installed in the environment via `cmake --version`.
    # If not installed, you can follow the commands below to download and unzip the cmake package, and then add it to the PATH variable.
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/cmake-3.23.0-linux-x86_64.tar.gz
    tar -zxvf cmake-3.23.0-linux-x86_64.tar.gz
    rm -f cmake-3.23.0-linux-x86_64.tar.gz
    export PATH=$PWD/cmake-3.23.0-linux-x86_64/bin:$PATH

    # It is recommended to install the PyMesh library in setup mode, the command is as follows:
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/PyMesh.tar.gz
    tar -zxvf PyMesh.tar.gz

    # You can also use git command to download, the speed may be relatively slow
    # git clone https://github.com/PyMesh/PyMesh.git
    # git submodule update --init --recursive --progress

    # Install necessary dependency packages
    apt-get install \
        libeigen3-dev \
        libgmp-dev \
        libgmpxx4ldbl \
        libmpfr-dev \
        libboost-dev \
        libboost-thread-dev \
        libtbb-dev \
        python3-dev

    cd PyMesh
    export PYMESH_PATH=`pwd`
    python -m pip install -r $PYMESH_PATH/python/requirements.txt
    python setup.py build
    python setup.py install
    ```

    !!! warning "Installation Precautions"

        1. Due to network problems, some submodules may fail to clone during `git submodule update`. Just repeatedly execute `git submodule update --init --recursive --progress` until all libraries are cloned successfully.

        2. After all submodules are cloned successfully, please check if there are empty folders under `PyMesh/third_party/`. If so, you need to manually find and delete these empty folders, and then execute the `git submodule update --init --recursive --progress` command to restore these empty folders to a normal state containing files, and then continue to execute the remaining installation commands.

        3. Since the self-test tool nose has not adapted to Python>=3.10, executing `pymesh.test()` will report an error, **but this does not affect the normal use of pymesh**.

#### 1.4.3 Install Third-party Libraries [Optional]

PaddleScience provides a variety of third-party libraries for users to use during development. These libraries are located in the `ppsci/externals` directory and can be downloaded and installed via the `git submodule` command, or directly installed using the whl packages we provide. The specific operation steps are as follows:

=== "Install"

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
        python -m pip install ppsci/externals/paddle_scatter --no-build-isolation
        ```

    === "paddle_sparse"

        ``` sh
        cd PaddleScience
        git submodule update --init ppsci/externals/paddle_sparse
        # install from source(recommended)
        python -m pip install ppsci/externals/paddle_sparse --no-build-isolation
        ```

    === "paddle_cluster"

        ``` sh
        cd PaddleScience
        git submodule update --init ppsci/externals/paddle_cluster
        # install from source(recommended)
        python -m pip install ppsci/externals/paddle_cluster --no-build-isolation
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

=== "Usage"

    Taking `tensorly` as an example, the usage method is as follows:

    ``` python
    >>> from ppsci import externals
    >>> print(externals.__all__)
    ['deepali', 'neuraloperator', 'open3d', 'paddle_harmonics', 'paddle_scatter', 'paddle_sparse', 'paddle_cluster', 'tensorly', 'warp']

    >>> tl = externals.tensorly
    >>> tl.set_backend("paddle")

    >>> x = tl.tensor(np.ones((3, 3)))
    Tensor(shape=[3, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])
    ```

Please download, install and use the third-party libraries you need according to the above steps.

## 2. Verify Installation

- Execute the following code to verify whether the basic functions of installed PaddleScience are normal.

    ``` sh
    python -c "import ppsci; ppsci.run_check()"
    ```

    If `PaddleScience is installed successfully.✨ 🍰 ✨` appears, the installation verification is successful.

- [Optional] If the 4 dependent libraries have been correctly installed according to [1.4.2 Install Mesh Geometry](#142-install-mesh-geometry), you can execute the following code to verify whether the `ppsci.geometry.Mesh` module of PaddleScience works properly.

    ``` sh
    python -c "import ppsci; ppsci.run_check_mesh()"
    ```

    If `ppsci.geometry.Mesh module running successfully.✨ 🍰 ✨` appears, the module is running normally.

## 3. Start Using

- Run built-in examples (taking **ldc2d_unsteady_Re10.py** as an example)

    ``` sh
    cd examples/ldc/
    python ./ldc2d_unsteady_Re10.py
    ```

- Write your own example (assuming the example name is demo)

    It is recommended to create a new `demo` folder under `examples/`, then create `demo.py` in the `demo` folder, and finally use the [API](./api/arch.md) provided by PaddleScience to write code in the `demo.py` file.

    ``` py linenums="1" title="examples/demo/demo.py"
    import ppsci

    # write your code here...
    ```

    After writing, run your code

    ``` sh
    cd examples/demo
    python ./demo.py
    ```

    If you don't know how to write code based on PaddleScience next, it is recommended to refer to [**Quick Start**](./quickstart.md) and documents and codes of other examples to further understand how to use modules under `ppsci` to write your own examples.
