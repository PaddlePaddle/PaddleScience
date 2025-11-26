#!/bin/bash
set -ex

echo "==== Step 1: Checking Python Environment ===="
which python
python -V
echo "==== Python Environment Verified ===="


echo "==== Step 2: Checking CMake Environment ===="
# Verify if cmake exists in current environment. If not, download a temporary version.
if command -v cmake >/dev/null 2>&1; then
    echo "CMake found: $(cmake --version | head -n 1)"
else
    echo "CMake not found. Installing temporary CMake 3.23.0 ..."
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/cmake-3.23.0-linux-x86_64.tar.gz
    tar -zxvf cmake-3.23.0-linux-x86_64.tar.gz
    rm -f cmake-3.23.0-linux-x86_64.tar.gz
    export PATH=$PWD/cmake-3.23.0-linux-x86_64/bin:$PATH
    echo "Temporary CMake installed: $(cmake --version | head -n 1)"
fi
echo "CMake environment ready."


echo "==== Step 3: Downloading PyMesh Package ===="
# Download PyMesh package if not already present.
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/PyMesh.tar.gz
echo "Download completed. Extracting package..."
tar -zxvf PyMesh.tar.gz
echo "PyMesh package extracted."


echo "==== Step 4: Entering PyMesh Directory ===="
cd PyMesh
export PYMESH_PATH=$(pwd)
echo "PYMESH_PATH is set to: $PYMESH_PATH"


echo "==== Step 5: Installing GCC-9 Toolchain ===="
# Install gcc-9 and configure build toolchain.
apt-get install gcc-9 -y
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export LDFLAGS="-Wl,--allow-multiple-definition"
echo "GCC-9 and linker flags configured."


echo "==== Step 6: Installing Required System Libraries ===="
# Install core dependencies required by PyMesh.
apt-get install \
    libeigen3-dev \
    libgmp-dev \
    libgmpxx4ldbl \
    libmpfr-dev \
    libboost-dev \
    libboost-thread-dev \
    libtbb-dev -y
echo "System dependencies installed."


echo "==== Step 7: Installing Python Dependencies ===="
# Install Python dependencies including PyBind11, pysdf, and Open3D.
python -m pip install -r $PYMESH_PATH/python/requirements.txt
python -m pip install pybind11 -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install pysdf -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install open3d -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "Python dependencies installed."


echo "==== Step 8: Building PyMesh ===="
# Build PyMesh using setup.py
python setup.py build
echo "PyMesh build completed."


echo "==== Step 9: Installing PyMesh ===="
# Install PyMesh into current Python environment.
python setup.py install
echo "PyMesh installation completed."


echo "==== Step 10: Cleaning Temporary Files ===="
cd ..

# Remove downloaded archive and build artifacts.
rm -f PyMesh.tar.gz
rm -rf PyMesh/build

# Remove temporary CMake directory if it was installed.
if [ -d "./cmake-3.23.0-linux-x86_64" ]; then
    echo "Removing temporary CMake directory..."
    rm -rf ./cmake-3.23.0-linux-x86_64
fi

echo "Cleanup completed."


# Optional test section
# echo "==== Step 11: Running PyMesh Tests ===="
# python -c "import pymesh; pymesh.test()"
# echo "PyMesh tests finished."
