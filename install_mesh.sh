#!/bin/bash
set -ex

echo "==== Step 1: Checking Python Environment ===="
which python
python -V
echo "==== Python Environment Verified ===="

echo "==== Step 2: Downloading PyMesh Package ===="
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/PyMesh.tar.gz
echo "Download completed. Extracting package..."
tar -zxvf PyMesh.tar.gz
echo "PyMesh package extracted."

echo "==== Step 3: Entering PyMesh Directory ===="
cd PyMesh
export PYMESH_PATH=`pwd`
echo "PYMESH_PATH is set to: $PYMESH_PATH"

echo "==== Step 4: Installing GCC-9 Toolchain ===="
apt-get install gcc-9 -y
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export LDFLAGS="-Wl,--allow-multiple-definition"
echo "GCC-9 and build flags configured."

echo "==== Step 5: Installing Required System Libraries ===="
apt-get install \
    libeigen3-dev \
    libgmp-dev \
    libgmpxx4ldbl \
    libmpfr-dev \
    libboost-dev \
    libboost-thread-dev \
    libtbb-dev -y
echo "System dependencies installed."

echo "==== Step 6: Installing Python Dependencies ===="
python -m pip install -r $PYMESH_PATH/python/requirements.txt
python -m pip install pybind11 -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install pysdf -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install open3d -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "Python dependencies installed."

echo "==== Step 7: Building PyMesh ===="
python setup.py build
echo "PyMesh build completed."

echo "==== Step 8: Installing PyMesh ===="
python setup.py install
echo "PyMesh installation completed."

# echo "==== Step 9: Running PyMesh Tests ===="
# python -c "import pymesh; pymesh.test()"
# echo "PyMesh tests finished."
