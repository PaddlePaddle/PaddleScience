FROM nvcr.io/nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    g++ gcc cmake wget xz-utils git make zlib1g-dev openssl libssl-dev libffi-dev libeigen3-dev sqlite3 libsqlite3-dev libx11-dev libgl1-mesa-dev bzip2 libbz2-dev liblzma-dev\
    # pymesh dependency
    libeigen3-dev libgmp-dev libgmpxx4ldbl libmpfr-dev libboost-dev libboost-thread-dev libtbb-dev && \
    apt-get clean autoclean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://www.python.org/ftp/python/3.7.11/Python-3.7.11.tgz && \
    tar -xf Python-3.7.11.tgz && \
    cd Python-3.7.11 && \
    ./configure --with-ssl && \
    make -j -notest && \
    make install && \
    cd .. && rm -rf Python-3.7.11 && rm Python-3.7.11.tgz && \
    ln -s /usr/local/bin/python3.7 /usr/local/bin/python
    
RUN python3 -m pip install --upgrade pip && pip install pysdf open3d db-sqlite3

# install PaddleScience and pymesh
COPY ./PaddleScience /src/PaddleScience
COPY ./pymesh.tar.xz /src/pymesh.tar.xz 

RUN cd /src && \
    ls /src && \
    tar -xvf pymesh.tar.xz && \
    cd pymesh && \
    python3 setup.py install

RUN cd /src && \
    python3 -m ensurepip --upgrade && \
    pip3 config set global.index-url https://pypi.doubanio.com/simple/ && \
    # ensure paddle install normally
    ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/local/cuda/lib64/libcudnn.so && \
    ln -s /usr/local/cuda/lib64/libcublas.so.11 /usr/local/cuda/lib64/libcublas.so && \
    echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64/"' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu"' >> /etc/profile && \
    source ~/.bashrc && \
    source /etc/profile && \
    python3 -m pip install paddlepaddle-gpu==0.0.0.post116 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html && \
    cd PaddleScience && pip3 install -r requirements.txt && pip install -e . && \
    rm -rf /src
