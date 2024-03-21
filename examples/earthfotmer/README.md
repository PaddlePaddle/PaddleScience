# Earthformer: Exploring Space-Time Transformers for Earth System Forecasting

Conventionally, Earth system (e.g., weather and climate) forecasting relies on
numerical simulation with complex physical models and hence is both expensive
in computation and demanding on domain expertise. With the explosive growth of
spatiotemporal Earth observation data in the past decade, data-driven models that
apply Deep Learning (DL) are demonstrating impressive potential for various Earth
system forecasting tasks. The Transformer as an emerging DL architecture, despite
its broad success in other domains, has limited adoption in this area. In this paper,
we propose Earthformer, a space-time Transformer for Earth system forecasting.
Earthformer is based on a generic, flexible and efficient space-time attention block,
named Cuboid Attention. The idea is to decompose the data into cuboids and apply
cuboid-level self-attention in parallel. These cuboids are further connected with a
collection of global vectors. We conduct experiments on the MovingMNIST dataset
and a newly proposed chaotic N -body MNIST dataset to verify the effectiveness
of cuboid attention and figure out the best design of Earthformer. Experiments on
two real-world benchmarks about precipitation nowcasting and El Niño/Southern
Oscillation (ENSO) forecasting show that Earthformer achieves state-of-the-art
performance.

<div align=center>
    <img src="doc/fig_arch1.jpg" width="70%" height="auto" >
</div>

## Installation

### 1. Install PaddlePaddle

Please install the <font color="red"><b>2.6.0</b></font>  or <font color="red"><b>develop</b></font> version of PaddlePaddle according to your environment on the official website of [PaddlePaddle](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html).

For example, if your environment is linux and CUDA 11.2, you can install PaddlePaddle by the following command.

``` shell
python -m pip install paddlepaddle-gpu==2.6.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

After installation, run the following command to verify if PaddlePaddle has been successfully installed.

``` shell
python -c "import paddle; paddle.utils.run_check()"
```

If `"PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now."` appears, to verify that the installation was successful.

### 2. Install PaddleScience

Clone the code of PaddleScience from [here](https://github.com/PaddlePaddle/PaddleScience.git) and install requirements.

``` shell
git clone -b develop https://github.com/PaddlePaddle/PaddleScience.git
cd PaddleScience
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
export PYTHONPATH=$PWD
```

## Example Usage

### 1. Download the data and model weights

``` shell
cd examples/yinglong
wget https://paddle-org.bj.bcebos.com/paddlescience/datasets/yinglong/hrrr_example_24vars.tar
tar -xvf hrrr_example_24vars.tar
wget https://paddle-org.bj.bcebos.com/paddlescience/datasets/yinglong/hrrr_example_69vars.tar
tar -xvf hrrr_example_69vars.tar
wget https://paddle-org.bj.bcebos.com/paddlescience/models/yinglong/inference.tar
tar -xvf inference.tar
```

### 2. Run the code

The following code runs the Yinglong model, and the model output will be saved in 'output/result.npy'.

``` shell
# YingLong-12 Layers
python predict_12layers.py mode=infer
# YingLong-24 Layers
python predict_24layers.py mode=infer
```

We also visualized the predicted wind speed at 10 meters above ground level, with an initial field of 0:00 on January 1, 2022. Click [here](https://paddle-org.bj.bcebos.com/paddlescience/docs/Yinglong/result.gif) to view the prediction results.
