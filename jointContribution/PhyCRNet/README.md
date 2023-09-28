# PhyCRNet

Physics-informed convolutional-recurrent neural networks for solving spatiotemporal PDEs

Paper link: [[Journal Paper](https://www.sciencedirect.com/science/article/pii/S0045782521006514)], [[ArXiv](https://arxiv.org/pdf/2106.14103.pdf)]

By: [Pu Ren](https://scholar.google.com/citations?user=7FxlSHEAAAAJ&hl=en), [Chengping Rao](https://github.com/Raocp), [Yang Liu](https://coe.northeastern.edu/people/liu-yang/), [Jian-Xun Wang](http://sites.nd.edu/jianxun-wang/) and [Hao Sun](https://web.mit.edu/haosun/www/#/home)

## Highlights

- Present a Physics-informed discrete learning framework for solving spatiotemporal PDEs without any labeled data
- Proposed an encoder-decoder convolutional-recurrent scheme for low-dimensional feature extraction
- Employ hard-encoding of initial and boundary conditions
- Incorporate autoregressive and residual connections to explicitly simulate the time marching

## 参考

- <https://github.com/isds-neu/PhyCRNet/>

## 原仓库环境

- Python 3.6.13，使用Pytorch 1.6.0
- [Pytorch](https://pytorch.org/) 1.6.0，random_fields.py使用的torch.ifft在更高版本不支持。如果不生成数据集，可以使用其他版本。
- matplotlib, numpy, scipy
- post_process 中 `x = x[:-1]` 一行需要注释

## 数据集

创建目录和生成测试数据集

``` shell
mkdir -p output/data/2dBurgers/ output/data/2dFN/
mkdir -p output/figures/2dBurgers/ output/figures/2dFN/

python Datasets/Burgers_2d_solver_HighOrder.py
# 暂时没有用到
# python Datasets/FN_2d_solver_HighOrder.py
```

## 运行

``` shell
python Codes/PhyCRNet_burgers.py
```

## 注意

训练网络 steps 可以从较小步骤开始，比如100，然后修改为200
