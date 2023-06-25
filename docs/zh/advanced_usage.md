# 进阶使用指南

本文档介绍如何使用 PaddleScience 中分布式训练(暂时只支持数据并行)、混合精度训练、梯度累加等功能。

## 1. 分布式训练

### 1.1 数据并行

接下来以 `examples/pipe/poiseuille_flow.py` 为例，介绍如何正确使用 PaddleScience 的数据并行功能。分布式训练细节可以参考：[Paddle-使用指南-分布式训练-快速开始-数据并行](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/cluster_quick_start_collective_cn.html)。

1. 在 constraint 实例化完毕后，重新修改 `ITERS_PER_EPOCH` 为经过了自动多卡数据切分后的 `dataloader` 的长度（一般情况下其长度等于单卡 dataloader 的长度除以卡数，向上取整），如代码中黄色高亮行所示。

    ``` py linenums="146" title="examples/pipe/poiseuille_flow.py" hl_lines="24"
    --8<--
    examples/pipe/poiseuille_flow.py:146:172
    --8<--
    ```

2. 使用分布式训练命令启动训练

    ``` sh
    # 指定 0,1,2,3 张卡启动分布式训练
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python -m paddle.distributed.launch --gpus="0,1,2,3" poiseuille_flow.py
    ```

<!-- ### 1.2 模型并行

TODO -->

## 2. 自动混合精度训练

接下来介绍如何正确使用 PaddleScience 的自动混合精度精度功能。自动混合精度训练细节可以参考：[Paddle-使用指南-性能调优-自动混合精度训练（AMP）](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/performance_improving/amp_cn.html)。

实例化 `Solver` 时加上 2 个参数: `use_amp=True`, `amp_level="O1"`(或`amp_level="O2"`)。如代码中黄色高亮行所示，通过指定 `use_amp=True`，开启自动混合精度功能，其次通过设置 `amp_level="O1"`，指定混合精度所用的模式，`O1` 为自动混合精度，`O2` 为更激进的纯 fp16 训练模式，一般推荐使用 `O1`。

``` py linenums="1" hl_lines="5 6"
# initialize solver
solver = ppsci.solver.Solver(
    ...,
    ...,
    use_amp=True,
    amp_level="O1", # or amp_level="O2"
)
```

## 3. 梯度累加

接下来介绍如何正确使用 PaddleScience 的梯度累加功能。自动混合精度训练细节可以参考：[Paddle-使用指南-性能调优-自动混合精度训练（AMP）-动态图下使用梯度累加](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/performance_improving/amp_cn.html#dongtaituxiashiyongtiduleijia)。

实例化 `Solver` 时加上 1 个参数: `update_freq=N_ACCUMULATE`。如代码中黄色高亮行所示，`N_ACCUMULATE` 可以是 2 或者更大的倍数，推荐使用 2、4、8，此时全局 `batch size` 等于 `update_freq * batch size`。梯度累加对于部分训练任务有一定的性能提升。

``` py linenums="1" hl_lines="5"
# initialize solver
solver = ppsci.solver.Solver(
    ...,
    ...,
    update_freq=2, # or 4, 8
)
```
