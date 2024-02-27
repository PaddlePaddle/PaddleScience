# NowcastNet

=== "模型训练命令"

    暂无

=== "模型评估命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/nowcastnet/mrms.tar
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/nowcastnet/mrms.tar --output mrms.tar
    mkdir ./datasets
    tar -xvf mrms.tar -C ./datasets/
    python nowcastnet.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/nowcastnet/nowcastnet_pretrained.pdparams
    ```

## 1. 背景简介

近年来，深度学习方法已被应用于天气预报，尤其是雷达观测的降水预报。这些方法利用大量雷达复合观测数据来训练神经网络模型，以端到端的方式进行训练，无需明确参考降水过程的物理定律。
这里复现了一个针对极端降水的非线性短临预报模型——NowcastNet，该模型将物理演变方案和条件学习法统一到一个神经网络框架中，实现了端到端的优化。

## 2. 模型原理

本章节仅对 NowcastNet 的模型原理进行简单地介绍，详细的理论推导请阅读 [Skilful nowcasting of extreme precipitation with NowcastNet](https://www.nature.com/articles/s41586-023-06184-4#Abs1)。

模型的总体结构如图所示：

<figure markdown>
  ![nowcastnet-arch](https://paddle-org.bj.bcebos.com/paddlescience/docs/nowcastnet/nowcastnet.png){ loading=lazy style="margin:0 auto"}
  <figcaption>NowcastNet 网络模型</figcaption>
</figure>

模型使用预训练权重推理，接下来将介绍模型的推理过程。

## 3. 模型构建

在该案例中，用 PaddleScience 代码表示如下：

``` py linenums="24" title="examples/nowcastnet/nowcastnet.py"
--8<--
examples/nowcastnet/nowcastnet.py:24:36
--8<--
```

``` yaml linenums="35" title="examples/nowcastnet/conf/nowcastnet.yaml"
--8<--
examples/nowcastnet/conf/nowcastnet.yaml:35:53
--8<--
```

其中，`input_keys` 和 `output_keys` 分别代表网络模型输入、输出变量的名称。

## 4. 模型评估可视化

完成上述设置之后，将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`：

``` py linenums="57" title="examples/nowcastnet/nowcastnet.py"
--8<--
examples/nowcastnet/nowcastnet.py:57:61
--8<--
```

然后构建 VisualizerRadar 生成图片结果：

``` py linenums="69" title="examples/nowcastnet/nowcastnet.py"
--8<--
examples/nowcastnet/nowcastnet.py:69:82
--8<--
```

## 5. 完整代码

``` py linenums="1" title="examples/nowcastnet/nowcastnet.py"
--8<--
examples/nowcastnet/nowcastnet.py
--8<--
```

## 6. 结果展示

下图展示了模型的预测结果和真值结果。

<figure markdown>
  ![result](https://paddle-org.bj.bcebos.com/paddlescience/docs/nowcastnet/pd.gif){ loading=lazy style="margin:0 auto;"}
  <figcaption>模型预测结果</figcaption>
</figure>

<figure markdown>
  ![result](https://paddle-org.bj.bcebos.com/paddlescience/docs/nowcastnet/gt.gif){ loading=lazy style="margin:0 auto;"}
  <figcaption>模型真值结果</figcaption>
</figure>

## 7. 参考资料

- [Skilful nowcasting of extreme precipitation with NowcastNet](https://www.nature.com/articles/s41586-023-06184-4)
