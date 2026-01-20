# NowcastNet

=== "Model Training Command"

    None

=== "Model Evaluation Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/nowcastnet/mrms.tar
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/nowcastnet/mrms.tar -o mrms.tar
    mkdir ./datasets
    tar -xvf mrms.tar -C ./datasets/
    python nowcastnet.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/nowcastnet/nowcastnet_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python nowcastnet.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/nowcastnet/mrms.tar
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/nowcastnet/mrms.tar -o mrms.tar
    mkdir ./datasets
    tar -xvf mrms.tar -C ./datasets/
    python nowcastnet.py mode=infer
    ```

## 1. Background Introduction

In recent years, deep learning methods have been applied to weather forecasting, especially precipitation forecasting from radar observations. These methods utilize large amounts of radar composite observation data to train neural network models in an end-to-end manner, without explicitly referring to the physical laws of precipitation processes.
Here, we reproduce NowcastNet, a nonlinear nowcasting model for extreme precipitation, which unifies physical evolution schemes and conditional learning methods into a neural network framework, achieving end-to-end optimization.

## 2. Model Principle

This chapter only briefly introduces the model principle of NowcastNet. For detailed theoretical derivation, please read [Skilful nowcasting of extreme precipitation with NowcastNet](https://www.nature.com/articles/s41586-023-06184-4#Abs1).

The overall structure of the model is shown in the figure:

<figure markdown>
  ![nowcastnet-arch](https://paddle-org.bj.bcebos.com/paddlescience/docs/nowcastnet/nowcastnet.png){ loading=lazy style="margin:0 auto"}
  <figcaption>NowcastNet Network Model</figcaption>
</figure>

The model uses pre-trained weights for inference. Next, the inference process of the model will be introduced.

## 3. Model Construction

In this case, expressed in PaddleScience code as follows:

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

Among them, `input_keys` and `output_keys` represent the names of input and output variables of the network model respectively.

## 4. Model Evaluation Visualization

After completing the above settings, pass the instantiated objects to `ppsci.solver.Solver` in order:

``` py linenums="57" title="examples/nowcastnet/nowcastnet.py"
--8<--
examples/nowcastnet/nowcastnet.py:57:61
--8<--
```

Then build VisualizerRadar to generate image results:

``` py linenums="69" title="examples/nowcastnet/nowcastnet.py"
--8<--
examples/nowcastnet/nowcastnet.py:69:82
--8<--
```

## 5. Complete Code

``` py linenums="1" title="examples/nowcastnet/nowcastnet.py"
--8<--
examples/nowcastnet/nowcastnet.py
--8<--
```

## 6. Result Display

The figure below shows the model's prediction results and ground truth results.

<figure markdown>
  ![result](https://paddle-org.bj.bcebos.com/paddlescience/docs/nowcastnet/pd.gif){ loading=lazy style="margin:0 auto;"}
  <figcaption>Model Prediction Result</figcaption>
</figure>

<figure markdown>
  ![result](https://paddle-org.bj.bcebos.com/paddlescience/docs/nowcastnet/gt.gif){ loading=lazy style="margin:0 auto;"}
  <figcaption>Model Ground Truth Result</figcaption>
</figure>

## 7. References

- [Skilful nowcasting of extreme precipitation with NowcastNet](https://www.nature.com/articles/s41586-023-06184-4)
