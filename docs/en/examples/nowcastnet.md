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

Deep learning has recently emerged as a powerful tool for weather forecasting, particularly for precipitation nowcasting using radar data. These methods leverage vast amounts of radar composite observations to train end-to-end neural networks, often without explicit reliance on physical laws.

Here, we reproduce NowcastNet, a nonlinear model designed for extreme precipitation nowcasting. NowcastNet unifies physical evolution schemes with conditional learning within a neural network framework, enabling effective end-to-end optimization.

## 2. Model Principle

This chapter only briefly introduces the model principle of NowcastNet. For detailed theoretical derivation, please read [Skilful nowcasting of extreme precipitation with NowcastNet](https://www.nature.com/articles/s41586-023-06184-4#Abs1).

The model architecture is illustrated below:

<figure markdown>
  ![nowcastnet-arch](https://paddle-org.bj.bcebos.com/paddlescience/docs/nowcastnet/nowcastnet.png){ loading=lazy style="margin:0 auto"}
  <figcaption>NowcastNet Network Model</figcaption>
</figure>

The model utilizes pre-trained weights for inference. We detail the inference process below.

## 3. Model Construction

The PaddleScience implementation is as follows:

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

Here, `input_keys` and `output_keys` denote the input and output variable names of the network model.

## 4. Model Evaluation Visualization

After configuration, pass the instantiated objects to `ppsci.solver.Solver`:

``` py linenums="57" title="examples/nowcastnet/nowcastnet.py"
--8<--
examples/nowcastnet/nowcastnet.py:57:61
--8<--
```

Next, initialize `VisualizerRadar` to generate visualization results:

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

The figures below display the model's predictions compared to the ground truth.

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
