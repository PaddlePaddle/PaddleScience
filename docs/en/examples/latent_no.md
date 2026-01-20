# Latent Neural Operator LatentNO(or LNO)

=== "Model Training Command"

    ``` sh
    # Darcy
    # linux
    wget -c -P ./datas/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/LatentNO/Darcy_{train,val}.npy
    # windows
    # foreach ($f in "train","val") {curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/LatentNO/Darcy_$f.npy --create-dirs -o ./datas/Darcy_$f.npy}
    python LatentNO-steady.py --config-name=LatentNO-Darcy.yaml

    # Elasticity
    # linux
    wget -c -P ./datas/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/LatentNO/Elasticity_{train,val}.npy
    # windows
    # foreach ($f in "train","val") {curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/LatentNO/Elasticity_$f.npy --create-dirs -o ./datas/Elasticity_$f.npy}
    python LatentNO-steady.py --config-name=LatentNO-Elasticity.yaml

    # Pipe
    # linux
    wget -c -P ./datas/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/LatentNO/Pipe_{train,val}.npy
    # windows
    # foreach ($f in "train","val") {curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/LatentNO/Pipe_$f.npy --create-dirs -o ./datas/Pipe_$f.npy}
    python LatentNO-steady.py --config-name=LatentNO-Pipe.yaml

    # NS2d
    # linux
    wget -c -P ./datas/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/LatentNO/NS2d_{train,val}.npy
    # windows
    # foreach ($f in "train","val") {curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/LatentNO/NS2d_$f.npy --create-dirs -o ./datas/Pipe_$f.npy}
    python LatentNO-time.py --config-name=LatentNO-NS2d.yaml
    ```

=== "Model Evaluation Command"

    ``` sh
    # Darcy
    # linux
    wget -c -P ./datas/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/LatentNO/Darcy_{train,val}.npy
    # windows
    # foreach ($f in "train","val") {curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/LatentNO/Darcy_$f.npy --create-dirs -o ./datas/Darcy_$f.npy}
    python LatentNO-steady.py --config-name=LatentNO-Darcy.yaml mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/LatentNO/LatentNO_Darcy_pretrained.pdparams

    # Elasticity
    # linux
    wget -c -P ./datas/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/LatentNO/Elasticity_{train,val}.npy
    # windows
    # foreach ($f in "train","val") {curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/LatentNO/Elasticity_$f.npy --create-dirs -o ./datas/Elasticity_$f.npy}
    python LatentNO-steady.py --config-name=LatentNO-Elasticity.yaml mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/LatentNO/LatentNO_Elasticity_pretrained.pdparams

    # Pipe
    # linux
    wget -c -P ./datas/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/LatentNO/Pipe_{train,val}.npy
    # windows
    # foreach ($f in "train","val") {curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/LatentNO/Pipe_$f.npy --create-dirs -o ./datas/Pipe_$f.npy}
    python LatentNO-steady.py --config-name=LatentNO-Pipe.yaml mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/LatentNO/LatentNO_Pipe_pretrained.pdparams

    # NS2d
    # linux
    wget -c -P ./datas/ https://paddle-org.bj.bcebos.com/paddlescience/datasets/LatentNO/NS2d_{train,val}.npy
    # windows
    # foreach ($f in "train","val") {curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/LatentNO/NS2d_$f.npy --create-dirs -o ./datas/Pipe_$f.npy}
    python LatentNO-time.py --config-name=LatentNO-NS2d.yaml mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/LatentNO/LatentNO_NS2d_pretrained.pdparams
    ```

## 1. Background Introduction

The forward problem of solving partial differential equations (PDEs) refers to finding the solution function given the specific form of the equation and the initial and boundary conditions. It can be unified into an operator learning task, thereby generalized into a sequence-to-sequence transformation framework. Neural operator models can learn the mapping from input functions to output functions in a data-driven manner based on paired training data, where both input and output functions are represented by sequences of sampling points.
In recent years, the Transformer architecture has dominated the construction of neural operators. The attention mechanism models the long-range non-linear interaction relationships between all objects in the sequence, naturally fitting the sequence-to-sequence representation in the PDE solving process, and can provide more accurate modeling results compared to traditional fully connected structures. However, the time complexity of the attention mechanism is quadratic with respect to the sequence length, so the computational cost of using the attention mechanism to build neural operators increases dramatically. To reduce computational costs, some existing works attempt to replace the original attention mechanism with variants of linear time complexity attention mechanisms, but due to their limited modeling capabilities, they often sacrifice the solving accuracy of PDEs. Another part of existing works attempts to solve PDEs using a small number of physical features in the latent space, thereby getting rid of the intricate interaction relationships between a large number of sampling points in the original geometric space, and capturing the correlation between physical features in a compact latent space. However, these methods either rely on manually specified basis function features or fail to construct a persistent latent space.
Therefore, this case proposes a physical cross-attention module, which decouples the positions of the input observation samples and the output samples to be predicted, and autonomously learns a persistent latent space from the data. Based on the physical cross-attention module, a latent neural operator model is further designed.

<figure markdown>
  ![pipe](https://paddle-org.bj.bcebos.com/paddlescience/docs/LatentNO/LatentNO_1.jpg){ loading=lazy }
  <figcaption>Structure diagram of Latent Neural Operator</figcaption>
</figure>

## 2. Implementation of Latent Neural Operator

This section will explain how to implement the construction, training, testing and evaluation of the latent neural operator model based on PaddleScience code. The directory structure of the case is as follows.
``` log
LatentNO/
├── config
│     ├── LatentNO-Darcy.yaml
│     └── ...
├── datas
│   ├── Darcy_train.npy
│   ├── Darcy_val.npy
│   └── ...
├── LatentNO-steady.py
├── LatentNO-time.py
└── utils.py

```

### 2.1 Dataset Construction and Loading

For different tasks involved in this project, the dataset of this example can be divided into two categories: one is static data (Darcy, Pipe, Elasticity); the other is time-dependent data (NS2d). To be compatible with the automatic training process under the PaddleScience framework, this case designed and implemented dedicated dataset classes, corresponding to static scenarios and dynamic scenarios respectively, named `LatentNODataset` and `LatentNODataset_time`. Next, the construction of `LatentNODataset` will be explained in detail first.

For static data tasks, data is first stored in the `./datas` directory in the form of `.npy` files. Each file is named according to the data name and mode (training set or validation set), such as `Darcy_train.npy` or `Darcy_val.npy`. These files internally store dictionaries containing three key variables x, y1 and y2. Both x and y1 will be used as inputs to the model, while y2 is the final prediction target. During the loading phase, data will be converted to Paddle tensor format, and the shape will be adjusted according to requirements to meet the model's input requirements, and x and y1 will be concatenated when necessary.

``` py linenums="123" title="ppsci/data/dataset/latent_no_dataset.py"
--8<--
ppsci/data/dataset/latent_no_dataset.py:123:142
--8<--
```

To enhance the training stability and generalization ability of the model, a normalization module is also built into the dataset class. This module calculates the mean and standard deviation of each variable during the initialization phase, and automatically performs normalization when loading data. At the same time, an interface for denormalization is provided to facilitate restoration to the true physical scale during inference or visualization.

``` py linenums="144" title="ppsci/data/dataset/latent_no_dataset.py"
--8<--
ppsci/data/dataset/latent_no_dataset.py:144:149
--8<--
```

``` py linenums="12" title="ppsci/data/dataset/latent_no_dataset.py"
--8<--
ppsci/data/dataset/latent_no_dataset.py:12:82
--8<--
```

During the training process, by calling the `__getitem__` method, the input, label and corresponding weight of a piece of data can be returned by index, thereby seamlessly connecting to the training pipeline.

``` py linenums="166" title="ppsci/data/dataset/latent_no_dataset.py"
--8<--
ppsci/data/dataset/latent_no_dataset.py:166:186
--8<--
```

The entire data is stored in a dictionary in the format agreed by PaddleScience. input is used to provide input tensors, label is used to provide supervision signals, and weight_dict allows users to assign weights to different loss components.

For time-dependent data tasks, the construction of the dataset is similar. The main difference is that `LatentNODataset_time` retains x, y1 and y2 in the input dictionary at the same time, allowing the model to directly obtain time-related context information to assist training, while the label part is still y2 to supervise the final prediction result. This design ensures the capture of time-dependent characteristics during the training process, and also provides a good data interface for subsequent long-term evolution prediction.

### 2.2 Model Construction

The latent neural operator includes three processes: encoding, latent space operator fitting, and decoding. In processing static data tasks, the forward propagation process of the model is expressed in PaddleScience as follows:

``` py linenums="244" title="ppsci/arch/latent_no.py"
--8<--
ppsci/arch/latent_no.py:244:275
--8<--
```

<figure markdown>
  ![pipe](https://paddle-org.bj.bcebos.com/paddlescience/docs/LatentNO/LatentNO_2.jpg){ loading=lazy }
  <figcaption>Physical cross-attention module used in encoding and decoding stages</figcaption>
</figure>

The encoding process includes two parts: input projection and input function encoding. The input projection operation promotes the tuple composed of the sampling position of the observation function input in sequence form in the geometric space and the corresponding physical quantity value to a higher vector dimension. Geometric space is the original space of PDE input or output, which contains several sample points, each sample consisting of multi-dimensional spatial position coordinates and multi-dimensional physical quantity values. Through the input projection operation, the observation function can be projected into a space where it is easier to capture non-local features. The input function encoding operation maps the projected input data from the geometric space to the latent space. The latent neural operator model uses the representation Token of the imaginary sampling position in the latent space to re-represent the input function, where the number of imaginary sampling positions is much smaller than the number of sampling points of the input function in the geometric space, achieving the purpose of sequence compression. The latent neural operator model uses physical cross-attention to complete the encoding operation of the input function from geometric space to latent space. The relevant code for the encoding operation is expressed in PaddleScience as follows:

``` py linenums="258" title="ppsci/arch/latent_no.py"
--8<--
ppsci/arch/latent_no.py:258:268
--8<--
```

``` py linenums="225" title="ppsci/arch/latent_no.py"
--8<--
ppsci/arch/latent_no.py:225:227
--8<--
```

``` py linenums="35" title="ppsci/arch/latent_no.py"
--8<--
ppsci/arch/latent_no.py:35:82
--8<--
```

After completing the encoding of the input function, the length of the sequence to be processed is significantly reduced to the number of imaginary sampling point positions in the latent space, so extracting and converting the features of the input function in the latent space is more efficient than in the original geometric space. The latent neural operator model fits the solution operator of the PDE problem in the latent space, uses stacked Transformer layers, and uses the self-attention mechanism as the kernel integral operator. Each layer performs information aggregation on the representation Tokens at the imaginary sampling positions in the latent space, thereby converting the features of the input function into the features of the output function. Fitting the solution operator based on shorter feature sequences in the latent space gives the latent neural operator model higher solving efficiency on PDE problems, and is also compatible with kernel integral operators with stronger modeling capabilities, thereby ensuring excellent solving accuracy on PDE problems. The stacked structure in the latent space is expressed in PaddleScience as follows:

``` py linenums="269" title="ppsci/arch/latent_no.py"
--8<--
ppsci/arch/latent_no.py:269:270
--8<--
```

``` py linenums="230" title="ppsci/arch/latent_no.py"
--8<--
ppsci/arch/latent_no.py:230:232
--8<--
```

``` py linenums="140" title="ppsci/arch/latent_no.py"
--8<--
ppsci/arch/latent_no.py:140:189
--8<--
```

The decoding process includes two parts: output function decoding and output projection. The output function decoding operation maps the representation Token at the imaginary sampling position converted by the stacked Transformer layer back to the geometric space. The latent neural operator model uses physical cross-attention again to decode the representation vector of the output function representation sequence in the latent space at the corresponding position to be predicted according to the query position of the output function. The output projection operation projects the decoded representation vector at the position to be predicted into the predicted low-dimensional physical quantity value. The relevant code for the decoding process is expressed in PaddleScience as follows:

``` py linenums="272" title="ppsci/arch/latent_no.py"
--8<--
ppsci/arch/latent_no.py:272:273
--8<--
```

``` py linenums="228" title="ppsci/arch/latent_no.py"
--8<--
ppsci/arch/latent_no.py:228:228
--8<--
```

When dealing with time-dependent data, the overall structure of the model remains unchanged, but in order to meet the automatic training requirements of PaddleScience, the `LatentNO_time` class rewrites the forward propagation function to implement a time-unroll / autoregressive process. Inside the time iteration, `LatentNO_time` introduces two different next-step input sources: during training, externally provided y2 (label information) is additionally used, and the aligned segment y2[..., t:t+step] is cut out from y2 as part of the next input; during inference, the model's pred_step is used as the next input and stop_gradient=True is executed on it to block cross-step gradient propagation. Regardless of which source is used, the next current_y is updated through a sliding window method of "keeping the trunk part + discarding the earliest time slots + splicing new fragments at the end". Expressed in PaddleScience as follows:

``` py linenums="392" title="ppsci/arch/latent_no.py"
--8<--
ppsci/arch/latent_no.py:392:462
--8<--
```

In the training or validation function, the model is instantiated by the following code.

``` py linenums="10" title="examples/LatentNO/LatentNO-steady.py"
--8<--
examples/LatentNO/LatentNO-steady.py:10:10
--8<--
```

``` py linenums="10" title="examples/LatentNO/LatentNO-time.py"
--8<--
examples/LatentNO/LatentNO-time.py:10:10
--8<--
```

### 2.3 Constraint Construction

This case adopts supervised learning. According to the API structure description of PaddleScience, the built-in `SupervisedConstraint` is used to construct supervised constraints. Expressed in PaddleScience code as follows (testing constraints are similar, the difference is that in some tasks, denormalization operations need to be performed when calculating test loss, that is, obtain the normalizer of the training set through `sup_constraint.data_loader.dataset.normalizer` and pass it into `RelLpLoss` as a parameter)

``` py linenums="57" title="examples/LatentNO/LatentNO-steady.py"
--8<--
examples/LatentNO/LatentNO-steady.py:57:66
--8<--
```

The loss function is relative Lp loss. For static tasks, the loss function `RelLpLoss` is expressed as follows.

``` py linenums="9" title="examples/LatentNO/utils.py"
--8<--
examples/LatentNO/utils.py:9:54
--8<--
```

Similarly, adjustments have been made for time-dependent tasks to adapt to the automatic training framework. `RelLpLoss_time` implements gradient backpropagation update using cumulative error per time step through the `use_full_sequence` parameter, and uses full sequence one-time error as the evaluation metric.

``` py linenums="57" title="examples/LatentNO/utils.py"
--8<--
examples/LatentNO/utils.py:57:138
--8<--
```

### 2.4 Optimizer Construction

The trainer uses the AdamW optimizer, the learning rate setting is given by the configuration file, and OneCycleLR is used to control the learning rate change. Expressed in PaddleScience code as follows:

``` py linenums="57" title="examples/LatentNO/LatentNO-steady.py"
--8<--
examples/LatentNO/LatentNO-steady.py:57:77
--8<--
```

### 2.5 Model Training

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training. Expressed in PaddleScience code as follows:

``` py linenums="89" title="examples/LatentNO/LatentNO-steady.py"
--8<--
examples/LatentNO/LatentNO-steady.py:89:97
--8<--
```

## 3. Complete Code

``` py linenums="1" title="examples/LatentNO/LatentNO-steady.py"
--8<--
examples/LatentNO/LatentNO-steady.py
--8<--
```

## 4. Result Display

The following shows the performance of the latent neural operator in several PDE forward problems.

<figure markdown>
  ![pipe](https://paddle-org.bj.bcebos.com/paddlescience/docs/LatentNO/LatentNO_3.jpg){ loading=lazy }
  <figcaption>Performance of latent neural operator in several PDE forward problems</figcaption>
</figure>

## 5. References

[1] Wang T, Wang C. Latent neural operator for solving forward and inverse pde problems[J]. Advances in Neural Information Processing Systems, 2024, 37: 33085-33107.
