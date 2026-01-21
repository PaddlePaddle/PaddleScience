# tempoGAN(temporally Generative Adversarial Networks)

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6521709" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat -P datasets/tempoGAN/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat -P datasets/tempoGAN/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat --create-dirs -o ./datasets/tempoGAN/2d_train.mat
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat --create-dirs -o ./datasets/tempoGAN/2d_valid.mat
    python tempoGAN.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat -P datasets/tempoGAN/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat -P datasets/tempoGAN/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat --create-dirs -o ./datasets/tempoGAN/2d_train.mat
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat --create-dirs -o ./datasets/tempoGAN/2d_valid.mat
    python tempoGAN.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/tempoGAN/tempogan_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python tempoGAN.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat -P datasets/tempoGAN/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat --create-dirs -o ./datasets/tempoGAN/2d_valid.mat
    python tempoGAN.py mode=infer
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [tempogan_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/tempoGAN/tempogan_pretrained.pdparams) | MSE: 4.21e-5<br>PSNR: 47.19<br>SSIM: 0.9974 |

## 1. Background Introduction

In fluid simulation problems, capturing the complex details of turbulence has always been a long-standing challenge for numerical simulation. Solving these details with discrete models will incur huge computational costs, which will soon become infeasible for flows on human spatial and temporal scales. Therefore, the demand for fluid super-resolution has emerged, which aims to recover high-resolution fluid simulation results from low-resolution results through fluid dynamics simulation and deep learning technology, so as to reduce the huge computational cost in the process of generating high-resolution fluids. This technology can be applied to various fluid simulations, such as water flow, air flow, flame simulation, etc.

Generative Adversarial Networks (GAN) is a deep learning network using unsupervised learning methods. GAN networks (at least) contain two models: Generator and Discriminator. The generator is used to generate the output of the problem, and the discriminator is used to judge whether the output is true or false. Both optimize together in mutual game, and finally make the output of the generator close to the true value.

Based on the GAN network, tempoGAN adds a time-related discriminator Discriminator_tempo. The network structure of this discriminator is the same as the basic discriminator, but the input is several consecutive frames of data in time, rather than a single frame of data, thereby taking timing into consideration.

This problem mainly uses this network to obtain corresponding high-density fluid data through input low-density fluid data, greatly saving time and computational costs.

## 2. Problem Definition

This problem includes three models: Generator, Discriminator and time-related discriminator (Discriminator_tempo). According to the training process of the GAN network, these three models are trained alternately, and the training order is: Discriminator, Discriminator_tempo, Generator.
GAN network is unsupervised learning. In the network design of this problem, the target value is used as an input value and input into the network for training.

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods. In order to quickly understand PaddleScience, only key steps such as model construction and constraint construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Dataset Introduction

The dataset is a 2d fluid dataset generated using the open source code package [mantaflow](http://mantaflow.com/install.html). The dataset includes numerical values converted from low and high-density fluid images of a certain number of continuous frames, stored in dictionary form in `.mat` files.

Before running the code for this problem, please download [training dataset](https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat) and [validation dataset](https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat), and store them separately in the path after downloading:

``` yaml linenums="27"
--8<--
examples/tempoGAN/conf/tempogan.yaml:27:28
--8<--
```

### 3.2 Model Construction

<figure markdown>
  ![tempoGAN-arch](https://paddle-org.bj.bcebos.com/paddlescience/docs/tempoGAN/tempoGAN_arch.png){ loading=lazy style="margin:0 auto"}
  <figcaption> tempoGAN network model</figcaption>
</figure>

The figure above is the complete model structure diagram of tempoGAN, but this problem only deals with relatively simple cases, and does not involve parts including velocity and vorticity input, 3d, data augmentation, advection operator, etc. If you are interested in contents not included in these documents, you can modify the code yourself and conduct further experiments.

As shown in the figure above, the input of Generator is the interpolation of low-density fluid data, and the output is the generated high-density fluid simulation data. The input of Discriminator is the concatenation of interpolation of low-density fluid data with high-density fluid simulation data generated by Generator and target high-density fluid data respectively. The input of Discriminator_tempo is multi-frame continuous high-density fluid simulation data generated by Generator and target high-density fluid data.

Although the composition of input and output looks complicated, they are essentially fluid density data, so the mapping functions of the 3 networks are all $f: \mathbb{R}^1 \to \mathbb{R}^1$.

Different from simple MLP networks, depending on different problems to be solved, GAN generators and discriminators have a variety of network structures to choose from, which will not be repeated here. Due to this uniqueness, the tempoGAN network in this problem is not built into PaddleScience and needs to be implemented additionally.

The Generator in this problem is a model with 4 layers of improved Res Block, Discriminator and Discriminator_tempo are the same model with 4 layers of convolution results, both have the same network structure but different inputs. The network parameters of Generator, Discriminator and Discriminator_tempo also need to be defined additionally.

For specific code, please refer to the gan.py file in [Complete Code](#4-complete-code).

Since the intermediate results of the generator and discriminator in the GAN network need to be called mutually and participate in each other's loss calculation, Model List is used for implementation, expressed in PaddleScience code as follows:

``` py linenums="57"
--8<--
examples/tempoGAN/tempoGAN.py:57:76
--8<--
```

Note that the network input defined in the above code is not exactly the same as the actual network input, so transform needs to be performed on the input.

### 3.3 transform Construction

The input of Generator is the interpolation of low-density fluid data, while the dataset stores the original low-density fluid data, so an interpolation transform is required.

``` py linenums="270"
--8<--
examples/tempoGAN/functions.py:270:275
--8<--
```

Discriminator and Discriminator_tempo have more complex transforms on input, respectively:

``` py linenums="360"
--8<--
examples/tempoGAN/functions.py:360:394
--8<--
```

Where:

``` py linenums="369"
--8<--
examples/tempoGAN/functions.py:369:369
--8<--
```

Indicates stopping the calculation gradient of parameters. This is set because this variable is only used as input for Discriminator and Discriminator_tempo here, and should not participate in gradient backpropagation during reverse calculation. If such setting is not made, since this variable comes from the output of Generator, the gradient will be transmitted to Generator along this variable during backpropagation, thereby changing the parameters in Generator, which is obviously not what we want.

In this way, we instantiate a neural network model `model list` possessing Generator, Discriminator and Discriminator_tempo and containing input transform.

### 3.4 Parameter and Hyperparameter Setting

We need to specify problem-related parameters, such as dataset path, weight parameters for various losses, etc.

``` yaml linenums="27"
--8<--
examples/tempoGAN/conf/tempogan.yaml:27:37
--8<--
```

Note that it contains 3 bool type variables `use_amp`, `use_spatialdisc` and `use_tempodisc`, which respectively represent whether to use mixed precision training (AMP), whether to use Discriminator and whether to use Discriminator_tempo. When both `use_spatialdisc` and `use_tempodisc` are set to `False`, the network structure of this problem will become a pure Generator model and is no longer a GAN network.

At the same time, hyperparameters such as training epochs and learning rate need to be specified. Note that since the GAN network training process is different from general single-model networks, the setting of `EPOCHS` is also different.

``` yaml linenums="73"
--8<--
examples/tempoGAN/conf/tempogan.yaml:73:76
--8<--
```

### 3.5 Optimizer Construction

The training uses the Adam optimizer, and the learning rate is reduced to $1/20$ of the original when `Epoch` reaches half, so the `Step` method is used as the learning rate strategy. If `by_epoch` is set to True, the learning rate will change according to the training `Epoch`, otherwise it will change according to `Iteration`.

``` py linenums="78"
--8<--
examples/tempoGAN/tempoGAN.py:78:94
--8<--
```

### 3.6 Constraint Construction

This problem adopts unsupervised learning method. Although it is not trained in a supervised learning manner, supervised constraint `SupervisedConstraint` can still be used here. Before defining constraints, data reading configurations such as file path need to be specified for supervised constraints. Since tempoGAN belongs to self-supervised learning, there is no label data in the dataset, but a part of input data is used as `label`, so `output_expr` of the constraint needs to be set.

``` py linenums="122"
--8<--
examples/tempoGAN/tempoGAN.py:122:125
--8<--
```

#### 3.6.1 Constraints of Generator

The following is the specific content of the constraint, note the `output_expr` mentioned above:

``` py linenums="98"
--8<--
examples/tempoGAN/tempoGAN.py:98:127
--8<--
```

The first parameter of `SupervisedConstraint` is the reading configuration of supervised constraint, where `dataset` field represents the training dataset information used, and each field respectively represents:

1. `name`: Dataset type, here `NamedArrayDataset` represents dataset of `.mat` type read from Array;
2. `input`: Input data of Array type;
3. `label`: Label data of Array type;
4. `transforms`: All data transform methods, here `FunctionalTransform` is a custom data transform class reserved by PaddleScience, which supports custom transform of input data when writing code. For specific code, please refer to [Custom loss and data transform](#38-loss-data-transform);

`batch_size` field represents the size of batch;

`sampler` field represents sampling method, where each field represents:

1. `name`: Sampler type, here `BatchSampler` represents batch sampler;
2. `drop_last`: Whether to discard the last samples that cannot make up a mini-batch, default is False;
3. `shuffle`: Whether to shuffle the order when generating sample subscripts, default is False;

The second parameter is the loss function. Here `FunctionalLoss` is a custom loss function class reserved by PaddleScience, which supports custom loss calculation method when writing code, rather than using existing methods such as `MSE`. For specific code, please refer to [Custom loss and data transform](#38-loss-data-transform).

The third parameter is `output_expr` of the constraint condition. As mentioned above, it is to allow the program to use input data as `label`.

The fourth parameter is the name of the constraint condition. We need to name each constraint condition for subsequent indexing.

After the constraints are constructed, encapsulate them into a dictionary with the name we just named as the keyword for subsequent access. Since `use_spatialdisc` and `use_tempodisc` are set in this problem, some constraints of Generator may not exist, so first encapsulate the certainly existing constraints into the dictionary, and when other constraints exist, add constraint elements to the dictionary.

``` py linenums="129"
--8<--
examples/tempoGAN/tempoGAN.py:129:160
--8<--
```

#### 3.6.2 Constraints of Discriminator

``` py linenums="164"
--8<--
examples/tempoGAN/tempoGAN.py:164:201
--8<--
```

The meaning of each parameter is the same as [Constraints of Generator](#361-generator).

#### 3.6.3 Constraints of Discriminator_tempo

``` py linenums="205"
--8<--
examples/tempoGAN/tempoGAN.py:205:244
--8<--
```

The meaning of each parameter is the same as [Constraints of Generator](#361-generator).

### 3.7 Visualizer Construction

Because of the characteristics of GAN network training, this problem does not use the built-in visualizer in PaddleScience, but customizes a function for implementing inference. This function reads validation set data, obtains inference results and saves the results in image form. Calling this function at regular intervals during the training process can monitor the training effect during the training process.

``` py linenums="154"
--8<--
examples/tempoGAN/functions.py:154:230
--8<--
```

### 3.8 Custom loss and data transform

Since this problem adopts unsupervised learning and there is no label data in the data, loss is calculated, so loss needs to be customized. The method is to define relevant functions first, and then pass the function name as a parameter to `FunctionalLoss`. It should be noted that the input and output parameters of the custom loss function need to be consistent with other functions such as `MSE` in PaddleScience, that is, the input is dictionary variables such as model output `output_dict`, and the output is loss value `paddle.Tensor`.

#### 3.8.1 Loss of Generator

The loss of Generator provides l1 loss, l2 loss, loss judged by Discriminator on output and loss judged by Discriminator_tempo on output. Whether these losses exist is controlled according to weight parameters. If the weight parameter of a certain loss item is 0, it means that the loss item is not added during training.

``` py linenums="277"
--8<--
examples/tempoGAN/functions.py:277:346
--8<--
```

#### 3.8.2 Loss of Discriminator

Discriminator is a discriminator, its function is to judge whether data is true data or false data, so its loss is the loss generated by judging data generated by Generator as false and the loss generated by judging target value data as true.

``` py linenums="396"
--8<--
examples/tempoGAN/functions.py:396:410
--8<--
```

#### 3.8.3 Loss of Discriminator_tempo

The loss composition of Discriminator_tempo is the same as Discriminator, only the required data is different.

``` py linenums="412"
--8<--
examples/tempoGAN/functions.py:412:428
--8<--
```

#### 3.8.4 Custom data transform

This problem provides an input data processing method, randomly cropping a piece of input fluid density data, then judging the density value. If the density value of the cropped block is lower than the threshold, it is re-cropped until the density meets the condition or the number of cropping times reaches the threshold. This is mainly done to reduce the video memory required for training, and the judgment of the block density value ensures the richness of information in the block. In [Parameter and Hyperparameter Setting](#34), `tile_ratio` indicates how many times the original size is compared to the block size, that is, if `tile_ratio` is 2, the size of the cropped block is one-fourth of the entire original image.

``` py linenums="431"
--8<--
examples/tempoGAN/functions.py:431:489
--8<--
```

Note that the code here only provides the idea of data transform. The simple block method in the current code will obviously affect the training effect due to less information contained in the input. Therefore, in this problem, when the video memory is sufficient, `tile_ratio` should be set to 1. When the video memory is insufficient, it is also recommended to prioritize using mixed precision training to reduce current memory usage.

### 3.9 Model Training

After completing the above settings, first pass the above instantiated objects to `ppsci.solver.Solver` in order, and then start training.

``` py linenums="247"
--8<--
examples/tempoGAN/tempoGAN.py:247:258
--8<--
```

Note that the GAN type network training method is alternating training of multiple models, which is different from single model or multi-model phased training, and `solver.train` API cannot be simply used. For specific code, please refer to tempoGAN.py file in [Complete Code](#4).

### 3.10 Model Evaluation

#### 3.10.1 Evaluation during training

During training, only target results and model output results of specific images are saved at specific `Epoch`, and an evaluation is performed on the output result of the last `Epoch` after training ends, so as to intuitively evaluate the model optimization effect. Do not use the built-in evaluator in PaddleScience, nor evaluate during the training process:

``` py linenums="287"
--8<--
examples/tempoGAN/tempoGAN.py:287:293
--8<--
```

``` py linenums="307"
--8<--
examples/tempoGAN/tempoGAN.py:307:323
--8<--
```

For specific code, please refer to tempoGAN.py file in [Complete Code](#4).

#### 3.10.2 Evaluation in eval

The evaluation metric for this problem is to compare the super-resolution result output by the model with the actual high-resolution image, and use three indicators MSE (Mean-Square Error), PSNR (Peak Signal-to-Noise Ratio), and SSIM (Structural SIMilarity) to evaluate image similarity. Therefore, the built-in evaluator in PaddleScience is not used, nor is there a `Solver.eval()` process.

``` py linenums="326"
--8<--
examples/tempoGAN/tempoGAN.py:326:406
--8<--
```

In addition, where:

``` py linenums="396"
--8<--
examples/tempoGAN/tempoGAN.py:396:403
--8<--
```

Provides the option to save the model output result, so as to see the result after super-resolution more intuitively. Whether to open is specified by `save_outs` in configuration file `EVAL`:

``` yaml linenums="91"
--8<--
examples/tempoGAN/conf/tempogan.yaml:91:94
--8<--
```

## 4. Complete Code

The complete code contains PaddleScience specific training process code tempoGAN.py and all custom function code functions.py. In addition, network structure code gan.py is added to `ppsci.arch`, which is shown below together. If you need to customize the network structure, you can use it as a reference.

``` py linenums="1" title="tempoGAN.py"
--8<--
examples/tempoGAN/tempoGAN.py
--8<--
```

``` py linenums="1" title="functions.py"
--8<--
examples/tempoGAN/functions.py
--8<--
```

``` py linenums="1" title="gan.py"
--8<--
ppsci/arch/gan.py
--8<--
```

## 5. Result Display

After using mixed precision training, evaluate MSE, PSNR, SSIM between target on test set. The values of evaluation indicators are:

| MSE | PSNR | SSIM |
| :---: | :---: | :---: |
| 4.21e-5 | 47.19 | 0.9974 |

The input of a fluid super-resolution example, model prediction result, and result directly generated by open source code package mantaflow in [Dataset Introduction](#31) are as follows. The model prediction result is basically consistent with the generated target result.

<figure markdown>
  ![input](https://paddle-org.bj.bcebos.com/paddlescience/docs/tempoGAN/input.gif){ loading=lazy }
  <figcaption>Input low-density fluid</figcaption>
</figure>

<figure markdown>
  ![pred-amp02](https://paddle-org.bj.bcebos.com/paddlescience/docs/tempoGAN/pred_amp02.gif){ loading=lazy }
  <figcaption>High-density fluid obtained by inference after mixed precision training</figcaption>
</figure>

<figure markdown>
  ![target](https://paddle-org.bj.bcebos.com/paddlescience/docs/tempoGAN/target.gif){ loading=lazy }
  <figcaption> Target high-density fluid</figcaption>
</figure>

## 6. References

- [tempoGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow](https://dl.acm.org/doi/10.1145/3197517.3201304)

- [Reference Code](https://github.com/thunil/tempoGAN)
