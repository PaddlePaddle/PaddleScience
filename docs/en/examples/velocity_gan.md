# VelocityGAN

!!! note

    1. Before running, it is recommended to quickly understand [Dataset](#31) and [Data Reading Method](#32-dataset-api).
    2. Download [OpenFWI Dataset](https://openfwi-lanl.github.io/docs/data.html#vel) to the corresponding subdirectory in `FWIOpenData` directory (e.g. `Flatvel_A`).
    3. Correspond the `anno` parameter in the yaml configuration file to the dataset.

=== "Model Training Command"

    ``` sh
    python velocityGAN.py
    ```

=== "Model Evaluation Command"

    ``` sh
    python velocityGAN.py model=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/velocitygan/velocitygan_pretrained.pdparams
    ```

| Pretrained Model | Metrics                                        |
| :--------- | :------------------------------------------ |
| [velocitygan_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/velocitygan/velocitygan_pretrained.pdparams) | MAE: 0.0669<br>RMSE: 0.0947<br>SSIM: 0.8511 |

## 1. Background Introduction

Underground velocity images play an important role in the field of earth sciences. They reflect the propagation speed of seismic waves in various underground areas and provide key information for detecting the internal structure of the earth. Seismic waveform inversion methods are widely used to reconstruct underground velocity imaging. Traditional physics-driven solution methods are numerical optimization processes that require multiple iterations and solving wave equations. This is not only computationally expensive, but usually only achieves local optimal solutions, resulting in limited image accuracy. Data-driven deep learning methods can alleviate these problems and generate higher-precision velocity images in a shorter time.

VelocityGAN is a specific example. It is an end-to-end framework that can generate high-quality velocity images directly from raw seismic waveform data. The paper shows that VelocityGAN outperforms traditional physics-driven waveform inversion methods and achieves SOTA performance in data-driven benchmarks.

## 2. Model Principle

As a data-driven deep learning method, VelocityGAN can directly learn the mapping relationship from waveform data to velocity images without solving wave equations. This paragraph only briefly introduces the model principle. For specific details, please read [VelocityGAN: Data-Driven Full-Waveform Inversion Using Conditional Adversarial Networks](https://arxiv.org/abs/1809.10262v6).

### 2.1 Model Structure

VelocityGAN is a conditional adversarial network containing an image-to-image generator and a CNN discriminator. The figure below shows the overall structure of the model.

![velocityGAN](https://paddle-org.bj.bcebos.com/paddlescience/docs/velocitygan/velocityGAN.png)

- `Generator` is a convolutional neural network with Encoder-Decoder structure. The Encoder extracts features from seismic waveform data and gradually compresses them into latent vectors; the Decoder infers the corresponding velocity map based on this latent vector.

- `Discriminator` is a model composed of 9 convolutional blocks. Input velocity image, output image authenticity score.

### 2.2 Loss Function

The discriminator's loss function uses Wasserstein loss and gradient penalty. Its expression is:

$$
L_d = \underset{\tilde{x} \sim \mathbb{P}_g}{\mathbb{E}} D(\tilde{x}) - \underset{x \sim \mathbb{P}_r}{\mathbb{E}}D(x) + \lambda \underset{\hat{x} \sim \mathbb{P}_{\hat{x}}}{\mathbb{E}} \left[ \left( \| \nabla_{\hat{x}} D(\hat{x}) \|_2 - 1 \right)^2 \right]
$$

Where $\mathbb{P}_g$ is the generator distribution, $\mathbb{P}_r$ is the real data distribution, and $\mathbb{P}_{\hat{x}}$ is a mixed interpolation sample from $\mathbb{P}_g$ and $\mathbb{P}_r$.

The generator's loss function is a combination of adversarial loss [$- \underset{\tilde{x} \sim \mathbb{P}_g}{\mathbb{E}}D(\tilde{x})$] and content loss (MAE, MSE). Its expression is:

$$
L_g = - \underset{\tilde{x} \sim \mathbb{P}_g}{\mathbb{E}}D(\tilde{x}) + \frac{\lambda_1}{w\cdot h} \sum_{i=1}^{w} \sum_{j=1}^{h} \left| \tilde{v}(i,j) - v(i,j) \right| + \frac{\lambda_2}{w\cdot h}\sum_{i=1}^{w} \sum_{j=1}^{h} \left( \tilde{v}(i,j) - v(i,j) \right)^2
$$

Where $w$ and $h$ are the width and height of the velocity map respectively, $v(\cdot)$ and $\tilde{v}(\cdot)$ represent the true pixel value and predicted pixel value of the velocity map respectively. $\lambda_1$ and $\lambda_2$ are hyperparameters used to adjust the relative importance of the two losses.

## 3. Model Construction

Next, we will explain how to use the PaddleScience framework to implement VelocityGAN. The following content only elaborates on key steps. For other details, please refer to [API Documentation](https://paddlescience-docs.readthedocs.io/zh-cn/latest/zh/api/arch/).

### 3.1 Dataset Introduction

The dataset uses the [OpenFWI](https://openfwi-lanl.github.io/docs/data.html#vel) dataset open sourced by [SMILE Team](https://smileunc.github.io/).

OpenFWI has a total of 12 datasets, divided into four categories: Vel Family, Fault Family, Style Family and Kimberlina Family. This case mainly uses the first two categories, and their configuration information is as follows:

![image-20240830153600238](https://paddle-org.bj.bcebos.com/paddlescience/docs/velocitygan/vel_family.png)

![image-20240830153613634](https://paddle-org.bj.bcebos.com/paddlescience/docs/velocitygan/fault_family.png)

Among them, each dataset contains waveform data and corresponding velocity images. The figure below shows an example of velocity images in each dataset.

![image-20240830154311787](https://paddle-org.bj.bcebos.com/paddlescience/docs/velocitygan/data.png)

It can be seen that Vel Family includes two cases of straight and curved geological interfaces, while Fault Family adds some geological faults on this basis.

Each sample contains a velocity image and five waveform data, as shown in the figure below.

![image-20240830154807670](https://paddle-org.bj.bcebos.com/paddlescience/docs/velocitygan/sample.png)

Among them, 5 red stars lined up represent five seismic sources on the ground, and 70 receivers are also arranged on the ground. Seismic waves propagate downwards and bounce back, and the receivers record data every 0.001 seconds, totaling 1000. Therefore, a seismic waveform dataset with a shape of (5, 1000, 70) is generated.

Note: All data are not real collected data, but simulated. For specific details, please read [OpenFWI: Large-Scale Multi-Structural Benchmark Datasets for Seismic Full Waveform Inversion](https://arxiv.org/abs/2111.02926).

### 3.2 Build dataset API

Since a dataset consists of 120 data files, passing in all file paths is cumbersome. In order to facilitate data reading, all paths can be packaged into a text file. By parsing the paths in turn, all data can be read. Due to this special reading method, we cannot use the built-in dataset API of PaddleScience, so we customized `ppsci.data.dataset.FWIDataset`.

The configuration code of dataloader is given below:
``` py linenums="120"
--8<--
examples/velocityGAN/velocityGAN.py:120:141
--8<--
```
Among them, `dataset` uses our customized `FWIDataset`, and `anno` passes in the path of the text file, which contains the paths of all data files.

### 3.3 Model Construction

VelocityGAN in this case is not built into PaddleScience and needs to be implemented additionally, so we customized `ppsci.arch.VelocityGenerator` and `ppsci.arch.VelocityDiscriminator`.

The model construction code is as follows:

``` py linenums="112"
--8<--
examples/velocityGAN/velocityGAN.py:112:114
--8<--
```

The parameter configuration is as follows:
``` yaml linenums="41"
--8<--
examples/velocityGAN/conf/velocityGAN.yaml:41:58
--8<--
```

### 3.4 Custom loss

The loss function of VelocityGAN is a bit complicated and needs to be implemented by ourselves. PaddleScience provides an API for customizing loss functions - `ppsci.loss.FunctionalLoss`. The method is to define the loss function first, and then pass the function name as a parameter to `FunctionalLoss`. Note that the input and output of the custom loss function need to be in the format of a dictionary.

#### 3.4.1 Loss of Generator

The loss of Generator includes L1 loss, L2 loss and adversarial loss. These three losses all have corresponding weights. If the weight of a certain loss is 0, it means that the loss item is not added during training.

``` py linenums="24"
--8<--
examples/velocityGAN/functions.py:24:53
--8<--
```

#### 3.4.2 Loss of Discriminator

The loss of Discriminator includes Wasserstein loss and gradient penalty. Among them, only the gradient penalty term has weight parameters.
``` py linenums="68"
--8<--
examples/velocityGAN/functions.py:68:119
--8<--
```

Note:

``` py linenums="80"
--8<--
examples/velocityGAN/functions.py:80:80
--8<--
```

Indicates that the pred variable does not participate in gradient calculation. This is because pred is only used as the input of Discriminator and its gradient does not need to be considered. Moreover, pred is the output of Generator. If gradient calculation is not stopped, the parameter gradient of Generator will accumulate during discriminator training and eventually affect the training of the first batch of the generator.

### 3.5 Constraint Construction

This case uses `ppsci.constraint.SupervisedConstraint` to construct constraints.

The construction code is as follows:

``` py linenums="143"
--8<--
examples/velocityGAN/velocityGAN.py:143:158
--8<--
```

Among them, `output_expr` specifies how to construct `output_dict`, and `name` is the name of the constraint, which is convenient for subsequent indexing.

After the constraint construction is completed, it needs to be created in the form of a dictionary for easy passing to `ppsci.solver.Solver` later.

### 3.6 Optimizer Construction

VelocityGAN uses AdamW optimizer, which can be directly constructed by calling `ppsci.optimizer.AdamW`, code as follows:

``` py linenums="160"
--8<--
examples/velocityGAN/velocityGAN.py:160:165
--8<--
```

### 3.7 Solver Construction

Pass the constructed model, constraints, optimizer and other parameters to `ppsci.solver.Solver`.

``` py linenums="167"
--8<--
examples/velocityGAN/velocityGAN.py:167:184
--8<--
```

### 3.8 Model Training

``` py linenums="186"
--8<--
examples/velocityGAN/velocityGAN.py:186:190
--8<--
```

### 3.9 Custom metric

The evaluation indicators of this case are: MAE (Mean Absolute Error), RMSE (Root Mean Squared Error) and SSIM (Structural SIMilarity). Among them, PaddleScience provides APIs for MAE and RMSE, while SSIM requires us to implement it additionally.

PaddleScience provides an API for customizing metric functions - `ppsci.metric.FunctionalMetric`. The method is to define the metric function first, and then pass the function name as a parameter to `FunctionalMetric`. Note that the input and output of the custom metric function need to be in the format of a dictionary.

The implementation code of SSIM is as follows:
``` py linenums="199"
--8<--
examples/velocityGAN/functions.py:199:312
--8<--
```

### 3.10 Validator Construction

This case uses `ppsci.validate.SupervisedValidator` to construct the validator.

``` py linenums="56"
--8<--
examples/velocityGAN/velocityGAN.py:56:68
--8<--
```

### 3.11 Model Evaluation

After passing the model, validator and weight path to `ppsci.solver.Solver`, start evaluation through `solver.eval()`.

``` py linenums="70"
--8<--
examples/velocityGAN/velocityGAN.py:70:78
--8<--
```

### 3.12 Visualization

After evaluation, we visualize the results in the form of images, code as follows:

``` py linenums="80"
--8<--
examples/velocityGAN/velocityGAN.py:80:94
--8<--
```

## 4. Complete Code

``` py linenums="1" title="velocityGAN.py"
--8<--
examples/velocityGAN/velocityGAN.py
--8<--
```

## 5. Result Display

Training results using [FlatVel-A](https://drive.google.com/drive/folders/1NIdjiYhjWSV9NHn7ZEFYTpJxzvzxqYRb) dataset.

|  MAE   |  RMSE  |  SSIM  |
| :----: | :----: | :----: |
| 0.0669 | 0.0947 | 0.8511 |

![image-20240914192445180](https://paddle-org.bj.bcebos.com/paddlescience/docs/velocitygan/flatvel_a_1.png)

![image-20240914192456002](https://paddle-org.bj.bcebos.com/paddlescience/docs/velocitygan/flatvel_a_2.png)

## 6. References

- [VelocityGAN: Data-Driven Full-Waveform Inversion Using Conditional Adversarial Networks](https://arxiv.org/abs/1809.10262v6)

- [OpenFWI: Large-Scale Multi-Structural Benchmark Datasets for Seismic Full Waveform Inversion](https://arxiv.org/abs/2111.02926)

- [Reference Code](https://github.com/lanl/OpenFWI?tab=readme-ov-file#ref2)
