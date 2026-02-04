# WGANGP

!!! note

    1. Before running, download [Cifar10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and update data_path in wgangp_cifar10.yaml
    2. Before running, download [MINST](http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz) and update data_path in wgangp_mnist.yaml

=== "Model Training Command"

```sh
# CIFAR10 Experiment
python wgangp_cifar10.py
# MNIST Experiment
python wgangp_mnist.py
# Toy Dataset Experiment
python wgangp_toy.py
```

=== "Model Evaluation Command"

```sh
# CIFAR10 Experiment
python wgangp_cifar10.py mode=eval EVAL.pretrained_gen_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/wgangp/model_generator_cifar10.pdparams #EVAL.pretrained_dis_model_path is the model address after downloading from https://paddle-org.bj.bcebos.com/paddlescience/models/wgangp/model_discriminator_cifar10.pdparams
# MNIST Experiment
python wgangp_mnist.py mode=eval EVAL.pretrained_gen_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/wgangp/model_generator_mnist.pdparams #EVAL.pretrained_dis_model_path is the model address after downloading from https://paddle-org.bj.bcebos.com/paddlescience/models/wgangp/model_discriminator_mnist.pdparams
# Toy Dataset Experiment
python wgangp_toy.py mode=eval EVAL.pretrained_gen_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/wgangp/model_generator_toy_8gaussians.pdparams #EVAL.pretrained_dis_model_path is the model address after downloading from https://paddle-org.bj.bcebos.com/paddlescience/models/wgangp/model_discriminator_toy_8gaussians.pdparams
```

| Pretrained Model                                                                                      | Metric      |
|:-------------------------------------------------------------------------------------------|:--------|
| [wgangp_cifar10_gen_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/wgangp/model_generator_cifar10.pdparams) <br> [wgangp_cifar10_dis_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/wgangp/model_discriminator_cifar10.pdparams) | IS: 5.2 |



## 1. Background Introduction

In the fields of digital image processing and machine learning, Generative Adversarial Networks (GANs) have attracted widespread attention due to their excellent image generation capabilities. However, traditional GAN architectures may encounter instability problems during training, especially when generating high-resolution or complex scene images. To solve these problems, researchers proposed Wasserstein Generative Adversarial Networks with Gradient Penalty (WGAN-GP), which not only enhances the stability of the training process, but also significantly improves the quality of generated images.

WGAN-GP minimizes the difference between the real data distribution and the generated data distribution by improving the loss function, and introduces a gradient penalty mechanism to ensure smoothness and stability during the training process. This optimization method overcomes the common mode collapse problem in traditional GANs, while promoting more efficient training and more realistic image generation.

## 2. Model Principle

WGAN-GP proposes an alternative to weight clipping: penalize the norm of the gradient of the critic's input. Stabilize the training of multiple GAN architectures with almost no hyperparameter tuning.

### 2.1 Model Structure

WGAN-GP is a conditional adversarial network containing a noise-to-image generator and a CNN discriminator. The overall structure of the model is shown below.

```
    noise===>generator===>fake_image==
                                      ==>discriminator===>Wasserstein Loss+Gradient Penalty
                               image==
```

- `Generator` is a convolutional neural network.

- `Discriminator` is a model composed of convolutional blocks. Input image, output image authenticity score.

### 2.2 Loss Function

The discriminator's loss function uses Wasserstein loss and gradient penalty. Its expression is:

$$
L_d = \underset{\tilde{x} \sim \mathbb{P}_g}{\mathbb{E}} D(\tilde{x}) - \underset{x \sim \mathbb{P}_r}{\mathbb{E}}D(x) + \lambda \underset{\hat{x} \sim \mathbb{P}_{\hat{x}}}{\mathbb{E}} \left[ \left( \| \nabla_{\hat{x}} D(\hat{x}) \|_2 - 1 \right)^2 \right]
$$

Where $\mathbb{P}_g$ is the generator distribution, $\mathbb{P}_r$ is the real data distribution, and $\mathbb{P}_{\hat{x}}$ is a mixed interpolation sample from $\mathbb{P}_g$ and $\mathbb{P}_r$.

The generator's loss function is adversarial loss [$- \underset{\tilde{x} \sim \mathbb{P}_g}{\mathbb{E}}D(\tilde{x})$]. Its expression is:

$$
L_g = - \underset{\tilde{x} \sim \mathbb{P}_g}{\mathbb{E}}D(\tilde{x})
$$

Where $\mathbb{P}_g$ is the generator distribution

## 3. Model Construction

Next, we will explain how to use the PaddleScience framework to implement WGAN-GP. The following content only elaborates on key steps. For other details, please refer to [API Documentation](https://paddlescience-docs.readthedocs.io/en/latest/en/api/arch/).

### 3.1 Dataset Introduction

The dataset uses [Cifar10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) dataset, [MNIST](http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz) and toy datasets (swissroll/8gaussians/25gaussians).

Cifar10 dataset contains 60000 32x32 color images, divided into 10 categories, with 6000 images per category.

Cifar10 dataset has 3 versions

| Version          | Size        | md5sum                               |
|:-----------------|:------------|:-------------------------------------|
| CIFAR-100 python | 161 MB      | eb9058c3a382ffc7106e4002c42a8d85     |
| CIFAR-100 Matlab | 175 MB      | 6a4bfa1dcd5c9453dda6bb54194911f4     |
| CIFAR-100 binary | 161 MB      | 03b5dce01913d631647c71ecec9e9cb8     |

This implementation uses CIFAR-100 python version

MNIST dataset contains 60000 28x28 grayscale images, divided into 10 categories, with 6000 images per category.

Toy datasets

Swissroll: Three-dimensional nonlinear manifold dataset, presenting a continuous curled spiral structure,

8gaussians: Two-dimensional synthetic dataset containing eight symmetrically distributed Gaussian clusters, with centers uniformly distributed on a circle,

25gaussians: High-density Gaussian mixture dataset consisting of 25 regularly arranged two-dimensional Gaussian distributions with compact cluster spacing.

### 3.2 Build dataset API

Since the Cifar10 dataset consists of 5 data files, due to the dataset organization method, we cannot directly use the built-in dataset API of PaddleScience, so read all data first, and then use ```ppsci.data.dataset.array_dataset.NamedArrayDataset```.

The code for reading Cifar10 dataset is given below:
``` py linenums="167"
--8<--
examples/wgangp/functions.py:167:177
--8<--
```
Where `data_path` passes in the path of CIFAR-10.

The configuration code of dataloader is given below:
``` py linenums="108"
--8<--
examples/wgangp/wgangp_cifar10.py:108:122
--8<--
```

Since the MNIST dataset cannot directly use the built-in dataset API of PaddleScience, read all data first, and then use ```ppsci.data.dataset.array_dataset.NamedArrayDataset```.

The code for reading MNIST dataset is given below:
``` py linenums="368"
--8<--
examples/wgangp/functions.py:368:377
--8<--
```

The configuration code of dataloader is given below:
``` py linenums="101"
--8<--
examples/wgangp/wgangp_mnist.py:101:114
--8<--
```

Since the toy dataset cannot directly use the built-in dataset API of PaddleScience, generate all data first, and then use ```ppsci.data.dataset.array_dataset.NamedArrayDataset```.

The generation code of toy dataset is given below
``` py linenums="194"
--8<--
examples/wgangp/functions.py:194:236
--8<--
```

The configuration code of dataloader is given below:
``` py linenums="94"
--8<--
examples/wgangp/wgangp_toy.py:94:107
--8<--
```

### 3.3 Model Construction

WGAN-GP in this case is not built into PaddleScience and needs to be implemented additionally, so we customized `WganGpCifar10Generator` and `WganGpCifar10Discriminator`, `WganGpMnistGenerator` and `WganGpMnistDiscriminator`, `WganGpToyGenerator` and `WganGpToyDiscriminator`.

The model construction code is as follows:

`WganGpCifar10Generator` and `WganGpCifar10Discriminator`
``` py linenums="92"
--8<--
examples/wgangp/wgangp_cifar10.py:92:93
--8<--
```

`WganGpMnistGenerator` and `WganGpMnistDiscriminator`
``` py linenums="87"
--8<--
examples/wgangp/wgangp_mnist.py:87:88
--8<--
```

`WganGpToyGenerator` and `WganGpToyDiscriminator`
``` py linenums="80"
--8<--
examples/wgangp/wgangp_toy.py:80:81
--8<--
```

Parameter configuration is as follows:

`WganGpCifar10Generator` and `WganGpCifar10Discriminator`
```yaml linenums="29"
--8<--
examples/wgangp/conf/wgangp_cifar10.yaml:29:43
--8<--
```

`WganGpMnistGenerator` and `WganGpMnistDiscriminator`
```yaml linenums="29"
--8<--
examples/wgangp/conf/wgangp_mnist.yaml:29:38
--8<--
```

`WganGpToyGenerator` and `WganGpToyDiscriminator`
```yaml linenums="29"
--8<--
examples/wgangp/conf/wgangp_toy.yaml:29:37
--8<--
```

### 3.4 Custom loss

The loss function of WGAN-GP is relatively complex and needs to be implemented by ourselves. PaddleScience provides an API for customizing loss functions - `ppsci.loss.FunctionalLoss`. The method is to define the loss function first, and then pass the function name as a parameter to `FunctionalLoss`. Note that the input and output of the custom loss function need to be in the format of a dictionary.

#### 3.4.1 Loss of Generator

The loss of Cifar10_Generator contains adversarial loss and classification loss. Both losses have corresponding weights. If the weight of a certain loss is 0, it means that the loss item is not added during training.
``` py linenums="16"
--8<--
examples/wgangp/functions.py:16:44
--8<--
```

The loss of MNIST_Generator only contains adversarial loss.
``` py linenums="313"
--8<--
examples/wgangp/functions.py:313:328
--8<--
```

The loss of Toy_Generator only contains adversarial loss.
``` py linenums="238"
--8<--
examples/wgangp/functions.py:238:254
--8<--
```

#### 3.4.2 Loss of Discriminator

The loss of Cifar10_Discriminator contains Wasserstein loss, gradient penalty and classification loss. Among them, only the classification loss item has weight parameters.
``` py linenums="46"
--8<--
examples/wgangp/functions.py:46:95
--8<--
```

The loss of MNIST_Discriminator contains Wasserstein loss and gradient penalty.
``` py linenums="330"
--8<--
examples/wgangp/functions.py:330:366
--8<--
```

The loss of Toy_Discriminator contains Wasserstein loss and gradient penalty.
``` py linenums="256"
--8<--
examples/wgangp/functions.py:256:292
--8<--
```

### 3.5 Constraint Construction

All cases use `ppsci.constraint.SupervisedConstraint` to construct constraints.

The construction code is as follows:

For Cifar10 experiment
``` py linenums="125"
--8<--
examples/wgangp/wgangp_cifar10.py:125:141
--8<--
```

For MNIST experiment
``` py linenums="117"
--8<--
examples/wgangp/wgangp_mnist.py:117:132
--8<--
```

For toy dataset experiment
``` py linenums="110"
--8<--
examples/wgangp/wgangp_toy.py:110:125
--8<--
```

### 3.6 Optimizer Construction

WGANGP uses Adam optimizer, which can be directly constructed by calling `ppsci.optimizer.Adam`, code as follows:

For Cifar10 experiment
``` py linenums="144"
--8<--
examples/wgangp/wgangp_cifar10.py:144:158
--8<--
```

For MNIST experiment
``` py linenums="135"
--8<--
examples/wgangp/wgangp_mnist.py:135:137
--8<--
```

For toy dataset experiment
``` py linenums="128"
--8<--
examples/wgangp/wgangp_toy.py:128:131
--8<--
```

### 3.7 Solver Construction

Pass the constructed model, constraints, optimizer and other parameters to `ppsci.solver.Solver`.

For Cifar10 experiment
``` py linenums="161"
--8<--
examples/wgangp/wgangp_cifar10.py:161:178
--8<--
```

For MNIST experiment
``` py linenums="140"
--8<--
examples/wgangp/wgangp_mnist.py:140:157
--8<--
```

For toy dataset experiment
``` py linenums="134"
--8<--
examples/wgangp/wgangp_toy.py:134:151
--8<--
```

### 3.8 Model Training

For Cifar10 experiment
``` py linenums="181"
--8<--
examples/wgangp/wgangp_cifar10.py:181:186
--8<--
```

For MNIST experiment
``` py linenums="160"
--8<--
examples/wgangp/wgangp_mnist.py:160:165
--8<--
```

For toy dataset experiment
``` py linenums="154"
--8<--
examples/wgangp/wgangp_toy.py:154:159
--8<--
```

### 3.9 Custom metric

In the cases, only the case for Cifar10 has an evaluation metric as Inception Score, while MNIST and Toy cases do not have evaluation metrics. Since an error will be reported if metric is empty, an invalid metric is customized.

So we implemented two additional metrics

PaddleScience provides an API for customizing metric functions - `ppsci.metric.FunctionalMetric`. The method is to define the metric function first, and then pass the function name as a parameter to `FunctionalMetric`. Note that the input and output of the custom metric function need to be in the format of a dictionary.

The implementation code of Inception Score is as follows:
``` py linenums="97"
--8<--
examples/wgangp/functions.py:97:154
--8<--
```

The code of invalid_metric is as follows
``` py linenums="389"
--8<--
examples/wgangp/functions.py:389:391
--8<--
```

### 3.10 Validator Construction

This case uses `ppsci.validate.SupervisedValidator` to construct the validator.

For Cifar10 experiment
``` py linenums="53"
--8<--
examples/wgangp/wgangp_cifar10.py:53:62
--8<--
```

For MNIST experiment
``` py linenums="46"
--8<--
examples/wgangp/wgangp_mnist.py:46:54
--8<--
```

For toy dataset experiment
``` py linenums="46"
--8<--
examples/wgangp/wgangp_toy.py:46:52
--8<--
```

### 3.11 Model Evaluation

After passing the model, validator and weight path to `ppsci.solver.Solver`, start evaluation through `solver.eval()`.

For Cifar10 experiment
``` py linenums="65"
--8<--
examples/wgangp/wgangp_cifar10.py:65:74
--8<--
```

For MNIST experiment
``` py linenums="56"
--8<--
examples/wgangp/wgangp_mnist.py:56:65
--8<--
```

For toy dataset experiment
``` py linenums="55"
--8<--
examples/wgangp/wgangp_toy.py:55:63
--8<--
```

### 3.12 Visualization

After evaluation, we visualize the results in the form of images, code as follows:

For Cifar10 experiment
``` py linenums="76"
--8<--
examples/wgangp/wgangp_cifar10.py:76:87
--8<--
```

For MNIST experiment
``` py linenums="67"
--8<--
examples/wgangp/wgangp_mnist.py:67:83
--8<--
```

For toy dataset experiment
``` py linenums="65"
--8<--
examples/wgangp/wgangp_toy.py:65:75
--8<--
```

## 4. Complete Code

For Cifar10 experiment
``` py
--8<--
examples/wgangp/wgangp_cifar10.py
--8<--
```

For MNIST experiment
``` py
--8<--
examples/wgangp/wgangp_mnist.py
--8<--
```

For toy dataset experiment
``` py
--8<--
examples/wgangp/wgangp_toy.py
--8<--
```

## 6. References

- [Improved Training of Wasserstein GANs Paper](https://arxiv.org/abs/1704.00028)

- [Reference Code](https://github.com/igul222/improved_wgan_training)
