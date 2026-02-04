# Predicting the Strength of Composites

=== "Model Training Command"

    ``` sh
    python main.py
    ```

=== "Model Evaluation Command"

    ``` sh
    python main.py mode=eval
    ```

## Download Pretrained Model

| [resnet18-v5-fold1](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) |
[resnet18-v5-fold2](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) |
[resnet18-v5-fold3](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) |
[resnet18-v5-fold4](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) |
[resnet18-v5-fold5](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) |

## Download Necessary Model Parameters

| [Saved_Output](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/Saved_Output.tar.gz) |

## Background Introduction

The ultimate tensile strength (UTS) of a material is a core indicator for measuring the tensile failure of composite materials, directly determining its application safety and reliability. It is a key basis for structural design, ensuring that components do not fail under tensile loads; it is also an important criterion for material selection, matching the strength requirements of different scenarios, and ultimately ensuring the performance upper limit of composite products. However, due to complex morphology-property relationships, predicting its mechanical properties remains difficult, and traditional machine learning methods struggle to make effective predictions.

Aiming at the problem of material structural strength prediction in the field of materials science, the ultimate tensile strength (UTS) of polymer-ceramic composites is predicted through X-ray CT images. Compared with traditional material strength prediction methods which have strict requirements for data and models and consume long time costs, this project uses deep learning technology to achieve high-precision UTS value prediction under small sample dataset conditions, providing a faster and more accurate tool. Helping researchers quickly understand material properties and optimize material design.

This study uses Convolutional Neural Networks (CNN) to analyze X-ray Computed Tomography (CT) images of cold-sintered polymer-ceramic composites to address this issue. Traditional machine learning models with morphological features as input produce limited accuracy, while using pre-trained convolutional neural networks and further optimizing the model using ensemble learning. Alternative machine learning methods using small datasets to reveal morphology-structure-property relationships in composites provide a more precise and efficient solution for measuring composite performance.

## Directory Structure

```
CNN_UTS/
│
├─ conf/
│    └─ resnet.yaml
├─ data_utils.py
├─ model_utils.py
├─ main.py
├─ requirements.txt
├─ readme.md
├─ resnet18-v5-finetune/
├─ outputs/
├─ Saved_Output/
└─ Dataset/
     ├─ Train_val/
     └─ Test/
```

## 2. Model Principle

This chapter introduces the principle of the material tensile strength prediction model based on convolutional neural networks.

The main idea of this method is to establish a non-linear mapping relationship between material microstructure images and tensile strength (UTS) through convolutional neural networks. The model adopts ResNet architecture, which can effectively extract deep feature information in images.

This case adopts ResNet-18 as the basic model architecture, mainly including the following parts:

1. Input layer: Receives 224×224×3 RGB image data
2. Convolutional layer: Multiple convolutional blocks, including residual connections
3. Pooling layer: Max pooling operation, reducing feature map size
4. Fully connected layer: Mapping features to final predicted value
5. Output layer: Output predicted UTS value (MPa)

In this way, we can automatically learn key features in material microstructure images, establish mapping relationship between images and performance, and achieve accurate tensile strength prediction.

## 3. Model Implementation

This chapter explains how to implement the material tensile strength prediction model based on PaddleScience code. This case uses 5-fold cross-validation for model training and evaluation, and uses various built-in functional modules of PaddleScience.

### 3.1 Data Format Description

Dataset download link: <https://paddle-org.bj.bcebos.com/paddlescience/datasets/CNN_UTS/Dataset.zip>

| Image Name         | ...Feature Columns... | UTS (MPa) | ... |
|--------------------|--------------|-----------|-----|
| IPP_10__40060.jpg  | ...          | 0.56      | ... |
| ...                | ...          | ...       | ... |

The dataset used in this case contains material microstructure images and corresponding tensile strength labels. The dataset is divided into the following parts:

1. Training set: `Dataset/Train_val/`
2. Test set: `Dataset/Test/`

Dataset structure is as follows:

- Each sample contains an RGB image and corresponding UTS label
- Images are preprocessed and uniformly adjusted to 224×224 size
- Normalized using standardization parameters of ImageNet pretrained weights

To facilitate data processing, we use the `make_dataset` function to create the dataset:

``` py linenums="73" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:73:74
--8<--
```

### 3.2 Model Construction

This case uses `paddle.vision.models.resnet18` built in PaddlePaddle to construct the ResNet-18 model. The main parameters of the model include:

1. Network structure: ResNet-18 (2,2,2,2)
2. Input channels: 3 (RGB images)
3. Output dimension: 1 (UTS predicted value)
4. Pretrained weights: ImageNet

Model definition code is as follows:

``` py linenums="112" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:112:115
--8<--
```

### 3.3 Data Augmentation

To improve the generalization ability of the model, we implemented a variety of data augmentation strategies:

1. Random horizontal flip
2. Random vertical flip
3. Center crop to 224×224
4. Standardization processing

Data augmentation configuration is as follows:

``` py linenums="53" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:53:70
--8<--
```

### 3.4 Training Strategy

This case adopts a 5-fold cross-validation strategy for model training:

1. Divide training data into 5 folds
2. Train an independent model for each fold
3. Finally use prediction results of all folds for ensemble

Training process includes:

``` py linenums="85" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:85:98
--8<--
```

### 3.5 Loss Function and Optimizer

Use mean squared error loss function for regression task:

``` py linenums="116" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:116:116
--8<--
```

Use Adam optimizer for parameter update:

``` py linenums="117" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:117:119
--8<--
```

### 3.6 Model Evaluation

Evaluation process includes:

1. Calculate MSE and R² metrics
2. Generate parity plot and violin plot
3. Perform ensemble prediction

Validator construction code is as follows:

``` py linenums="156" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py:156:188
--8<--
```

## 4. Complete Code

``` py linenums="1" title="examples/CNN_UTS/main.py"
--8<--
examples/CNN_UTS/main.py
--8<--
```

## References

- [Predicting the Strength of Composites with Computer Vision Using Small Experimental Datasets](<https://pubs.acs.org/doi/10.1021/acsmaterialslett.4c02424>)
