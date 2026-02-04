# XRDMatch

## 1. Model Training and Evaluation
=== "Model Training Command"
``` sh
    python main.py
```
=== "Model Evaluation Command"
``` sh
    python main.py --mode eval --exp_id x --epoch x
```

## 2. Background Introduction

XRDMatch is a semi-supervised learning example of XRD data based on PaddleScience, using the FlexMatch algorithm for material classification. This example demonstrates how to use a small amount of labeled data and a large amount of unlabeled data to train a high-performance classification model, which is particularly suitable for XRD spectral analysis in materials science.

X-ray diffraction (XRD) is an important characterization technique in materials science that can provide information on the crystal structure of materials. In practical applications, obtaining a large amount of labeled XRD data is costly and time-consuming, while semi-supervised learning can make full use of a large amount of unlabeled data to improve model performance and reduce annotation costs.

The purpose of this work is to use XRD data of lithium-ion solid electrolyte materials for training to obtain the corresponding structure and performance relationship. Through the FlexMatch algorithm, combined with technologies such as data augmentation, pseudo-label generation, dynamic thresholding, and consistency regularization, efficient semi-supervised learning is achieved.


## 3. Model Principle

The main idea of this method is to establish a non-linear mapping relationship between XRD spectral data and material properties through a convolutional neural network. The model uses the VGG network as a feature extractor, combined with the FlexMatch semi-supervised learning algorithm, which can effectively use a large amount of unlabeled data to improve model performance.

This case uses the VGG network as the basic model architecture, mainly including the following parts:

1. Input layer: Receive 1×4501 XRD spectral data
2. Convolutional layer: Multi-layer convolutional block to extract local feature patterns
3. Pooling layer: Dimensionality reduction and feature aggregation
4. Fully connected layer: Feature mapping to classification results
5. Output layer: 2-class classification (positive/negative)

Through the FlexMatch algorithm, the model can:
- Generate pseudo labels based on weak augmented data
- Use strong augmented data for consistency training
- Dynamically adjust selection thresholds to balance samples of each category

### 3.1 Data Format Description

The dataset contains material XRD spectral data and corresponding performance labels:
- **Data Link**:
```
https://paddle-org.bj.bcebos.com/paddlescience/datasets/xrdmatch/lbs.csv
https://paddle-org.bj.bcebos.com/paddlescience/datasets/xrdmatch/ulbs.csv
```
- **`xrd_data/lbs.csv`**: Labeled data
  - Contains sample name, ID, label and XRD spectral data (4501-dimensional features)
  - Label: 0 (positive class), 1 (negative class)

- **`xrd_data/ulbs.csv`**: Unlabeled data
  - Contains sample name, ID and XRD spectral data (4501-dimensional features)
  - No label information, used for semi-supervised learning

### 3.2 Data Preprocessing and Augmentation Strategy

1. **Normalization**: Normalize XRD intensity values to [0,1] range
2. **Noise Processing**: Remove low-intensity noise (threshold < 0.1)
3. **Data Augmentation**:
   - **Weak Augmentation**: Add small amount of noise (10%) and shift (100 pixels)
   - **Strong Augmentation**: Scaling (15%), elimination (15%), large shift (200 pixels) and noise (20%)
``` py linenums="42" title="examples/xrdmatch/main.py"
--8<--
examples/xrdmatch/main.py:42:89
--8<--
```
### 3.3 Custom Dataset Class
``` py linenums="201" title="examples/xrdmatch/main.py"
--8<--
examples/xrdmatch/main.py:201:239
--8<--
```
### 3.4 FlexMatch Semi-supervised Loss Function

1. **Labeled Data Training**: Use cross-entropy loss for supervised learning
2. **Unlabeled Data Processing**:
   - Generate weak augmented and strong augmented versions
   - Generate pseudo labels based on weak augmented versions
   - Use strong augmented versions for consistency training
3. **Dynamic Threshold**: Dynamically adjust selection threshold based on category confidence
``` py linenums="242" title="examples/xrdmatch/main.py"
--8<--
examples/xrdmatch/main.py:242:327
--8<--
```

### 3.5 Loss Function
```python
total_loss = loss_lb + lambda_u * loss_ulb
```

Where:
- `loss_lb`: Cross-entropy loss of labeled data
- `loss_ulb`: Consistency loss of unlabeled data
- `lambda_u`: Unlabeled loss weight (default 1.0)

### 3.6 Training Configuration

- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.01)
- **Batch Size**: Labeled 32, Unlabeled 96
- **Number of Experiments**: 100 independent experiments
- **Training Epochs**: 100 epochs per experiment (10 iterations per epoch)
- **Data Split**: First 20 positive classes, first 75 negative classes used for training
- **Model Saving**: Save model only when F1 score ≥ 0.7

## 3.7 Evaluation Metrics

- **Accuracy**: Proportion of correctly classified samples
- **Precision**: Proportion of actually positive samples among those predicted as positive
- **Recall**: Proportion of correctly predicted samples among actual positive samples
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed distribution of prediction results for each category
- **Evaluation Method**: Supports two modes: evaluation during training and independent evaluation
- **Evaluation during training**: Automatically called during training, logs saved to saved_models_ppsci/exp_*/log.txt file for each experiment
- **Independent evaluation**: Use `--mode eval` parameter to evaluate saved models, results saved to eval_log.txt file
- **Model saving strategy**: Save model only when F1 score ≥ 0.7
- **Built-in evaluation implementation**: This function will be automatically called during training, and logs will be saved to saved_models_ppsci/exp_*/log.txt file for each experiment. Code implementation:
``` py linenums="412" title="examples/xrdmatch/main.py"
--8<--
examples/xrdmatch/main.py:412:457
--8<--
```

## 4. Result Example

### Training Log Example

```
Epoch: 0
[2025-8-27 02:40:12,747 INFO] confusion matrix
[2025-8-27 02:40:12,748 INFO] [[0.22222222 0.77777778]
 [0.2        0.8       ]]
[2025-8-27 02:40:12,748 INFO] evaluation metric
[2025-8-27 02:40:12,748 INFO] acc: 0.7188
[2025-8-27 02:40:12,748 INFO] precision: 0.5083
[2025-8-27 02:40:12,750 INFO] recall: 0.5111
[2025-8-27 02:40:12,750 INFO] f1: 0.5060
F1 score 0.5060 < 0.7, model not saved at epoch 0
```

### Performance Metrics

Typical performance on standard test set:

| Metric | Value |
|------|-----|
| Accuracy | 0.797 |
| Precision | 0.673 |
| Recall | 0.789 |
| F1 Score | 0.695 |

### Evaluation Log Example

```
Evaluating experiment 0 epoch 11 model...
Starting prediction...
confusion matrix
[[0.77777778 0.22222222]
 [0.16363636 0.83636364]]
evaluation metric
acc: 0.6480
precision: 0.6480
recall: 0.6480
f1: 0.6480
```

## 5. Complete Code
``` py linenums="1" title="examples/xrdmatch/main.py"
--8<--
examples/xrdmatch/main.py
--8<--
```

## References

Zheng Wan., et al.** "XRDMatch: a semi-supervised learning framework to efficiently discover room temperature lithium superionic conductors." *Energy Environ. Sci.*, 2024, 17, 9487. (https://pubs.rsc.org/en/content/articlelanding/2024/ee/d4ee02970d)
