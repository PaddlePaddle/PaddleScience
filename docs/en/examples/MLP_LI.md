# Battery_LI (Lithium-ion Battery Electrode Material Performance Prediction)

## Background Introduction

Lithium-ion Battery (LIB), as the core of modern energy storage technology, is widely used in consumer electronics, electric vehicles, and renewable energy storage. Electrode materials are the key to the performance of lithium-ion batteries, directly determining the energy density, power density, lifespan, and safety of batteries. However, the research and development of electrode materials is a complex and time-consuming process, usually requiring a combination of experimental testing and theoretical calculation, which consumes a lot of time and resources.

## Model Principle

This Multi-Layer Perceptron (MLP) model aims to use features extracted from the Materials Project dataset to predict the electrochemical performance of lithium-ion battery electrode materials. Input features include stoichiometric properties, crystal structure characteristics, electronic structure properties, and other battery attributes. The output is average voltage, specific energy, and specific capacity.

## Dataset Introduction

| Dataset Name | Download Link |
|-----------|---------|
| Training Set + Validation Set | [MP_data_down_loading(train+validate).csv](https://paddle-org.bj.bcebos.com/paddlescience/docs/MP_data_down_loading(train+validate).csv) |
| Training Set + Validation Set + Test Set | [MP_data_down_loading(train+validate+test).csv](https://paddle-org.bj.bcebos.com/paddlescience/docs/MP_data_down_loading(train+validate+test).csv) |

Data reading requires additional dependency `bayesian-optimization`. Please run the installation command `pip install bayesian-optimization`.

## Model

To view the specific implementation of this model, please refer to the following code file: `MLP_LI.py`
(Implementation of evaluation part not added)

## Trained Model Weight File

| Pretrained Model                        |
|-----------------------------------|
| [MLP_LI_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/MLP_LI_pretrained.pdparams) |


## Model Training Command
=== "Model Training Command"

    ``` sh
    # Train model
    python MLP_LI.py --train

    # Download pretrained model (if needed)
    wget -c "https://paddle-org.bj.bcebos.com/paddlescience/models/MLP_LI/MLP_LI_pretrained.pdparams"
    ```

## Complete Code

``` py linenums="1" title="examples/MLP_LI/MLP_LI.py"
--8<--
examples/MLP_LI/MLP_LI.py
--8<--
```

## Model Performance

The performance of the model on the test set is as follows:

- **Test Loss**: 0.0058

- **VRMSE Voltage**: 0.73
- **CRMSE Specific Capacity**: 165.01
- **ERMSE Specific Energy**: 238.64
- **Average RMSE**: 134.79

In addition, the average absolute error (MAE) of the model on various output indicators is as follows:

- **VMAE Voltage**: 0.55
- **CMAE Specific Capacity**: 73.34
- **EMAE Specific Energy**: 180.10
- **Average MAE**: 84.66

These results indicate that the model has high accuracy in predicting voltage, while there is still room for improvement in predicting specific capacity and specific energy.

### Charts

#### 1. Performance Prediction of Voltage (Original Scale)
This chart shows the performance prediction of voltage. Comparison between predicted values and true values is used to evaluate the accuracy of the model.

![Performance Prediction of Voltage (Original Scale)](https://paddle-org.bj.bcebos.com/paddlescience%2Fdocs%2Fperformance_prediction_voltage.png)

#### 2. Performance Prediction (Original Scale)
This chart shows the overall prediction performance of the model for all three electrochemical properties (voltage, specific energy, and specific capacity).

![Performance Prediction (Original Scale)](https://paddle-org.bj.bcebos.com/paddlescience%2Fdocs%2Fperformance_prediction_original.png )

#### 3. Initial Training Loss
The following figure shows the changes in training and validation loss during the initial training phase (by Epochs).

![Initial Training Loss](https://paddle-org.bj.bcebos.com/paddlescience%2Fdocs%2Finitial_training_loss.png)

## Conclusion
The MLP model shows strong predictive capability on the provided dataset, especially in voltage prediction. However, there is still room for further improvement in the prediction of specific capacity and specific energy. In the future, richer feature engineering, more complex model architectures, and optimized hyperparameter tuning can be used to improve the predictive performance of the model.

## Next Steps
1. Consider adding additional features or performing feature engineering to improve model prediction accuracy.
2. Try different neural network architectures or optimization strategies to improve performance.
3. Continue hyperparameter optimization to obtain better model performance.


## References

Yang, X., Li, Y., Liu, Z., & Zhang, W. (2022)
(https://doi.org/10.1016/j.gee.2022.10.002)
