# Multi-Device Support

The prosperity of the PaddlePaddle ecosystem depends on the contributions of developers and users. We warmly welcome more models contributed to PaddlePaddle's multi-hardware adaptation.

## 1. Hardware Support List

The list of models adapted for various hardware in PaddleScience is summarized as follows (excluding cases only on AIStudio):

=== "Mathematics (AI for Math)"

    | Problem Type | Case Name | NVIDIA | Hygon | Taichu | MetaX |
    |-----|-----|-----|-----|-----|-----|
    | Helmholtz Equation | [SPINN(Helmholtz3D)](./examples/spinn.md) | ✅ | | ✅ | ✅ |
    | Phase Field Equation | [Allen-Cahn](./examples/allen_cahn.md) | ✅ | | | ✅ |
    | Differential Equation | [Laplace Equation](./examples/laplace2d.md) | ✅ | | ✅ | ✅ |
    | Differential Equation | [Burgers Equation](./examples/deephpms.md) | ✅ | | | ✅ |
    | Differential Equation | [Non-linear Partial Differential Equation](./examples/pirbn.md) | ✅ | | | ✅ |
    | Differential Equation | [Lorenz Equation](./examples/lorenz.md) | ✅ | | ✅ | ✅ |
    | Differential Equation | [Rossler Equation](./examples/rossler.md) | ✅ | | ✅ | ✅ |
    | Operator Learning | [DeepONet](./examples/deeponet.md) | ✅ | | ✅ | ✅ |
    | Integral Equation | [Volterra Integral Equation](./examples/volterra_ide.md) | ✅ | | ✅ | ✅ |
    | Optical Rogue Wave | [Optical rogue wave](./examples/nlsmb.md) | ✅ | | ✅ | ✅ |
    | Domain Decomposition | [XPINN](./examples/xpinns.md) | ✅ | | ✅ | ✅ |
    | Brusselator Diffusion System | [3D-Brusselator](./examples/brusselator3d.md) | ✅ | | | |
    | Symbolic Regression | [Transformer4SR](./examples/transformer4sr.md) | ✅ | | | |

=== "Technology Science (AI for Technology)"

    | Problem Type | Case Name | NVIDIA | Hygon | Taichu | MetaX |
    |-----|-----|-----|-----|-----|-----|
    | Car Surface Drag Prediction | [DrivAerNet](./examples/drivaernet.md) | ✅ | | | ✅ |
    | 1D Linear Convection Problem | [1D Linear Convection](./examples/adv_cvit.md) | ✅ | | | ✅ |
    | Unsteady Incompressible Fluid | [2D Lid-Driven Cavity Buoyancy Flow](./examples/ns_cvit.md) | ✅ | | | ✅ |
    | Steady Incompressible Fluid | [Re3200 2D Steady Lid-Driven Cavity Flow](./examples/ldc2d_steady.md) | ✅ | | | ✅ |
    | Steady Incompressible Fluid | [2D Darcy Flow](./examples/darcy2d.md) | ✅ | | ✅ | ✅ |
    | Steady Incompressible Fluid | [2D Pipe Flow](./examples/labelfree_DNN_surrogate.md) | ✅ | | | ✅ |
    | Steady Incompressible Fluid | [3D Intracranial Aneurysm](./examples/aneurysm.md) | ✅ | | | ✅ |
    | Steady Incompressible Fluid | [Flow Around Arbitrary 2D Geometry](./examples/deepcfd.md) | ✅ | | ✅ | ✅ |
    | Unsteady Incompressible Fluid | [2D Unsteady Lid-Driven Cavity Flow](./examples/ldc2d_unsteady.md) | ✅ | | ✅ | ✅ |
    | Unsteady Incompressible Fluid | [Re100 2D Flow Around Cylinder](./examples/cylinder2d_unsteady.md) | ✅ | | | ✅ |
    | Unsteady Incompressible Fluid | [Re100~750 2D Flow Around Cylinder](./examples/cylinder2d_unsteady_transformer_physx.md) | ✅ | | | ✅ |
    | Compressible Fluid | [2D Air Shock Wave](./examples/shock_wave.md) | ✅ | | | ✅ |
    | General Flow Field Simulation | [Aerodynamic Shape Design](./examples/amgnet.md) | ✅ | | | |
    | Fluid-Structure Interaction | [Vortex-Induced Vibration](./examples/viv.md) | ✅ | | | ✅ |
    | Multiphase Flow | [Gas-Liquid Two-Phase Flow](./examples/bubble.md) | ✅ | | | ✅ |
    | High-Resolution Flow Field Reconstruction | [2D Turbulent Flow Field Reconstruction](./examples/tempoGAN.md) | ✅ | | | ✅ |
    | Solver Coupling | [CFD-GCN](./examples/cfdgcn.md) | ✅ | | | ✅ |
    | Stress Analysis | [1D Euler Beam Deformation](./examples/euler_beam.md) | ✅ | | ✅ | ✅ |
    | Stress Analysis | [2D Plate Deformation](./examples/biharmonic2d.md) | ✅ | | | ✅ |
    | Stress Analysis | [3D Bracket Deformation](./examples/bracket.md) | ✅ | | | ✅ |
    | Stress Analysis | [Structural Vibration Simulation](./examples/phylstm.md) | ✅ | | | ✅ |
    | Stress Analysis | [2D Elastic-Plastic Structure](./examples/epnn.md) | ✅ | | | ✅ |
    | Stress Analysis and Inverse Problem | [3D Car Control Arm Deformation](./examples/control_arm.md) | ✅ | | | ✅ |
    | Stress Analysis and Inverse Problem | [3D Heart Simulation](./examples/heart.md) | ✅ | | | ✅ |
    | Topology Optimization | [2D Topology Optimization](./examples/topopt.md) | ✅ | | | ✅ |
    | Thermal Simulation | [1D Heat Exchanger Thermal Simulation](./examples/heat_exchanger.md) | ✅ | | | ✅ |
    | Thermal Simulation | [2D Thermal Simulation](./examples/heat_pinn.md) | ✅ | | | ✅ |
    | Thermal Simulation | [2D Chip Thermal Simulation](./examples/chip_heat.md) | ✅ | | | |
    | Operator Learning | [NeuralOperator](./examples/neuraloperator.md) | ✅ | | | |
    | Car Surface Drag Prediction | [DrivAerNetPlusPlus](./examples/drivaernetplusplus.md) | ✅ | | | |
    | Solver Coupling | [NSFNets](./examples/nsfnet.md) | ✅ | | | ✅ |
    | High-Resolution Flow Field Reconstruction | [PhyCRNet](./examples/phycrnet.md) | ✅ | | | |
    | Solver Coupling | [NSFNet4](./examples/nsfnet4.md) | ✅ | | | |

=== "Material Science (AI for Material)"

    | Problem Type | Case Name | NVIDIA | Hygon | Taichu | MetaX |
    |-----|-----|-----|-----|-----|-----|
    | Material Design | [Diffuser Design (Inverse Problem)](./examples/hpinns.md) | ✅ | | ✅ | ✅ |
    | Material Design | [CGCNN](./examples/cgcnn.md) | ✅ | | ✅ | |

=== "Earth Science (AI for Earth Science)"

    | Problem Type | Case Name | NVIDIA | Hygon | Taichu | MetaX |
    |-----|-----|-----|-----|-----|-----|
    | Weather Forecasting | [Extformer-MoE Weather Forecasting](./examples/extformer_moe.md) | ✅ | | ✅ | ✅ |
    | Weather Forecasting | [FourCastNet Weather Forecasting](./examples/fourcastnet.md) | ✅ | | ✅ | |
    | Weather Forecasting | [NowCastNet Weather Forecasting](./examples/nowcastnet.md) | ✅ | | | ✅ |
    | Weather Forecasting | [GraphCast Weather Forecasting](./examples/graphcast.md) | ✅ | | ✅ | ✅ |
    | Weather Forecasting | [DGMR Weather Forecasting](./examples/dgmr.md) | ✅ | | | |
    | Seismic Waveform Inversion | [VelocityGAN Seismic Waveform Inversion](./examples/velocity_gan.md) | ✅ | | | ✅ |
    | Traffic Prediction | [TGCN Traffic Flow Prediction](./examples/tgcn.md) | ✅ | | | ✅ |
    | Weather Forecasting | [EarthFormer Weather Forecasting](./examples/earthformer.md) | ✅ | | | ✅ |
    | Traffic Prediction | [IOPS Traffic Flow Prediction](./examples/iops.md) | ✅ | | | ✅ |
    | Weather Forecasting | [Pangu-Weather Weather Forecasting](./examples/pangu_weather.md) | ✅ | | |  ✅|
    | Weather Forecasting | [FengWu Weather Forecasting](./examples/fengwu.md) | ✅ | | | ✅ |

=== "Chemical Science (AI for Chemistry)"

    | Problem Type | Case Name | NVIDIA | Hygon | Taichu | MetaX |
    |-----|-----|-----|-----|-----|-----|
    | Chemical Molecule Generation | [Moflow](./examples/moflow.md) | ✅ | | | ✅ |
    | Chemical Reaction Prediction | [IFM](./examples/ifm.md) | ✅ | | | ✅ |

## 2. Running Guide

For hardware already supported by PaddleScience, we provide a running example for each hardware, taking [1D Euler Beam Deformation](./examples/euler_beam.md) as an example.

!!! note

    Please ensure that you have correctly installed PaddlePaddle corresponding to the computing hardware in your environment, otherwise please refer to [PaddleCustomDevice](https://github.com/PaddlePaddle/PaddleCustomDevice) to connect your hardware code to PaddlePaddle.

=== "NVIDIA"

    ``` sh
    # Install PaddleScience
    git clone -b develop https://github.com/PaddlePaddle/PaddleScience.git
    # If github clone is slow, you can use gitee clone
    # git clone -b develop https://gitee.com/paddlepaddle/PaddleScience.git

    cd PaddleScience

    # install paddlesci with editable mode
    python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
    cd examples/euler_beam
    ```

    === "Model Training Command"

        ``` sh
        python euler_beam.py
        ```

    === "Model Evaluation Command"

        ``` sh
        python euler_beam.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/euler_beam/euler_beam_pretrained.pdparams
        ```

    === "Model Export Command"

        ``` sh
        python euler_beam.py mode=export
        ```

    === "Model Inference Command"

        ``` sh
        python euler_beam.py mode=infer
        ```

=== "Hygon"

    ``` sh
    # Install PaddleScience
    git clone -b develop https://github.com/PaddlePaddle/PaddleScience.git
    # If github clone is slow, you can use gitee clone
    # git clone -b develop https://gitee.com/paddlepaddle/PaddleScience.git

    cd PaddleScience

    # install paddlesci with editable mode
    python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
    cd examples/euler_beam
    ```

    === "Model Training Command"

        ``` sh
        python euler_beam.py
        ```

    === "Model Evaluation Command"

        ``` sh
        # Test your trained model
        python euler_beam.py mode=eval EVAL.pretrained_model_path=$YOUR_MODEL_PATH
        # Test officially provided pre-trained model
        python euler_beam.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/euler_beam/euler_beam_pretrained.pdparams
        ```

    === "Model Export Command"

        ``` sh
        python euler_beam.py mode=export
        ```

    === "Model Inference Command"

        ``` sh
        python euler_beam.py mode=infer
        ```

=== "Taichu"

    ``` sh
    # Install PaddleScience
    git clone -b develop https://github.com/PaddlePaddle/PaddleScience.git
    # If github clone is slow, you can use gitee clone
    # git clone -b develop https://gitee.com/paddlepaddle/PaddleScience.git

    cd PaddleScience

    # install paddlesci with editable mode
    python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
    cd examples/euler_beam
    ```

    === "Model Training Command"

        ``` sh
        python euler_beam.py
        ```

    === "Model Evaluation Command"

        ``` sh
        # Test your trained model
        python euler_beam.py mode=eval EVAL.pretrained_model_path=$YOUR_MODEL_PATH
        # Test officially provided pre-trained model
        python euler_beam.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/euler_beam/euler_beam_pretrained.pdparams
        ```

    === "Model Export Command"

        ``` sh
        python euler_beam.py mode=export
        ```

    === "Model Inference Command"

        ``` sh
        python euler_beam.py mode=infer INFER.device=sdaa
        ```

=== "MetaX"

    ``` sh
    # Install PaddleScience
    git clone -b develop https://github.com/PaddlePaddle/PaddleScience.git
    # If github clone is slow, you can use gitee clone
    # git clone -b develop https://gitee.com/paddlepaddle/PaddleScience.git

    cd PaddleScience

    # install paddlesci with editable mode
    python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
    cd examples/euler_beam
    ```

    === "Model Training Command"

        ``` sh
        python euler_beam.py
        ```

    === "Model Evaluation Command"

        ``` sh
        # Test your trained model
        python euler_beam.py mode=eval EVAL.pretrained_model_path=$YOUR_MODEL_PATH
        # Test officially provided pre-trained model
        python euler_beam.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/euler_beam/euler_beam_pretrained.pdparams
        ```

    === "Model Export Command"

        ``` sh
        python euler_beam.py mode=export
        ```

    === "Model Inference Command"

        ``` sh
        python euler_beam.py mode=infer
        ```

## 3. Contribution Guide

We provide reference accuracy and corresponding pre-trained model weights based on NVIDIA CUDA training at the beginning of the public case documentation. If you need to run on specified hardware, you can refer to the following steps:

1. If your hardware type has not been connected to PaddlePaddle, you can refer to the [PaddleCustomDevice](https://github.com/PaddlePaddle/PaddleCustomDevice) official documentation to connect to the PaddlePaddle framework. If your hardware type has been connected to PaddlePaddle but has not been added to the hardware support list of PaddleScience, please add your hardware type in [ppsci/utils/config.py](https://github.com/PaddlePaddle/PaddleScience/blob/develop/ppsci/utils/config.py#L215) and [deploy/python_infer/base.py](https://github.com/PaddlePaddle/PaddleScience/blob/develop/deploy/python_infer/base.py#L217).

2. Prepare necessary datasets according to the steps given in the case documentation.

3. If the model documentation provides a model training command, you need to perform full training on your hardware, save training logs, record the best model accuracy and the best model weights. These contents are generally automatically saved in the case folder during the training process.

4. If the model documentation provides a model evaluation command, you need to evaluate the accuracy of the best model saved in the third step on your hardware, save evaluation logs, and record evaluation accuracy. These contents are generally automatically saved in the case folder during the evaluation process.

    !!! note

        For full model training accuracy, the default requirement is that the best accuracy aligns with NVIDIA CUDA accuracy. Specifically, if the case accuracy indicator is relative error (such as L2 relative error), the indicator cannot exceed the reference value by ± 0.5%. If the case accuracy indicator is an error like MSE/MAE, it should be in the same order of magnitude as the reference value.

5. If the model documentation provides model export and inference commands, please verify whether model export and inference can be executed normally on the new hardware and align with CUDA inference results according to the model export and inference commands.

6. After completing the above steps, you can add your hardware support information (✅) to the corresponding model in the table of [1. Hardware Support List](#1), and then submit a PR to PaddleScience. Your PR should include at least the following contents:
    * Add running instructions for using the model based on your hardware environment in [2. Running Guide](#2)
    * Best model weight file saved during training (`.pdparams` file)
    * Running logs such as training/evaluation (`.log` file)
    * Software versions used to verify model accuracy, including but not limited to:
        * PaddlePaddle version
        * PaddleCustomDevice version (if any)
    * Machine environment used to verify model accuracy, including but not limited to:
        * Chip model
        * System version
        * Hardware driver version
        * Operator library version, etc.

## 4. More Documentation

For more documents on PaddlePaddle multi-hardware adaptation and usage, please refer to:

* [PaddlePaddle User Guide](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/guides/index_en.html)
* [PaddlePaddle Hardware Support](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/hardware_support/index_en.html)
* [PaddleCustomDevice](https://github.com/PaddlePaddle/PaddleCustomDevice)
