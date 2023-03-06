[//]: <> (title: Flow around a cylinder use case tutorial, author: Yanbo Zhang @zhangyanbo at baidu.com)


# 3D Flow around a Cylinder 

This guide introduces to how to build a PINN model for simulating the 3D flow around a cylinder in PaddleScience.
In this example, two versions are provided. It is recommended to pay attention to the baseline version first. 
If you want higher training speed or want to run on distributed systems, please pay attention to the optimize version.


## Baseline
This guide introduces to how to build a PINN model for simulating the flow around a cylinder in PaddleScience.
Run the command as follows:
```
python3.7 cylinder3d_unsteady.py
```

## Parameters
If you want to configure out the hyper-parameters of this model, you can modify the following file.
```
config.yaml
```

## Loss processor
If you want to determine 5 loss curves of this model, you shall firstly input the loss file name in 'loss_processor.py'.
Run the command as follows:
```
python3.7 loss_processor.py
```
It will gives you all 5 losses curves with current best loss record. And will show you loss ratio.

## VTK generator
The baseline comparasion can be founded in 'vtk_generator'. You can compute the Quantitative error and try to minimize them!. Either you can get *.vtu files(open in Paraview)
Run the command as follows:
```
python3.7 vtk_generator.py
```