[//]: <> (title: Flow around a cylinder use case tutorial, author: Xiandong Liu @liuxiandong at baidu.com)


# Flow around a Cylinder

This guide introduces to how to build a PINN model for simulating the flow around a cylinder in PaddleScience.
In this example, two versions are provided. It is recommended to pay attention to the baseline version first. 
If you want higher training speed or want to run on distributed systems, please pay attention to the optimize version.


## Baseline
This guide introduces to how to build a PINN model for simulating the flow around a cylinder in PaddleScience.
Run the command as follows:
```
cd baseline
python3.7 cylinder3d_unsteady.py
```

## Optimize
If you want to try out the advanced automatic differentiation function of static graph, you can run the following command.
```
cd optimize
python3.7 cylinder3d_unsteady_optimize.py
```
On this basis, if you want to use a distributed system, you can run the following command:
```
cd optimize
python3.7 -m paddle.distributed.launch --gpus=1,2 cylinder3d_unsteady_optimize.py
```
