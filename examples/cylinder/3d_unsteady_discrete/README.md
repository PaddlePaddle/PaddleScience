[//]: <> (title: Flow around a cylinder use case tutorial, author: Yanbo Zhang @zhangyanbo at baidu.com)


# 3D unsteady flow around a cylinder with Re=3900
This guide introduces how to build a PINN model for simulating the 3D flow around a cylinder in PaddleScience.
In this example,  you may directly load pre-check models and compare loss curves with them. Wish you can reach a better result. 
The working directory is:
```
*/PaddleScience/examples/cylinder/3d_unsteady_discrete/
```

## Configuration
If you want to configure out the hyper-parameters of this model, you can modify the following file.
```
vim config.yaml
```

## Train
This guide introduces to how to build a PINN model for simulating the flow around a cylinder in PaddleScience.
Run the command as follows:
```
python3.7 cylinder3d_unsteady.py
```
This is the main process of this example, files from <font color="#dd0000">./data</font> are input. Eventually, <font color="#dd0000">*.pdparams</font> are build under <font color="#dd0000">./checkpoint</font> and losses are recorded in <font color="#dd0000">./output</font>

## VTK generator
The baseline comparasion can be founded in 'vtk_generator'. You can compute the Quantitative error (baseline-based error) and try to minimize them! Either you gain a serious of *.vtu files (open them in Paraview).
Run the command as follows:
```
python3.7 vtk_generator.py
```

## Loss processor
A simple loss analysis is deployed in this file, inputs loss file (txt in <font color="#dd0000">./output</font>).
Run the command as follows:
```
python3.7 loss_processor.py
```
In the first step, the scripts will generates the <font color="#dd0000">total_loss</font> which consist of 1 total loss diagram, 4 categories of losses, and 1 learning-rate curve. For each diagram, there is a reference curve for comparisio. And will show you loss ratio. Secondly, an intuitional photo named <font color="#dd0000">loss_ratio.jpg</font> will be formed based on loss data. It tells how each loss evovles in ratio. At last, since the model is logged by each 1000 epochs, the option : "evaluate with the best model" will be neccessary. Once this evaluation is performed in vtk_generator and "error.csv" is obtained, the baseline-based error can be visulized in <font color="#dd0000">train_validation_cmp.jpg</font>.
