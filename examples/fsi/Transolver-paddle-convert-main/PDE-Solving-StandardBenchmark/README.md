# Transolver for PDE Solving

We evaluate [Transolver](https://arxiv.org/abs/2402.02366) with six widely used PDE-solving benchmarks, which is provided by [FNO and GeoFNO](https://github.com/neuraloperator/neuraloperator).

**Transolver achieves 22% averaged relative promotion over the previous second-best model, presenting favorable efficiency and scalibility.**

<p align="center">
<img src=".\fig\standard_benchmark.png" height = "300" alt="" align=center />
<br><br>
<b>Table 1.</b> Comparison in six standard benchmarks. Relative L2 is recorded.
</p>


## Get Started

1. Install Python 3.8. For convenience, execute the following command.

```bash
pip install -r requirements.txt
```

2. Prepare Data. You can obtain experimental datasets from the following links.


| Dataset       | Task                                    | Geometry        | Link                                                         |
| ------------- | --------------------------------------- | --------------- | ------------------------------------------------------------ |
| Elasticity    | Estimate material inner stress          | Point Cloud     | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Plasticity    | Estimate material deformation over time | Structured Mesh | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Navier-Stokes | Predict future fluid velocity           | Regular Grid    | [[Google Cloud]](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) |
| Darcy         | Estimate fluid pressure through medium  | Regular Grid    | [[Google Cloud]](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) |
| AirFoil       | Estimate airﬂow velocity around airfoil | Structured Mesh | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Pipe          | Estimate fluid velocity in a pipe       | Structured Mesh | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |

3. Train and evaluate model. We provide the experiment scripts of all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```bash
bash scripts/Transolver_Elas.sh # for Elasticity
bash scripts/Transolver_Plas.sh # for Plasticity
bash scripts/Transolver_NS.sh # for Navier-Stokes
bash scripts/Transolver_Darcy.sh # for Darcy
bash scripts/Transolver_Airfoil.sh # for Airfoil
bash scripts/Transolver_Pipe.sh # for Pipe
```

 Note: You need to change the argument `--data_path` to your dataset path.

4. Develop your own model. Here are the instructions:

   - Add the model file under folder `./models/`.
   - Add the model name into `./model_dict.py`.
   - Add a script file under folder `./scripts/` and change the argument `--model`.

## Visualization

Transolver can handle PDEs under various geometrics well, such as predicting the future fluid and estimating the [[shock wave]](https://en.wikipedia.org/wiki/Shock_wave) around airfoil. 

<p align="center">
<img src=".\fig\showcase.png" height = "400" alt="" align=center />
<br><br>
<b>Figure 1.</b> Case study of different models.
</p>

## PDE Solving at Scale

To align with previous model, we only experiment with 8-layer Transolver in the main text. Actually, you can easily obtain a better performance by **scaling up Transolver**. The relative L2 generally decreases when we adding more layers.

<p align="center">
<img src=".\fig\scalibility.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 2.</b> Scaling up Transolver: relative L2 curve w.r.t. model layers.
</p>

## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{wu2024Transolver,
  title={Transolver: A Fast Transformer Solver for PDEs on General Geometries},
  author={Haixu Wu and Huakun Luo and Haowen Wang and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

## Contact

If you have any questions or want to use the code, please contact [wuhx23@mails.tsinghua.edu.cn](mailto:wuhx23@mails.tsinghua.edu.cn).

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/neuraloperator/neuraloperator

https://github.com/neuraloperator/Geo-FNO

https://github.com/thuml/Latent-Spectral-Models
