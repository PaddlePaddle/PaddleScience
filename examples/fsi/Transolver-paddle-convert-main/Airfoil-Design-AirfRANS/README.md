# Transolver for Airfoil Design

We test [Transolver](https://arxiv.org/abs/2402.02366) on practical design tasks. The airfoil design task requires the model to estimate the surrounding and surface physical quantities of a 2D airfoil under different Reynolds and angles of attacks.

<p align="center">
<img src=".\fig\task.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 1.</b> Airfoil design task. Left: surrounding pressure; Right: x-direction wind speed.
</p>

## Get Started

This part of code is developed based on the [[AirfRANS]](https://github.com/Extrality/AirfRANS).

1. Install Python 3.8. For convenience, execute the following command.

```bash
pip install -r requirements.txt
```

Note: You need to install [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric).

2. Prepare Data.

The experiment data is provided by [[AirfRANS]](https://github.com/Extrality/AirfRANS). You can directly download it with this [link](https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip) (9.3GB).

3. Train and evaluate model. We provide the experiment scripts under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```bash
bash scripts/Transolver.sh # for Training Transolver (will take 20-24 hours on one single A100)
bash scripts/Evaluation.sh # for Evaluation
bash scripts/GraphSAGE.sh # for Training GraphSAGE (will take 30-36 hours on one single A100)
```

Note: You need to change the argument `--my_path` to your dataset path.

4. Test model with different settings. This benchmark supports four types of settings.

| Settings                                     | Argument      |
| -------------------------------------------- | ------------- |
| Use full data                                | `-t full`     |
| Use scarce data                              | `-t scarce`   |
| Test on out-of-distribution Reynolds         | `-t reynolds` |
| Test on out-of-distribution Angle of Attacks | `-t aoa`      |

5. Develop your own model. Here are the instructions:

   - Add the model file under folder `./models/`.

   - Add the training details in `./params.yaml`. If you donot want to change setting, just copy other models' configuration.

   - Add the model configuration into `./main.py`.

   - Add a script file under folder `./scripts/` and change the argument `--model`.

## Main Results

Transolver achieves the consistent best performance in practical design tasks.

<p align="center">
<img src=".\fig\results.png" height = "300" alt="" align=center />
<br><br>
<b>Table 1.</b> Model comparisons on the practical design tasks.
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

https://github.com/Extrality/AirfRANS
