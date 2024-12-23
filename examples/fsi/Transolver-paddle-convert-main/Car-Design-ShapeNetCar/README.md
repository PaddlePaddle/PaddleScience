# Transolver for Car Design

We test [Transolver](https://arxiv.org/abs/2402.02366) on practical design tasks. The car design task requires the model to estimate the surrounding wind speed and surface pressure for a driving car.

<p align="center">
<img src=".\fig\task.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 1.</b> Car design task. 
</p>

Relative error of surrounding wind, surface pressure and [drag coefficient](https://en.wikipedia.org/wiki/Drag_coefficient) are recorded, as well as [Spearman's rank correlations](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient), which can be used to quantify the model's capability in ranking different designs.

<p align="center">
<img src=".\fig\results.png" height = "300" alt="" align=center />
<br><br>
<b>Table 1.</b> Model comparisons of the car design task. 
</p>


## Get Started

1. Install Python 3.8. For convenience, execute the following command.

```bash
pip install -r requirements.txt
```

Note: You need to install [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric).

2. Prepare Data.

The raw data can be found [[here]](http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip), which is provided by [Nobuyuki Umetani](https://dl.acm.org/doi/abs/10.1145/3197517.3201325).

3. Train and evaluate model. We provide the experiment scripts under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```bash
bash scripts/Transolver.sh # for Training (will take 8-10 hours on one single A100)
bash scripts/Evaluation.sh # for Evaluation
```

Note: You need to change the argument `--data_dir` and `--save_dir` to your dataset path. Here `data_dir` is for the raw data and `save_dir` is to save the preprocessed data.

If you have already downloaded or generated the preprocecessed data, you can change `--preprocessed` as True for speed up.

4. Develop your own model. Here are the instructions:

   - Add the model file under folder `./models/`.
   - Add the model configuration into `./main.py`.
   - Add a script file under folder `./scripts/` and change the argument `--model`.

## Slice Visualization

Transolver proposes to **learn physical states** hidden under the unwieldy meshes. 

The following visualization demonstrates that Transolver can successfully learn to ascribe the points under similar physical state to the same slice, such as windshield, license plate and headlight.

<p align="center">
<img src=".\fig\car_slice_surf.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 2.</b> Visualization for Transolver learned physical states. 
</p>


## Showcases

Transolver achieves the best performance in complex geometries and hybrid physics.

<p align="center">
<img src=".\fig\case_study.png" height = "150" alt="" align=center />
<br><br>
<b>Figure 3.</b> Case study of Transolver and other models. 
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

We appreciate the following papers a lot for their valuable code base or datasets:

https://dl.acm.org/doi/abs/10.1145/3197517.3201325

https://openreview.net/forum?id=EyQO9RPhwN
