# VelocityGAN

| 预训练模型 | 指标                                        |
| :--------- | :------------------------------------------ |
|            | MAE: 0.0669<br>PSNR: 0.0947<br>SSIM: 0.8511 |

## 1. 模型简介

地震波形反演方法被广泛应用于重构地下速度成像。波形反演问题是一种典型的非线性且病态的逆问题。现有的物理驱动的计算方法会遇到周期跳跃和局部最小值的问题，并且伴随着很大的计算开销。相比于物理驱动的方法，数据驱动的深度学习方法能在更短的时间生成更精确的地下速度成像。

VelocityGAN是一种数据驱动的深度学习网络，包含两个模型：生成器（Generator）和判别器（Discriminator）。生成器输入地震波形数据，生成地下速度图像；判别器区分真实图像和生成图像，二者对抗训练，最终使得生成器的输出接近真实值。





## 2. 问题定义

本问题使用VelocityGAN，通过输入地震波形数据，输出对应的地下速度成像。根据GAN网络的训练流程，判别器一直训练，生成器每隔一定批次训练一次。





## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。为了快速理解 PaddleScience，接下来仅对模型构建、约束构建等关键步骤进行阐述，而其余细节请参考 [API文档](https://paddlescience-docs.readthedocs.io/zh-cn/latest/zh/api/arch/)。



### 3.1 数据集介绍

数据集使用[OpenFWI](https://openfwi-lanl.github.io)开源的[全波形反演数据集](https://openfwi-lanl.github.io/docs/data.html#vel)，包括地震波形数据和速度成像数据，以.npy文件保存。

运行本问题代码前请下载数据集，如[FlatVel-A](https://drive.google.com/drive/folders/1NIdjiYhjWSV9NHn7ZEFYTpJxzvzxqYRb)。

注意：一份数据集包含多个.npy文件，所以，数据读取的方式是通过构建一个数据集路径的txt文件，通过解析文件中每一条路径，从而依次读取数据信息的。正因为这种读取方式，我们需要自定义dataset 类。详细代码，可以查看ppsci.data.dataset.fwidataset。

```
# 包含数据集路径的txt文件示例
Dataset_directory/data1.npy		Dataset_directory/model1.npy
Dataset_directory/data2.npy		Dataset_directory/model2.npy
...
Dataset_directory/data48.npy	Dataset_directory/model3.npy
```



### 3.2 模型求解

![image-20240825160034203](VelocityGAN/image-20240825160034203.png)

上图为VelocityGAN完整的模型结构图。

生成器由encoder和decoder构成，输入地震波形数据，输出生成图像；判别器是全卷积的结构，输入速度图像，输出真实性分数。生成器的encoder包含了14个卷积块，decoder由5个转置卷积块和6个卷积块交替组合。判别器由9个卷积块构成。

VelocityGAN没有内置在PaddleScience中，需要额外实现。

具体代码请参考[完整代码](#4)中 velocitygan.py文件。



### 3.3 输入数据的transform构建

地震波形数据在输入生成器之前需要做一些转换和归一化操作。

```python
def create_transform(ctx, k):
    log_data_min = log_transform(ctx['data_min'], k)
    log_data_max = log_transform(ctx['data_max'], k)
    transform_data = Compose([
        LogTransform(k),
        MinMaxNormalize(log_data_min, log_data_max)
    ])
    transform_label = Compose([
        MinMaxNormalize(ctx['label_min'], ctx['label_max'])
    ])

    return transform_data, transform_label
```



### 3.4 超参数设置

我们需要指定问题的相关参数，如数据集路径、各项loss的权重。

```python
# general settings
mode: train # running mode: train/eval
seed: 42
output_dir: ${hydra:run.dir}
log_freq: 20
DATASET: "flatvel-a"
DATASET_CONFIG: './dataset_config.json'
k: 1
ANNO_PATH_TRAIN: "./flatvel_a_train_single.txt"
ANNO_PATH_VAL: "./flatvel_a_val_small.txt"

WEIGHT_DICT:
  lambda_g1v: 100.0
  lambda_g2v: 0.0
  lambda_adv: 1.0
  lambda_gp: 10.0
```

注意：由于一份数据集划分了多个文件，为了方便处理，这里ANNO_PATH_TRAIN不是数据集的路径，而是包含了数据集路径的.txt文件。



### 3.5 优化器构建

训练使用AdamW优化器，学习率固定为0.0001。

```python
# set optimizer
optimizer = ppsci.optimizer.AdamW(learning_rate=cfg.TRAIN.learning_rate, 	weight_decay=cfg.TRAIN.weight_decay)
optimizer_g = optimizer(model_gen)
optimizer_d = optimizer(model_dis)
```

### 3.6 约束构建

本问题采用ppsci.constraint.SupervisedConstraint。首先定义好dataloader_cfg。

```python
# set dataloader config
dataloader_cfg = {
    "dataset": {
        "name": "FWIDataset",
        "input_keys": ("data", ),
        "label_keys": ("real_image", ),
        "weight_dict":cfg.WEIGHT_DICT,
        "anno": "flatvel_a_train.txt",
        "preload": True,
        "sample_ratio": 1,
        "file_size": ctx['file_size'],
        "transform_data": transform_data,
        "transform_label": transform_label,
    },
    "sampler": {
        "name": "BatchSampler",
        "shuffle": True,
        "drop_last": True,
    },
    "batch_size": cfg.TRAIN.batch_size,
    "use_shared_memory": True,
    "num_workers": 16,
}
```

然后，分别构建生成器和判别器的约束。

```python
constraient_gen = ppsci.constraint.SupervisedConstraint(
    dataloader_cfg=dataloader_cfg,
    loss=ppsci.loss.FunctionalLoss(gen_funcs.loss_func_gen),
    output_expr={"fake_image": lambda out: out["fake_image"]},
    name="cst_gen"
)
constraient_gen_dict = {constraient_gen.name: constraient_gen}

constraient_dis = ppsci.constraint.SupervisedConstraint(
    dataloader_cfg=dataloader_cfg,
    loss=ppsci.loss.FunctionalLoss(dis_funcs.loss_func_dis),
    output_expr={"fake_image": lambda out: out["fake_image"]},
    name="cst_dis"
)
constraient_dis_dict = {constraient_dis.name: constraient_dis}
```

`SupervisedConstraint` 的第一个参数是监督约束的读取配置，其中 `dataset` 字段表示使用的训练数据集信息，各个字段分别表示：

1. `name`： 数据集类型，此处 `FWIDataset`是我们自定义的数据集类；
2. `input_keys`： 输入数据的key；
3. `label_keys`： 标签数据的key；
4. `transforms`： 传入之前定义好的transfrom_data/label；

`batch_size` 字段表示 batch的大小；

`sampler` 字段表示采样方法，其中各个字段表示：

1. `name`： 采样器类型，此处 `BatchSampler` 表示批采样器；
2. `drop_last`： 是否需要丢弃最后无法凑整一个 mini-batch 的样本，默认值为 True；
3. `shuffle`： 是否需要在生成样本下标时打乱顺序，默认值为 True；

第二个参数是损失函数，此处的 `FunctionalLoss` 为 PaddleScience 预留的自定义 loss 函数类，该类支持编写代码时自定义 loss 的计算方法，而不是使用诸如 `MSE` 等现有方法。

第三个参数是约束条件的 `output_expr`，指定了如何构建output_dict。

第四个参数是约束条件的名字，我们需要给每一个约束条件命名，方便后续对其索引。



### 3.7 自定义loss

#### 3.7.1 Generator的loss

生成器的loss包含L1loss、L2loss、输出经过判别器判别的adv_loss。这些 loss 是否存在根据权重参数控制，若某一项 loss 的权重参数为 0，则表示训练中不添加该 loss 项。

```python
def loss_func_gen(self, output_dict, label_dict, weight_dict):
    l1loss = paddle.nn.L1Loss()
    l2loss = paddle.nn.MSELoss()

    pred = output_dict["fake_image"]
    label = label_dict["real_image"]

    loss_g1v = l1loss(pred, label)
    loss_g2v = l2loss(pred, label)

    loss = weight_dict["lambda_g1v"][0] * loss_g1v + weight_dict["lambda_g2v"][0] * loss_g2v
    if self.model_dis is not None:
        loss_adv = -paddle.mean(self.model_dis({"image": pred})["score"])
        loss += weight_dict["lambda_adv"][0] * loss_adv

    return {"loss_g": loss}
```

#### 3.7.2 Discriminator的loss

判别器的loss包含真假图像的分数之差和梯度惩罚项。

```python
def compute_gradient_penalty(self, real_samples, fake_samples):
    alpha = paddle.rand([real_samples.shape[0], 1, 1, 1], dtype=real_samples.dtype)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.stop_gradient = False  # Allow gradients to be calculated
    d_interpolates = self.model_dis({"image":interpolates})["score"]

    gradients = paddle.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=paddle.ones([real_samples.shape[0], d_interpolates.shape[1]], dtype='float32'),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.reshape([gradients.shape[0], -1])
    gradient_penalty = paddle.mean((paddle.norm(gradients, p=2, axis=1) - 1) ** 2)
    return gradient_penalty

def loss_func_dis(self, output_dict, label_dict, weight_dict):

    pred = output_dict["fake_image"]
    label = label_dict["real_image"]

    gradient_penalty = self.compute_gradient_penalty(label, pred)
    loss_real = paddle.mean(self.model_dis({"image": label})["score"])
    loss_fake = paddle.mean(self.model_dis({"image": pred})["score"])
    loss = -loss_real + loss_fake + gradient_penalty * weight_dict["lambda_gp"][0]
    return {"loss_d": loss}
```

### 3.8 模型训练

完成上述设置之后，首先需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练。

```python
solver_gen = ppsci.solver.Solver(
    model=model_gen,
    output_dir=cfg.output_dir,
    constraint=constraient_gen_dict,
    optimizer=optimizer_g,
    epochs=cfg.TRAIN.epochs_gen,
    iters_per_epoch=cfg.TRAIN.iters_per_epoch_gen
)

solver_dis = ppsci.solver.Solver(
    model=model_gen,
    output_dir=cfg.output_dir,
    constraint=constraient_dis_dict,
    optimizer=optimizer_d,
    epochs=cfg.TRAIN.epochs_dis,
    iters_per_epoch=cfg.TRAIN.iters_per_epoch_dis
)
```

注意 GAN 类型的网络训练方法为两个模型交替训练，与单一模型或多模型分阶段训练不同，不能简单的使用 `solver.train` API，具体代码请参考 [完整代码](###4. 完整代码)中 VelocityGAN.py 文件。

### 3.9 模型评估

本问题的评估指标为：MAE(Mean Absolute Error), RMSE(Root Mean Squared Error)和SSIM(Structural SIMilarity)。

```python
def evaluate(cfg: DictConfig):
    with open('dataset_config.json') as f:
        try:
            ctx = json.load(f)[cfg.DATASET]
        except KeyError:
            print('Unsupported dataset.')
            sys.exit()

    if cfg.file_size is not None:
        ctx['file_size'] = cfg.file_size

    transform_data, transform_label = func_module.create_transform(ctx, cfg.k)

    # set model
    model_gen = ppsci.arch.VelocityGenerator(**cfg.MODEL.gen_net)

    # set valid_dataloader_cfg
    valid_dataloader_cfg = {
        "dataset": {
            "name": "FWIDataset",
            "input_keys": ("data", ),
            "label_keys": ("real_image", ),
            "weight_dict": {},
            "anno": cfg.ANNO_PATH_VAL,
            "preload": True,
            "sample_ratio": 1,
            "file_size": ctx['file_size'],
            "transform_data": transform_data,
            "transform_label": transform_label,
        },
        "sampler": {
            "name": "BatchSampler",
            "shuffle": True,
            "drop_last": False,
        },
        "batch_size": cfg.EVAL.batch_size,
        "use_shared_memory": True,  # True的时候出现错误
        "num_workers": 16,
    }

    # set validator
    validator = ppsci.validate.SupervisedValidator(
        dataloader_cfg=valid_dataloader_cfg,
        loss=ppsci.loss.MAELoss("mean"),
        output_expr={"real_image": lambda out: out["fake_image"]},
        metric={"MAE": ppsci.metric.MAE(),
                "RMSE": ppsci.metric.RMSE(),
                "SSIM": FunctionalMetric(func_module.ssim_metirc)},
        name="val"
    )

    validator_dict = {validator.name: validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model=model_gen,
        validator=validator_dict,
        pretrained_model_path="generator.pdparams",
    )

    solver.eval()
```

### 3.10 可视化

评估完成后，我们以图片的形式对结果进行可视化，如下所示。

```python
if cfg.VIS.vis:
    with solver.no_grad_context_manager(True):
        for batch_idx, (input_, label_, _) in enumerate(validator.data_loader):
            if batch_idx + 1 > cfg.VIS.vb:
                break
            fake_image = model_gen(input_)["fake_image"].numpy()
            real_image = label_["real_image"].numpy()
            for i in range(cfg.VIS.vsa):
                plot_velocity(fake_image[i, 0], real_image[i, 0], f'{cfg.output_dir}/V_{batch_idx}_{i}.png')
    print(f"The visualizations are saved to {cfg.output_dir}")
```



## 4. 完整代码

```python
import os

os.environ['FLAGS_embedding_deterministic'] = '1'
os.environ['FLAGS_cudnn_deterministic'] = '1'
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
os.environ['NCCL_ALGO'] = 'Tree'

import sys
import json
import hydra

import paddle
import ppsci
from ppsci.metric import FunctionalMetric
import functions as func_module
from omegaconf import DictConfig
from ppsci.utils import logger
from functions import plot_velocity



def evaluate(cfg: DictConfig):
    with open('dataset_config.json') as f:
        try:
            ctx = json.load(f)[cfg.DATASET]
        except KeyError:
            print('Unsupported dataset.')
            sys.exit()

    if cfg.file_size is not None:
        ctx['file_size'] = cfg.file_size

    transform_data, transform_label = func_module.create_transform(ctx, cfg.k)

    # set model
    model_gen = ppsci.arch.VelocityGenerator(**cfg.MODEL.gen_net)

    # set valid_dataloader_cfg
    valid_dataloader_cfg = {
        "dataset": {
            "name": "FWIDataset",
            "input_keys": ("data", ),
            "label_keys": ("real_image", ),
            "weight_dict": {},
            "anno": cfg.ANNO_PATH_VAL,
            "preload": True,
            "sample_ratio": 1,
            "file_size": ctx['file_size'],
            "transform_data": transform_data,
            "transform_label": transform_label,
        },
        "sampler": {
            "name": "BatchSampler",
            "shuffle": True,
            "drop_last": False,
        },
        "batch_size": cfg.EVAL.batch_size,
        "use_shared_memory": True,  # True的时候出现错误
        "num_workers": 16,
    }

    # set validator
    validator = ppsci.validate.SupervisedValidator(
        dataloader_cfg=valid_dataloader_cfg,
        loss=ppsci.loss.MAELoss("mean"),
        output_expr={"real_image": lambda out: out["fake_image"]},
        metric={"MAE": ppsci.metric.MAE(),
                "RMSE": ppsci.metric.RMSE(),
                "SSIM": FunctionalMetric(func_module.ssim_metirc)},
        name="val"
    )

    validator_dict = {validator.name: validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model=model_gen,
        validator=validator_dict,
        pretrained_model_path="generator.pdparams",
    )

    solver.eval()

    if cfg.VIS.vis:
        with solver.no_grad_context_manager(True):
            for batch_idx, (input_, label_, _) in enumerate(validator.data_loader):
                if batch_idx + 1 > cfg.VIS.vb:
                    break
                fake_image = model_gen(input_)["fake_image"].numpy()
                real_image = label_["real_image"].numpy()
                for i in range(cfg.VIS.vsa):
                    plot_velocity(fake_image[i, 0], real_image[i, 0], f'{cfg.output_dir}/V_{batch_idx}_{i}.png')
        print(f"The visualizations are saved to {cfg.output_dir}")




def train(cfg: DictConfig):
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", os.path.join(cfg.output_dir, "train.log"), "info")

    with open(cfg.DATASET_CONFIG) as f:
        try:
            ctx = json.load(f)[cfg.DATASET]
        except KeyError:
            print('Unsupported dataset.')
            sys.exit()

    if cfg.file_size is not None:
        ctx['file_size'] = cfg.file_size


    transform_data, transform_label = func_module.create_transform(ctx, cfg.k)

    # set model
    model_gen = ppsci.arch.VelocityGenerator(**cfg.MODEL.gen_net)

    model_dis = ppsci.arch.VelocityDiscriminator(**cfg.MODEL.dis_net)

    # set class for loss function
    gen_funcs = func_module.GenFuncs()
    gen_funcs.model_dis = model_dis
    dis_funcs = func_module.DisFuncs()
    dis_funcs.model_dis = model_dis

    # set dataloader config
    dataloader_cfg = {
        "dataset": {
            "name": "FWIDataset",
            "input_keys": ("data", ),
            "label_keys": ("real_image", ),
            "weight_dict":cfg.WEIGHT_DICT,
            "anno": "flatvel_a_train.txt",
            "preload": True,
            "sample_ratio": 1,
            "file_size": ctx['file_size'],
            "transform_data": transform_data,
            "transform_label": transform_label,
        },
        "sampler": {
            "name": "BatchSampler",
            "shuffle": True,
            "drop_last": True,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "use_shared_memory": True,
        "num_workers": 16,
    }

    constraient_gen = ppsci.constraint.SupervisedConstraint(
        dataloader_cfg=dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(gen_funcs.loss_func_gen),
        output_expr={"fake_image": lambda out: out["fake_image"]},
        name="cst_gen"
    )
    constraient_gen_dict = {constraient_gen.name: constraient_gen}

    constraient_dis = ppsci.constraint.SupervisedConstraint(
        dataloader_cfg=dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(dis_funcs.loss_func_dis),
        output_expr={"fake_image": lambda out: out["fake_image"]},
        name="cst_dis"
    )
    constraient_dis_dict = {constraient_dis.name: constraient_dis}

    # set optimizer
    optimizer = ppsci.optimizer.AdamW(learning_rate=cfg.TRAIN.learning_rate, weight_decay=cfg.TRAIN.weight_decay)
    optimizer_g = optimizer(model_gen)
    optimizer_d = optimizer(model_dis)

    solver_gen = ppsci.solver.Solver(
        model=model_gen,
        output_dir=cfg.output_dir,
        constraint=constraient_gen_dict,
        optimizer=optimizer_g,
        epochs=cfg.TRAIN.epochs_gen,
        iters_per_epoch=cfg.TRAIN.iters_per_epoch_gen
    )

    solver_dis = ppsci.solver.Solver(
        model=model_gen,
        output_dir=cfg.output_dir,
        constraint=constraient_dis_dict,
        optimizer=optimizer_d,
        epochs=cfg.TRAIN.epochs_dis,
        iters_per_epoch=cfg.TRAIN.iters_per_epoch_dis
    )


    for i in range(cfg.TRAIN.epochs):
        logger.message(f"\nEpoch: {i + 1}\n")
        solver_dis.train()
        solver_gen.train()

    paddle.save(model_gen.state_dict(), 'model_gen.pdparams')


@hydra.main(version_base=None, config_path="./conf", config_name="demo.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
```

## 5. 结果展示

使用[FlatVel-A](https://drive.google.com/drive/folders/1NIdjiYhjWSV9NHn7ZEFYTpJxzvzxqYRb)数据集的训练结果。

|  MAE   |  RMSE  |  SSIM  |
| :----: | :----: | :----: |
| 0.0669 | 0.0947 | 0.8511 |

![](VelocityGAN.assets/V_7_1.jpg)

![](VelocityGAN.assets/V_6_0.jpg)

## 6. 参考文献

- [VelocityGAN: Data-Driven Full-Waveform Inversion Using Conditional Adversarial Networks](https://arxiv.org/abs/1809.10262v6)

- [OpenFWI: Large-Scale Multi-Structural Benchmark Datasets for Seismic Full Waveform Inversion](https://arxiv.org/abs/2111.02926)

- [参考代码](https://github.com/lanl/OpenFWI?tab=readme-ov-file#ref2)

