# Predicting the Strength of Composites

## 工作简述

针对材料科学领域中材料结构强度预测这一问题，通过X射线CT图像预测聚合物-陶瓷复合材料的极限抗拉强度（UTS）。相较于传统材料强度预测方法对于数据和模型的需求严苛，且需要耗费较长的时间成本，本项目通过深度学习技术，在小样本数据集的条件下，实现了较高精度的UTS值预测，提供了更快速且准确的工具。帮助研究人员快速了解材料的特性，并优化材料设计

## 参考

Po-Hao Lai, et al. "Predicting the Strength of Composites with Computer Vision Using Small Experimental Datasets"
<https://doi.org/10.5281/zenodo.14803929>

## 目录结构

```
CNN_UTS/
│
├─ conf/  
│    └─ resnet.yaml
├─ data_utils.py  
├─ model_utils.py  
├─ main.py  
├─ requirements.txt  
├─ readme.md  
├─ resnet18-v5-finetune/  
├─ outputs/  
├─ Saved_Output/  
└─ Dataset/  
     ├─ Train_val/  
     └─ Test/  
```

## 环境依赖

见 requirements.txt

## 数据格式说明

数据集下载链接:<https://paddle-org.bj.bcebos.com/paddlescience/datasets/CNN_UTS/Dataset.zip>

- `Dataset/Train_val/` 和 `Dataset/Test/` 下为若干子文件夹，每个子文件夹代表一个样本组。
- 每个子文件夹内包含若干 `.jpg` 图像和一个 `.csv` 文件。
- `.csv` 文件示例（每行对应一张图片，需包含 `Image Name`、若干特征列、`UTS (MPahttps://paddle-org.bj.bcebos.com/paddlescience/datasets/CNN_UTS/Dataset.zip)` 等标签）：

| Image Name         | ...特征列... | UTS (MPa) | ... |
|--------------------|--------------|-----------|-----|
| IPP_10__40060.jpg  | ...          | 0.56      | ... |
| ...                | ...          | ...       | ... |

## 训练好的模型权重文件

| 预训练模型参数                        |
|-----------------------------------|
| [Saved_Output.zip](下载链接待填写) |
| [resnet18-v5-fold1](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) |
 [resnet18-v5-fold2](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) |
 [resnet18-v5-fold3](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) |
 [resnet18-v5-fold4](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) |
 [resnet18-v5-fold5](https://paddle-org.bj.bcebos.com/paddlescience/models/CNN_UTS/resnet18-v5-fold1.pdparams) ||

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置参数

编辑 `conf/resnet.yaml`，可自定义训练/评估参数：

```yaml
mode: "eval"
seed: 42
device: "cuda:0"
data:
  train_path: "./Dataset/Train_val"
  test_path: "./Dataset/Test"
  N: 1
train:
  epochs: 32
  n_splits: 5
  batch_size: 32
  lr: 0.0009761248347350309
output_dir: "./Saved_Output"
```

### 3. 训练模型

```bash
python main.py mode=train
```

### 4. 评估模型

```bash
python main.py mode=eval
```

### 5. 可视化与结果

- 训练和评估后，预测结果、统计指标、可视化图片会自动保存在 `Saved_Output/` 目录下。
。
## 完整代码
import os

import hydra
import numpy as np
import paddle
import tqdm
from data_utils import device2str
from data_utils import make_dataset
from model_utils import set_seed
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedGroupKFold

import ppsci


def train(cfg):
    # 设置随机种子
    set_seed(cfg.seed)
    device = device2str(cfg.device)
    num_epochs = cfg.train.epochs
    n_splits = cfg.train.n_splits
    Batch_size = cfg.train.batch_size
    lr = cfg.train.lr
    N_skip = cfg.data.N

    # 数据增强配置
    transforms_list = [paddle.vision.transforms.CenterCrop(size=224)]
    # 这里可根据cfg添加更多增强
    transforms_list.append(paddle.vision.transforms.RandomHorizontalFlip(prob=0.5))
    transforms_list.append(paddle.vision.transforms.RandomVerticalFlip(prob=0.5))
    transforms_list.append(
        paddle.vision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    )
    online_transforms = paddle.vision.transforms.Compose(transforms=transforms_list)
    val_xform_list = [paddle.vision.transforms.CenterCrop(size=224)]
    val_xform_list.append(
        paddle.vision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    )
    offline_transforms = paddle.vision.transforms.Compose(transforms=val_xform_list)

    # 数据集
    train_val_dataset = make_dataset(cfg.data.train_path, N=N_skip, device=device)
    test_dataset = make_dataset(cfg.data.test_path, N=N_skip, device=device)

    kf = StratifiedGroupKFold(n_splits=n_splits)
    uts_label = [it[1][0] for it in train_val_dataset]
    sample_id = [it[1][1] for it in train_val_dataset]

    val_loss_all_fold = []
    test_loss_all_fold = []
    min_epoch_all_fold = []
    test_preds_history = []

    for fold, (train_index, val_index) in enumerate(
        kf.split(train_val_dataset, uts_label, sample_id)
    ):
        print(f"\n===== Fold {fold+1}/{n_splits} 开始 =====")
        set_seed(cfg.seed)
        train_dataset = paddle.io.Subset(dataset=train_val_dataset, indices=train_index)
        val_dataset = paddle.io.Subset(dataset=train_val_dataset, indices=val_index)
        train_loader = paddle.io.DataLoader(
            dataset=train_dataset,
            batch_size=Batch_size,
            shuffle=True,
        )
        val_loader = paddle.io.DataLoader(
            dataset=val_dataset,
            batch_size=128,
            shuffle=False,
        )
        test_loader = paddle.io.DataLoader(
            dataset=test_dataset,
            batch_size=128,
            shuffle=False,
        )
        # 定义模型
        # model = paddle.vision.models.resnet18(pretrained=True)
        model = ppsci.arch.ResNet(
            num_blocks=(2, 2, 2, 2),  # ResNet18结构
            num_classes=1,  # 回归任务
            in_channels=3,  # 彩色图像
        )
        model.to(device)
        criterion = paddle.nn.MSELoss()
        optimizer = paddle.optimizer.Adam(
            parameters=model.parameters(), learning_rate=lr, weight_decay=0.0
        )
        val_loss_history = []
        test_loss_history = []
        test_preds_best = None
        pbar = tqdm.tqdm(range(num_epochs))
        for epoch in pbar:
            print(f"\n--- Fold {fold+1} Epoch {epoch+1}/{num_epochs} ---")
            model.train()
            set_seed(cfg.seed)
            batch_losses = []
            for i, (images, group, labels) in enumerate(train_loader):
                images = online_transforms(images)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                optimizer.clear_gradients(set_to_zero=False)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
                print(
                    f"[Fold {fold+1}][Epoch {epoch+1}][Batch {i+1}/{len(train_loader)}] Loss: {loss.item():.6f}"
                )
            avg_train_loss = (
                sum(batch_losses) / len(batch_losses) if batch_losses else 0
            )
            print(f"[Fold {fold+1}][Epoch {epoch+1}] 训练集平均Loss: {avg_train_loss:.6f}")
            model.eval()
            with paddle.no_grad():
                val_loss = 0
                k = 0
                preds_val = []
                true_labels_val = []
                for images, group, labels in val_loader:
                    images = offline_transforms(images)
                    outputs = model(images)
                    val_loss += criterion(outputs.squeeze(), labels).item() * len(
                        images
                    )
                    k += len(images)
                    preds_val.append(outputs.squeeze())
                    true_labels_val.append(labels)
                val_loss /= k
                preds_val = paddle.concat(x=preds_val, axis=0).detach().cpu().numpy()
                true_labels_val = (
                    paddle.concat(x=true_labels_val, axis=0).detach().cpu().numpy()
                )
                print(f"[Fold {fold+1}][Epoch {epoch+1}] 验证集Loss: {val_loss:.6f}")
                test_loss = 0
                k = 0
                preds_test = []
                true_labels_test = []
                for images, group, labels in test_loader:
                    images = offline_transforms(images)
                    outputs = model(images)
                    test_loss += criterion(outputs.squeeze(), labels).item() * len(
                        images
                    )
                    k += len(images)
                    preds_test.append(outputs.squeeze())
                    true_labels_test.append(labels)
                test_loss /= k
                preds_test = paddle.concat(x=preds_test, axis=0).detach().cpu().numpy()
                true_labels_test = (
                    paddle.concat(x=true_labels_test, axis=0).detach().cpu().numpy()
                )
                print(f"[Fold {fold+1}][Epoch {epoch+1}] 测试集Loss: {test_loss:.6f}")
            val_loss_history.append(val_loss)
            test_loss_history.append(test_loss)
            if val_loss == np.min(val_loss_history):
                test_preds_best = preds_test.copy()
                paddle.save(
                    obj=model.state_dict(),
                    path=f"./resnet18-v5-finetune/resnet18-v5-fold{fold + 1}.pdparams",
                )
                print(f"[Fold {fold+1}][Epoch {epoch+1}] 验证集Loss新低，已保存模型参数！")
            pbar.set_postfix_str(
                f"Train {avg_train_loss:.3e}, Val {val_loss:.3e}, Test {test_loss:.3e}"
            )
        min_epoch = np.argmin(val_loss_history)
        val_loss_all_fold.append(val_loss_history[min_epoch])
        min_epoch_all_fold.append(min_epoch + 1)
        test_loss_all_fold.append(test_loss_history[min_epoch])
        test_preds_history.append(test_preds_best)
        print(
            f"===== Fold {fold+1} 完成，最佳验证集Loss: {val_loss_all_fold[-1]:.6f}，最佳epoch: {min_epoch+1} =====\n"
        )
    # 结果统计与输出
    print(f"Validation for five folds: {val_loss_all_fold}")
    print(
        f"Lowest validation loss for each fold occurred at epochs: {min_epoch_all_fold}"
    )
    print(
        f"Mean validation loss: {np.mean(val_loss_all_fold):.4f} ± {np.std(val_loss_all_fold):.4f}"
    )
    print(
        f"Mean test loss: {np.mean(test_loss_all_fold):.4f} ± {np.std(test_loss_all_fold):.4f}"
    )


def evaluate(cfg):
    """
    完整的评估流程，包括模型加载、推理、指标计算、可视化和集成预测
    """
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedGroupKFold

    set_seed(cfg.seed)
    device = device2str(cfg.device)
    n_splits = cfg.train.n_splits
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    train_val_dataset = make_dataset(cfg.data.train_path, N=cfg.data.N, device=device)
    test_dataset = make_dataset(cfg.data.test_path, N=cfg.data.N, device=device)

    # 定义离线变换（用于推理）
    offline_transforms = paddle.vision.transforms.Compose(
        [
            paddle.vision.transforms.CenterCrop(size=224),
            paddle.vision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    kf = StratifiedGroupKFold(n_splits=n_splits)
    uts_label = [it[1][0] for it in train_val_dataset]
    sample_id = [it[1][1] for it in train_val_dataset]

    # 存储所有fold的结果
    train_mse_all_fold = []
    val_mse_all_fold = []
    test_mse_all_fold = []
    train_rsquared_all_fold = []
    val_rsquared_all_fold = []
    test_rsquared_all_fold = []
    test_preds_history = []

    for fold, (train_index, val_index) in enumerate(
        kf.split(train_val_dataset, y=uts_label, groups=sample_id)
    ):
        print(f"Fold {fold + 1}/{n_splits}")
        set_seed(cfg.seed)

        # 准备保存预测结果的文件名
        output_train_file = os.path.join(output_dir, f"preds_train_fold{fold + 1}.npy")
        output_val_file = os.path.join(output_dir, f"preds_val_fold{fold + 1}.npy")
        output_test_file = os.path.join(output_dir, f"preds_test_fold{fold + 1}.npy")
        output_unique_groups_file = os.path.join(
            output_dir, f"unique_val_groups_fold_{fold + 1}.npy"
        )
        output_sample_id_file = os.path.join(
            output_dir, f"unique_sample_id_fold_{fold + 1}.npy"
        )

        # 检查是否已有保存的结果
        if (
            os.path.exists(output_train_file)
            and os.path.exists(output_val_file)
            and os.path.exists(output_test_file)
        ):
            print("Loading saved outputs for this fold.")
            preds_train = np.load(output_train_file)
            true_labels_train = np.load(
                output_train_file.replace("preds", "true_labels")
            )
            preds_val = np.load(output_val_file)
            true_labels_val = np.load(output_val_file.replace("preds", "true_labels"))
            preds_test = np.load(output_test_file)
            true_labels_test = np.load(output_test_file.replace("preds", "true_labels"))

            test_preds_history.append(preds_test)

            # 加载分组信息
            unique_val_groups = np.load(output_unique_groups_file, allow_pickle=True)
            unique_sample_id = np.load(output_sample_id_file, allow_pickle=True)
            print(f"Loaded unique validation groups: {unique_val_groups}")
            print(f"Loaded unique sample IDs: {unique_sample_id}")

            true_labels_val_flat = true_labels_val
            unique_true_labels_val = sorted(set(true_labels_val_flat))
            print(
                f"Validation groups for fold {fold+1} contains unique UTS: {unique_true_labels_val}"
            )

            # 加载样本ID
            train_samples_id = np.load(output_train_file.replace("preds", "sample_ids"))
            val_samples_id = np.load(output_val_file.replace("preds", "sample_ids"))
            test_samples_id = np.load(output_test_file.replace("preds", "sample_ids"))

        else:
            print("Computing new predictions for this fold.")
            # 创建训练和验证数据集
            train_dataset = paddle.io.Subset(train_val_dataset, train_index)
            val_dataset = paddle.io.Subset(train_val_dataset, val_index)

            # 创建数据加载器
            train_loader = paddle.io.DataLoader(
                train_dataset, batch_size=32, shuffle=False, num_workers=0
            )
            val_loader = paddle.io.DataLoader(
                val_dataset, batch_size=128, shuffle=False, num_workers=0
            )
            test_loader = paddle.io.DataLoader(
                test_dataset, batch_size=128, shuffle=False, num_workers=0
            )

            # 加载模型
            model = paddle.load(
                f"./resnet18-v5-finetune/resnet18-v5-fold{fold + 1}.pdparams"
            )
            model.eval()
            model.to(device)

            true_labels_train, preds_train = [], []
            true_labels_val, preds_val = [], []
            true_labels_test, preds_test = [], []

            # 存储分组信息
            train_groups_fold, train_samples_id = [], []
            val_groups_fold, val_samples_id = [], []
            test_groups_fold, test_samples_id = [], []

            with paddle.no_grad():
                # 训练集推理
                for images, groups, labels, features in train_loader:
                    images = offline_transforms(images)
                    outputs = model(images)
                    true_labels_train.append(labels.numpy())
                    preds_train.append(outputs.numpy())
                    train_groups_fold.extend(groups[0].numpy())
                    train_samples_id.extend(groups[1].numpy())

                # 验证集推理
                for images, groups, labels, features in val_loader:
                    images = offline_transforms(images)
                    outputs = model(images)
                    true_labels_val.append(labels.numpy())
                    preds_val.append(outputs.numpy())
                    val_groups_fold.extend(groups[0].numpy())
                    val_samples_id.extend(groups[1].numpy())

                # 测试集推理
                for images, groups, labels, features in test_loader:
                    images = offline_transforms(images)
                    outputs = model(images)
                    true_labels_test.append(labels.numpy())
                    preds_test.append(outputs.numpy())
                    test_groups_fold.extend(groups[0].numpy())
                    test_samples_id.extend(groups[1].numpy())

            # 展平结果
            true_labels_train = np.concatenate(true_labels_train)
            preds_train = np.concatenate(preds_train)
            true_labels_val = np.concatenate(true_labels_val)
            preds_val = np.concatenate(preds_val)
            true_labels_test = np.concatenate(true_labels_test)
            preds_test = np.concatenate(preds_test)

            test_preds_history.append(preds_test)

            unique_val_groups = sorted(set(val_groups_fold))
            print(
                f"Validation groups for fold {fold+1} contains UTS groups: {unique_val_groups}"
            )

            unique_sample_id = sorted(set(val_samples_id))
            print(
                f"Validation groups for fold {fold+1} contains sample ID: {unique_sample_id}"
            )

            true_labels_val_flat = true_labels_val
            unique_true_labels_val = sorted(set(true_labels_val_flat))
            print(
                f"Validation groups for fold {fold+1} contains unique UTS: {unique_true_labels_val}"
            )

            # 保存预测结果
            np.save(output_train_file, preds_train)
            np.save(output_val_file, preds_val)
            np.save(output_test_file, preds_test)
            np.save(
                output_train_file.replace("preds", "true_labels"), true_labels_train
            )
            np.save(output_val_file.replace("preds", "true_labels"), true_labels_val)
            np.save(output_test_file.replace("preds", "true_labels"), true_labels_test)
            np.save(output_unique_groups_file, unique_val_groups)
            np.save(output_sample_id_file, unique_sample_id)
            np.save(output_train_file.replace("preds", "sample_ids"), train_samples_id)
            np.save(output_val_file.replace("preds", "sample_ids"), val_samples_id)
            np.save(output_test_file.replace("preds", "sample_ids"), test_samples_id)

            print(f"Saved predictions for fold {fold + 1}.")

        # 计算指标
        r_squared_train = r2_score(true_labels_train, preds_train)
        mse_train = mean_squared_error(true_labels_train, preds_train)
        r_squared_val = r2_score(true_labels_val, preds_val)
        mse_val = mean_squared_error(true_labels_val, preds_val)
        r_squared_test = r2_score(true_labels_test, preds_test)
        mse_test = mean_squared_error(true_labels_test, preds_test)
        print(f"MSE: Train: {mse_train}, Validation: {mse_val}, Test: {mse_test}")

        # 存储指标
        train_rsquared_all_fold.append(r_squared_train)
        val_rsquared_all_fold.append(r_squared_val)
        test_rsquared_all_fold.append(r_squared_test)
        train_mse_all_fold.append(mse_train)
        val_mse_all_fold.append(mse_val)
        test_mse_all_fold.append(mse_test)

        # 绘制parity plot
        fig, ax = plt.subplots()
        ax.scatter(
            true_labels_train,
            preds_train,
            s=30,
            marker=".",
            label=f"Train R-squared: {r_squared_train:.4f}",
        )
        ax.scatter(
            true_labels_val,
            preds_val,
            s=10,
            marker="*",
            label=f"Validation R-squared: {r_squared_val:.4f}",
        )
        ax.scatter(
            true_labels_test,
            preds_test,
            s=10,
            marker="o",
            color="red",
            label=f"Test R-squared: {r_squared_test:.4f}",
        )
        ax.plot(
            [true_labels_train.min(), true_labels_train.max()],
            [true_labels_train.min(), true_labels_train.max()],
            color="black",
            linestyle="--",
            label="Ideal fit",
        )
        ax.set_xlabel("True UTS (MPa)")
        ax.set_ylabel("Predicted UTS (MPa)")
        ax.set_aspect("equal")
        ax.set_title(f"Fold {fold + 1} Parity Plot")
        ax.legend()
        plt.savefig(
            os.path.join(output_dir, f"parity_plot_fold{fold+1}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # 绘制violin plot
        fig, ax = plt.subplots(figsize=(6, 6))
        train_label_added = False
        val_label_added = False
        test_label_added = False

        # 确保数组长度一致
        min_train_length = min(
            len(preds_train), len(train_samples_id), len(true_labels_train)
        )
        preds_train_aligned = preds_train[:min_train_length]
        train_samples_id_aligned = train_samples_id[:min_train_length]
        true_labels_train_aligned = true_labels_train[:min_train_length]

        # 训练集violin plot
        for i, label in enumerate(np.unique(train_samples_id_aligned)):
            mask = train_samples_id_aligned == label
            preds_for_label = preds_train_aligned[mask]
            true_for_label = true_labels_train_aligned[mask]
            parts = ax.violinplot(
                preds_for_label,
                positions=[np.mean(true_for_label)],
                showmeans=False,
                showmedians=True,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor("tab:blue")
                pc.set_edgecolor("black")
                pc.set_alpha(0.5)
            parts["cmedians"].set_color("tab:blue")
            parts["cmins"].set_color("tab:blue")
            parts["cmaxes"].set_color("tab:blue")
            if not train_label_added:
                ax.plot(
                    true_for_label,
                    preds_for_label,
                    "o",
                    color="tab:blue",
                    markersize=4,
                    label=f"Train $R^2$: {r_squared_train:.4f}",
                    alpha=0.6,
                )
                train_label_added = True
            else:
                ax.plot(
                    true_for_label,
                    preds_for_label,
                    "o",
                    color="tab:blue",
                    markersize=4,
                    alpha=0.6,
                )

        # 验证集violin plot
        for i, label in enumerate(np.unique(val_samples_id)):
            mask = val_samples_id == label
            preds_for_label = preds_val[mask]
            true_for_label = true_labels_val[mask]
            parts = ax.violinplot(
                preds_for_label,
                positions=[np.mean(true_for_label)],
                showmeans=False,
                showmedians=True,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor("tab:orange")
                pc.set_edgecolor("black")
                pc.set_alpha(0.5)
            parts["cmedians"].set_color("tab:orange")
            parts["cmins"].set_color("tab:orange")
            parts["cmaxes"].set_color("tab:orange")
            if not val_label_added:
                ax.plot(
                    true_for_label,
                    preds_for_label,
                    "d",
                    color="tab:orange",
                    markersize=4,
                    label=f"Val $R^2$: {r_squared_val:.4f}",
                    alpha=0.6,
                )
                val_label_added = True
            else:
                ax.plot(
                    true_for_label,
                    preds_for_label,
                    "d",
                    color="tab:orange",
                    markersize=4,
                    alpha=0.6,
                )

        # 测试集violin plot
        for i, label in enumerate(np.unique(test_samples_id)):
            mask = test_samples_id == label
            preds_for_label = preds_test[mask]
            true_for_label = true_labels_test[mask]
            parts = ax.violinplot(
                preds_for_label,
                positions=[np.mean(true_for_label)],
                showmeans=False,
                showmedians=True,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor("tab:red")
                pc.set_edgecolor("black")
                pc.set_alpha(0.5)
            parts["cmedians"].set_color("tab:red")
            parts["cmins"].set_color("tab:red")
            parts["cmaxes"].set_color("tab:red")
            if not test_label_added:
                ax.plot(
                    true_for_label,
                    preds_for_label,
                    "x",
                    color="tab:red",
                    markersize=4,
                    label=f"Test $R^2$: {r_squared_test:.4f}",
                    alpha=0.6,
                )
                test_label_added = True
            else:
                ax.plot(
                    true_for_label,
                    preds_for_label,
                    "x",
                    color="tab:red",
                    markersize=4,
                    alpha=0.6,
                )

        ax.plot(
            [true_labels_train_aligned.min(), true_labels_train_aligned.max()],
            [true_labels_train_aligned.min(), true_labels_train_aligned.max()],
            label="Ideal fit",
            color="black",
            linestyle="--",
        )
        ax.legend(prop={"size": 11})
        ax.set_xlabel("True UTS (MPa)", fontsize=18)
        ax.set_ylabel("Predicted UTS (MPa)", fontsize=18)
        ax.tick_params(axis="x", direction="in", top=True, length=3, width=1)
        ax.tick_params(axis="y", direction="in", right=True, length=3, width=1)
        ax.set_title(f"Fold {fold+1} Parity Violin Plot")
        plt.xticks(np.arange(0, 5.2, 1), fontsize=16)
        plt.yticks(np.arange(0, 5.2, 1), fontsize=16)
        plt.savefig(
            os.path.join(output_dir, f"violin_plot_fold{fold+1}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    # 最终统计结果
    print("\nFinal Statistics Across All Folds:")
    print(
        f"Train MSE: {np.mean(train_mse_all_fold):.4f} ± {np.std(train_mse_all_fold):.4f}"
    )
    print(
        f"Train R-squared: {np.mean(train_rsquared_all_fold):.4f} ± {np.std(train_rsquared_all_fold):.4f}"
    )
    print(
        f"Validation MSE: {np.mean(val_mse_all_fold):.4f} ± {np.std(val_mse_all_fold):.4f}"
    )
    print(
        f"Validation R-squared: {np.mean(val_rsquared_all_fold):.4f} ± {np.std(val_rsquared_all_fold):.4f}"
    )
    print(
        f"Test MSE: {np.mean(test_mse_all_fold):.4f} ± {np.std(test_mse_all_fold):.4f}"
    )
    print(
        f"Test R-squared: {np.mean(test_rsquared_all_fold):.4f} ± {np.std(test_rsquared_all_fold):.4f}"
    )

    # 集成学习（Ensemble）
    print("\nEnsemble Learning for Test Data:")
    test_preds_history = np.array(test_preds_history)

    # 计算中位数和均值预测
    median_preds_test = np.median(np.vstack(test_preds_history), axis=0)
    mean_preds_test = np.mean(np.vstack(test_preds_history), axis=0)

    # 加载测试集真实标签（使用最后一个fold的结果）
    true_labels_test = np.load(
        os.path.join(output_dir, f"true_labels_test_fold{n_splits}.npy")
    )

    # 计算集成指标
    median_test_mse = np.mean((median_preds_test - true_labels_test) ** 2)
    median_test_r2 = r2_score(true_labels_test, median_preds_test)
    mean_test_mse = np.mean((mean_preds_test - true_labels_test) ** 2)
    mean_test_r2 = r2_score(true_labels_test, mean_preds_test)

    print(
        f"Median Test MSE: {median_test_mse:.4f}, Median Test R-squared: {median_test_r2:.4f}"
    )
    print(
        f"Mean Test MSE: {mean_test_mse:.4f}, Mean Test R-squared: {mean_test_r2:.4f}"
    )

    # 绘制集成预测的parity plot
    fig, ax = plt.subplots()
    ax.scatter(
        true_labels_test,
        median_preds_test,
        s=10,
        marker="o",
        label=f"Median R-squared: {median_test_r2:.4f}",
    )
    ax.scatter(
        true_labels_test,
        mean_preds_test,
        s=10,
        marker="x",
        label=f"Mean R-squared: {mean_test_r2:.4f}",
    )
    ax.plot(
        [true_labels_test.min(), true_labels_test.max()],
        [true_labels_test.min(), true_labels_test.max()],
        color="black",
        linestyle="--",
        label="Ideal fit",
    )
    ax.set_xlabel("True UTS (MPa)")
    ax.set_ylabel("Predicted UTS (MPa)")
    ax.set_aspect("equal")
    ax.set_title("Test Data Parity Plot (Ensemble Predictions)")
    ax.legend()
    plt.savefig(
        os.path.join(output_dir, "ensemble_parity_plot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # 绘制集成预测的violin plot
    fig, ax = plt.subplots(figsize=(6, 6))
    test_label_added = False
    for i, label in enumerate(np.unique(true_labels_test)):
        mask = true_labels_test == label
        preds_for_label = mean_preds_test[mask]
        parts = ax.violinplot(
            preds_for_label, positions=[label], showmeans=False, showmedians=True
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("tab:red")
            pc.set_edgecolor("black")
            pc.set_alpha(0.5)
        parts["cmedians"].set_color("tab:red")
        parts["cmins"].set_color("tab:red")
        parts["cmaxes"].set_color("tab:red")
        if not test_label_added:
            ax.plot(
                [label] * len(preds_for_label),
                preds_for_label,
                "rx",
                markersize=4,
                label=f"Test $R^2$: {mean_test_r2:.4f}",
                alpha=0.6,
            )
            test_label_added = True
        else:
            ax.plot(
                [label] * len(preds_for_label),
                preds_for_label,
                "rx",
                markersize=4,
                alpha=0.6,
            )

    ax.plot(
        [true_labels_test.min(), true_labels_test.max() + 0.4],
        [true_labels_test.min(), true_labels_test.max() + 0.4],
        label="Ideal fit",
        color="black",
        linestyle="--",
    )
    ax.set_xlabel("True UTS (MPa)", fontsize=18)
    ax.set_ylabel("Predicted UTS (MPa)", fontsize=18)
    ax.tick_params(axis="x", direction="in", top=True, length=3, width=1)
    ax.tick_params(axis="y", direction="in", right=True, length=3, width=1)
    ax.legend(loc="upper left", prop={"size": 12})
    plt.xticks(np.arange(0, 4.2, 1), fontsize=16)
    plt.yticks(np.arange(0, 4.2, 1), fontsize=16)
    plt.savefig(
        os.path.join(output_dir, "ensemble_violin_plot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # 保存集成预测结果
    np.save(
        os.path.join(output_dir, "ensemble_median_preds_test.npy"), median_preds_test
    )
    np.save(os.path.join(output_dir, "ensemble_mean_preds_test.npy"), mean_preds_test)
    np.save(os.path.join(output_dir, "ensemble_true_labels_test.npy"), true_labels_test)

    print(f"\nAll results saved to {output_dir}")


@hydra.main(version_base=None, config_path="./conf", config_name="resnet.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
