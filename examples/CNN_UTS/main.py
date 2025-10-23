import os

import hydra
import matplotlib.pyplot as plt
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


def print_gpu_memory():
    """打印GPU内存使用情况"""
    if paddle.device.cuda.device_count() > 0:
        try:
            memory_allocated = paddle.device.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = paddle.device.cuda.memory_reserved() / 1024**3  # GB
            print(f"GPU内存使用: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
        except Exception:
            print("无法获取GPU内存信息")


def train(cfg):
    # 设置随机种子
    set_seed(cfg.seed)
    device = device2str(cfg.device)

    # 检查GPU可用性
    if "gpu" in device:
        if not paddle.device.cuda.device_count():
            print("警告：配置为GPU训练但未检测到GPU，将使用CPU")
            device = "cpu"
        else:
            print(f"使用GPU设备: {device}")
            # 设置GPU设备
            paddle.set_device(device)

    num_epochs = cfg.train.epochs
    n_splits = cfg.train.n_splits
    Batch_size = cfg.train.batch_size
    lr = cfg.train.lr
    N_skip = cfg.data.N

    # 创建输出目录
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

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
        print_gpu_memory()
        set_seed(cfg.seed)
        train_dataset = paddle.io.Subset(dataset=train_val_dataset, indices=train_index)
        val_dataset = paddle.io.Subset(dataset=train_val_dataset, indices=val_index)
        train_loader = paddle.io.DataLoader(
            dataset=train_dataset,
            batch_size=Batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = paddle.io.DataLoader(
            dataset=val_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,
        )
        test_loader = paddle.io.DataLoader(
            dataset=test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,
        )
        # 定义模型
        model = paddle.vision.models.resnet18(pretrained=True)
        # 修改最后一层以适应回归任务
        model.fc = paddle.nn.Linear(model.fc.weight.shape[0], 1)
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
            for i, (images, groups, labels) in enumerate(train_loader):
                try:
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
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("GPU内存不足，跳过当前批次")
                        paddle.device.cuda.empty_cache()
                        continue
                    else:
                        print(f"训练过程中出现错误: {e}")
                        continue
                except Exception as e:
                    print(f"训练过程中出现错误: {e}")
                    continue
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
                for images, groups, labels in val_loader:
                    try:
                        images = offline_transforms(images)
                        outputs = model(images)
                        val_loss += criterion(outputs.squeeze(), labels).item() * len(
                            images
                        )
                        k += len(images)
                        preds_val.append(outputs.squeeze())
                        true_labels_val.append(labels)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print("验证阶段GPU内存不足，跳过当前批次")
                            paddle.device.cuda.empty_cache()
                            continue
                        else:
                            print(f"验证过程中出现错误: {e}")
                            continue
                    except Exception as e:
                        print(f"验证过程中出现错误: {e}")
                        continue
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
                for images, groups, labels in test_loader:
                    try:
                        images = offline_transforms(images)
                        outputs = model(images)
                        test_loss += criterion(outputs.squeeze(), labels).item() * len(
                            images
                        )
                        k += len(images)
                        preds_test.append(outputs.squeeze())
                        true_labels_test.append(labels)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print("测试阶段GPU内存不足，跳过当前批次")
                            paddle.device.cuda.empty_cache()
                            continue
                        else:
                            print(f"测试过程中出现错误: {e}")
                            continue
                    except Exception as e:
                        print(f"测试过程中出现错误: {e}")
                        continue
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

        # 收集当前fold的所有预测结果和标签
        print(f"收集 Fold {fold+1} 的预测结果...")

        # 训练集预测
        model.eval()
        with paddle.no_grad():
            preds_train = []
            true_labels_train = []
            train_groups_fold = []
            train_samples_id = []

            for images, groups, labels in train_loader:
                try:
                    images = offline_transforms(images)
                    outputs = model(images)
                    preds_train.append(outputs.numpy())
                    true_labels_train.append(labels.numpy())
                    train_groups_fold.extend(groups[0].numpy())
                    train_samples_id.extend(groups[1].numpy())
                except Exception as e:
                    print(f"训练集预测过程中出现错误: {e}")
                    continue

            # 验证集预测
            preds_val = []
            true_labels_val = []
            val_groups_fold = []
            val_samples_id = []

            for images, groups, labels in val_loader:
                try:
                    images = offline_transforms(images)
                    outputs = model(images)
                    preds_val.append(outputs.numpy())
                    true_labels_val.append(labels.numpy())
                    val_groups_fold.extend(groups[0].numpy())
                    val_samples_id.extend(groups[1].numpy())
                except Exception as e:
                    print(f"验证集预测过程中出现错误: {e}")
                    continue

            # 测试集预测
            preds_test = []
            true_labels_test = []
            test_groups_fold = []
            test_samples_id = []

            for images, groups, labels in test_loader:
                try:
                    images = offline_transforms(images)
                    outputs = model(images)
                    preds_test.append(outputs.numpy())
                    true_labels_test.append(labels.numpy())
                    test_groups_fold.extend(groups[0].numpy())
                    test_samples_id.extend(groups[1].numpy())
                except Exception as e:
                    print(f"测试集预测过程中出现错误: {e}")
                    continue

        # 展平结果
        preds_train = np.concatenate(preds_train)
        true_labels_train = np.concatenate(true_labels_train)
        preds_val = np.concatenate(preds_val)
        true_labels_val = np.concatenate(true_labels_val)
        preds_test = np.concatenate(preds_test)
        true_labels_test = np.concatenate(true_labels_test)

        # 获取unique groups和sample id
        unique_val_groups = sorted(set(val_groups_fold))
        unique_sample_id = sorted(set(val_samples_id))

        # 保存当前fold的预测结果和标签
        np.save(os.path.join(output_dir, f"preds_train_fold{fold+1}.npy"), preds_train)
        np.save(os.path.join(output_dir, f"preds_val_fold{fold+1}.npy"), preds_val)
        np.save(os.path.join(output_dir, f"preds_test_fold{fold+1}.npy"), preds_test)
        np.save(
            os.path.join(output_dir, f"true_labels_train_fold{fold+1}.npy"),
            true_labels_train,
        )
        np.save(
            os.path.join(output_dir, f"true_labels_val_fold{fold+1}.npy"),
            true_labels_val,
        )
        np.save(
            os.path.join(output_dir, f"true_labels_test_fold{fold+1}.npy"),
            true_labels_test,
        )

        # 保存样本ID
        np.save(
            os.path.join(output_dir, f"sample_ids_train_fold{fold+1}.npy"),
            train_samples_id,
        )
        np.save(
            os.path.join(output_dir, f"sample_ids_val_fold{fold+1}.npy"), val_samples_id
        )
        np.save(
            os.path.join(output_dir, f"sample_ids_test_fold{fold+1}.npy"),
            test_samples_id,
        )

        # 保存unique groups和sample id
        np.save(
            os.path.join(output_dir, f"unique_val_groups_fold_{fold+1}.npy"),
            unique_val_groups,
        )
        np.save(
            os.path.join(output_dir, f"unique_sample_id_fold_{fold+1}.npy"),
            unique_sample_id,
        )

        print(f"Fold {fold+1} 预测结果已保存")
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

    # 集成预测
    print("\n开始集成预测...")
    # 确保test_preds_history是numpy数组
    test_preds_history = np.array(test_preds_history)
    print(f"集成预测形状: {test_preds_history.shape}")

    # 确保数据形状正确，移除多余的维度
    if test_preds_history.ndim == 3 and test_preds_history.shape[-1] == 1:
        test_preds_history = test_preds_history.squeeze(-1)

    ensemble_mean_preds = np.mean(test_preds_history, axis=0)
    ensemble_median_preds = np.median(test_preds_history, axis=0)

    # 保存集成预测结果
    np.save(
        os.path.join(output_dir, "ensemble_mean_preds_test.npy"), ensemble_mean_preds
    )
    np.save(
        os.path.join(output_dir, "ensemble_median_preds_test.npy"),
        ensemble_median_preds,
    )

    # 获取测试集真实标签（从最后一个fold获取）
    true_labels_test = np.load(
        os.path.join(output_dir, f"true_labels_test_fold{n_splits}.npy")
    )

    # 验证所有fold的测试集标签是否一致
    print("验证测试集标签一致性...")
    all_test_labels = []
    for i in range(n_splits):
        fold_labels = np.load(
            os.path.join(output_dir, f"true_labels_test_fold{i+1}.npy")
        )
        all_test_labels.append(fold_labels)
        print(f"  Fold {i+1} 测试集标签形状: {fold_labels.shape}")

    # 检查标签是否一致
    labels_consistent = all(
        np.array_equal(all_test_labels[0], labels) for labels in all_test_labels[1:]
    )
    if labels_consistent:
        print("  ✅ 所有fold的测试集标签一致")
    else:
        print("  [WARNING] 不同fold的测试集标签不一致，这可能导致集成学习效果不佳")
        # 使用第一个fold的标签作为参考
        true_labels_test = all_test_labels[0]

    # 计算集成预测的性能指标
    ensemble_mean_mse = mean_squared_error(true_labels_test, ensemble_mean_preds)
    ensemble_mean_r2 = r2_score(true_labels_test, ensemble_mean_preds)
    ensemble_median_mse = mean_squared_error(true_labels_test, ensemble_median_preds)
    ensemble_median_r2 = r2_score(true_labels_test, ensemble_median_preds)

    print("\n集成预测性能指标:")
    print(f"  均值集成 - MSE: {ensemble_mean_mse:.4f}, R²: {ensemble_mean_r2:.4f}")
    print(f"  中位数集成 - MSE: {ensemble_median_mse:.4f}, R²: {ensemble_median_r2:.4f}")

    # 计算每个fold的性能
    fold_performances = []
    for i in range(n_splits):
        fold_mse = mean_squared_error(true_labels_test, test_preds_history[i])
        fold_r2 = r2_score(true_labels_test, test_preds_history[i])
        fold_performances.append((fold_mse, fold_r2))
        print(f"  Fold {i+1} - MSE: {fold_mse:.4f}, R²: {fold_r2:.4f}")

    # 计算单fold平均性能
    single_fold_mse = np.mean([perf[0] for perf in fold_performances])
    single_fold_r2 = np.mean([perf[1] for perf in fold_performances])
    print(f"  单fold平均 - MSE: {single_fold_mse:.4f}, R²: {single_fold_r2:.4f}")

    # 找出最佳单fold性能
    best_fold_idx = np.argmax([perf[1] for perf in fold_performances])
    best_fold_r2 = fold_performances[best_fold_idx][1]
    print(f"  最佳单fold (Fold {best_fold_idx+1}) - R²: {best_fold_r2:.4f}")

    # 检查集成学习是否有效
    print("\n集成学习效果分析:")
    if ensemble_mean_r2 > single_fold_r2:
        print(f"  [OK] 均值集成有效！比单fold平均提升了 {ensemble_mean_r2 - single_fold_r2:.4f} R²")
    else:
        print(
            f"  [WARNING] 均值集成效果不明显，比单fold平均差了 {single_fold_r2 - ensemble_mean_r2:.4f} R²"
        )

    if ensemble_mean_r2 > best_fold_r2:
        print(f"  [OK] 均值集成有效！比最佳单fold提升了 {ensemble_mean_r2 - best_fold_r2:.4f} R²")
    else:
        print(f"  [WARNING] 均值集成不如最佳单fold，差了 {best_fold_r2 - ensemble_mean_r2:.4f} R²")

    # 尝试加权集成
    print("\n尝试加权集成...")
    # 根据每个fold的性能计算权重
    fold_weights = np.array([perf[1] for perf in fold_performances])  # 使用R²作为权重
    fold_weights = fold_weights / np.sum(fold_weights)  # 归一化
    print(f"  Fold权重: {fold_weights}")

    weighted_preds = np.average(test_preds_history, axis=0, weights=fold_weights)
    weighted_mse = mean_squared_error(true_labels_test, weighted_preds)
    weighted_r2 = r2_score(true_labels_test, weighted_preds)
    print(f"  加权集成 - MSE: {weighted_mse:.4f}, R²: {weighted_r2:.4f}")

    if weighted_r2 > ensemble_mean_r2:
        print(f"  [OK] 加权集成比简单均值集成提升了 {weighted_r2 - ensemble_mean_r2:.4f} R²")
        # 保存加权集成结果
        np.save(
            os.path.join(output_dir, "ensemble_weighted_preds_test.npy"), weighted_preds
        )

    np.save(os.path.join(output_dir, "ensemble_true_labels_test.npy"), true_labels_test)

    # 生成可视化图表
    print("生成可视化图表...")

    # 1. 集成预测的parity plot
    plt.figure(figsize=(8, 6))
    plt.scatter(true_labels_test, ensemble_mean_preds, alpha=0.6)
    plt.plot(
        [true_labels_test.min(), true_labels_test.max()],
        [true_labels_test.min(), true_labels_test.max()],
        "r--",
        lw=2,
    )
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Ensemble Parity Plot")
    plt.savefig(
        os.path.join(output_dir, "ensemble_parity_plot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. 集成预测的violin plot
    plt.figure(figsize=(10, 6))
    data_to_plot = [true_labels_test, ensemble_mean_preds]
    plt.violinplot(data_to_plot, positions=[1, 2], showmeans=True, showmedians=True)
    plt.xticks([1, 2], ["True Values", "Predicted Values"])
    plt.ylabel("Values")
    plt.title("Ensemble Violin Plot")
    plt.savefig(
        os.path.join(output_dir, "ensemble_violin_plot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 3. 每个fold的parity plot
    for fold in range(n_splits):
        fold_preds = np.load(os.path.join(output_dir, f"preds_test_fold{fold+1}.npy"))
        fold_true = np.load(
            os.path.join(output_dir, f"true_labels_test_fold{fold+1}.npy")
        )

        plt.figure(figsize=(8, 6))
        plt.scatter(fold_true, fold_preds, alpha=0.6)
        plt.plot(
            [fold_true.min(), fold_true.max()],
            [fold_true.min(), fold_true.max()],
            "r--",
            lw=2,
        )
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Fold {fold+1} Parity Plot")
        plt.savefig(
            os.path.join(output_dir, f"parity_plot_fold{fold+1}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 4. 每个fold的violin plot
        plt.figure(figsize=(10, 6))
        data_to_plot = [fold_true, fold_preds]
        plt.violinplot(data_to_plot, positions=[1, 2], showmeans=True, showmedians=True)
        plt.xticks([1, 2], ["True Values", "Predicted Values"])
        plt.ylabel("Values")
        plt.title(f"Fold {fold+1} Violin Plot")
        plt.savefig(
            os.path.join(output_dir, f"violin_plot_fold{fold+1}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    print("集成预测结果和可视化图表已保存")


def evaluate(cfg):
    """
    完整的评估流程，包括模型加载、推理、指标计算、可视化和集成预测
    """
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedGroupKFold

    set_seed(cfg.seed)
    device = device2str(cfg.device)

    # 检查GPU可用性
    if "gpu" in device:
        if not paddle.device.cuda.device_count():
            print("警告：配置为GPU评估但未检测到GPU，将使用CPU")
            device = "cpu"
        else:
            print(f"使用GPU设备进行评估: {device}")
            paddle.set_device(device)

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
                for images, groups, labels in train_loader:
                    images = offline_transforms(images)
                    outputs = model(images)
                    true_labels_train.append(labels.numpy())
                    preds_train.append(outputs.numpy())
                    train_groups_fold.extend(groups[0].numpy())
                    train_samples_id.extend(groups[1].numpy())

                # 验证集推理
                for images, groups, labels in val_loader:
                    images = offline_transforms(images)
                    outputs = model(images)
                    true_labels_val.append(labels.numpy())
                    preds_val.append(outputs.numpy())
                    val_groups_fold.extend(groups[0].numpy())
                    val_samples_id.extend(groups[1].numpy())

                # 测试集推理
                for images, groups, labels in test_loader:
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
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            true_labels_train,
            preds_train,
            s=30,
            marker=".",
            alpha=0.6,
            label=f"Train R-squared: {r_squared_train:.4f}",
        )
        ax.scatter(
            true_labels_val,
            preds_val,
            s=30,
            marker="*",
            alpha=0.6,
            label=f"Validation R-squared: {r_squared_val:.4f}",
        )
        ax.scatter(
            true_labels_test,
            preds_test,
            s=30,
            marker="o",
            color="red",
            alpha=0.6,
            label=f"Test R-squared: {r_squared_test:.4f}",
        )
        ax.plot(
            [true_labels_train.min(), true_labels_train.max()],
            [true_labels_train.min(), true_labels_train.max()],
            color="black",
            linestyle="--",
            lw=2,
            label="Ideal fit",
        )
        ax.set_xlabel("True UTS (MPa)", fontsize=14)
        ax.set_ylabel("Predicted UTS (MPa)", fontsize=14)
        ax.set_aspect("equal")
        ax.set_title(f"Fold {fold + 1} Parity Plot", fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"parity_plot_fold{fold+1}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 绘制复杂的violin plot (类似图片中的样式)
        fig, ax = plt.subplots(figsize=(10, 8))

        # 准备数据
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

        # 添加理想拟合线
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
        ax.set_title(f"Fold {fold+1} Parity Violin Plot", fontsize=16)
        plt.xticks(np.arange(0, 5.2, 1), fontsize=16)
        plt.yticks(np.arange(0, 5.2, 1), fontsize=16)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"violin_plot_fold{fold+1}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

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
    print(f"集成预测形状: {test_preds_history.shape}")

    # 计算每个fold的预测方差，用于评估集成学习的多样性
    pred_variance = np.var(test_preds_history, axis=0)
    print(f"预测方差统计: 均值={np.mean(pred_variance):.4f}, 标准差={np.std(pred_variance):.4f}")

    # 计算中位数和均值预测
    # 确保数据形状正确，移除多余的维度
    if test_preds_history.ndim == 3 and test_preds_history.shape[-1] == 1:
        test_preds_history = test_preds_history.squeeze(-1)

    median_preds_test = np.median(test_preds_history, axis=0)
    mean_preds_test = np.mean(test_preds_history, axis=0)

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
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        true_labels_test,
        median_preds_test,
        s=30,
        marker="o",
        alpha=0.6,
        label=f"Median R-squared: {median_test_r2:.4f}",
    )
    ax.scatter(
        true_labels_test,
        mean_preds_test,
        s=30,
        marker="x",
        alpha=0.6,
        label=f"Mean R-squared: {mean_test_r2:.4f}",
    )
    ax.plot(
        [true_labels_test.min(), true_labels_test.max()],
        [true_labels_test.min(), true_labels_test.max()],
        color="black",
        linestyle="--",
        lw=2,
        label="Ideal fit",
    )
    ax.set_xlabel("True UTS (MPa)", fontsize=14)
    ax.set_ylabel("Predicted UTS (MPa)", fontsize=14)
    ax.set_aspect("equal")
    ax.set_title("Test Data Parity Plot (Ensemble Predictions)", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "ensemble_parity_plot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 绘制集成预测的复杂violin plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # 为每个真实值范围创建violin plot
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

    # 添加理想拟合线
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
    ax.set_title("Ensemble Parity Violin Plot", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "ensemble_violin_plot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

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
