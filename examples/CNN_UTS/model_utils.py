# model_utils.py
import matplotlib.pyplot as plt
import numpy as np
import paddle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def set_seed(seed):
    paddle.seed(seed=seed)
    np.random.seed(seed)
    import random

    random.seed(seed)
    if paddle.device.cuda.device_count() >= 1:
        paddle.seed(seed=seed)
        paddle.seed(seed=seed)


def visualize_results(true, pred, output_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(true, pred, alpha=0.5)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Parity Plot")
    plt.savefig(output_path)
    plt.close()


def compute_metrics(true, pred):
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    return {"rmse": rmse, "r2": r2}


def plot_parity(true, pred, title="Parity Plot", save_path=None, label=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(true, pred, s=10, marker="o", label=label)
    plt.plot(
        [true.min(), true.max()],
        [true.min(), true.max()],
        color="black",
        linestyle="--",
        label="Ideal fit",
    )
    plt.xlabel("True UTS (MPa)")
    plt.ylabel("Predicted UTS (MPa)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_violin(true, pred, group_ids, color="tab:blue", label=None, ax=None):
    """
    true: 真实标签（1D数组）
    pred: 预测值（1D数组）
    group_ids: 分组id（如样本id或UTS分组，1D数组）
    color: 颜色
    label: 图例标签
    ax: 可选，matplotlib的ax对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    unique_groups = np.unique(group_ids)
    for i, group in enumerate(unique_groups):
        mask = group_ids == group
        preds_for_group = pred[mask]
        true_for_group = true[mask]
        parts = ax.violinplot(
            preds_for_group,
            positions=[np.mean(true_for_group)],
            showmeans=False,
            showmedians=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor("black")
            pc.set_alpha(0.5)
        parts["cmedians"].set_color(color)
        parts["cmins"].set_color(color)
        parts["cmaxes"].set_color(color)
        ax.plot(
            true_for_group,
            preds_for_group,
            "o",
            color=color,
            markersize=4,
            alpha=0.6,
            label=label if i == 0 else None,
        )
    return ax


def plot_all_violin(
    true_train,
    pred_train,
    id_train,
    r2_train,
    true_val,
    pred_val,
    id_val,
    r2_val,
    true_test,
    pred_test,
    id_test,
    r2_test,
    save_path=None,
):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_violin(
        true_train,
        pred_train,
        id_train,
        color="tab:blue",
        label=f"Train $R^2$: {r2_train:.4f}",
        ax=ax,
    )
    plot_violin(
        true_val,
        pred_val,
        id_val,
        color="tab:orange",
        label=f"Val $R^2$: {r2_val:.4f}",
        ax=ax,
    )
    plot_violin(
        true_test,
        pred_test,
        id_test,
        color="tab:red",
        label=f"Test $R^2$: {r2_test:.4f}",
        ax=ax,
    )
    ax.plot(
        [
            min(true_train.min(), true_val.min(), true_test.min()),
            max(true_train.max(), true_val.max(), true_test.max()),
        ],
        [
            min(true_train.min(), true_val.min(), true_test.min()),
            max(true_train.max(), true_val.max(), true_test.max()),
        ],
        color="black",
        linestyle="--",
        label="Ideal fit",
    )
    ax.legend(prop={"size": 11})
    ax.set_xlabel("True UTS (MPa)", fontsize=18)
    ax.set_ylabel("Predicted UTS (MPa)", fontsize=18)
    ax.set_title("Parity Violin Plot")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
