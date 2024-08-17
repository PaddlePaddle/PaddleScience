import sys
import os
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import functools
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization

print("PaddlePaddle version:", paddle.__version__)

# 数据加载和预处理
filePath = "./MP_data_down_loading(train+validate).csv"
try:
    df = pd.read_csv(filePath, header=0)
    print("Data loaded successfully.")
except Exception as e:
    print("Error loading data:", e)
    raise

print("Data shape:", tuple(df.shape))
print("Data columns:", df.columns)

# 数据处理部分
df_charge_space_group_number = pd.get_dummies(
    df["charge_space_group_number"], prefix="charge_space_group_number"
)
df = df.join(df_charge_space_group_number)
df_discharge_space_group_number = pd.get_dummies(
    df["discharge_space_group_number"], prefix="discharge_space_group_number"
)
df = df.join(df_discharge_space_group_number)
df = df.drop(
    [
        "battery_id",
        "battery_formula",
        "framework_formula",
        "adj_pairs",
        "capacity_vol",
        "energy_vol",
        "formula_charge",
        "formula_discharge",
        "id_charge",
        "id_discharge",
        "working_ion",
        "num_steps",
        "stability_charge",
        "stability_discharge",
        "charge_crystal_system",
        "charge_energy_per_atom",
        "charge_formation_energy_per_atom",
        "charge_band_gap",
        "charge_efermi",
        "discharge_crystal_system",
        "discharge_energy_per_atom",
        "discharge_formation_energy_per_atom",
        "discharge_band_gap",
        "discharge_efermi",
    ],
    axis=1,
)
print("Data shape after dropping columns:", tuple(df.shape))

x_df = df.drop(["average_voltage", "capacity_grav", "energy_grav"], axis=1)
y_df = df[["average_voltage", "capacity_grav", "energy_grav"]]

pca = PCA(0.99)
x_df = pca.fit_transform(x_df)
x_df = pd.DataFrame(x_df)

min_max_scaler = MinMaxScaler()
x_df.columns = x_df.columns.astype(str)
x_df = min_max_scaler.fit_transform(x_df)

y_min = y_df.min()
y_max = y_df.max()
y_df = (y_df - y_min) / (y_max - y_min)

print("x_df shape:", tuple(x_df.shape))
print("y_df shape:", tuple(y_df.shape))

len_train_test = int(tuple(x_df.shape)[0] * 0.9)
x_train, x_test = x_df[:len_train_test], x_df[len_train_test:]
y_train, y_test = y_df[:len_train_test], y_df[len_train_test:]

print("x_train shape:", tuple(x_train.shape))
print("x_test shape:", tuple(x_test.shape))
print("y_train shape:", tuple(y_train.shape))
print("y_test shape:", tuple(y_test.shape))

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# 模型构建函数
def get_model(
    input_shape,
    learning_rate,
    nodes1,
    nodes2,
    nodes3,
    dropout_rate1,
    dropout_rate2,
    dropout_rate3,
):
    model = nn.Sequential(
        nn.Linear(input_shape[0], nodes1),
        nn.ReLU(),
        nn.Dropout(dropout_rate1),
        nn.Linear(nodes1, nodes2),
        nn.ReLU(),
        nn.Dropout(dropout_rate2),
        nn.Linear(nodes2, nodes3),
        nn.ReLU(),
        nn.Dropout(dropout_rate3),
        nn.Linear(nodes3, 3),
        nn.Sigmoid(),
    )

    optimizer = optim.RMSProp(learning_rate=learning_rate, momentum=0.9, centered=True, parameters=model.parameters())
    loss_fn = nn.MSELoss()

    return model, optimizer, loss_fn

# 可视化损失函数
def visualize_loss(history, title):
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(loss))
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show(block=False)  # 不阻塞程序的执行

# 模型训练和验证函数
def fit_model(
    input_shape,
    learning_rate,
    nodes1,
    nodes2,
    nodes3,
    dropout_rate1,
    dropout_rate2,
    dropout_rate3,
    epochs=1000,
):
    model, optimizer, loss_fn = get_model(
        input_shape,
        learning_rate,
        nodes1,
        nodes2,
        nodes3,
        dropout_rate1,
        dropout_rate2,
        dropout_rate3,
    )

    history = {'loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        x_train_tensor = paddle.to_tensor(x_train, dtype='float32')
        y_train_tensor = paddle.to_tensor(y_train, dtype='float32')
        preds = model(x_train_tensor)
        loss = loss_fn(preds, y_train_tensor)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        model.eval()
        x_test_tensor = paddle.to_tensor(x_test, dtype='float32')
        y_test_tensor = paddle.to_tensor(y_test, dtype='float32')
        val_preds = model(x_test_tensor)
        val_loss = loss_fn(val_preds, y_test_tensor)

        history['loss'].append(loss.numpy())
        history['val_loss'].append(val_loss.numpy())

    return model, history

# 贝叶斯优化使用的目标函数
def bayesian_optimization_target(
    input_shape,
    learning_rate,
    nodes1,
    nodes2,
    nodes3,
    dropout_rate1,
    dropout_rate2,
    dropout_rate3,
):
    _, history = fit_model(
        input_shape,
        learning_rate,
        int(nodes1),
        int(nodes2),
        int(nodes3),
        dropout_rate1,
        dropout_rate2,
        dropout_rate3,
        epochs=10,  # 在贝叶斯优化中通常减少训练轮数以加快速度
    )
    return -np.mean(history['val_loss'])  # 取负值，因为贝叶斯优化是找最大值

# 初始训练和评估模型
model, history = fit_model((x_train.shape[1],), 0.0001, 40, 30, 15, 0.2, 0.2, 0.2)
visualize_loss(history, "Training and Validation Loss")

# 贝叶斯优化部分
inputs_shape = (x_train.shape[1],)
pbounds = {
    "learning_rate": (1e-05, 0.001),
    "nodes1": (8, 256),
    "nodes2": (8, 256),
    "nodes3": (8, 256),
    "dropout_rate1": (0.1, 0.9),
    "dropout_rate2": (0.1, 0.9),
    "dropout_rate3": (0.1, 0.9),
}

fit_with_partial = functools.partial(bayesian_optimization_target, inputs_shape)

optimizer = BayesianOptimization(
    f=fit_with_partial, pbounds=pbounds, verbose=2, random_state=42
)
optimizer.maximize(init_points=10, n_iter=20)
print("optimizer.max: ", optimizer.max)

# 保存贝叶斯优化结果
os.makedirs("results_out", exist_ok=True)
csv_path = "results_out/bayesian_opt.csv"
json_path = "results_out/bayesian_opt.json"

with open(csv_path, mode="w") as f:
    writer = csv.writer(f)
    writer.writerow(["learning_rate", "nodes1", "nodes2", "nodes3", "dropout_rate1", "dropout_rate2", "dropout_rate3", "loss"])
    for res in optimizer.res:
        writer.writerow([
            res["params"]["learning_rate"],
            int(res["params"]["nodes1"]),
            int(res["params"]["nodes2"]),
            int(res["params"]["nodes3"]),
            res["params"]["dropout_rate1"],
            res["params"]["dropout_rate2"],
            res["params"]["dropout_rate3"],
            -res["target"]  # Convert the negative loss back to positive
        ])

import json
with open(json_path, mode="w") as f:
    json.dump(optimizer.max, f, indent=4)

# 使用优化后的参数进行最终模型训练和评估
best_params = optimizer.max["params"]
best_model, best_history = fit_model(
    inputs_shape,
    learning_rate=best_params["learning_rate"],
    nodes1=int(best_params["nodes1"]),
    nodes2=int(best_params["nodes2"]),
    nodes3=int(best_params["nodes3"]),
    dropout_rate1=best_params["dropout_rate1"],
    dropout_rate2=best_params["dropout_rate2"],
    dropout_rate3=best_params["dropout_rate3"],
    epochs=1000  # 这里可以增加训练轮数以获得更好的结果
)

# 保存最佳模型
paddle.save(best_model.state_dict(), "results_out/best_model.pdparams")

# 可视化最佳模型的训练过程
visualize_loss(best_history, "Best Model Training and Validation Loss")

# 在测试集上评估最佳模型
best_model.eval()
x_test_tensor = paddle.to_tensor(x_test, dtype='float32')
y_test_tensor = paddle.to_tensor(y_test, dtype='float32')
y_pred = best_model(x_test_tensor)
test_loss = nn.functional.mse_loss(y_pred, y_test_tensor)
print(f"Test loss: {test_loss.numpy()}")

# 你还可以添加进一步的评估指标或对模型的预测进行分析
