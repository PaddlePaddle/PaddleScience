from ppsci.arch import base
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MLPModel(base.Arch):
    def __init__(self, input_shape, learning_rate, nodes1, nodes2, nodes3, dropout_rate1, dropout_rate2, dropout_rate3):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, nodes1),
            nn.ReLU(),
            nn.Dropout(dropout_rate1),
            nn.Linear(nodes1, nodes2),
            nn.ReLU(),
            nn.Dropout(dropout_rate2),
            nn.Linear(nodes2, nodes3),
            nn.ReLU(),
            nn.Dropout(dropout_rate3),
            nn.Linear(nodes3, 3),
            nn.Sigmoid()
        )
        self.optimizer = optim.RMSProp(learning_rate=learning_rate, momentum=0.9, centered=True, parameters=self.model.parameters())
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def train(self, x_train, y_train, x_test, y_test, epochs=1000):
        history = {'loss': [], 'val_loss': []}
        x_train = paddle.to_tensor(x_train, dtype='float32')
        y_train = paddle.to_tensor(y_train, dtype='float32')
        x_test = paddle.to_tensor(x_test, dtype='float32')
        y_test = paddle.to_tensor(y_test, dtype='float32')

        for epoch in range(epochs):
            self.model.train()
            y_pred = self.model(x_train)
            loss = self.criterion(y_pred, y_train)
            loss.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()

            self.model.eval()
            val_pred = self.model(x_test)
            val_loss = self.criterion(val_pred, y_test)

            history['loss'].append(loss.numpy())
            history['val_loss'].append(val_loss.numpy())

        self.visualize_loss(history, "Training and Validation Loss")

    def visualize_loss(self, history, title):
        loss = history["loss"]
        val_loss = history["val_loss"]
        epochs = range(len(loss))
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, loss, "b", label="Training loss")
        plt.plot(epochs, val_loss, "r", label="Validation loss")
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show(block=False)  # 不阻塞程序的执行

    def evaluate(self, x_test, y_test):
        self.model.eval()
        x_test = paddle.to_tensor(x_test, dtype='float32')
        y_test = paddle.to_tensor(y_test, dtype='float32')
        y_pred = self.model(x_test)
        loss = self.criterion(y_pred, y_test)
        return loss.numpy()

# 数据预处理和模型训练示例代码
filePath = "C:/Users/ssm18/new0811/PaddleScience/ppsci/data/dataset/MP_data_down_loading(train+validate).csv"

df = pd.read_csv(filePath, header=0)

# 数据处理部分
df_charge_space_group_number = pd.get_dummies(df["charge_space_group_number"], prefix="charge_space_group_number")
df = df.join(df_charge_space_group_number)
df_discharge_space_group_number = pd.get_dummies(df["discharge_space_group_number"], prefix="discharge_space_group_number")
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

x_df = df.drop(["average_voltage", "capacity_grav", "energy_grav"], axis=1)
y_df = df[["average_voltage", "capacity_grav", "energy_grav"]]

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

pca = PCA(0.99)
x_df = pca.fit_transform(x_df)
x_df = pd.DataFrame(x_df)

min_max_scaler = MinMaxScaler()
x_df.columns = x_df.columns.astype(str)
x_df = min_max_scaler.fit_transform(x_df)

y_min = y_df.min()
y_max = y_df.max()
y_df = (y_df - y_min) / (y_max - y_min)

len_train_test = int(tuple(x_df.shape)[0] * 0.9)
x_train, x_test = x_df[:len_train_test], x_df[len_train_test:]
y_train, y_test = y_df[:len_train_test], y_df[len_train_test:]

# 初始化并训练模型
model = MLPModel(
    input_shape=x_train.shape[1],
    learning_rate=0.0001,
    nodes1=40,
    nodes2=30,
    nodes3=15,
    dropout_rate1=0.2,
    dropout_rate2=0.2,
    dropout_rate3=0.2
)

model.train(x_train, y_train, x_test, y_test, epochs=1000)
model.evaluate(x_test, y_test)
