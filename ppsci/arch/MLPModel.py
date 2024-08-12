from ppsci.arch import base
import tensorflow as tf
from tensorflow import keras
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functools
import csv
import math
import os

class MLPModel(base.Arch):
    def __init__(self, input_shape, learning_rate, nodes1, nodes2, nodes3, dropout_rate1, dropout_rate2, dropout_rate3):
        super(MLPModel, self).__init__()
        self.model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Dense(nodes1, activation="relu"),
            keras.layers.Dropout(dropout_rate1),
            keras.layers.Dense(nodes2, activation="relu"),
            keras.layers.Dropout(dropout_rate2),
            keras.layers.Dense(nodes3, activation="relu"),
            keras.layers.Dropout(dropout_rate3),
            keras.layers.Dense(3, activation="sigmoid"),
        ])
        self.model.compile(
            optimizer=keras.optimizers.RMSprop(
                learning_rate=learning_rate, momentum=0.9, centered=True
            ),
            loss="mse",
        )

    def forward(self, x):
        return self.model(x)

    def train(self, x_train, y_train, x_test, y_test, epochs=1000):
        history = self.model.fit(
            x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=0
        )
        self.visualize_loss(history, "Training and Validation Loss")

    def visualize_loss(self, history, title):
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
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
        loss = self.model.evaluate(x_test, y_test)
        return loss

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
    input_shape=(tuple(x_train.shape)[1],),
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
