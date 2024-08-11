import sys

sys.path.append("./utils")
import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import functools
from bayes_opt import BayesianOptimization
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

print("TensorFlow version:", tf.__version__)

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
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Dense(nodes1, activation="relu"),
            keras.layers.Dropout(dropout_rate1),
            keras.layers.Dense(nodes2, activation="relu"),
            keras.layers.Dropout(dropout_rate2),
            keras.layers.Dense(nodes3, activation="relu"),
            keras.layers.Dropout(dropout_rate3),
            keras.layers.Dense(3, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.RMSprop(
            learning_rate=learning_rate, momentum=0.9, centered=True
        ),
        loss="mse",
    )
    return model


def visualize_loss(history, title):
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


def fit_with(
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
    model = get_model(
        input_shape,
        learning_rate,
        nodes1,
        nodes2,
        nodes3,
        dropout_rate1,
        dropout_rate2,
        dropout_rate3,
    )
    history = model.fit(
        x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=0
    )
    visualize_loss(history, "Training and Validation Loss")
    return model, history


model, history = fit_with((tuple(x_train.shape)[1],), 0.0001, 40, 30, 15, 0.2, 0.2, 0.2)
visualize_loss(history, "Training and Validation Loss")

learning_rate = 0.0001
epochs = 10
nodes1 = 40
nodes2 = 30
nodes3 = 15
dropout_rate1 = 0.2
dropout_rate2 = 0.2
dropout_rate3 = 0.2
inputs_shape = (tuple(x_train.shape)[1],)
init_points = 10
n_iter = 20


def fit_with(
    inputs_shape,
    learning_rate,
    epochs,
    nodes1,
    nodes2,
    nodes3,
    dropout_rate1,
    dropout_rate2,
    dropout_rate3,
):
    nodes11 = int(nodes1) * 2
    nodes22 = int(nodes2) * 2
    nodes33 = int(nodes3) * 2
    model = get_model(
        inputs_shape,
        learning_rate,
        nodes11,
        nodes22,
        nodes33,
        dropout_rate1,
        dropout_rate2,
        dropout_rate3,
    )
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath="./v8.keras", monitor="val_loss", save_best_only=True
        )
    ]
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks_list,
        verbose=0,
    )
    visualize_loss(history, "Training and Validation Loss")
    best_model = keras.models.load_model("./v8.keras")
    loss = best_model.evaluate(x_test, y_test)
    return 1 / loss


fit_with_partial = functools.partial(fit_with, inputs_shape=inputs_shape, epochs=epochs)
pbounds = {
    "learning_rate": (1e-05, 0.001),
    "nodes1": (8, 256),
    "nodes2": (8, 256),
    "nodes3": (8, 256),
    "dropout_rate1": (0.1, 0.9),
    "dropout_rate2": (0.1, 0.9),
    "dropout_rate3": (0.1, 0.9),
}

optimizer = BayesianOptimization(
    f=fit_with_partial, pbounds=pbounds, verbose=2, random_state=42
)
optimizer.maximize(init_points=init_points, n_iter=n_iter)
print("optimizer.max: ", optimizer.max)

os.makedirs("results_out", exist_ok=True)
Rmses_V = []
Rmses_C = []
Rmses_E = []
Rmses = []
times = 5
epochs = 10

for i in range(times):
    print("Calculation times: ", i)

    def fit_with_for_train(
        inputs_shape,
        learning_rate,
        epochs,
        nodes1,
        nodes2,
        nodes3,
        dropout_rate1,
        dropout_rate2,
        dropout_rate3,
    ):
        nodes11 = int(nodes1) * 2
        nodes22 = int(nodes2) * 2
        nodes33 = int(nodes3) * 2
        model = get_model(
            inputs_shape,
            learning_rate,
            nodes11,
            nodes22,
            nodes33,
            dropout_rate1,
            dropout_rate2,
            dropout_rate3,
        )
        callbacks_list = [
            keras.callbacks.ModelCheckpoint(
                filepath="./v8-%i.keras" % i, monitor="val_loss", save_best_only=True
            )
        ]
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks_list,
            verbose=0,
        )
        best_model = keras.models.load_model("./v8-%i.keras" % i)
        return best_model

    model = fit_with_for_train(
        inputs_shape=(tuple(x_train.shape)[1],),
        epochs=epochs,
        learning_rate=0.0001,
        nodes1=40,
        nodes2=30,
        nodes3=15,
        dropout_rate1=0.2,
        dropout_rate2=0.2,
        dropout_rate3=0.2,
    )
    y_pred = model.predict(x_test)
    y_pred_average_voltage = y_pred[:, 0]
    y_pred_capacity_grav = y_pred[:, 1]
    y_pred_energy_grav = y_pred[:, 2]
    y_pred_average_voltage = (
        y_pred_average_voltage * (y_max.iloc[0] - y_min.iloc[0]) + y_min.iloc[0]
    )
    y_pred_capacity_grav = (
        y_pred_capacity_grav * (y_max.iloc[1] - y_min.iloc[1]) + y_min.iloc[1]
    )
    y_pred_energy_grav = (
        y_pred_energy_grav * (y_max.iloc[2] - y_min.iloc[2]) + y_min.iloc[2]
    )
    y_test_average_voltage = y_test[:, 0]
    y_test_capacity_grav = y_test[:, 1]
    y_test_energy_grav = y_test[:, 2]
    y_test_average_voltage = (
        y_test_average_voltage * (y_max.iloc[0] - y_min.iloc[0]) + y_min.iloc[0]
    )
    y_test_capacity_grav = (
        y_test_capacity_grav * (y_max.iloc[1] - y_min.iloc[1]) + y_min.iloc[1]
    )
    y_test_energy_grav = (
        y_test_energy_grav * (y_max.iloc[2] - y_min.iloc[2]) + y_min.iloc[2]
    )

    filename = "results_out/%i_predicted_results.csv" % i
    headers = [
        "battery_number",
        "average_voltage",
        "predicted_average_voltage",
        "capacity_grav",
        "predicted_capacity_grav",
        "energy_grav",
        "predicted_energy_grav",
    ]

    with open(filename, "w", newline="") as f:
        fcsv = csv.writer(f)
        fcsv.writerow(headers)

    for j in range(len(y_pred)):
        rows = [
            j + 1,
            y_test_average_voltage[j],
            y_pred_average_voltage[j],
            y_test_capacity_grav[j],
            y_pred_capacity_grav[j],
            y_test_energy_grav[j],
            y_pred_energy_grav[j],
        ]
        with open(filename, "a", newline="") as f:
            fcsv = csv.writer(f)
            fcsv.writerow(rows)

    battery_number = range(1, len(y_pred_average_voltage) + 1)

    plt.figure(figsize=(16, 4))
    plt.plot(battery_number, y_test_average_voltage, "r", label="DFT Voltage")
    plt.plot(battery_number, y_pred_average_voltage, "b", label="MLP Voltage")
    plt.title("DFT Vs. MLP")
    plt.xlabel("Battery number [-]")
    plt.ylabel("Voltage [V]")
    plt.legend()
    plt.show(block=False)  # 不阻塞程序的执行

    plt.figure(figsize=(16, 4))
    plt.plot(battery_number, y_test_capacity_grav, "r", label="DFT Capacity")
    plt.plot(battery_number, y_pred_capacity_grav, "b", label="MLP Capacity")
    plt.title("DFT Vs. MLP")
    plt.xlabel("Battery number [-]")
    plt.ylabel("Gravimetric Capacity [mAh/g]")
    plt.legend()
    plt.show(block=False)  # 不阻塞程序的执行

    plt.figure(figsize=(16, 4))
    plt.plot(battery_number, y_test_energy_grav, "r", label="DFT Energy")
    plt.plot(battery_number, y_pred_energy_grav, "b", label="MLP Energy")
    plt.title("DFT Vs. MLP")
    plt.xlabel("Battery number [-]")
    plt.ylabel("Gravimetric Energy [Wh/kg]")
    plt.legend()
    plt.show(block=False)  # 不阻塞程序的执行

    def get_mse(records_real, records_predict):
        if len(records_real) == len(records_predict):
            return sum(
                [((x - y) ** 2) for x, y in zip(records_real, records_predict)]
            ) / len(records_real)
        else:
            return None

    def get_rmse(records_real, records_predict):
        mse = get_mse(records_real, records_predict)
        if mse:
            return math.sqrt(mse)
        else:
            return None

    with open(filename) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        reals = []
        predicts = []
        for row in reader:
            real = float(row[1])
            reals.append(real)
            predict = float(row[2])
            predicts.append(predict)
    Rmse_V = get_rmse(reals, predicts)
    Rmses_V.append(Rmse_V)

    with open(filename) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        reals = []
        predicts = []
        for row in reader:
            real = float(row[3])
            reals.append(real)
            predict = float(row[4])
            predicts.append(predict)
    Rmse_C = get_rmse(reals, predicts)
    Rmses_C.append(Rmse_C)

    with open(filename) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        reals = []
        predicts = []
        for row in reader:
            real = float(row[5])
            reals.append(real)
            predict = float(row[6])
            predicts.append(predict)
    Rmse_E = get_rmse(reals, predicts)
    Rmses_E.append(Rmse_E)

    Rmse = (Rmse_V + Rmse_C + Rmse_E) / 3
    Rmses.append(Rmse)

print("\n\n")
print("V Rmse: ", Rmses_V)
print("C Rmse: ", Rmses_C)
print("E Rmse: ", Rmses_E)
print("Average Rmse: ", Rmses)
print("\n")

rmse_idx_V = np.argmin(Rmses_V)
rmse_min_V = np.amin(Rmses_V)
rmse_idx_C = np.argmin(Rmses_C)
rmse_min_C = np.amin(Rmses_C)
rmse_idx_E = np.argmin(Rmses_E)
rmse_min_E = np.amin(Rmses_E)
rmse_idx = np.argmin(Rmses)
rmse_min = np.amin(Rmses)

print("rmse_idx_V: ", rmse_idx_V)
print("rmse_min_V: ", rmse_min_V)
print("rmse_idx_C: ", rmse_idx_C)
print("rmse_min_C: ", rmse_min_C)
print("rmse_idx_E: ", rmse_idx_E)
print("rmse_min_E: ", rmse_min_E)
print("rmse_idx: ", rmse_idx)
print("rmse_min: ", rmse_min)

filename2 = "results_out/rmse_and_time.csv"
headers = [
    "rmse_idx_V",
    "rmse_min_V",
    "rmse_idx_C",
    "rmse_min_C",
    "rmse_idx_E",
    "rmse_min_E",
    "rmse_idx",
    "rmse_min",
    "optimizer.max",
]

with open(filename2, "w", newline="") as f:
    fcsv = csv.writer(f)
    fcsv.writerow(headers)
rows = [
    rmse_idx_V,
    rmse_min_V,
    rmse_idx_C,
    rmse_min_C,
    rmse_idx_E,
    rmse_min_E,
    rmse_idx,
    rmse_min,
    optimizer.max,
]

with open(filename2, "a", newline="") as f:
    fcsv = csv.writer(f)
    fcsv.writerow(rows)
