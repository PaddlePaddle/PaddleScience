import pickle

import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import ppsci


# 数据预处理函数
def preprocess_data(file_path):
    # 读取数据
    df = pd.read_excel(file_path)

    # 编码污染类型
    label_encoder = LabelEncoder()
    df["pollution_type"] = label_encoder.fit_transform(df["pollution_type"])

    # 特征和标签
    X = df[["PM2.5", "PM10", "SO2", "NO2", "CO"]].values
    y = df["pollution_type"].values

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, label_encoder


# 模型训练函数
def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    class_weights,
    label_classes,
    batch_size=32,
    epochs=100,
    patience=10,
):
    # 转换为 Paddle tensor
    X_train_tensor = paddle.to_tensor(X_train, dtype="float32")
    y_train_tensor = paddle.to_tensor(y_train, dtype="int64")
    X_test_tensor = paddle.to_tensor(X_test, dtype="float32")
    y_test_tensor = paddle.to_tensor(y_test, dtype="int64")

    # 使用 MLP 模型
    model = ppsci.arch.MLP(
        input_keys=["input"],  # 输入键
        output_keys=["output"],  # 输出键
        input_dim=5,  # 输入特征维度
        output_dim=len(label_classes),  # 输出分类数
        num_layers=3,  # 网络层数
        hidden_size=64,  # 隐藏层单元数量
        activation="ReLU",  # 激活函数
    )

    # 定义损失函数和优化器
    class_weights_tensor = paddle.to_tensor(class_weights, dtype="float32")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scheduler = paddle.optimizer.lr.StepDecay(
        learning_rate=0.001, step_size=50, gamma=0.5
    )
    optimizer = optim.Adam(parameters=model.parameters(), learning_rate=scheduler)

    # 早停参数
    best_val_loss = float("inf")
    early_stop_count = 0

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(len(X_train_tensor))
        X_train_tensor = X_train_tensor[indices]
        y_train_tensor = y_train_tensor[indices]

        # 批量训练
        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor[i : i + batch_size]
            y_batch = y_train_tensor[i : i + batch_size]

            # 前向传播
            logits = model({"input": X_batch})["output"]
            loss = loss_fn(logits, y_batch)

            # 反向传播和优化
            loss.backward()
            optimizer.step()
            optimizer.clear_gradients()

        # 验证阶段
        model.eval()
        with paddle.no_grad():
            val_logits = model({"input": X_test_tensor})["output"]
            val_loss = loss_fn(val_logits, y_test_tensor).numpy()
            val_predictions = paddle.argmax(val_logits, axis=1).numpy()
            val_accuracy = np.mean(val_predictions == y_test)

        print(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.numpy():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%"
        )

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
            paddle.save(model.state_dict(), "best_model.pdparams")
        else:
            early_stop_count += 1
            if early_stop_count > patience:
                print("早停机制触发，停止训练")
                break

    return model


# 模型评估和保存函数
def evaluate_and_save_model(model, X_test, y_test, label_encoder, scaler):
    # 测试集评估
    model.eval()
    with paddle.no_grad():
        X_test_tensor = paddle.to_tensor(X_test, dtype="float32")
        test_logits = model({"input": X_test_tensor})["output"]
        test_predictions = paddle.argmax(test_logits, axis=1).numpy()

    # 打印分类报告
    print(
        classification_report(
            y_test, test_predictions, target_names=label_encoder.classes_
        )
    )

    # 保存模型和处理器
    paddle.save(model.state_dict(), "pollution_model.pdparams")
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)


# 模型推理函数
def predict_pollution(sample, model, scaler, label_encoder):
    sample_scaled = scaler.transform(sample)
    sample_tensor = paddle.to_tensor(sample_scaled, dtype="float32")

    with paddle.no_grad():
        prediction = model({"input": sample_tensor})["output"]
        predicted_class = paddle.argmax(prediction, axis=1).numpy()[0]

    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label


# 主程序
if __name__ == "__main__":
    # 数据预处理
    X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_data(
        "./trainData.xlsx"
    )

    # 计算类别权重
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )

    # 模型训练
    model = train_model(
        X_train, y_train, X_test, y_test, class_weights, label_encoder.classes_
    )

    # 模型评估和保存
    evaluate_and_save_model(model, X_test, y_test, label_encoder, scaler)

    # 加载测试样本
    test_sample = [[28, 52, 3, 46, 0.5]]  # 样本数据
    predicted_label = predict_pollution(test_sample, model, scaler, label_encoder)
    print(f"预测的污染类型为: {predicted_label}")
