import paddle
# 导入必要的模块
import optuna
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
OUTPUT_TEST = True

# 获取当前脚本所在目录
current_dir = Path(__file__).resolve().parent

# 构建数据文件的完整路径
X_train_path = "C:/Users/ssm18/new0811/PaddleScience/examples/ML_Pipeline/data/data/cleaned/training.csv"
y_train_path = "C:/Users/ssm18/new0811/PaddleScience/examples/ML_Pipeline/data/data/cleaned/training_labels.csv"
X_val_path = "C:/Users/ssm18/new0811/PaddleScience/examples/ML_Pipeline/data/data/cleaned/validation.csv"
y_val_path = "C:/Users/ssm18/new0811/PaddleScience/examples/ML_Pipeline/data/data/cleaned/validation_labels.csv"
X_test_path = "C:/Users/ssm18/new0811/PaddleScience/examples/ML_Pipeline/data/data/cleaned/test.csv"
y_test_path = "C:/Users/ssm18/new0811/PaddleScience/examples/ML_Pipeline/data/data/cleaned/test_labels.csv"

import os

print("X_train_path exists:", os.path.exists(X_train_path))

# 读取数据并处理
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)
X_val = pd.read_csv(X_val_path)
y_val = pd.read_csv(y_val_path)

columns = X_train.columns
for col in columns:
    if "[" in col or "]" in col:
        old_name = col
        col = col.replace("[", "(")
        col = col.replace("]", ")")
        X_train = X_train.rename(columns={old_name: col})
        X_val = X_val.rename(columns={old_name: col})

X_train, X_verif, y_train, y_verif = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_verif = X_verif.reset_index(drop=True)
y_verif = y_verif.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)

# 使用 Optuna 进行超参数优化
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 15),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_loguniform("gamma", 1e-08, 1.0),
        "subsample": trial.suggest_loguniform("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.5, 0.9),
    }
    params["tree_method"] = "hist"
    optuna_model = XGBRegressor(**params)
    optuna_model.fit(X_train, y_train)
    verif_pred = optuna_model.predict(X_verif)
    verif_loss = mean_absolute_percentage_error(y_verif, verif_pred) * 100
    verif_error = mean_squared_error(y_verif, verif_pred, squared=False)
    error = verif_loss + verif_error
    return error

# 开始超参数优化
sampler = optuna.samplers.CmaEsSampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=50)

# 获取最佳参数
best_params = study.best_trial.params
print("Best trial:")
print("  Value: ", study.best_trial.value)
print("  Params: ")
for key, value in best_params.items():
    print(f"    {key}: {value}")

# 使用最佳参数初始化 XGBoost 模型
model = XGBRegressor(**best_params)
model.fit(X_train, y_train)

# 预测验证集
val_preds = model.predict(X_val)
val_loss = mean_squared_error(y_val, val_preds, squared=False)
print(f"Validation RMSE: {val_loss}")

# 加载测试数据
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)
# 对测试数据进行相同的列名处理，确保特征名称一致
columns_test = X_test.columns
for col in columns_test:
    if "[" in col or "]" in col:
        old_name = col
        col = col.replace("[", "(")
        col = col.replace("]", ")")
        X_test = X_test.rename(columns={old_name: col})

# 检查是否与训练集的列一致
X_test = X_test[X_train.columns]

# 将测试数据转换为 Paddle Tensor
test_inputs = {"x": paddle.to_tensor(X_test.values).astype("float32")}
test_labels = {"y": paddle.to_tensor(y_test.values).astype("float32")}
# 使用模型预测测试集
test_preds = model.predict(X_test)

# 计算测试集上的评估指标
test_rmse = mean_squared_error(y_test, test_preds, squared=False)
test_r2 = r2_score(y_test, test_preds)
adjusted_percent_error = test_rmse / y_test.mean() * 100

# 打印测试集结果
print(f"Test RMSE: {test_rmse}")
print(f"Test R2 Score: {test_r2}")
print(f"Adjusted Percent Error: {adjusted_percent_error}")

# 保存预测结果
predictions_dir = current_dir / "data" / "predictions" / "XG"
predictions_dir.mkdir(parents=True, exist_ok=True)

# 保存预测和真实值
pd.DataFrame(test_preds).to_csv(predictions_dir / "test_pred_xg.csv", index=False, header=False)
pd.DataFrame(y_test).to_csv(predictions_dir / "test_true_xg.csv", index=False, header=False)
