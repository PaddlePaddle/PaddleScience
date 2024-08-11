# get_ipython().run_line_magic('reset', '')
import optuna
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from scipy import stats
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
OUTPUT_TEST = True

# 获取当前脚本所在目录
current_dir = Path(__file__).resolve().parent

# 构建数据文件的完整路径
data_dir = current_dir.parent / "data" / "data" / "cleaned"
X_train_path = data_dir / "training.csv"
y_train_path = data_dir / "training_labels.csv"
X_val_path = data_dir / "validation.csv"
y_val_path = data_dir / "validation_labels.csv"
X_test_path = data_dir / "test.csv"
y_test_path = data_dir / "test_labels.csv"
# 读取数据
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
sanity_check = XGBRegressor()
sanity_check.fit(X_train, y_train)
val_preds = sanity_check.predict(X_val)
sanity_val_error = mean_squared_error(y_val, val_preds, squared=False)
val_true = y_val.to_numpy().squeeze()
sanity_val_r = r2_score(val_true, val_preds)
print("SANITY CHECK VALUES:")
print("Validation RMSE:", sanity_val_error)
print("Validation R:", sanity_val_r)
from sklearn.metrics import r2_score

sanity_val_error = mean_squared_error(y_val, val_preds, squared=False)
val_true = y_val.to_numpy().squeeze()
sanity_val_r = r2_score(val_true, val_preds)
print("SANITY CHECK VALUES:")
print("Validation RMSE:", sanity_val_error)
print("Validation R2 Score:", sanity_val_r)


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


sampler = optuna.samplers.CmaEsSampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=50)
print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
params = trial.params
model = XGBRegressor(**params)
print(params)
model.fit(X_train, y_train)

val_preds = model.predict(X_val)
train_preds = model.predict(X_train)
error = mean_squared_error(y_val, val_preds, squared=False)
r_error = r2_score(val_true, val_preds)
train_error = mean_squared_error(y_train, train_preds)
train_true = y_train.to_numpy().squeeze()
train_r_error = r2_score(train_true, train_preds)
print("Validation RMSE:", error)
print("Difference from sanity check:", error - sanity_val_error)
print("Validation R:", r_error)
print("Difference from sanity check:", r_error - sanity_val_r)
print("Validation PE", mean_absolute_percentage_error(val_true, val_preds))
print("Training RMSE:", train_error)
print("Training R:", train_r_error)
from sklearn.metrics import mean_squared_error, r2_score

val_preds = model.predict(X_val)
sanity_val_error = mean_squared_error(y_val, val_preds, squared=False)
val_true = y_val.to_numpy().squeeze()
sanity_val_r = r2_score(val_true, val_preds)
print("SANITY CHECK VALUES:")
print("Validation RMSE:", sanity_val_error)
print("Validation R2 Score:", sanity_val_r)
train_preds = model.predict(X_train)
error = mean_squared_error(y_val, val_preds, squared=False)
r_error = r2_score(val_true, val_preds)
train_error = mean_squared_error(y_train, train_preds)
train_true = y_train.squeeze()
print("Training RMSE:", train_error)
print("Training R2 Score:", r2_score(train_true, train_preds))
if not OUTPUT_TEST:
    raise ValueError(
        "OUTPUT_TEST set to False. If you would like to output final test values set to True and continue running from here"
    )
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)

columns = X_test.columns
for col in columns:
    if "[" in col or "]" in col:
        old_name = col
        col = col.replace("[", "(")
        col = col.replace("]", ")")
        X_test = X_test.rename(columns={old_name: col})
test_preds = model.predict(X_test)
train_preds = model.predict(X_train)
pred_data = pd.DataFrame(test_preds)
# 设置文件保存路径
predictions_dir = current_dir / "data" / "predictions" / "XG"
predictions_dir.mkdir(parents=True, exist_ok=True)

# 保存预测结果
pred_data = pd.DataFrame(test_preds)
pred_filepath = predictions_dir / "test_pred_xg.csv"
pred_data.to_csv(pred_filepath, index=False, header=False)

pred_data = pd.DataFrame(y_test)
pred_filepath = predictions_dir / "test_true_xg.csv"
pred_data.to_csv(pred_filepath, index=False, header=False)

pred_data = pd.DataFrame(train_preds)
pred_filepath = predictions_dir / "train_pred_xg.csv"
pred_data.to_csv(pred_filepath, index=False, header=False)

pred_data = pd.DataFrame(y_train)
pred_filepath = predictions_dir / "train_true_xg.csv"
pred_data.to_csv(pred_filepath, index=False, header=False)

pred_data = pd.DataFrame(X_train)
pred_filepath = predictions_dir / "train_input_xg.csv"
pred_data.to_csv(pred_filepath, index=False, header=False)

true_data = pd.DataFrame(X_test)
true_filepath = predictions_dir / "test_input_xg.csv"
true_data.to_csv(true_filepath, index=False, header=False)

# 加载预测结果
test_pred_data = np.genfromtxt(
    predictions_dir / "test_pred_xg.csv", delimiter=",", filling_values=np.nan
)
test_true_data = np.genfromtxt(
    predictions_dir / "test_true_xg.csv", delimiter=",", filling_values=np.nan
)
train_pred_data = np.genfromtxt(
    predictions_dir / "train_pred_xg.csv", delimiter=",", filling_values=np.nan
)
train_true_data = np.genfromtxt(
    predictions_dir / "train_true_xg.csv", delimiter=",", filling_values=np.nan
)

test_rmse = mean_squared_error(test_true_data, test_pred_data, squared=False)
test_r = r2_score(test_true_data, test_pred_data)
pearson_r = stats.pearsonr(test_true_data, test_pred_data)
train_rmse = mean_squared_error(train_true_data, train_pred_data, squared=False)
train_r = stats.pearsonr(train_true_data, train_pred_data)
print("Train:")
print(train_rmse)
print("Test:")
print(test_rmse)
print(test_r)
print(pearson_r)
print(
    "percent Error:",
    mean_absolute_percentage_error(test_true_data, test_pred_data) * 100,
)
split_df = pd.DataFrame({"true": test_true_data, "pred": test_pred_data})
split_df = split_df.sort_values(by="true")
split_df.reset_index(inplace=True, drop=True)
mid = (max(test_true_data) + min(test_true_data)) / 2
diff = 1000
idx = -1
for i in range(len(split_df)):
    new_diff = abs(split_df.iloc[i]["true"] - mid)
    if new_diff <= diff:
        diff = new_diff
        idx = i
print(len(split_df.iloc[idx:]["true"]) / len(split_df))
top_half_true = split_df.iloc[idx:]["true"].to_numpy().squeeze()
top_half_pred = split_df.iloc[idx:]["pred"].to_numpy().squeeze()
print(
    "adjusted percent Error:",
    mean_absolute_percentage_error(top_half_true, top_half_pred) * 100,
)
print("SANITY CHECK VALUES:")
print("Validation RMSE:", sanity_val_error)
print("Validation R2 Score:", sanity_val_r)
print("Training RMSE:", train_error)
train_r2 = r2_score(y_train, train_preds)
print("Training R2 Score:", train_r2)

# 使用模型预测测试集
test_preds = model.predict(X_test)
test_rmse = mean_squared_error(y_test, test_preds, squared=False)
test_r2 = r2_score(y_test, test_preds)
adjusted_percent_error = test_rmse / y_test.mean() * 100


print(f"Test RMSE: {test_rmse}")
print(f"Test R2 Score: {test_r2}")
print(f"Adjusted Percent Error: {adjusted_percent_error}")
from sklearn.metrics import mean_squared_error, r2_score

val_preds = model.predict(X_val)
sanity_val_error = mean_squared_error(y_val, val_preds, squared=False)
sanity_val_r = r2_score(y_val, val_preds)
print("SANITY CHECK VALUES:")
print("Validation RMSE:", sanity_val_error)
print("Validation R2 Score:", sanity_val_r)
train_preds = model.predict(X_train)
train_error = mean_squared_error(y_train, train_preds, squared=False)
train_r2 = r2_score(y_train, train_preds)
print("Training RMSE:", train_error)
print("Training R2 Score:", train_r2)
test_preds = model.predict(X_test)
test_rmse = mean_squared_error(y_test, test_preds, squared=False)
test_r2 = r2_score(y_test, test_preds)
adjusted_percent_error = test_rmse / y_test.mean() * 100
print(f"Test RMSE: {test_rmse}")
print(f"Test R2 Score: {test_r2}")
print(f"Adjusted Percent Error: {adjusted_percent_error}")
