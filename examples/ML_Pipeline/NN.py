import os
from os import path as osp
import hydra
from omegaconf import DictConfig
import numpy as np
import paddle
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from matplotlib import pyplot as plt

import ppsci
from ppsci.utils import logger
from ppsci.optimizer import lr_scheduler, optimizer 
from ppsci.constraint import SupervisedConstraint
from ppsci.validate import SupervisedValidator
from ppsci.solver import Solver

def weighted_loss(output_dict, target_dict, weight_dict=None):
   pred = output_dict["target"]
   true = target_dict["target"]
   epsilon = 1e-06
   n = len(true)
   weights = true / (paddle.sum(x=true) + epsilon)
   squared = (true - pred) ** 2
   weighted = squared * weights
   loss = paddle.sum(x=weighted) / n
   return {"weighted_mse": loss}

def create_tensor_dict(X, y):
   """创建输入和标签的tensor字典"""
   return {
       "input": paddle.to_tensor(X.values, dtype='float32'),
       "label": {"target": paddle.to_tensor(y.values, dtype='float32')}
   }

def create_constraint(input_dict, batch_size, shuffle=True):
   """创建监督约束"""
   return SupervisedConstraint(
       dataloader_cfg={
           "dataset": {
               "name": "NamedArrayDataset",
               "input": {"input": input_dict["input"]},
               "label": input_dict["label"],
           },
           "batch_size": batch_size,
           "sampler": {
               "name": "BatchSampler",
               "drop_last": False,
               "shuffle": shuffle,
           },
       },
       loss=weighted_loss,
       output_expr={"target": lambda out: out["target"]},
       name="train_constraint"
   )

def create_validator(input_dict, batch_size, name="validator"):
   """创建评估器"""
   return SupervisedValidator(
       dataloader_cfg={
           "dataset": {
               "name": "NamedArrayDataset",
               "input": {"input": input_dict["input"]},
               "label": input_dict["label"],
           },
           "batch_size": batch_size,
           "sampler": {
               "name": "BatchSampler",
               "drop_last": False,
               "shuffle": False,
           },
       },
       loss=weighted_loss,
       output_expr={"target": lambda out: out["target"]},
       metric={
           "RMSE": ppsci.metric.RMSE(),
           "MAE": ppsci.metric.MAE()
       },
       name=name
   )

def create_optimizer(model, optimizer_name, lr, train_cfg, data_size):
   """创建优化器和学习率调度器"""
   schedule = lr_scheduler.ExponentialDecay(
       epochs=train_cfg.epochs,
       iters_per_epoch=data_size // train_cfg.batch_size,
       learning_rate=lr,
       gamma=0.95,
       decay_steps=5,
       warmup_epoch=2,
       warmup_start_lr=1.0e-6
   )()
   
   if optimizer_name == 'Adam':
       return optimizer.Adam(learning_rate=schedule)(model)
   elif optimizer_name == 'RMSProp':
       return optimizer.RMSProp(learning_rate=schedule)(model)
   else:
       return optimizer.SGD(learning_rate=schedule)(model)

def define_model(trial, input_dim, output_dim):
   n_layers = trial.suggest_int('n_layers', 4, 6)
   hidden_sizes = []
   for i in range(n_layers):
       out_features = trial.suggest_int(f'n_units_l{i}', 10, input_dim // 2)
       hidden_sizes.append(out_features)
   
   model = ppsci.arch.MLP(
       input_keys=("input",),
       output_keys=("target",),
       num_layers=None,
       hidden_size=hidden_sizes,
       activation="relu",
       input_dim=input_dim,
       output_dim=output_dim
   )
   return model

def train(cfg: DictConfig):
    print("Starting training...")
    ppsci.utils.misc.set_random_seed(cfg.seed)
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")
    
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    # 读取和预处理数据
    X_train = pd.read_csv(cfg.data.train_features_path)
    y_train = pd.read_csv(cfg.data.train_labels_path)
    X_val = pd.read_csv(cfg.data.val_features_path)
    y_val = pd.read_csv(cfg.data.val_labels_path)

    for col in X_train.columns:
        if '[' in col or ']' in col:
            old_name = col
            new_name = col.replace('[', '(').replace(']', ')')
            X_train = X_train.rename(columns={old_name: new_name})
            X_val = X_val.rename(columns={old_name: new_name})
            
    X_train, X_verif, y_train, y_verif = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42)
    
    for df in [X_train, y_train, X_verif, y_verif, X_val, y_val]:
        df.reset_index(drop=True, inplace=True)

    def objective(trial):
        model = define_model(trial, cfg.model.input_dim, cfg.model.output_dim)
        
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSProp', 'SGD'])
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        
        train_dict = create_tensor_dict(X_train, y_train)
        verif_dict = create_tensor_dict(X_verif, y_verif)
        
        opt = create_optimizer(model, optimizer_name, lr, cfg.train, len(X_train))
        
        train_constraint = create_constraint(train_dict, cfg.train.batch_size)
        verif_validator = create_validator(verif_dict, cfg.eval.batch_size, "verif_validator")

        solver = Solver(
            model=model,
            constraint={"train": train_constraint},
            optimizer=opt,
            validator={"verif": verif_validator},
            output_dir=cfg.output_dir,
            epochs=cfg.train.epochs,
            iters_per_epoch=len(X_train) // cfg.train.batch_size,
            eval_during_train=True,
            eval_freq=5,
            save_freq=10,
            eval_with_no_grad=True,
            log_freq=50,
        )
        
        solver.train()
        
        verif_preds = solver.predict(
            {"input": verif_dict["input"]},
            return_numpy=True
        )["target"]
        
        verif_rmse = np.sqrt(mean_squared_error(y_verif.values, verif_preds))
        
        return verif_rmse

    study = optuna.create_study()
    study.optimize(objective, n_trials=50)
    
    best_params = study.best_trial.params
    print(f"\nBest hyperparameters: {best_params}")
    
    # 保存最优模型结构
    hidden_sizes = []
    for i in range(best_params['n_layers']):
        hidden_sizes.append(best_params[f'n_units_l{i}'])
    
    # 创建并训练最终模型
    final_model = define_model(study.best_trial, cfg.model.input_dim, cfg.model.output_dim)
    opt = create_optimizer(final_model, best_params['optimizer'], best_params['lr'], cfg.train, len(X_train))
    
    train_dict = create_tensor_dict(X_train, y_train)
    val_dict = create_tensor_dict(X_val, y_val)
    
    train_constraint = create_constraint(train_dict, cfg.train.batch_size)
    val_validator = create_validator(val_dict, cfg.eval.batch_size, "val_validator")

    solver = Solver(
        model=final_model,
        constraint={"train": train_constraint},
        optimizer=opt,
        validator={"valid": val_validator},
        output_dir=cfg.output_dir,
        epochs=cfg.train.epochs,
        iters_per_epoch=len(X_train) // cfg.train.batch_size,
        eval_during_train=cfg.train.eval_during_train,
        eval_freq=5,
        save_freq=10,
        eval_with_no_grad=cfg.eval.eval_with_no_grad,
        log_freq=50,
    )

    solver.train()
    
    # 保存模型结构和权重
    model_dict = {
        'state_dict': final_model.state_dict(),
        'hidden_size': hidden_sizes,
        'n_layers': best_params['n_layers'],
        'optimizer': best_params['optimizer'],
        'lr': best_params['lr']
    }
    paddle.save(model_dict, os.path.join(cfg.output_dir, 'checkpoints', 'best_model.pdparams'))
    print(f"Saved model structure and weights to {os.path.join(cfg.output_dir, 'checkpoints', 'best_model.pdparams')}")
    
    solver.plot_loss_history(by_epoch=True, smooth_step=1)
    solver.eval()
    
    visualize_results(solver, X_val, y_val, cfg.output_dir)

def evaluate(cfg: DictConfig):
    print("Starting evaluation...")
    ppsci.utils.misc.set_random_seed(cfg.seed)
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # 读取和预处理数据
    X_val = pd.read_csv(cfg.data.val_features_path)
    y_val = pd.read_csv(cfg.data.val_labels_path)
    
    for col in X_val.columns:
        if '[' in col or ']' in col:
            old_name = col
            new_name = col.replace('[', '(').replace(']', ')')
            X_val = X_val.rename(columns={old_name: new_name})

    # 加载模型结构和权重
    print(f"Loading model from {cfg.eval.pretrained_model_path}")
    model_dict = paddle.load(cfg.eval.pretrained_model_path)
    hidden_size = model_dict['hidden_size']
    print(f"Loaded model structure with hidden sizes: {hidden_size}")
    
    model = ppsci.arch.MLP(
        input_keys=("input",),
        output_keys=("target",),
        num_layers=None,
        hidden_size=hidden_size,
        activation="relu",
        input_dim=cfg.model.input_dim,
        output_dim=cfg.model.output_dim
    )
    
    # 加载模型权重
    model.set_state_dict(model_dict['state_dict'])
    print("Successfully loaded model weights")

    valid_dict = create_tensor_dict(X_val, y_val)
    valid_validator = create_validator(valid_dict, cfg.eval.batch_size, "valid_validator")

    solver = Solver(
        model=model,
        output_dir=cfg.output_dir,
        validator={"valid": valid_validator},
        eval_with_no_grad=cfg.eval.eval_with_no_grad,
    )

    # 评估模型
    print("Evaluating model...")
    solver.eval()
    
    # 生成预测结果
    predictions = solver.predict(
        {"input": valid_dict["input"]},
        return_numpy=True
    )["target"]
    
    # 计算多个评估指标
    rmse = np.sqrt(mean_squared_error(y_val.values, predictions))
    r2 = r2_score(y_val.values, predictions)
    mape = mean_absolute_percentage_error(y_val.values, predictions)
    
    print(f"Evaluation metrics:")
    print(f"RMSE: {rmse:.5f}")
    print(f"R2 Score: {r2:.5f}")
    print(f"MAPE: {mape:.5f}")
    
    # 可视化结果
    print("Generating visualization...")
    visualize_results(solver, X_val, y_val, cfg.output_dir)
    print("Evaluation completed.")

def visualize_results(solver, X_val, y_val, output_dir):
   pred_dict = solver.predict(
       {"input": paddle.to_tensor(X_val.values, dtype='float32')},
       return_numpy=True
   )
   val_preds = pred_dict["target"]
   val_true = y_val.values

   plt.figure(figsize=(10, 6))
   plt.grid(True, linestyle='--', alpha=0.7)
   plt.hist(val_true, bins=30, alpha=0.6, label='True Jsc', color='tab:blue')
   plt.hist(val_preds, bins=30, alpha=0.6, label='Predicted Jsc', color='orange')

   pred_mean = np.mean(val_preds)
   pred_std = np.std(val_preds)
   plt.axvline(pred_mean, color='black', linestyle='--')
   plt.axvline(pred_mean + pred_std, color='red', linestyle='--')
   plt.axvline(pred_mean - pred_std, color='red', linestyle='--')

   val_rmse = np.sqrt(mean_squared_error(val_true, val_preds))
   plt.title(f'Distribution of True Jsc vs Pred Jsc: RMSE {val_rmse:.5f}', pad=20)
   plt.xlabel('Jsc (mA/cm²)')
   plt.ylabel('Counts')
   plt.legend(fontsize=10)
   plt.tight_layout()
   plt.savefig(osp.join(output_dir, 'jsc_distribution.png'), dpi=300, bbox_inches='tight')
   plt.close()

@hydra.main(version_base=None, config_path="./conf", config_name="nn_optuna_ppsci.yaml")
def main(cfg: DictConfig):
   if cfg.mode == "train":
       train(cfg)
   elif cfg.mode == "eval":
       evaluate(cfg)
   else:
       raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")

if __name__ == "__main__":
   main()