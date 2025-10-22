import os
import tempfile
import urllib.request
from os import path as osp

import hydra
import numpy as np
import optuna
import paddle
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import ppsci
from ppsci.constraint import SupervisedConstraint
from ppsci.optimizer import lr_scheduler
from ppsci.optimizer import optimizer
from ppsci.solver import Solver
from ppsci.validate import SupervisedValidator


def download_model_from_url(url, local_path=None):
    """
    Download model from URL to local path.

    Args:
        url (str): URL to download the model from
        local_path (str, optional): Local path to save the model.
                                   If None, saves to a temporary file.

    Returns:
        str: Path to the downloaded model file
    """
    if local_path is None:
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        local_path = os.path.join(temp_dir, "downloaded_model.pdparams")

    print(f"Downloading model from {url} to {local_path}")

    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"Successfully downloaded model to {local_path}")
        return local_path
    except Exception as e:
        raise RuntimeError(f"Failed to download model from {url}: {str(e)}")


def load_model_from_path_or_url(model_path):
    """
    Load model from local path or URL.

    Args:
        model_path (str): Local path or URL to the model

    Returns:
        dict: Loaded model dictionary
    """
    if model_path.startswith(("http://", "https://")):
        # It's a URL, download first
        local_path = download_model_from_url(model_path)
        return paddle.load(local_path)
    else:
        # It's a local path
        return paddle.load(model_path)


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
    """Create Tensor Dictionary for Input and Labels"""
    return {
        "input": paddle.to_tensor(X.values, dtype="float32"),
        "label": {"target": paddle.to_tensor(y.values, dtype="float32")},
    }


def create_constraint(input_dict, batch_size, shuffle=True):
    """Create supervision constraints"""
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
        name="train_constraint",
    )


def create_validator(input_dict, batch_size, name="validator"):
    """Create an evaluator"""
    return SupervisedValidator(
        dataloader_cfg={
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {"input": input_dict["input"]},
                "label": input_dict["label"],
            },
            "batch_size": batch_size,
        },
        loss=weighted_loss,
        output_expr={"target": lambda out: out["target"]},
        metric={"RMSE": ppsci.metric.RMSE(), "MAE": ppsci.metric.MAE()},
        name=name,
    )


def create_optimizer(model, optimizer_name, lr, train_cfg, data_size):
    """Create optimizer and learning rate scheduler"""
    schedule = lr_scheduler.ExponentialDecay(
        epochs=train_cfg.epochs,
        iters_per_epoch=data_size // train_cfg.batch_size,
        learning_rate=lr,
        gamma=train_cfg.lr_scheduler.gamma,
        decay_steps=train_cfg.lr_scheduler.decay_steps,
        warmup_epoch=train_cfg.lr_scheduler.warmup_epoch,
        warmup_start_lr=train_cfg.lr_scheduler.warmup_start_lr,
    )()

    if optimizer_name == "Adam":
        return optimizer.Adam(learning_rate=schedule)(model)
    elif optimizer_name == "RMSProp":
        return optimizer.RMSProp(learning_rate=schedule)(model)
    else:
        return optimizer.SGD(learning_rate=schedule)(model)


def define_model(trial, input_dim, output_dim):
    n_layers = trial.suggest_int("n_layers", 4, 6)
    hidden_sizes = []
    for i in range(n_layers):
        out_features = trial.suggest_int(f"n_units_l{i}", 10, input_dim // 2)
        hidden_sizes.append(out_features)

    model = ppsci.arch.MLP(
        input_keys=("input",),
        output_keys=("target",),
        num_layers=None,
        hidden_size=hidden_sizes,
        activation="relu",
        input_dim=input_dim,
        output_dim=output_dim,
    )
    return model


def train(cfg: DictConfig):
    # Read and preprocess data
    X_train = pd.read_csv(cfg.data.train_features_path)
    y_train = pd.read_csv(cfg.data.train_labels_path)
    X_val = pd.read_csv(cfg.data.val_features_path)
    y_val = pd.read_csv(cfg.data.val_labels_path)

    for col in X_train.columns:
        if "[" in col or "]" in col:
            old_name = col
            new_name = col.replace("[", "(").replace("]", ")")
            X_train = X_train.rename(columns={old_name: new_name})
            X_val = X_val.rename(columns={old_name: new_name})

    X_train, X_verif, y_train, y_verif = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    for df in [X_train, y_train, X_verif, y_verif, X_val, y_val]:
        df.reset_index(drop=True, inplace=True)

    def objective(trial):
        model = define_model(trial, cfg.model.input_dim, cfg.model.output_dim)

        optimizer_name = trial.suggest_categorical(
            "optimizer", ["Adam", "RMSProp", "SGD"]
        )
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

        train_dict = create_tensor_dict(X_train, y_train)
        verif_dict = create_tensor_dict(X_verif, y_verif)

        opt = create_optimizer(model, optimizer_name, lr, cfg.TRAIN, len(X_train))

        train_constraint = create_constraint(train_dict, cfg.TRAIN.batch_size)
        verif_validator = create_validator(
            verif_dict, cfg.EVAL.batch_size, "verif_validator"
        )

        solver = Solver(
            model=model,
            constraint={"train": train_constraint},
            optimizer=opt,
            validator={"verif": verif_validator},
            output_dir=cfg.output_dir,
            epochs=cfg.TRAIN.search_epochs,
            iters_per_epoch=len(X_train) // cfg.TRAIN.batch_size,
            eval_during_train=cfg.TRAIN.eval_during_train,
            eval_freq=cfg.TRAIN.eval_freq,
            save_freq=cfg.TRAIN.save_freq,
            eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
            log_freq=cfg.TRAIN.log_freq,
        )

        solver.train()

        verif_preds = solver.predict({"input": verif_dict["input"]}, return_numpy=True)[
            "target"
        ]

        verif_rmse = np.sqrt(mean_squared_error(y_verif.values, verif_preds))

        return verif_rmse

    study = optuna.create_study()
    study.optimize(objective, n_trials=50)

    best_params = study.best_trial.params
    print("\nBest hyperparameters: " + str(best_params))

    # Save the optimal model structure
    hidden_sizes = []
    for i in range(best_params["n_layers"]):
        hidden_sizes.append(best_params[f"n_units_l{i}"])

    # Create and train the final model
    final_model = define_model(
        study.best_trial, cfg.model.input_dim, cfg.model.output_dim
    )
    opt = create_optimizer(
        final_model,
        best_params["optimizer"],
        best_params["lr"],
        cfg.TRAIN,
        len(X_train),
    )

    train_dict = create_tensor_dict(X_train, y_train)
    val_dict = create_tensor_dict(X_val, y_val)

    train_constraint = create_constraint(train_dict, cfg.TRAIN.batch_size)
    val_validator = create_validator(val_dict, cfg.EVAL.batch_size, "val_validator")

    solver = Solver(
        model=final_model,
        constraint={"train": train_constraint},
        optimizer=opt,
        validator={"valid": val_validator},
        output_dir=cfg.output_dir,
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=len(X_train) // cfg.TRAIN.batch_size,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        save_freq=cfg.TRAIN.save_freq,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
        log_freq=cfg.TRAIN.log_freq,
    )

    solver.train()

    # Save model structure and weights
    model_dict = {
        "state_dict": final_model.state_dict(),
        "hidden_size": hidden_sizes,
        "n_layers": best_params["n_layers"],
        "optimizer": best_params["optimizer"],
        "lr": best_params["lr"],
    }
    paddle.save(
        model_dict, os.path.join(cfg.output_dir, "checkpoints", "best_model.pdparams")
    )
    print(
        "Saved model structure and weights to "
        + os.path.join(cfg.output_dir, "checkpoints", "best_model.pdparams")
    )

    solver.plot_loss_history(by_epoch=True, smooth_step=1)
    solver.eval()

    visualize_results(solver, X_val, y_val, cfg.output_dir)


def evaluate(cfg: DictConfig):
    # Read and preprocess data
    X_val = pd.read_csv(cfg.data.val_features_path)
    y_val = pd.read_csv(cfg.data.val_labels_path)

    for col in X_val.columns:
        if "[" in col or "]" in col:
            old_name = col
            new_name = col.replace("[", "(").replace("]", ")")
            X_val = X_val.rename(columns={old_name: new_name})

    # Loading model structure and weights
    print(f"Loading model from {cfg.EVAL.pretrained_model_path}")
    model_dict = load_model_from_path_or_url(cfg.EVAL.pretrained_model_path)
    hidden_size = model_dict["hidden_size"]
    print(f"Loaded model structure with hidden sizes: {hidden_size}")

    model = ppsci.arch.MLP(
        input_keys=("input",),
        output_keys=("target",),
        num_layers=None,
        hidden_size=hidden_size,
        activation="relu",
        input_dim=cfg.model.input_dim,
        output_dim=cfg.model.output_dim,
    )

    # Load model weights
    model.set_state_dict(model_dict["state_dict"])
    print("Successfully loaded model weights")

    valid_dict = create_tensor_dict(X_val, y_val)
    valid_validator = create_validator(
        valid_dict, cfg.EVAL.batch_size, "valid_validator"
    )

    solver = Solver(
        model=model,
        output_dir=cfg.output_dir,
        validator={"valid": valid_validator},
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )

    # evaluation model
    print("Evaluating model...")
    solver.eval()

    # Generate prediction results
    predictions = solver.predict({"input": valid_dict["input"]}, return_numpy=True)[
        "target"
    ]

    # Calculate multiple evaluation indicators
    rmse = np.sqrt(mean_squared_error(y_val.values, predictions))
    r2 = r2_score(y_val.values, predictions)
    mape = mean_absolute_percentage_error(y_val.values, predictions)

    print("Evaluation metrics:")
    print(f"RMSE: {rmse:.5f}")
    print(f"R2 Score: {r2:.5f}")
    print(f"MAPE: {mape:.5f}")

    # Visualization results
    print("Generating visualization...")
    visualize_results(solver, X_val, y_val, cfg.output_dir)
    print("Evaluation completed.")


def visualize_results(solver, X_val, y_val, output_dir):
    pred_dict = solver.predict(
        {"input": paddle.to_tensor(X_val.values, dtype="float32")}, return_numpy=True
    )
    val_preds = pred_dict["target"]
    val_true = y_val.values

    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.hist(val_true, bins=30, alpha=0.6, label="True Jsc", color="tab:blue")
    plt.hist(val_preds, bins=30, alpha=0.6, label="Predicted Jsc", color="orange")

    pred_mean = np.mean(val_preds)
    pred_std = np.std(val_preds)
    plt.axvline(pred_mean, color="black", linestyle="--")
    plt.axvline(pred_mean + pred_std, color="red", linestyle="--")
    plt.axvline(pred_mean - pred_std, color="red", linestyle="--")

    val_rmse = np.sqrt(mean_squared_error(val_true, val_preds))
    plt.title(f"Distribution of True Jsc vs Pred Jsc: RMSE {val_rmse:.5f}", pad=20)
    plt.xlabel("Jsc (mA/cm²)")
    plt.ylabel("Counts")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(
        osp.join(output_dir, "jsc_distribution.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


@hydra.main(version_base=None, config_path="./conf", config_name="psc_nn.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
