import os
import paddle
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper:
"DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks".

This script trains and evaluates graph-based neural networks, specifically DragGNN models, for predicting aerodynamic drag.
"""
import time
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Dataset, DataLoader as GeoDataLoader
from tqdm import tqdm
import numpy as np
from DeepSurrogate_models import DragGNN, DragGNN_XL, EnhancedDragGNN
from DrivAerNetDataset import DrivAerNetGNNDataset
import pandas as pd
config = {'cuda': True, 'exp_name':
    'DragPrediction_DrivAerNet_DragGNN_100epochs_NeurIPS', 'seed': 1,
    'batch_size': 1, 'epochs': 100, 'lr': 0.001, 'optimizer': 'adam',
    'dataset_path': '../DrivAerNet_FEN_STLs', 'aero_coeff':
    '../Combined_AeroCoefficients_DrivAerNet.csv', 'subset_dir':
    '../subset_dir'}
device = str('cuda' if paddle.device.cuda.device_count() >= 1 and config[
    'cuda'] else 'cpu').replace('cuda', 'gpu')


def setup_seed(seed: int):
    """Set the seed for reproducibility."""
    paddle.seed(seed=seed)
    paddle.seed(seed=seed)
    np.random.seed(seed)


def r2_score(output, target):
    """Compute R-squared score."""
    target_mean = paddle.mean(x=target)
    ss_tot = paddle.sum(x=(target - target_mean) ** 2)
    ss_res = paddle.sum(x=(target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def initialize_model(config: dict) ->paddle.nn.Layer:
    model = DragGNN_XL().to(device)
    if config['cuda'] and paddle.device.cuda.device_count() > 1:
        model = paddle.DataParallel(layers=model)
    return model


def get_dataloaders(dataset_path: str, aero_coeff: str, subset_dir: str,
    batch_size: int) ->tuple:
    """
    Prepare and return the training, validation, and test DataLoader objects.

    Args:
        dataset_path (str): The file path to the dataset directory containing the STL files.
        aero_coeff (str): The path to the CSV file with metadata for the models.
        subset_dir (str): The directory containing the subset files (train, val, test).
        num_points (int): The number of points to sample from each point cloud in the dataset.
        batch_size (int): The number of samples per batch to load.

    Returns:
        tuple: A tuple containing the training DataLoader, validation DataLoader, and test DataLoader.
    """
    full_dataset = DrivAerNetGNNDataset(root_dir=dataset_path, csv_file=
        aero_coeff)

    def create_subset(dataset, ids_file):
        try:
            with open(os.path.join(subset_dir, ids_file), 'r') as file:
                subset_ids = file.read().split()
            subset_indices = dataset.data_frame[dataset.data_frame['Design'
                ].isin(subset_ids)].index.tolist()
            return paddle.io.Subset(dataset=dataset, indices=subset_indices)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f'Error loading subset file {ids_file}: {e}')
    train_dataset = create_subset(full_dataset, 'train_design_ids.txt')
    val_dataset = create_subset(full_dataset, 'val_design_ids.txt')
    test_dataset = create_subset(full_dataset, 'test_design_ids.txt')
    train_dataloader = GeoDataLoader(train_dataset, batch_size=batch_size,
        shuffle=True, drop_last=True, num_workers=4)
    val_dataloader = GeoDataLoader(val_dataset, batch_size=batch_size,
        shuffle=False, drop_last=True, num_workers=4)
    test_dataloader = GeoDataLoader(test_dataset, batch_size=batch_size,
        shuffle=False, drop_last=True, num_workers=4)
    return train_dataloader, val_dataloader, test_dataloader


def train_and_evaluate(model: paddle.nn.Layer, train_dataloader: paddle.io.
    DataLoader, val_dataloader: paddle.io.DataLoader, config: dict):
    """
    Train and evaluate the model using the provided dataloaders and configuration.

    Args:
        model (torch.nn.Module): The model to be trained and evaluated.
        train_dataloader (DataLoader): Dataloader for the training set.
        val_dataloader (DataLoader): Dataloader for the validation set.
        config (dict): Configuration dictionary containing training hyperparameters and settings.

    """
    train_losses, val_losses = [], []
    training_start_time = time.time()
    device = str('cuda' if paddle.device.cuda.device_count() >= 1 and
        config['cuda'] else 'cpu').replace('cuda', 'gpu')
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
        learning_rate=config['lr'], weight_decay=0.0001) if config['optimizer'
>>>>>>        ] == 'adam' else torch.optim.SGD(model.parameters(), lr=config['lr'
        ], momentum=0.9, weight_decay=0.0001)
    tmp_lr = paddle.optimizer.lr.ReduceOnPlateau(mode='min', patience=20,
        factor=0.1, verbose=True, learning_rate=optimizer.get_lr())
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
    best_mse = float('inf')
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()
        total_loss, total_r2 = 0, 0
        for data in tqdm(train_dataloader, desc=
            f"Epoch {epoch + 1}/{config['epochs']} [Training]"):
            data = data.to(device)
            optimizer.clear_gradients(set_to_zero=False)
            outputs = model(data)
            loss = paddle.nn.functional.mse_loss(input=outputs.squeeze(),
                label=data.y)
            r2 = r2_score(outputs.squeeze(), data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_r2 += r2.item()
        epoch_duration = time.time() - epoch_start_time
        avg_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_loss)
        print(
            f'Epoch {epoch + 1} Training Loss: {avg_loss:.6f} Time: {epoch_duration:.2f}s'
            )
        avg_r2 = total_r2 / len(train_dataloader)
        print(f'Average Training R²: {avg_r2:.4f}')
        model.eval()
        val_loss, val_r2 = 0, 0
        inference_times = []
        with paddle.no_grad():
            for data in tqdm(val_dataloader, desc=
                f"Epoch {epoch + 1}/{config['epochs']} [Validation]"):
                inference_start_time = time.time()
                data = data.to(device)
                outputs = model(data)
                loss = paddle.nn.functional.mse_loss(input=outputs.squeeze(
                    ), label=data.y)
                val_loss += loss.item()
                r2 = r2_score(outputs.squeeze(), data.y)
                val_r2 += r2.item()
                inference_duration = time.time() - inference_start_time
                inference_times.append(inference_duration)
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        avg_inference_time = sum(inference_times) / len(inference_times)
        print(
            f'Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}, Avg Inference Time: {avg_inference_time:.4f}s'
            )
        avg_val_r2 = val_r2 / len(val_dataloader)
        print(f'Average Validation R²: {avg_val_r2:.4f}')
        if avg_val_loss < best_mse:
            best_mse = avg_val_loss
            best_model_path = os.path.join('models',
                f"{config['exp_name']}_best_model.pth")
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            paddle.save(obj=model.state_dict(), path=best_model_path)
            print(
                f'New best model saved with MSE: {best_mse:.6f} and R²: {avg_val_r2:.4f}'
                )
        scheduler.step(avg_val_loss)
    training_duration = time.time() - training_start_time
    print(f'Total training time: {training_duration:.2f}s')
    model_path = os.path.join('models', f"{config['exp_name']}_final_model.pth"
        )
    paddle.save(obj=model.state_dict(), path=model_path)
    print(f'Model saved to {model_path}')
    np.save(os.path.join('models', f"{config['exp_name']}_train_losses.npy"
        ), np.array(train_losses))
    np.save(os.path.join('models', f"{config['exp_name']}_val_losses.npy"),
        np.array(val_losses))


def test_model(model: paddle.nn.Layer, test_dataloader: paddle.io.
    DataLoader, config: dict):
    """
    Test the model using the provided test DataLoader and calculate different metrics.

    Args:
        model (torch.nn.Module): The trained model to be tested.
        test_dataloader (DataLoader): DataLoader for the test set.
        config (dict): Configuration dictionary containing model settings.

    """
    model.eval()
    total_mse, total_mae, total_r2 = 0, 0, 0
    max_mae = 0
    total_inference_time = 0
    total_samples = 0
    with paddle.no_grad():
        for data in test_dataloader:
            start_time = time.time()
            data = data.to(device)
            outputs = model(data)
            end_time = time.time()
            inference_time = end_time - start_time
            total_inference_time += inference_time
            mse = paddle.nn.functional.mse_loss(input=outputs.squeeze(),
                label=data.y)
            mae = paddle.nn.functional.l1_loss(input=outputs.squeeze(),
                label=data.y)
            r2 = r2_score(outputs.squeeze(), data.y)
            total_mse += mse.item()
            total_mae += mae.item()
            total_r2 += r2.item()
            max_mae = max(max_mae, mae.item())
            total_samples += data.y.size(0)
    avg_mse = total_mse / len(test_dataloader)
    avg_mae = total_mae / len(test_dataloader)
    avg_r2 = total_r2 / len(test_dataloader)
    print(
        f'Test MSE: {avg_mse:.6f}, Test MAE: {avg_mae:.6f}, Max MAE: {max_mae:.6f}, Test R²: {avg_r2:.4f}'
        )
    print(
        f'Total inference time: {total_inference_time:.2f}s for {total_samples} samples'
        )


def load_and_test_model(model_path, test_dataloader, device):
    """Load a saved model and test it."""
    model = DragGNN_XL().to(device)
    model.set_state_dict(state_dict=paddle.load(path=str(model_path)))
    test_model(model, test_dataloader, config)


if __name__ == '__main__':
    setup_seed(config['seed'])
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config
        ['dataset_path'], config['aero_coeff'], config['subset_dir'],
        config['batch_size'])
    model = DragGNN_XL().to(device)
    train_and_evaluate(model, train_dataloader, val_dataloader, config)
    final_model_path = os.path.join('models',
        f"{config['exp_name']}_final_model.pth")
    print('Testing the final model:')
    load_and_test_model(final_model_path, test_dataloader, device)
    best_model_path = os.path.join('models',
        f"{config['exp_name']}_best_model.pth")
    print('Testing the best model:')
    load_and_test_model(best_model_path, test_dataloader, device)
