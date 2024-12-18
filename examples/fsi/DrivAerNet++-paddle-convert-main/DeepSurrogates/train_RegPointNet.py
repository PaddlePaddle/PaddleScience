import os
import paddle

"""

@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu, mohamed.elrefaie@tum.de

This module is part of the research presented in the paper:
"DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks".

"""
import numpy as np
import time
from tqdm import tqdm
from DeepSurrogate_models import RegPointNet
import pandas as pd
from DrivAerNetDataset import DrivAerNetDataset

config = {'exp_name':
              'DragPrediction_DrivAerNet_PointNet_r2_batchsize4_200epochs_100kpoints_tsne_NeurIPS',
          'cuda': True,
          'seed': 1,
          'num_points': 100000,
          'lr': 0.001,
          'batch_size': 4,
          'epochs': 200,
          'dropout': 0.0,
          'emb_dims': 1024,
          'k': 100,
          'optimizer': 'adam',
          'channels': [6, 64, 128, 256, 512, 1024],
          'linear_sizes': [128, 64, 32, 16],
          'output_channels': 1,
          'dataset_path': '../DrivAerNetPlusPlus_Processed_Point_Clouds_100k_paddle',
          'aero_coeff': '../DrivAerNetPlusPlus_Drag_8k.csv',
          'subset_dir': '../subset_dir'}

# 设置可见的 GPU 设备
os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'

# 根据配置选择设备
if paddle.device.cuda.device_count() >= 1 and config['cuda']:
    device = 'gpu'
else:
    device = 'cpu'

# 打印当前使用的设备
print(f"Using device: {device}")


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


def initialize_model(config: dict) -> paddle.nn.Layer:
    """
    Initialize and return the RegDGCNN model.
    Args:
        config (dict): A dictionary containing configuration parameters for the model, including:
            - k: The number of nearest neighbors to consider in the graph construction.
            - channels: A list defining the number of output channels for each graph convolutional layer.
            - linear_sizes: A list defining the sizes of each linear layer following the convolutional layers.
            - emb_dims: The size of the global feature vector obtained after the graph convolutional and pooling layers.
            - dropout: The dropout rate applied after each linear layer for regularization.
            - output_channels: The number of output channels in the final layer, equal to the number of prediction targets.

    Returns:
        torch.nn.Module: The initialized RegDGCNN model, potentially wrapped in a DataParallel module if multiple GPUs are used.
    """
    model = RegPointNet(args=config).to(device)
    if config['cuda'] and paddle.device.cuda.device_count() > 1:
        model = paddle.DataParallel(layers=model)
    return model


def get_dataloaders(dataset_path: str, aero_coeff: str, subset_dir: str,
                    num_points: int, batch_size: int, train_frac: float = 1.0) -> tuple:
    """
    Prepare and return the training, validation, and test DataLoader objects.

    Args:
        dataset_path (str): The file path to the dataset directory containing the STL files.
        aero_coeff (str): The path to the CSV file with metadata for the models.
        subset_dir (str): The directory containing the subset files (train, val, test).
        num_points (int): The number of points to sample from each point cloud in the dataset.
        batch_size (int): The number of samples per batch to load.
        train_frac (float): Fraction of the training data to be used for training.

    Returns:
        tuple: A tuple containing the training DataLoader, validation DataLoader, and test DataLoader.
    """
    full_dataset = DrivAerNetDataset(root_dir=dataset_path, csv_file=aero_coeff, num_points=num_points, pointcloud_exist=True)

    def create_subset(dataset, ids_file):
        try:
            with open(os.path.join(subset_dir, ids_file), 'r') as file:
                subset_ids = file.read().split()
            subset_indices = dataset.data_frame[dataset.data_frame['Design'].isin(subset_ids)].index.tolist()
            return paddle.io.Subset(dataset=dataset, indices=subset_indices)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f'Error loading subset file {ids_file}: {e}')

    train_dataset = create_subset(full_dataset, 'train_design_ids.txt')

    if train_frac < 1.0:
        train_size = int(len(train_dataset) * train_frac)
        train_dataset, _ = paddle.io.random_split(dataset=train_dataset,
                                                  lengths=[train_size, len(train_dataset) - train_size])

    train_dataloader = paddle.io.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, shuffle=True, drop_last=True, num_workers=32)
    val_dataset = create_subset(full_dataset, 'val_design_ids.txt')

    val_dataloader = paddle.io.DataLoader(dataset=val_dataset, batch_size=
    batch_size, shuffle=False, drop_last=True, num_workers=32)
    test_dataset = create_subset(full_dataset, 'test_design_ids.txt')

    test_dataloader = paddle.io.DataLoader(dataset=test_dataset, batch_size
    =batch_size, shuffle=False, drop_last=True, num_workers=32)
    return train_dataloader, val_dataloader, test_dataloader


def train_and_evaluate(model: paddle.nn.Layer, train_dataloader: paddle.io.DataLoader, val_dataloader: paddle.io.DataLoader, config: dict):
    """
    Train and evaluate the model using the provided dataloaders and configuration. This function handles the training
    loop, including forward and backward propagation, and evaluates the model's performance on the validation set at
    the end of each epoch. It saves the best model based on the lowest validation loss and also saves the final model
    state at the end of all epochs.

    Args:
        model (torch.nn.Module): The model to be trained and evaluated.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        config (dict): Configuration dictionary containing settings such as learning rate, batch size, number of epochs,
                       optimizer choice, etc.

    Returns:
        tuple: A tuple containing paths to the best and final saved model states.
    """
    train_losses, val_losses = [], []
    training_start_time = time.time()
    if config['optimizer'] == 'adam':
        optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                          learning_rate=config['lr'], weight_decay=0.0001)
    else:
        optimizer = paddle.optimizer.SGD(model.parameters(), lr=config['lr'],
                                         momentum=0.9, weight_decay=0.0001)
    tmp_lr = paddle.optimizer.lr.ReduceOnPlateau(mode='min', patience=50,
                                                 factor=0.1, verbose=True, learning_rate=optimizer.get_lr())
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
    best_mse = float('inf')
    best_model_path = None
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()
        total_loss, total_r2 = 0, 0
        for data, targets in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']} [Training]"):
            data, targets = data.to(device), targets.to(device).squeeze()
            data = data.transpose(perm=[0, 2, 1])
            optimizer.clear_gradients(set_to_zero=False)
            outputs = model(data)
            loss = paddle.nn.functional.mse_loss(input=outputs.squeeze(),
                                                 label=targets)
            r2 = r2_score(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_r2 += r2.item()
        avg_loss = total_loss / len(train_dataloader)
        avg_r2 = total_r2 / len(train_dataloader)
        train_losses.append(avg_loss)
        epoch_duration = time.time() - epoch_start_time
        print(
            f'Epoch {epoch + 1} Training Loss: {avg_loss:.6f} Time: {epoch_duration:.2f}s'
        )
        print(f'Average Training R²: {avg_r2:.4f}')
        model.eval()
        val_loss, val_r2 = 0, 0
        inference_times = []
        with paddle.no_grad():
            for data, targets in tqdm(val_dataloader, desc=
            f"Epoch {epoch + 1}/{config['epochs']} [Validation]"):
                inference_start_time = time.time()
                data, targets = data.to(device), targets.to(device).squeeze()
                data = data.transpose(perm=[0, 2, 1])
                outputs = model(data)
                loss = paddle.nn.functional.mse_loss(input=outputs.squeeze(
                ), label=targets)
                val_loss += loss.item()
                val_r2 += r2_score(outputs.squeeze(), targets).item()
                inference_duration = time.time() - inference_start_time
                inference_times.append(inference_duration)
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        avg_inference_time = sum(inference_times) / len(inference_times)
        print(
            f'Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}, Avg Inference Time: {avg_inference_time:.4f}s'
        )
        print(f'Average Validation R²: {val_r2 / len(val_dataloader):.4f}')
        if avg_val_loss < best_mse:
            best_mse = avg_val_loss
            best_model_path = os.path.join('models',
                                           f"{config['exp_name']}_best_model.pth")
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            paddle.save(obj=model.state_dict(), path=best_model_path)
            print(f'New best model saved with MSE: {best_mse:.6f}')
        scheduler.step(avg_val_loss)
    final_model_path = os.path.join('models',
                                    f"{config['exp_name']}_final_model.pth")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    paddle.save(obj=model.state_dict(), path=final_model_path)
    print(f'Model saved to {final_model_path}')
    np.save(os.path.join('models', f"{config['exp_name']}_train_losses.npy"
                         ), np.array(train_losses))
    np.save(os.path.join('models', f"{config['exp_name']}_val_losses.npy"),
            np.array(val_losses))
    return best_model_path, final_model_path


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
        for data, targets in test_dataloader:
            start_time = time.time()
            data, targets = data.to(device), targets.to(device).squeeze()
            data = data.transpose(perm=[0, 2, 1])
            outputs = model(data)
            end_time = time.time()
            inference_time = end_time - start_time
            total_inference_time += inference_time
            mse = paddle.nn.functional.mse_loss(input=outputs.squeeze(),
                                                label=targets)
            mae = paddle.nn.functional.l1_loss(input=outputs.squeeze(),
                                               label=targets)
            r2 = r2_score(outputs.squeeze(), targets)
            total_mse += mse.item()
            total_mae += mae.item()
            total_r2 += r2.item()
            max_mae = max(max_mae, mae.item())
            total_samples += targets.shape[0]
    avg_mse = total_mse / len(test_dataloader)
    avg_mae = total_mae / len(test_dataloader)
    avg_r2 = total_r2 / len(test_dataloader)
    print(
        f'Test MSE: {avg_mse:.6f}, Test MAE: {avg_mae:.6f}, Max MAE: {max_mae:.6f}, Test R²: {avg_r2:.4f}'
    )
    print(
        f'Total inference time: {total_inference_time:.2f}s for {total_samples} samples'
    )
    return {'MSE': avg_mse, 'MAE': avg_mae, 'Max MAE': max_mae, 'R2': avg_r2}


def load_and_test_model(model_path, test_dataloader, device):
    """Load a saved model and test it, returning the test results."""
    model = RegPointNet(args=config).to(device)
    model = paddle.DataParallel(layers=model)
    model.set_state_dict(state_dict=paddle.load(path=str(model_path)))
    return test_model(model, test_dataloader, config)


from sklearn.manifold import TSNE


def save_features_incrementally(features, filename):
    """ Save features incrementally to avoid large memory overhead. """
    with open(filename, 'ab') as f:
        np.save(f, features)


def extract_features_and_outputs(model_path, dataloader, device, config):
    """Load a saved model and extract features and outputs from the specified DataLoader, saving them to files."""
    model = RegPointNet(args=config).to(device)
    model = paddle.DataParallel(layers=model)
    model.set_state_dict(state_dict=paddle.load(path=str(model_path)))
    model.eval()
    features_list = []
    outputs_list = []
    total_batches = len(dataloader)
    tsne_save_path = os.path.join('models',
                                  f"{config['exp_name']}_train_tsne.npy")
    with paddle.no_grad():
        for i, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device).squeeze()
            data = data.transpose(perm=[0, 2, 1])
            output, features = model(data)
            features_list.append(features.cpu().numpy())
            if i % 10 == 0:
                print(f'Processed {i + 1}/{total_batches} batches.')
    print('Saving Results')
    features_array = np.concatenate(features_list, axis=0)
    print('Applying t-SNE...')
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features_array)
    tsne_save_path = os.path.join('models',
                                  f"{config['exp_name']}_train_tsne.npy")
    np.save(tsne_save_path, tsne_results)
    print('t-SNE results saved to {tsne_save_path}')


if __name__ == '__main__':
    setup_seed(config['seed'])
    train_fractions = [1.0]
    results = {}
    for frac in train_fractions:
        print(f'Training on {frac * 100}% of the training data')
        model = initialize_model(config).to(device)
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
            config['dataset_path'], config['aero_coeff'], config[
                'subset_dir'], config['num_points'], config['batch_size'],
            train_frac=frac)

        print('Start trainning')
        best_model_path, final_model_path = train_and_evaluate(model,
                                                               train_dataloader, val_dataloader, config)
        print('Testing the best model:')
        best_results = load_and_test_model(best_model_path, test_dataloader,
                                           device)
        print('Testing the final model:')
        final_results = load_and_test_model(final_model_path,
                                            test_dataloader, device)
