"""
Training script for TurbDiff model in PaddleScience.
"""

import os
import argparse
import time
import yaml
import paddle
import numpy as np

from model import DenoisingModel
from diffusion import GaussianDiffusion
from data_utils import (
    Variable, TurbulenceDataset, Normalization, 
    create_dataloader
)
from conditioning import Conditioning, CellTypeEmbedding, ConditioningType


def parse_args():
    parser = argparse.ArgumentParser(description='Train TurbDiff model')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Output directory for checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='gpu',
                      help='Device to use (gpu or cpu)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_environment(args, config):
    """Set up training environment."""
    # Set random seed
    paddle.seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    paddle.set_device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)


def create_variables(config):
    """Create variable definitions from config."""
    variables = []
    for var_config in config['variables']:
        variables.append(Variable(var_config['name'], var_config['dims']))
    return variables


def create_model(config, variables):
    """Create model from configuration."""
    # Calculate total dimensions for variables
    vars_dim = sum(var.dims for var in variables)
    
    # Create cell type embedding if specified
    cell_type_embedding = None
    if config['model'].get('cell_type_features', True):
        cell_type_embedding = CellTypeEmbedding.create(
            config['model'].get('cell_type_embedding_type', 'learned'),
            config['model'].get('cell_type_embedding_dim', 4),
            config['model'].get('num_cell_types', 5)
        )
    
    # Create conditioning module
    conditioning = Conditioning(
        cell_type_embedding=cell_type_embedding,
        use_cell_pos=config['model'].get('cell_pos_features', False)
    )
    
    # Create denoising model
    model = DenoisingModel(
        in_features=vars_dim,
        out_features=vars_dim * (2 if config['diffusion'].get('learned_variances', False) else 1),
        c_local_features=conditioning.local_conditioning_dim,
        c_global_features=conditioning.global_conditioning_dim,
        timesteps=config['diffusion'].get('timesteps', 1000),
        dim=config['model'].get('dim', 32),
        u_net_levels=config['model'].get('u_net_levels', 4),
        actfn=getattr(paddle.nn, config['model'].get('actfn', 'Silu')),
        norm_type=config['model'].get('norm_type', 'instance'),
        with_geometry_embedding=config['model'].get('with_geometry_embedding', True),
    )
    
    # Wrap with diffusion model
    diffusion = GaussianDiffusion(
        model,
        timesteps=config['diffusion'].get('timesteps', 1000),
        loss_type=config['diffusion'].get('loss_type', 'l2'),
        beta_schedule=config['diffusion'].get('beta_schedule', 'sigmoid'),
        clip_denoised=config['diffusion'].get('clip_denoised', False),
        noise_bcs=config['diffusion'].get('noise_bcs', False),
        learned_variances=config['diffusion'].get('learned_variances', False),
        elbo_weight=config['diffusion'].get('elbo_weight', None),
        detach_elbo_mean=config['diffusion'].get('detach_elbo_mean', True),
    )
    
    return diffusion, conditioning


def create_datasets(args, config, variables):
    """Create datasets from configuration."""
    # Create normalization
    normalization = Normalization(
        variables, 
        mode=config['data'].get('normalization_mode', 'mean-std')
    )
    
    # Create training dataset
    train_dataset = TurbulenceDataset(
        data_dir=args.data_dir,
        split='train',
        variables=variables,
    )
    
    # Create validation dataset
    val_dataset = TurbulenceDataset(
        data_dir=args.data_dir,
        split='val',
        variables=variables,
    )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0)
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0)
    )
    
    return train_dataset, val_dataset, train_loader, val_loader, normalization


def create_optimizer(model, config, total_steps):
    """Create optimizer and learning rate scheduler."""
    # Create learning rate scheduler
    learning_rate = config['training']['learning_rate']
    min_learning_rate = config['training'].get('min_learning_rate', learning_rate / 10)
    
    # Cosine decay with linear warmup
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=learning_rate,
        T_max=total_steps,
        eta_min=min_learning_rate
    )
    
    if config['training'].get('warmup_steps', 0) > 0:
        scheduler = paddle.optimizer.lr.LinearWarmup(
            scheduler,
            warmup_steps=config['training']['warmup_steps'],
            start_lr=learning_rate / 10,
            end_lr=learning_rate
        )
    
    # Create optimizer
    optimizer = paddle.optimizer.Adam(
        learning_rate=scheduler,
        parameters=model.parameters(),
        weight_decay=config['training'].get('weight_decay', 0.0),
        beta1=config['training'].get('beta1', 0.9),
        beta2=config['training'].get('beta2', 0.999),
    )
    
    return optimizer, scheduler


def train_epoch(model, train_loader, optimizer, conditioning, normalization, epoch, device, log_interval=10):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        # Prepare data
        x = batch['x']
        
        # Normalize data
        x_normalized = normalization.normalize(x, train_loader.dataset.stats)
        
        # Prepare conditioning
        C = {}
        if 'cell_type' in batch:
            local_cond = conditioning.prepare_local_conditioning(batch)
            if local_cond is not None:
                C[ConditioningType.LOCAL] = local_cond
        
        # Forward pass
        cell_idx = batch.get('cell_idx', None)
        cell_mask = batch.get('cell_mask', None)
        loss, t = model(x_normalized, C, cell_idx, cell_mask)
        
        # Backward pass and optimize
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        
        # Log progress
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            print(f'Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | '
                  f'Loss {loss.item():.4f} | {elapsed:.2f}s elapsed')
    
    # Return average loss
    return total_loss / len(train_loader)


def validate(model, val_loader, conditioning, normalization, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with paddle.no_grad():
        for batch in val_loader:
            # Prepare data
            x = batch['x']
            
            # Normalize data
            x_normalized = normalization.normalize(x, val_loader.dataset.stats)
            
            # Prepare conditioning
            C = {}
            if 'cell_type' in batch:
                local_cond = conditioning.prepare_local_conditioning(batch)
                if local_cond is not None:
                    C[ConditioningType.LOCAL] = local_cond
            
            # Forward pass
            cell_idx = batch.get('cell_idx', None)
            cell_mask = batch.get('cell_mask', None)
            loss, _ = model(x_normalized, C, cell_idx, cell_mask)
            
            # Update statistics
            total_loss += loss.item()
    
    # Return average loss
    return total_loss / len(val_loader)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    paddle.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up environment
    setup_environment(args, config)
    
    # Create variables from config
    variables = create_variables(config)
    
    # Create datasets and loaders
    train_dataset, val_dataset, train_loader, val_loader, normalization = create_datasets(
        args, config, variables
    )
    
    # Create model
    model, conditioning = create_model(config, variables)
    
    # Create optimizer and scheduler
    total_steps = args.epochs * len(train_loader)
    optimizer, lr_scheduler = create_optimizer(model, config, total_steps)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, conditioning, 
            normalization, epoch, args.device
        )
        
        # Validate
        val_loss = validate(
            model, val_loader, conditioning, 
            normalization, args.device
        )
        
        # Print metrics
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pdparams")
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, "best_model.pdparams")
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)
            print(f"New best model saved with val_loss: {val_loss:.4f}")
    
    print("Training completed!")


if __name__ == '__main__':
    main()
