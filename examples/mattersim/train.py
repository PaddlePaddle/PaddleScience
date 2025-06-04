"""
Training script for MatterSim model in PaddleScience.

This script enables fine-tuning of the MatterSim model on custom datasets.
"""

import os
import argparse
import yaml
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.optimizer import Adam
from paddle.io import DataLoader

from model import MatterSimModel
from data_utils import AtomisticDataset, collate_atomistic_batch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MatterSim model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--val_data_path', type=str, default=None,
                        help='Path to validation data (optional)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimension of element embeddings')
    parser.add_argument('--num_message_blocks', type=int, default=3,
                        help='Number of message passing blocks')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Dimension of hidden layers')
    parser.add_argument('--num_radial_basis', type=int, default=128,
                        help='Number of radial basis functions')
    parser.add_argument('--cutoff_distance', type=float, default=6.0,
                        help='Interatomic cutoff distance')
    parser.add_argument('--with_temperature', action='store_true',
                        help='Whether to include temperature conditioning')
    parser.add_argument('--with_pressure', action='store_true',
                        help='Whether to include pressure conditioning')
    
    # Training arguments
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint to start from')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Checkpoint save interval (epochs)')
    parser.add_argument('--device', type=str, default='gpu',
                        choices=['gpu', 'cpu'],
                        help='Device to use for training')
    
    return parser.parse_args()


def load_or_create_model(args):
    """Load model from checkpoint or create a new one."""
    model = MatterSimModel(
        embedding_dim=args.embedding_dim,
        num_message_blocks=args.num_message_blocks,
        hidden_dim=args.hidden_dim,
        num_radial_basis=args.num_radial_basis,
        cutoff_distance=args.cutoff_distance,
        with_temperature=args.with_temperature,
        with_pressure=args.with_pressure
    )
    
    # Load checkpoint if provided
    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        state_dict = paddle.load(args.checkpoint_path)
        model.set_state_dict(state_dict)
    
    return model


def train_epoch(model, dataloader, optimizer, epoch, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    energy_loss = 0.0
    force_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        # Unpack batch
        atomic_numbers = batch['atomic_numbers']
        positions = batch['positions']
        edge_index = batch['edge_index']
        
        # Ground truth
        target_energy = batch.get('energy')
        target_forces = batch.get('forces')
        
        # Optional conditioning
        temperature = batch.get('temperature')
        pressure = batch.get('pressure')
        
        # Forward pass
        outputs = model(
            atomic_numbers,
            positions,
            edge_index,
            temperature=temperature,
            pressure=pressure
        )
        
        # Compute loss
        loss = 0.0
        
        if target_energy is not None:
            e_loss = F.mse_loss(outputs['energy'], target_energy)
            loss += e_loss
            energy_loss += e_loss.item()
        
        if target_forces is not None:
            f_loss = F.mse_loss(outputs['forces'], target_forces)
            loss += 10.0 * f_loss  # Scale force loss
            force_loss += f_loss.item()
        
        # Backward pass and optimization
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f}")
    
    # Compute average losses
    avg_loss = total_loss / len(dataloader)
    avg_energy_loss = energy_loss / len(dataloader)
    avg_force_loss = force_loss / len(dataloader)
    
    return avg_loss, avg_energy_loss, avg_force_loss


def validate(model, dataloader, device):
    """Validate model performance."""
    model.eval()
    
    total_loss = 0.0
    energy_loss = 0.0
    force_loss = 0.0
    
    with paddle.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch
            atomic_numbers = batch['atomic_numbers']
            positions = batch['positions']
            edge_index = batch['edge_index']
            
            # Ground truth
            target_energy = batch.get('energy')
            target_forces = batch.get('forces')
            
            # Optional conditioning
            temperature = batch.get('temperature')
            pressure = batch.get('pressure')
            
            # Forward pass
            outputs = model(
                atomic_numbers,
                positions,
                edge_index,
                temperature=temperature,
                pressure=pressure
            )
            
            # Compute loss
            loss = 0.0
            
            if target_energy is not None:
                e_loss = F.mse_loss(outputs['energy'], target_energy)
                loss += e_loss
                energy_loss += e_loss.item()
            
            if target_forces is not None:
                f_loss = F.mse_loss(outputs['forces'], target_forces)
                loss += 10.0 * f_loss  # Scale force loss
                force_loss += f_loss.item()
            
            total_loss += loss.item()
    
    # Compute average losses
    avg_loss = total_loss / len(dataloader)
    avg_energy_loss = energy_loss / len(dataloader)
    avg_force_loss = force_loss / len(dataloader)
    
    return avg_loss, avg_energy_loss, avg_force_loss


def save_checkpoint(model, optimizer, epoch, loss, output_dir):
    """Save model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pdparams")
    paddle.save(model.state_dict(), checkpoint_path)
    
    optimizer_path = os.path.join(output_dir, f"optimizer_epoch_{epoch}.pdopt")
    paddle.save(optimizer.state_dict(), optimizer_path)
    
    # Save latest checkpoint separately
    latest_path = os.path.join(output_dir, "latest_checkpoint.pdparams")
    paddle.save(model.state_dict(), latest_path)
    
    print(f"Checkpoint saved to {checkpoint_path}")


def save_training_config(args, output_dir):
    """Save training configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    config = vars(args)
    config_path = os.path.join(output_dir, "training_config.yaml")
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Training config saved to {config_path}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Set device
    paddle.set_device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save training configuration
    save_training_config(args, args.output_dir)
    
    # Load or create model
    model = load_or_create_model(args)
    
    # Create optimizer
    optimizer = Adam(
        learning_rate=args.learning_rate,
        parameters=model.parameters(),
        weight_decay=args.weight_decay
    )
    
    # Create datasets and dataloaders
    train_dataset = AtomisticDataset(args.data_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_atomistic_batch
    )
    
    val_loader = None
    if args.val_data_path is not None:
        val_dataset = AtomisticDataset(args.val_data_path)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_atomistic_batch
        )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        
        # Train
        train_loss, train_energy_loss, train_force_loss = train_epoch(
            model, train_loader, optimizer, epoch, args.device
        )
        
        print(f"Train Loss: {train_loss:.4f} | "
              f"Energy Loss: {train_energy_loss:.4f} | "
              f"Force Loss: {train_force_loss:.4f}")
        
        # Validate
        if val_loader is not None:
            val_loss, val_energy_loss, val_force_loss = validate(
                model, val_loader, args.device
            )
            
            print(f"Val Loss: {val_loss:.4f} | "
                  f"Energy Loss: {val_energy_loss:.4f} | "
                  f"Force Loss: {val_force_loss:.4f}")
            
            # Save checkpoint if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    os.path.join(args.output_dir, "best_model")
                )
        
        # Save checkpoint at regular intervals
        if epoch % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                os.path.join(args.output_dir, "checkpoints")
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.num_epochs, train_loss,
        os.path.join(args.output_dir, "final_model")
    )
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
