"""
Inference script for TurbDiff model in PaddleScience.
"""

import os
import argparse
import yaml
import paddle
import numpy as np
import h5py
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from model import DenoisingModel
from diffusion import GaussianDiffusion
from data_utils import (
    Variable, TurbulenceDataset, Normalization, 
    create_dataloader
)
from conditioning import Conditioning, CellTypeEmbedding, ConditioningType


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with TurbDiff model')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                      help='Output directory for generated samples')
    parser.add_argument('--num_samples', type=int, default=10,
                      help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for inference')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='gpu',
                      help='Device to use (gpu or cpu)')
    parser.add_argument('--mode', type=str, default='sample',
                      choices=['sample', 'interpolate', 'unconditional'],
                      help='Inference mode')
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize generated samples')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_environment(args, config):
    """Set up inference environment."""
    # Set random seed
    paddle.seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    paddle.set_device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)


def create_variables(config):
    """Create variable definitions from config."""
    variables = []
    for var_config in config['variables']:
        variables.append(Variable(var_config['name'], var_config['dims']))
    return variables


def load_model(args, config, variables):
    """Load the trained model."""
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
    
    # Load checkpoint
    checkpoint = paddle.load(args.model_path)
    diffusion.set_state_dict(checkpoint['model_state_dict'])
    
    # Switch to evaluation mode
    diffusion.eval()
    
    return diffusion, conditioning


def load_test_dataset(args, config, variables):
    """Load test dataset for conditioning and evaluation."""
    # Create test dataset
    test_dataset = TurbulenceDataset(
        data_dir=args.data_dir,
        split='test',
        variables=variables,
    )
    
    # Create data loader
    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0)
    )
    
    # Create normalization
    normalization = Normalization(
        variables, 
        mode=config['data'].get('normalization_mode', 'mean-std')
    )
    
    return test_dataset, test_loader, normalization


def sample_from_model(model, conditioning, batch, normalization, stats, num_timesteps=None):
    """Sample from the model using the provided conditioning."""
    # Prepare conditioning
    C = {}
    if 'cell_type' in batch:
        local_cond = conditioning.prepare_local_conditioning(batch)
        if local_cond is not None:
            C[ConditioningType.LOCAL] = local_cond
    
    # Get shape from the input batch
    shape = batch['x'].shape
    
    # Generate random noise
    noise = paddle.randn(shape)
    
    # Sample from the model
    cell_idx = batch.get('cell_idx', None)
    cell_mask = batch.get('cell_mask', None)
    
    # Start timer
    start_time = time.time()
    
    with paddle.no_grad():
        sample = model.p_sample_loop(
            noise, 
            C, 
            cell_idx=cell_idx, 
            cell_mask=cell_mask, 
            verbose=True,
            num_timesteps=num_timesteps
        )
    
    # End timer
    sample_time = time.time() - start_time
    
    # Denormalize the sample
    denormalized_sample = normalization.denormalize(sample, stats)
    
    return denormalized_sample, sample_time


def interpolate_samples(model, conditioning, batch1, batch2, normalization, stats, num_steps=5, num_timesteps=None):
    """Interpolate between two conditioning sets."""
    # Prepare conditioning for first batch
    C1 = {}
    if 'cell_type' in batch1:
        local_cond1 = conditioning.prepare_local_conditioning(batch1)
        if local_cond1 is not None:
            C1[ConditioningType.LOCAL] = local_cond1
    
    # Prepare conditioning for second batch
    C2 = {}
    if 'cell_type' in batch2:
        local_cond2 = conditioning.prepare_local_conditioning(batch2)
        if local_cond2 is not None:
            C2[ConditioningType.LOCAL] = local_cond2
    
    # Get shape from the input batch
    shape = batch1['x'].shape
    
    # Generate same random noise for all interpolations
    noise = paddle.randn(shape)
    
    # Sample with interpolated conditioning
    samples = []
    for alpha in np.linspace(0, 1, num_steps):
        # Interpolate conditioning
        C = {}
        if ConditioningType.LOCAL in C1 and ConditioningType.LOCAL in C2:
            C[ConditioningType.LOCAL] = (1 - alpha) * C1[ConditioningType.LOCAL] + alpha * C2[ConditioningType.LOCAL]
        
        # Get cell indices from first batch (could be modified to interpolate these too)
        cell_idx = batch1.get('cell_idx', None)
        cell_mask = batch1.get('cell_mask', None)
        
        with paddle.no_grad():
            sample = model.p_sample_loop(
                noise, 
                C, 
                cell_idx=cell_idx, 
                cell_mask=cell_mask, 
                verbose=False,
                num_timesteps=num_timesteps
            )
            
            # Denormalize the sample
            denormalized_sample = normalization.denormalize(sample, stats)
            samples.append(denormalized_sample)
    
    return samples


def generate_unconditional_samples(model, shape, normalization, stats, num_samples=1, num_timesteps=None):
    """Generate unconditional samples."""
    samples = []
    
    for _ in range(num_samples):
        # Generate random noise
        noise = paddle.randn(shape)
        
        # Sample from the model without conditioning
        with paddle.no_grad():
            sample = model.p_sample_loop(
                noise, 
                {}, 
                verbose=False,
                num_timesteps=num_timesteps
            )
            
            # Denormalize the sample
            denormalized_sample = normalization.denormalize(sample, stats)
            samples.append(denormalized_sample)
    
    return samples


def save_samples(samples, variables, output_dir, prefix='sample'):
    """Save generated samples to HDF5 files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle both single sample and batch of samples
    if not isinstance(samples, list):
        samples = [samples]
    
    for i, sample in enumerate(samples):
        # Create HDF5 file
        file_path = os.path.join(output_dir, f"{prefix}_{i:04d}.h5")
        with h5py.File(file_path, 'w') as f:
            # Split sample by variables
            start_idx = 0
            for var in variables:
                end_idx = start_idx + var.dims
                var_data = sample[:, start_idx:end_idx].numpy()
                
                # Save variable data
                if var.dims == 1:
                    # Scalar field
                    f.create_dataset(var.name, data=var_data[0])
                else:
                    # Vector field components
                    f.create_dataset(var.name, data=var_data[0])
                
                # Move to next variable
                start_idx = end_idx
            
            # Save metadata
            f.attrs['generated'] = True
            f.attrs['generated_time'] = np.string_(time.strftime("%Y-%m-%d %H:%M:%S"))
    
    print(f"Saved {len(samples)} samples to {output_dir}")


def visualize_sample(sample, variables, output_dir, idx=0):
    """Visualize a generated sample."""
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # Get the first batch item
    if sample.ndim > 4:  # [B, C, H, W, D]
        sample = sample[0]  # [C, H, W, D]
    
    # Get grid dimensions
    _, h, w, d = sample.shape
    
    # Create a mesh grid for visualization
    y, x, z = np.meshgrid(
        np.linspace(0, 1, h),
        np.linspace(0, 1, w),
        np.linspace(0, 1, d),
    )
    
    # Start variable index
    start_idx = 0
    
    # Process each variable
    for var in variables:
        end_idx = start_idx + var.dims
        var_data = sample[start_idx:end_idx].numpy()
        
        if var.name == 'U' and var.dims == 3:
            # Vector field (velocity)
            u = var_data[0]
            v = var_data[1]
            w = var_data[2]
            
            # Calculate velocity magnitude
            magnitude = np.sqrt(u**2 + v**2 + w**2)
            
            # Create 3D plot for velocity magnitude
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot a slice of the velocity field
            slice_idx = d // 2
            sc = ax.scatter(
                x[:, :, slice_idx].flatten(),
                y[:, :, slice_idx].flatten(),
                z[:, :, slice_idx].flatten(),
                c=magnitude[:, :, slice_idx].flatten(),
                cmap='viridis',
                s=2
            )
            
            # Add colorbar
            plt.colorbar(sc, ax=ax, label='Velocity Magnitude')
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Velocity Magnitude (Z-slice at {slice_idx/d:.2f})')
            
            # Save figure
            plt.savefig(os.path.join(output_dir, 'visualizations', f'velocity_magnitude_{idx:04d}.png'))
            plt.close()
            
            # Create 2D slices plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # X-slice
            im0 = axes[0].imshow(magnitude[w//2, :, :].T, cmap='viridis', origin='lower')
            axes[0].set_title(f'X-slice at {w//2/w:.2f}')
            plt.colorbar(im0, ax=axes[0])
            
            # Y-slice
            im1 = axes[1].imshow(magnitude[:, h//2, :].T, cmap='viridis', origin='lower')
            axes[1].set_title(f'Y-slice at {h//2/h:.2f}')
            plt.colorbar(im1, ax=axes[1])
            
            # Z-slice
            im2 = axes[2].imshow(magnitude[:, :, d//2], cmap='viridis', origin='lower')
            axes[2].set_title(f'Z-slice at {d//2/d:.2f}')
            plt.colorbar(im2, ax=axes[2])
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'visualizations', f'velocity_slices_{idx:04d}.png'))
            plt.close()
            
        elif var.name == 'p' and var.dims == 1:
            # Scalar field (pressure)
            p = var_data[0]
            
            # Create 2D slices plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # X-slice
            im0 = axes[0].imshow(p[w//2, :, :].T, cmap='coolwarm', origin='lower')
            axes[0].set_title(f'Pressure X-slice at {w//2/w:.2f}')
            plt.colorbar(im0, ax=axes[0])
            
            # Y-slice
            im1 = axes[1].imshow(p[:, h//2, :].T, cmap='coolwarm', origin='lower')
            axes[1].set_title(f'Pressure Y-slice at {h//2/h:.2f}')
            plt.colorbar(im1, ax=axes[1])
            
            # Z-slice
            im2 = axes[2].imshow(p[:, :, d//2], cmap='coolwarm', origin='lower')
            axes[2].set_title(f'Pressure Z-slice at {d//2/d:.2f}')
            plt.colorbar(im2, ax=axes[2])
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'visualizations', f'pressure_slices_{idx:04d}.png'))
            plt.close()
        
        # Move to next variable
        start_idx = end_idx


def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up environment
    setup_environment(args, config)
    
    # Create variables from config
    variables = create_variables(config)
    
    # Load model
    model, conditioning = load_model(args, config, variables)
    
    # Load test dataset
    test_dataset, test_loader, normalization = load_test_dataset(args, config, variables)
    
    # Set number of timesteps for inference
    num_timesteps = config['inference'].get('num_timesteps', None)
    
    # Generate samples based on mode
    if args.mode == 'sample':
        print(f"Generating {args.num_samples} samples from test set conditions...")
        
        # Limit to requested number of samples
        sample_count = 0
        
        for batch in tqdm(test_loader):
            if sample_count >= args.num_samples:
                break
                
            # Generate sample
            sample, sample_time = sample_from_model(
                model, 
                conditioning, 
                batch, 
                normalization, 
                test_dataset.stats,
                num_timesteps
            )
            
            # Save sample
            save_samples(sample, variables, args.output_dir, prefix=f'sample_{sample_count:04d}')
            
            # Visualize if requested
            if args.visualize:
                visualize_sample(sample, variables, args.output_dir, sample_count)
            
            # Print timing
            print(f"Sample {sample_count} generated in {sample_time:.2f}s")
            
            sample_count += 1
    
    elif args.mode == 'interpolate':
        print("Generating interpolated samples...")
        
        # Get two different conditioning samples
        if len(test_loader) < 2:
            print("Error: Need at least 2 samples in test set for interpolation")
            return
        
        # Get the first two batches
        batches = []
        for i, batch in enumerate(test_loader):
            batches.append(batch)
            if i >= 1:
                break
        
        # Interpolate between the two batches
        interpolated_samples = interpolate_samples(
            model,
            conditioning,
            batches[0],
            batches[1],
            normalization,
            test_dataset.stats,
            num_steps=args.num_samples,
            num_timesteps=num_timesteps
        )
        
        # Save interpolated samples
        save_samples(interpolated_samples, variables, args.output_dir, prefix='interpolation')
        
        # Visualize if requested
        if args.visualize:
            for i, sample in enumerate(interpolated_samples):
                visualize_sample(sample, variables, args.output_dir, i)
    
    elif args.mode == 'unconditional':
        print("Generating unconditional samples...")
        
        # Get shape from test dataset
        for batch in test_loader:
            shape = batch['x'].shape
            break
        
        # Generate unconditional samples
        unconditional_samples = generate_unconditional_samples(
            model,
            shape,
            normalization,
            test_dataset.stats,
            num_samples=args.num_samples,
            num_timesteps=num_timesteps
        )
        
        # Save unconditional samples
        save_samples(unconditional_samples, variables, args.output_dir, prefix='unconditional')
        
        # Visualize if requested
        if args.visualize:
            for i, sample in enumerate(unconditional_samples):
                visualize_sample(sample, variables, args.output_dir, i)
    
    print("Inference completed!")


if __name__ == '__main__':
    main()
