"""
Evaluation script for TurbDiff model in PaddleScience.
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
from skimage.metrics import structural_similarity as ssim
import pandas as pd

from model import DenoisingModel
from diffusion import GaussianDiffusion
from data_utils import (
    Variable, TurbulenceDataset, Normalization, 
    create_dataloader
)
from conditioning import Conditioning, CellTypeEmbedding, ConditioningType
from infer import load_model, load_config, setup_environment, create_variables


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TurbDiff model')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Output directory for evaluation results')
    parser.add_argument('--num_samples', type=int, default=50,
                      help='Number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='gpu',
                      help='Device to use (gpu or cpu)')
    parser.add_argument('--metrics', type=str, nargs='+', 
                      default=['mse', 'mae', 'psnr', 'ssim', 'energy_spectrum'],
                      help='Metrics to evaluate')
    parser.add_argument('--save_samples', action='store_true',
                      help='Save generated samples')
    return parser.parse_args()


def calculate_mse(x, y):
    """Calculate Mean Squared Error."""
    return paddle.mean((x - y) ** 2).item()


def calculate_mae(x, y):
    """Calculate Mean Absolute Error."""
    return paddle.mean(paddle.abs(x - y)).item()


def calculate_psnr(x, y, data_range=None):
    """Calculate Peak Signal-to-Noise Ratio."""
    if data_range is None:
        data_range = paddle.max(y) - paddle.min(y)
    
    mse = calculate_mse(x, y)
    if mse == 0:
        return float('inf')
    
    return 20 * np.log10(data_range) - 10 * np.log10(mse)


def calculate_ssim(x, y):
    """Calculate Structural Similarity Index."""
    # Convert to numpy arrays
    x_np = x.numpy()
    y_np = y.numpy()
    
    # Calculate SSIM for each channel and average
    ssim_values = []
    for c in range(x_np.shape[1]):
        for d in range(x_np.shape[4]):  # For each slice in depth
            ssim_val = ssim(
                x_np[0, c, :, :, d], 
                y_np[0, c, :, :, d],
                data_range=np.max(y_np[0, c, :, :, d]) - np.min(y_np[0, c, :, :, d])
            )
            ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)


def calculate_energy_spectrum(velocity_field, dx=1.0):
    """
    Calculate energy spectrum for a velocity field.
    
    Args:
        velocity_field: Tensor of shape [3, H, W, D] (u, v, w components)
        dx: Grid spacing
        
    Returns:
        k: Wave numbers
        E: Energy spectrum
    """
    # Get shape and dimensions
    if velocity_field.ndim > 3:
        # If batch dimension is present, take first item
        velocity_field = velocity_field[0]
    
    # Convert to numpy
    u = velocity_field[0].numpy()
    v = velocity_field[1].numpy() if velocity_field.shape[0] > 1 else np.zeros_like(u)
    w = velocity_field[2].numpy() if velocity_field.shape[0] > 2 else np.zeros_like(u)
    
    # Get grid shape
    nx, ny, nz = u.shape
    
    # Compute FFTs of velocity components
    u_hat = np.fft.fftn(u) / (nx * ny * nz)
    v_hat = np.fft.fftn(v) / (nx * ny * nz)
    w_hat = np.fft.fftn(w) / (nx * ny * nz)
    
    # Create wavenumber grid
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dx)
    kz = 2 * np.pi * np.fft.fftfreq(nz, dx)
    
    # Create meshgrid
    kxx, kyy, kzz = np.meshgrid(kx, ky, kz, indexing='ij')
    k_squared = kxx**2 + kyy**2 + kzz**2
    
    # Energy in spectral space
    E_hat = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)
    
    # Compute energy spectrum by binning
    k_min = 2 * np.pi / max(nx, ny, nz)
    k_max = np.sqrt(3) * np.pi * min(nx, ny, nz)
    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), 32)
    
    # Initialize energy spectrum
    E = np.zeros_like(k_bins[:-1])
    
    # Bin energy
    for i in range(len(k_bins) - 1):
        k_lower = k_bins[i]
        k_upper = k_bins[i+1]
        
        # Find wavenumbers in this bin
        mask = (k_squared > k_lower**2) & (k_squared <= k_upper**2)
        
        # Accumulate energy
        if np.any(mask):
            E[i] = np.sum(E_hat[mask])
    
    # Wave number for each bin (use midpoint)
    k = 0.5 * (k_bins[1:] + k_bins[:-1])
    
    return k, E


def compute_energy_spectrum_error(pred_k, pred_E, true_k, true_E):
    """
    Compute error between predicted and true energy spectra.
    
    Args:
        pred_k: Predicted wave numbers
        pred_E: Predicted energy spectrum
        true_k: True wave numbers
        true_E: True energy spectrum
        
    Returns:
        Error metric
    """
    # Interpolate spectra to common wave numbers if they're different
    if not np.array_equal(pred_k, true_k):
        # Use a common k-space for comparison
        common_k = np.unique(np.concatenate([pred_k, true_k]))
        common_k.sort()
        
        # Interpolate to common k-space
        from scipy.interpolate import interp1d
        
        # Handle zero values with small epsilon to enable log interpolation
        epsilon = 1e-12
        
        # Interpolate in log-log space
        pred_interp = interp1d(
            np.log10(pred_k), 
            np.log10(pred_E + epsilon), 
            kind='linear', 
            bounds_error=False, 
            fill_value='extrapolate'
        )
        
        true_interp = interp1d(
            np.log10(true_k), 
            np.log10(true_E + epsilon), 
            kind='linear', 
            bounds_error=False, 
            fill_value='extrapolate'
        )
        
        # Get interpolated values
        pred_E_common = 10 ** pred_interp(np.log10(common_k)) - epsilon
        true_E_common = 10 ** true_interp(np.log10(common_k)) - epsilon
        
        # Use only valid region (where both spectra have data)
        valid_mask = (
            (common_k >= max(pred_k[0], true_k[0])) & 
            (common_k <= min(pred_k[-1], true_k[-1]))
        )
        
        pred_E_valid = pred_E_common[valid_mask]
        true_E_valid = true_E_common[valid_mask]
    else:
        # Same wave numbers, no interpolation needed
        pred_E_valid = pred_E
        true_E_valid = true_E
    
    # Compute relative error in energy spectrum
    rel_error = np.mean(np.abs(pred_E_valid - true_E_valid) / (true_E_valid + 1e-8))
    
    return rel_error


def evaluate_sample(generated, target, variables, metrics):
    """
    Evaluate a generated sample against ground truth.
    
    Args:
        generated: Generated sample tensor [B, C, H, W, D]
        target: Ground truth tensor [B, C, H, W, D]
        variables: List of variables
        metrics: List of metrics to evaluate
        
    Returns:
        Dictionary of metrics results
    """
    results = {}
    
    # Calculate global metrics for all variables
    if 'mse' in metrics:
        results['mse'] = calculate_mse(generated, target)
    
    if 'mae' in metrics:
        results['mae'] = calculate_mae(generated, target)
    
    if 'psnr' in metrics:
        results['psnr'] = calculate_psnr(generated, target)
    
    if 'ssim' in metrics:
        results['ssim'] = calculate_ssim(generated, target)
    
    # Calculate per-variable metrics
    start_idx = 0
    for var in variables:
        end_idx = start_idx + var.dims
        
        # Extract variable data
        generated_var = generated[:, start_idx:end_idx]
        target_var = target[:, start_idx:end_idx]
        
        # Calculate metrics for this variable
        var_results = {}
        
        if 'mse' in metrics:
            var_results['mse'] = calculate_mse(generated_var, target_var)
        
        if 'mae' in metrics:
            var_results['mae'] = calculate_mae(generated_var, target_var)
        
        if 'psnr' in metrics:
            var_results['psnr'] = calculate_psnr(generated_var, target_var)
        
        if 'ssim' in metrics:
            var_results['ssim'] = calculate_ssim(generated_var, target_var)
        
        # Calculate energy spectrum for velocity field
        if 'energy_spectrum' in metrics and var.name == 'U' and var.dims == 3:
            # Get energy spectrum for generated sample
            gen_k, gen_E = calculate_energy_spectrum(generated_var)
            
            # Get energy spectrum for target
            true_k, true_E = calculate_energy_spectrum(target_var)
            
            # Calculate error
            var_results['energy_spectrum_error'] = compute_energy_spectrum_error(
                gen_k, gen_E, true_k, true_E
            )
            
            # Save spectra for later plotting
            var_results['gen_spectrum'] = (gen_k, gen_E)
            var_results['true_spectrum'] = (true_k, true_E)
        
        # Store variable results
        results[var.name] = var_results
        
        # Move to next variable
        start_idx = end_idx
    
    return results


def plot_energy_spectra(results, output_dir):
    """
    Plot energy spectra from evaluation results.
    
    Args:
        results: Dictionary of evaluation results
        output_dir: Output directory for plots
    """
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    # Check if we have energy spectrum results for velocity
    if 'U' in results and 'samples' in results and 'gen_spectrum' in results['U']:
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Get sample results
        sample_results = results['samples']
        
        # Get average spectra
        true_k, true_E = results['U']['true_spectrum']
        gen_k, gen_E = results['U']['gen_spectrum']
        
        # Plot spectra
        plt.loglog(true_k, true_E, 'b-', linewidth=2, label='Ground Truth')
        plt.loglog(gen_k, gen_E, 'r-', linewidth=2, label='Generated')
        
        # Plot Kolmogorov -5/3 scaling law for reference
        k_range = np.logspace(np.log10(true_k[5]), np.log10(true_k[-5]), 100)
        # Scale to match true spectrum
        scale_idx = len(true_k) // 3
        scale_factor = true_E[scale_idx] / (true_k[scale_idx] ** (-5/3))
        kolmogorov = scale_factor * k_range ** (-5/3)
        plt.loglog(k_range, kolmogorov, 'k--', linewidth=1, label='k^(-5/3)')
        
        # Add labels and legend
        plt.xlabel('Wave Number (k)')
        plt.ylabel('Energy Spectrum E(k)')
        plt.title('Turbulence Energy Spectrum Comparison')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plots', 'energy_spectrum.png'), dpi=300)
        plt.close()


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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize results storage
    all_results = {
        'samples': [],
        'global': {metric: [] for metric in args.metrics if metric != 'energy_spectrum'},
    }
    
    # Add per-variable metrics
    for var in variables:
        all_results[var.name] = {metric: [] for metric in args.metrics if metric != 'energy_spectrum'}
        if var.name == 'U' and 'energy_spectrum' in args.metrics:
            all_results[var.name]['energy_spectrum_error'] = []
    
    # Process samples
    print(f"Evaluating model on {min(args.num_samples, len(test_loader))} test samples...")
    
    # Set number of timesteps for inference
    num_timesteps = config['inference'].get('num_timesteps', None)
    
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        if batch_idx >= args.num_samples:
            break
        
        # Get ground truth
        ground_truth = batch['x']
        
        # Prepare conditioning
        C = {}
        if 'cell_type' in batch:
            local_cond = conditioning.prepare_local_conditioning(batch)
            if local_cond is not None:
                C[ConditioningType.LOCAL] = local_cond
        
        # Generate sample
        cell_idx = batch.get('cell_idx', None)
        cell_mask = batch.get('cell_mask', None)
        
        # Generate random noise
        shape = ground_truth.shape
        noise = paddle.randn(shape)
        
        # Sample from the model
        with paddle.no_grad():
            sample = model.p_sample_loop(
                noise, 
                C, 
                cell_idx=cell_idx, 
                cell_mask=cell_mask, 
                verbose=False,
                num_timesteps=num_timesteps
            )
        
        # Denormalize
        denormalized_sample = normalization.denormalize(sample, test_dataset.stats)
        denormalized_ground_truth = normalization.denormalize(ground_truth, test_dataset.stats)
        
        # Evaluate sample
        results = evaluate_sample(
            denormalized_sample, 
            denormalized_ground_truth, 
            variables, 
            args.metrics
        )
        
        # Store results
        all_results['samples'].append(results)
        
        # Update global metrics
        for metric in args.metrics:
            if metric == 'energy_spectrum':
                continue
            if metric in results:
                all_results['global'][metric].append(results[metric])
        
        # Update per-variable metrics
        for var in variables:
            if var.name in results:
                for metric_name, value in results[var.name].items():
                    if not metric_name.startswith('gen_') and not metric_name.startswith('true_'):
                        all_results[var.name][metric_name].append(value)
        
        # Save sample if requested
        if args.save_samples:
            sample_dir = os.path.join(args.output_dir, 'samples')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save as HDF5
            sample_path = os.path.join(sample_dir, f'sample_{batch_idx:04d}.h5')
            with h5py.File(sample_path, 'w') as f:
                # Save generated sample
                f.create_dataset('generated', data=denormalized_sample.numpy())
                
                # Save ground truth
                f.create_dataset('ground_truth', data=denormalized_ground_truth.numpy())
    
    # Calculate average metrics
    avg_results = {'global': {}}
    
    # Global metrics
    for metric in all_results['global']:
        values = all_results['global'][metric]
        avg_results['global'][metric] = np.mean(values)
    
    # Per-variable metrics
    for var in variables:
        avg_results[var.name] = {}
        for metric in all_results[var.name]:
            values = all_results[var.name][metric]
            avg_results[var.name][metric] = np.mean(values)
    
    # Print results
    print("\nEvaluation Results:")
    print("===================")
    
    print("\nGlobal Metrics:")
    for metric, value in avg_results['global'].items():
        print(f"  {metric}: {value:.6f}")
    
    print("\nPer-Variable Metrics:")
    for var in variables:
        print(f"  {var.name}:")
        for metric, value in avg_results[var.name].items():
            if not metric.startswith('gen_') and not metric.startswith('true_'):
                print(f"    {metric}: {value:.6f}")
    
    # Save results to CSV
    results_df = pd.DataFrame()
    
    # Add global metrics
    for metric, value in avg_results['global'].items():
        results_df.loc['global', metric] = value
    
    # Add per-variable metrics
    for var in variables:
        for metric, value in avg_results[var.name].items():
            if not metric.startswith('gen_') and not metric.startswith('true_'):
                results_df.loc[var.name, metric] = value
    
    # Save DataFrame
    results_df.to_csv(os.path.join(args.output_dir, 'evaluation_results.csv'))
    
    # Save full results
    with open(os.path.join(args.output_dir, 'all_results.yaml'), 'w') as f:
        # Filter out non-serializable items
        serializable_results = {
            'global': all_results['global'],
        }
        
        for var in variables:
            serializable_results[var.name] = {}
            for metric, values in all_results[var.name].items():
                if not metric.startswith('gen_') and not metric.startswith('true_'):
                    serializable_results[var.name][metric] = values
        
        yaml.dump(serializable_results, f)
    
    # Plot energy spectra if applicable
    if 'energy_spectrum' in args.metrics:
        plot_energy_spectra(all_results, args.output_dir)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
