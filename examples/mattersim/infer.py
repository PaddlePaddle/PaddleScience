"""
Inference script for MatterSim model in PaddleScience.

This script enables inference with the MatterSim model for predicting
energies, forces, and other properties of atomic structures.
"""

import os
import argparse
import yaml
import json
import numpy as np
import paddle
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read, write
from ase.visualize.plot import plot_atoms

from model import MatterSimModel, MatterSimForceField
from data_utils import read_xyz_file, build_edge_index


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with MatterSim model')
    
    # Input arguments
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input structure file (XYZ format)')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Directory to save inference results')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--cutoff_distance', type=float, default=6.0,
                        help='Interatomic cutoff distance')
    parser.add_argument('--with_temperature', action='store_true',
                        help='Whether to include temperature conditioning')
    parser.add_argument('--with_pressure', action='store_true',
                        help='Whether to include pressure conditioning')
    
    # Inference arguments
    parser.add_argument('--temperature', type=float, default=None,
                        help='Temperature for inference (in Kelvin)')
    parser.add_argument('--pressure', type=float, default=None,
                        help='Pressure for inference (in GPa)')
    parser.add_argument('--save_format', type=str, default='all',
                        choices=['all', 'json', 'xyz', 'png'],
                        help='Format to save inference results')
    parser.add_argument('--device', type=str, default='gpu',
                        choices=['gpu', 'cpu'],
                        help='Device to use for inference')
    parser.add_argument('--relax', action='store_true',
                        help='Perform structure relaxation')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum number of relaxation steps')
    
    return parser.parse_args()


def load_model(args):
    """Load MatterSim model from checkpoint."""
    model = MatterSimModel(
        cutoff_distance=args.cutoff_distance,
        with_temperature=args.with_temperature,
        with_pressure=args.with_pressure
    )
    
    # Load checkpoint
    print(f"Loading model from {args.model_path}")
    state_dict = paddle.load(args.model_path)
    model.set_state_dict(state_dict)
    
    # Set device
    paddle.set_device(args.device)
    model.eval()
    
    return model


def run_inference(model, input_file, temperature=None, pressure=None, device='gpu'):
    """
    Run inference with MatterSim model.
    
    Args:
        model: MatterSim model
        input_file: Path to input structure file
        temperature: Optional temperature for inference
        pressure: Optional pressure for inference
        device: Device to use for inference
        
    Returns:
        Dictionary with inference results
    """
    # Read input structure
    data = read_xyz_file(input_file)
    
    # Extract data
    atomic_numbers = data['atomic_numbers']
    positions = data['positions']
    
    # Build edge index
    edge_index = build_edge_index(positions, model.cutoff_distance)
    
    # Convert to paddle tensors
    atomic_numbers_tensor = paddle.to_tensor(
        atomic_numbers, dtype='int64'
    ).unsqueeze(0)  # Add batch dimension
    
    positions_tensor = paddle.to_tensor(
        positions, dtype='float32'
    ).unsqueeze(0)  # Add batch dimension
    
    edge_index_tensor = paddle.to_tensor(edge_index, dtype='int64')
    
    # Prepare temperature and pressure tensors if provided
    temp_tensor = None
    if temperature is not None and model.with_temperature:
        temp_tensor = paddle.to_tensor([temperature], dtype='float32')
    
    press_tensor = None
    if pressure is not None and model.with_pressure:
        press_tensor = paddle.to_tensor([pressure], dtype='float32')
    
    # Run inference
    with paddle.no_grad():
        outputs = model(
            atomic_numbers_tensor,
            positions_tensor,
            edge_index_tensor,
            temperature=temp_tensor,
            pressure=press_tensor
        )
    
    # Extract results
    energy = outputs['energy'].numpy()[0]
    atom_energies = outputs['atom_energies'].numpy()[0][:len(atomic_numbers)]
    
    # Create results dictionary
    results = {
        'energy': float(energy),
        'energy_per_atom': float(energy) / len(atomic_numbers),
        'atom_energies': atom_energies.tolist()
    }
    
    # Add forces if computed
    if 'forces' in outputs:
        forces = outputs['forces'].numpy()[0][:len(atomic_numbers)]
        results['forces'] = forces.tolist()
    
    # Add stress if computed
    if 'stress' in outputs:
        stress = outputs['stress'].numpy()[0]
        results['stress'] = stress.tolist()
    
    # Add metadata
    results['num_atoms'] = len(atomic_numbers)
    results['elements'] = [int(z) for z in atomic_numbers]
    results['positions'] = positions.tolist()
    
    if temperature is not None:
        results['temperature'] = float(temperature)
    
    if pressure is not None:
        results['pressure'] = float(pressure)
    
    return results


def relax_structure(model, atomic_numbers, positions, temperature=None, pressure=None,
                   max_steps=100, step_size=0.05, force_tol=1e-3):
    """
    Relax atomic structure using forces from the model.
    
    Args:
        model: MatterSim model
        atomic_numbers: Array of atomic numbers
        positions: Array of atomic positions
        temperature: Optional temperature
        pressure: Optional pressure
        max_steps: Maximum number of relaxation steps
        step_size: Step size for atomic position updates
        force_tol: Force tolerance for convergence
        
    Returns:
        Dictionary with relaxed structure and energies
    """
    # Create force field calculator
    calculator = MatterSimForceField(
        model=model,
        with_temperature=model.with_temperature,
        with_pressure=model.with_pressure
    )
    
    # Copy initial positions
    current_pos = positions.copy()
    
    # Relaxation loop
    energies = []
    max_forces = []
    
    for step in range(max_steps):
        # Compute energy and forces
        results = calculator.compute(
            atomic_numbers, current_pos, temperature, pressure
        )
        
        energy = results['energy']
        forces = results['forces']
        
        # Calculate maximum force magnitude
        force_magnitudes = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_magnitudes)
        
        # Save current values
        energies.append(energy)
        max_forces.append(max_force)
        
        # Print progress
        print(f"Step {step+1}/{max_steps} | "
              f"Energy: {energy:.6f} eV | "
              f"Max Force: {max_force:.6f} eV/Å")
        
        # Check convergence
        if max_force < force_tol:
            print(f"Relaxation converged at step {step+1}")
            break
        
        # Update positions
        current_pos += step_size * forces
    
    # Final computation
    final_results = calculator.compute(
        atomic_numbers, current_pos, temperature, pressure
    )
    
    # Prepare output
    relaxation_results = {
        'initial_positions': positions.tolist(),
        'relaxed_positions': current_pos.tolist(),
        'initial_energy': float(energies[0]),
        'final_energy': float(final_results['energy']),
        'energy_difference': float(energies[0] - final_results['energy']),
        'steps': len(energies),
        'converged': max_forces[-1] < force_tol,
        'energies': energies,
        'max_forces': max_forces
    }
    
    return relaxation_results, current_pos


def visualize_structure(atomic_numbers, positions, output_path):
    """
    Visualize atomic structure and save to file.
    
    Args:
        atomic_numbers: Array of atomic numbers
        positions: Array of atomic positions
        output_path: Path to save visualization
    """
    # Create ASE Atoms object
    atoms = Atoms(numbers=atomic_numbers, positions=positions)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot atoms
    plot_atoms(atoms, ax, rotation=('0x,0y,0z'))
    
    # Add title
    ax.set_title(f"Structure with {len(atomic_numbers)} atoms")
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_xyz_with_properties(atomic_numbers, positions, results, output_path):
    """
    Save structure to XYZ file with properties.
    
    Args:
        atomic_numbers: Array of atomic numbers
        positions: Array of atomic positions
        results: Dictionary with inference results
        output_path: Path to save XYZ file
    """
    # Create ASE Atoms object
    atoms = Atoms(numbers=atomic_numbers, positions=positions)
    
    # Add energy to atoms
    atoms.info['energy'] = results['energy']
    atoms.info['energy_per_atom'] = results['energy_per_atom']
    
    # Add temperature and pressure if available
    if 'temperature' in results:
        atoms.info['temperature'] = results['temperature']
    
    if 'pressure' in results:
        atoms.info['pressure'] = results['pressure']
    
    # Add forces if available
    if 'forces' in results:
        atoms.arrays['forces'] = np.array(results['forces'])
    
    # Write to XYZ file
    write(output_path, atoms)


def plot_relaxation_results(relaxation_results, output_dir):
    """
    Plot relaxation results.
    
    Args:
        relaxation_results: Dictionary with relaxation results
        output_dir: Directory to save plots
    """
    # Create figure for energy vs. step
    plt.figure(figsize=(10, 6))
    plt.plot(relaxation_results['energies'], marker='o')
    plt.xlabel('Step')
    plt.ylabel('Energy (eV)')
    plt.title('Energy vs. Relaxation Step')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'relaxation_energy.png'), dpi=300)
    plt.close()
    
    # Create figure for max force vs. step
    plt.figure(figsize=(10, 6))
    plt.plot(relaxation_results['max_forces'], marker='o')
    plt.xlabel('Step')
    plt.ylabel('Max Force (eV/Å)')
    plt.title('Maximum Force vs. Relaxation Step')
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'relaxation_force.png'), dpi=300)
    plt.close()


def main():
    """Main inference function."""
    args = parse_args()
    
    # Set device
    paddle.set_device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args)
    
    # Read input structure
    data = read_xyz_file(args.input_file)
    atomic_numbers = data['atomic_numbers']
    positions = data['positions']
    
    # Run relaxation if requested
    if args.relax:
        print(f"Relaxing structure with {len(atomic_numbers)} atoms...")
        relaxation_results, relaxed_positions = relax_structure(
            model, atomic_numbers, positions,
            temperature=args.temperature,
            pressure=args.pressure,
            max_steps=args.max_steps
        )
        
        # Save relaxation results
        with open(os.path.join(args.output_dir, 'relaxation_results.json'), 'w') as f:
            json.dump(relaxation_results, f, indent=2)
        
        # Plot relaxation results
        plot_relaxation_results(relaxation_results, args.output_dir)
        
        # Update positions for inference
        positions = relaxed_positions
    
    # Run inference
    print(f"Running inference on structure with {len(atomic_numbers)} atoms...")
    results = run_inference(
        model, args.input_file,
        temperature=args.temperature,
        pressure=args.pressure,
        device=args.device
    )
    
    # Print summary
    print("\nInference Results:")
    print(f"Energy: {results['energy']:.6f} eV")
    print(f"Energy per atom: {results['energy_per_atom']:.6f} eV/atom")
    
    if 'forces' in results:
        max_force = max(np.linalg.norm(np.array(results['forces']), axis=1))
        print(f"Max force: {max_force:.6f} eV/Å")
    
    if 'stress' in results:
        stress = np.array(results['stress'])
        print(f"Pressure: {-np.trace(stress)/3:.6f} GPa")
    
    # Save results
    if args.save_format in ['all', 'json']:
        with open(os.path.join(args.output_dir, 'inference_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    if args.save_format in ['all', 'xyz']:
        save_xyz_with_properties(
            atomic_numbers, positions, results,
            os.path.join(args.output_dir, 'structure_with_properties.xyz')
        )
    
    if args.save_format in ['all', 'png']:
        visualize_structure(
            atomic_numbers, positions,
            os.path.join(args.output_dir, 'structure_visualization.png')
        )
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
