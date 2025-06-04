"""
Example script for using MatterSim model to predict materials properties.

This script demonstrates basic usage of the MatterSim model to predict
energy, forces, and other properties of atomic structures.
"""

import os
import argparse
import yaml
import numpy as np
import paddle
from ase.build import bulk
from ase.io import write
from ase.units import GPa

from model import MatterSimForceField


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MatterSim example script')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='./example_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='gpu',
                        choices=['gpu', 'cpu'],
                        help='Device to use for inference')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_example_structures():
    """Create example crystal structures using ASE."""
    structures = {
        'Si_diamond': bulk('Si', 'diamond', a=5.43),
        'NaCl': bulk('NaCl', 'rocksalt', a=5.64),
        'Al_fcc': bulk('Al', 'fcc', a=4.05),
        'Fe_bcc': bulk('Fe', 'bcc', a=2.87),
        'Cu_fcc': bulk('Cu', 'fcc', a=3.61),
    }
    
    return structures


def main():
    """Main example function."""
    args = parse_args()
    config = load_config(args.config)
    
    # Set device
    paddle.set_device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create example structures
    structures = create_example_structures()
    
    # Initialize MatterSim force field
    print("Initializing MatterSim force field...")
    calculator = MatterSimForceField(
        cutoff=config['model']['cutoff_distance'],
        with_temperature=config['model']['with_temperature'],
        with_pressure=config['model']['with_pressure'],
        device=args.device
    )
    
    # Set temperature and pressure for calculations
    temperature = config['inference']['temperature']
    pressure = config['inference']['pressure']
    
    # Calculate properties for each structure
    results = {}
    
    for name, structure in structures.items():
        print(f"\nCalculating properties for {name}...")
        
        # Extract atomic numbers and positions
        atomic_numbers = structure.get_atomic_numbers()
        positions = structure.get_positions()
        
        # Calculate properties
        properties = calculator.compute(
            atomic_numbers, positions, temperature, pressure
        )
        
        # Print results
        print(f"  Energy: {properties['energy']:.6f} eV")
        print(f"  Energy per atom: {properties['energy']/len(atomic_numbers):.6f} eV/atom")
        
        if 'forces' in properties:
            max_force = np.max(np.linalg.norm(properties['forces'], axis=1))
            print(f"  Maximum force: {max_force:.6f} eV/Å")
        
        if 'stress' in properties:
            stress = properties['stress']
            pressure_gpa = -np.trace(stress)/3 / GPa
            print(f"  Pressure: {pressure_gpa:.2f} GPa")
        
        # Store results
        results[name] = properties
        
        # Save structure with computed properties
        structure.calc = None  # Remove calculator to avoid serialization issues
        structure.info['energy'] = float(properties['energy'])
        structure.info['energy_per_atom'] = float(properties['energy']/len(atomic_numbers))
        
        if 'forces' in properties:
            structure.arrays['forces'] = properties['forces']
        
        # Write structure to XYZ file
        output_path = os.path.join(args.output_dir, f"{name}.xyz")
        write(output_path, structure)
    
    print(f"\nResults saved to {args.output_dir}")
    
    # Example of calculating lattice constants by energy minimization
    print("\nCalculating optimal lattice constant for Si diamond...")
    
    # Create range of lattice constants
    lattice_constants = np.linspace(5.2, 5.6, 9)
    energies = []
    
    for a in lattice_constants:
        # Create Si structure with current lattice constant
        si = bulk('Si', 'diamond', a=a)
        atomic_numbers = si.get_atomic_numbers()
        positions = si.get_positions()
        
        # Calculate energy
        properties = calculator.compute(atomic_numbers, positions)
        energy_per_atom = properties['energy'] / len(atomic_numbers)
        energies.append(energy_per_atom)
        
        print(f"  a = {a:.3f} Å, E = {energy_per_atom:.6f} eV/atom")
    
    # Find optimal lattice constant
    min_idx = np.argmin(energies)
    optimal_a = lattice_constants[min_idx]
    min_energy = energies[min_idx]
    
    print(f"\nOptimal lattice constant: {optimal_a:.3f} Å")
    print(f"Minimum energy: {min_energy:.6f} eV/atom")
    
    # Create plot of energy vs. lattice constant
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(lattice_constants, energies, 'o-')
    plt.axvline(x=optimal_a, color='r', linestyle='--', 
                label=f'Optimal a = {optimal_a:.3f} Å')
    plt.xlabel('Lattice Constant (Å)')
    plt.ylabel('Energy per Atom (eV)')
    plt.title('Energy vs. Lattice Constant for Si Diamond')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.savefig(os.path.join(args.output_dir, 'Si_lattice_optimization.png'), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
