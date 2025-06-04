"""
Generate synthetic test data for MatterSim model.

This script creates a set of synthetic atomic structures with computed properties
for testing the MatterSim model implementation.
"""

import os
import argparse
import numpy as np
from ase.build import bulk, molecule
from ase.io import write
from ase.collections import g2


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate test data for MatterSim')
    
    parser.add_argument('--output_dir', type=str, default='./test_data',
                        help='Directory to save generated data')
    parser.add_argument('--num_structures', type=int, default=10,
                        help='Number of structures to generate')
    parser.add_argument('--add_noise', action='store_true',
                        help='Add random noise to positions')
    parser.add_argument('--include_molecules', action='store_true',
                        help='Include molecular structures')
    
    return parser.parse_args()


def create_crystal_structures():
    """Create a set of crystal structures using ASE."""
    structures = {
        'Si_diamond': bulk('Si', 'diamond', a=5.43),
        'NaCl': bulk('NaCl', 'rocksalt', a=5.64),
        'Al_fcc': bulk('Al', 'fcc', a=4.05),
        'Fe_bcc': bulk('Fe', 'bcc', a=2.87),
        'Cu_fcc': bulk('Cu', 'fcc', a=3.61),
        'Ni_fcc': bulk('Ni', 'fcc', a=3.52),
        'Pd_fcc': bulk('Pd', 'fcc', a=3.89),
        'Pt_fcc': bulk('Pt', 'fcc', a=3.92),
        'Au_fcc': bulk('Au', 'fcc', a=4.08),
        'Ag_fcc': bulk('Ag', 'fcc', a=4.09),
    }
    
    return structures


def create_molecular_structures():
    """Create a set of molecular structures from G2 database."""
    structures = {}
    
    # Get names of all G2 molecules
    names = list(g2.names)
    
    # Create molecules (use first 20 or fewer)
    for name in names[:20]:
        try:
            mol = molecule(name)
            structures[f'mol_{name}'] = mol
        except Exception as e:
            print(f"Error creating molecule {name}: {e}")
    
    return structures


def generate_random_energy(atoms):
    """Generate a random but physically plausible energy for a structure."""
    # Typical binding energies are a few eV per atom
    num_atoms = len(atoms)
    
    # Generate energy based on structure type and number of atoms
    if 'mol_' in atoms.info.get('name', ''):
        # For molecules: ~0.1-1 eV per atom
        energy_per_atom = np.random.uniform(-1.0, -0.1)
    else:
        # For crystals: ~1-5 eV per atom
        energy_per_atom = np.random.uniform(-5.0, -1.0)
    
    return energy_per_atom * num_atoms


def generate_random_forces(atoms):
    """Generate random but physically plausible forces for a structure."""
    num_atoms = len(atoms)
    
    # Forces are typically small in equilibrium (0.01-0.1 eV/Å)
    force_scale = np.random.uniform(0.01, 0.1)
    
    # Generate random forces
    forces = np.random.normal(0, force_scale, (num_atoms, 3))
    
    return forces


def apply_random_perturbation(atoms, scale=0.1):
    """Apply random perturbation to atomic positions."""
    positions = atoms.get_positions()
    noise = np.random.normal(0, scale, positions.shape)
    atoms.set_positions(positions + noise)
    return atoms


def main():
    """Main function to generate test data."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create crystal and molecular structures
    crystal_structures = create_crystal_structures()
    structures = {**crystal_structures}
    
    if args.include_molecules:
        molecular_structures = create_molecular_structures()
        structures.update(molecular_structures)
    
    # Select a subset if needed
    if len(structures) > args.num_structures:
        keys = list(structures.keys())
        selected_keys = np.random.choice(keys, args.num_structures, replace=False)
        structures = {k: structures[k] for k in selected_keys}
    
    # Generate data for each structure
    for name, atoms in structures.items():
        # Set structure name
        atoms.info['name'] = name
        
        # Apply random perturbation if requested
        if args.add_noise:
            atoms = apply_random_perturbation(atoms)
        
        # Generate random properties
        energy = generate_random_energy(atoms)
        forces = generate_random_forces(atoms)
        
        # Set structure properties
        atoms.info['energy'] = float(energy)
        atoms.arrays['forces'] = forces
        
        # Add temperature and pressure for some structures
        if np.random.random() < 0.5:
            temperature = np.random.uniform(100, 1000)
            atoms.info['temperature'] = float(temperature)
        
        if np.random.random() < 0.5:
            pressure = np.random.uniform(0, 10)
            atoms.info['pressure'] = float(pressure)
        
        # Save structure to XYZ file
        output_path = os.path.join(args.output_dir, f"{name}.xyz")
        write(output_path, atoms)
        
        print(f"Generated structure: {name}")
    
    print(f"\nGenerated {len(structures)} structures in {args.output_dir}")


if __name__ == "__main__":
    main()
