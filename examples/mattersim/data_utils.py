"""
Data utilities for MatterSim model in PaddleScience.

This module provides dataset classes and utility functions for handling
atomistic data for the MatterSim model.
"""

import os
import glob
import numpy as np
import paddle
from paddle.io import Dataset


def read_xyz_file(file_path):
    """
    Read atomic structure from XYZ file.
    
    Args:
        file_path: Path to XYZ file
        
    Returns:
        Dictionary with atomic numbers, positions, and optional properties
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # First line is number of atoms
    num_atoms = int(lines[0].strip())
    
    # Second line may contain properties
    properties = {}
    property_line = lines[1].strip()
    
    # Try to parse properties (key=value pairs)
    if '=' in property_line:
        for item in property_line.split():
            if '=' in item:
                key, value = item.split('=')
                try:
                    # Try to convert to float
                    properties[key] = float(value)
                except ValueError:
                    # Keep as string if not a number
                    properties[key] = value
    
    # Read atom data (lines 2 to 2+num_atoms)
    atomic_symbols = []
    positions = []
    
    for i in range(2, 2 + num_atoms):
        parts = lines[i].strip().split()
        atomic_symbols.append(parts[0])
        positions.append([float(x) for x in parts[1:4]])
    
    # Convert to numpy arrays
    positions = np.array(positions)
    
    # Convert atomic symbols to atomic numbers
    atomic_numbers = []
    for symbol in atomic_symbols:
        atomic_numbers.append(symbol_to_atomic_number(symbol))
    
    atomic_numbers = np.array(atomic_numbers)
    
    # Create output dictionary
    data = {
        'atomic_numbers': atomic_numbers,
        'positions': positions
    }
    
    # Add properties if available
    if 'energy' in properties:
        data['energy'] = properties['energy']
    
    if 'temperature' in properties:
        data['temperature'] = properties['temperature']
    
    if 'pressure' in properties:
        data['pressure'] = properties['pressure']
    
    # Check for forces (may be in the file or in a separate file)
    if len(parts) >= 7:  # XYZ file contains forces
        forces = []
        for i in range(2, 2 + num_atoms):
            parts = lines[i].strip().split()
            forces.append([float(x) for x in parts[4:7]])
        
        data['forces'] = np.array(forces)
    
    return data


def symbol_to_atomic_number(symbol):
    """
    Convert atomic symbol to atomic number.
    
    Args:
        symbol: Atomic symbol (e.g., 'H', 'C', 'O')
        
    Returns:
        Atomic number
    """
    # Define mapping of atomic symbols to atomic numbers
    symbol_map = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
        'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
        'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
        'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
        'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
        'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
        'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
        'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
        'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94
    }
    
    # Handle case sensitivity
    symbol = symbol.capitalize()
    
    # Return atomic number if symbol exists, otherwise return 0
    return symbol_map.get(symbol, 0)


def build_edge_index(positions, cutoff=6.0):
    """
    Build edge index based on distance cutoff.
    
    Args:
        positions: Array of atomic positions [num_atoms, 3]
        cutoff: Distance cutoff for connecting atoms
        
    Returns:
        Edge index array [2, num_edges]
    """
    num_atoms = len(positions)
    
    # Create all possible pairs
    sources = []
    targets = []
    
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:  # Exclude self-loops
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= cutoff:
                    sources.append(i)
                    targets.append(j)
    
    # Combine into edge index
    edge_index = np.array([sources, targets])
    
    return edge_index


class AtomisticDataset(Dataset):
    """Dataset for atomistic structures."""
    
    def __init__(
        self,
        data_path,
        cutoff=6.0,
        transform=None,
        with_forces=True,
        with_temperature=True,
        with_pressure=True
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to directory containing XYZ files or single XYZ file
            cutoff: Distance cutoff for building edge index
            transform: Optional transform function to apply to data
            with_forces: Whether to include forces in the dataset
            with_temperature: Whether to include temperature in the dataset
            with_pressure: Whether to include pressure in the dataset
        """
        super().__init__()
        
        self.cutoff = cutoff
        self.transform = transform
        self.with_forces = with_forces
        self.with_temperature = with_temperature
        self.with_pressure = with_pressure
        
        # Find all XYZ files
        if os.path.isdir(data_path):
            self.file_paths = sorted(glob.glob(os.path.join(data_path, '*.xyz')))
        elif os.path.isfile(data_path) and data_path.endswith('.xyz'):
            self.file_paths = [data_path]
        else:
            raise ValueError(f"Invalid data path: {data_path}")
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No XYZ files found in {data_path}")
        
        print(f"Found {len(self.file_paths)} structures in the dataset")
    
    def __len__(self):
        """Return dataset size."""
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """Get dataset item by index."""
        # Read XYZ file
        data = read_xyz_file(self.file_paths[idx])
        
        # Build edge index
        edge_index = build_edge_index(data['positions'], self.cutoff)
        
        # Create output dictionary
        item = {
            'atomic_numbers': data['atomic_numbers'],
            'positions': data['positions'],
            'edge_index': edge_index
        }
        
        # Add optional properties if available
        if 'energy' in data:
            item['energy'] = data['energy']
        
        if self.with_forces and 'forces' in data:
            item['forces'] = data['forces']
        
        if self.with_temperature and 'temperature' in data:
            item['temperature'] = data['temperature']
        
        if self.with_pressure and 'pressure' in data:
            item['pressure'] = data['pressure']
        
        # Apply transform if available
        if self.transform is not None:
            item = self.transform(item)
        
        return item


def collate_atomistic_batch(batch):
    """
    Collate batch of atomistic structures.
    
    Args:
        batch: List of dictionaries from dataset
        
    Returns:
        Dictionary with batched tensors
    """
    # Get all keys from the first item
    keys = batch[0].keys()
    
    # Initialize output dictionary
    output = {}
    
    for key in keys:
        if key == 'atomic_numbers':
            # Convert list of arrays to padded tensor
            max_atoms = max(item[key].shape[0] for item in batch)
            padded = []
            
            for item in batch:
                num_atoms = item[key].shape[0]
                padded_item = np.zeros(max_atoms, dtype=item[key].dtype)
                padded_item[:num_atoms] = item[key]
                padded.append(padded_item)
            
            output[key] = paddle.to_tensor(np.stack(padded), dtype='int64')
        
        elif key == 'positions':
            # Convert list of arrays to padded tensor
            max_atoms = max(item[key].shape[0] for item in batch)
            padded = []
            
            for item in batch:
                num_atoms = item[key].shape[0]
                padded_item = np.zeros((max_atoms, 3), dtype=item[key].dtype)
                padded_item[:num_atoms] = item[key]
                padded.append(padded_item)
            
            output[key] = paddle.to_tensor(np.stack(padded), dtype='float32')
        
        elif key == 'forces' and batch[0][key] is not None:
            # Convert list of arrays to padded tensor
            max_atoms = max(item[key].shape[0] for item in batch)
            padded = []
            
            for item in batch:
                num_atoms = item[key].shape[0]
                padded_item = np.zeros((max_atoms, 3), dtype=item[key].dtype)
                padded_item[:num_atoms] = item[key]
                padded.append(padded_item)
            
            output[key] = paddle.to_tensor(np.stack(padded), dtype='float32')
        
        elif key == 'edge_index':
            # Handle edge index separately (more complex)
            # This is a simplified approach - in a real implementation,
            # we would need to shift indices for different graphs in the batch
            batch_edge_indices = []
            offset = 0
            
            for i, item in enumerate(batch):
                num_atoms = item['atomic_numbers'].shape[0]
                edge_index = item[key].copy()
                
                if i > 0:
                    # Shift indices by offset
                    edge_index += offset
                
                batch_edge_indices.append(edge_index)
                offset += num_atoms
            
            # Concatenate all edge indices
            output[key] = paddle.to_tensor(
                np.concatenate(batch_edge_indices, axis=1),
                dtype='int64'
            )
        
        elif key == 'energy' and batch[0][key] is not None:
            # Convert list of scalars to tensor
            values = [item[key] for item in batch]
            output[key] = paddle.to_tensor(values, dtype='float32')
        
        elif key == 'temperature' and batch[0][key] is not None:
            # Convert list of scalars to tensor
            values = [item[key] for item in batch]
            output[key] = paddle.to_tensor(values, dtype='float32')
        
        elif key == 'pressure' and batch[0][key] is not None:
            # Convert list of scalars to tensor
            values = [item[key] for item in batch]
            output[key] = paddle.to_tensor(values, dtype='float32')
    
    return output
