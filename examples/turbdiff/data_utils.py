"""
Data handling utilities for TurbDiff model.
"""

import os
import h5py
import numpy as np
import paddle
from paddle.io import Dataset, DataLoader


class Variable:
    """Represents a physical variable in the flow field."""
    
    def __init__(self, name, dims, index=None):
        """
        Initialize variable.
        
        Args:
            name: Variable name
            dims: Number of dimensions (components)
            index: Optional index for multi-variable storage
        """
        self.name = name
        self.dims = dims
        self.index = index
        
    def __repr__(self):
        return f"Variable({self.name}, dims={self.dims})"


class TurbulenceDataset(Dataset):
    """Dataset for 3D turbulence data."""
    
    def __init__(self, data_dir, split='train', variables=None, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing HDF5 data files
            split: Data split ('train', 'val', 'test')
            variables: List of variables to load
            transform: Optional transform to apply to data
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Default variables if none specified
        self.variables = variables or [
            Variable('U', 3),  # Velocity
            Variable('p', 1),  # Pressure
        ]
        
        # Find all HDF5 files for the given split
        self.data_files = []
        for file in os.listdir(os.path.join(data_dir, split)):
            if file.endswith('.h5'):
                self.data_files.append(os.path.join(data_dir, split, file))
        
        # Load dataset statistics
        self.stats = self._load_statistics()
        
        # Cache for data samples
        self.cache = {}
        self.max_cache_size = 100  # Adjust based on memory availability
    
    def _load_statistics(self):
        """Load dataset statistics for normalization."""
        stats_file = os.path.join(self.data_dir, 'statistics.h5')
        if not os.path.exists(stats_file):
            print(f"Warning: Statistics file {stats_file} not found. Using default normalization.")
            return self._default_statistics()
        
        with h5py.File(stats_file, 'r') as f:
            stats = {}
            for var in self.variables:
                if var.name in f:
                    stats[var.name] = {
                        'mean': paddle.to_tensor(f[var.name]['mean'][()]),
                        'std': paddle.to_tensor(f[var.name]['std'][()]),
                        'min': paddle.to_tensor(f[var.name]['min'][()]),
                        'max': paddle.to_tensor(f[var.name]['max'][()]),
                    }
                else:
                    print(f"Warning: Statistics for {var.name} not found. Using defaults.")
                    stats[var.name] = self._default_variable_statistics(var)
        
        return stats
    
    def _default_statistics(self):
        """Create default statistics if no statistics file is available."""
        stats = {}
        for var in self.variables:
            stats[var.name] = self._default_variable_statistics(var)
        return stats
    
    def _default_variable_statistics(self, var):
        """Create default statistics for a variable."""
        if var.name == 'U':
            # Velocity statistics
            return {
                'mean': paddle.zeros([var.dims]),
                'std': paddle.ones([var.dims]),
                'min': paddle.full([var.dims], -10.0),
                'max': paddle.full([var.dims], 10.0),
            }
        elif var.name == 'p':
            # Pressure statistics
            return {
                'mean': paddle.zeros([var.dims]),
                'std': paddle.ones([var.dims]),
                'min': paddle.full([var.dims], -5.0),
                'max': paddle.full([var.dims], 5.0),
            }
        else:
            # Default statistics for unknown variables
            return {
                'mean': paddle.zeros([var.dims]),
                'std': paddle.ones([var.dims]),
                'min': paddle.full([var.dims], -1.0),
                'max': paddle.full([var.dims], 1.0),
            }
    
    def __len__(self):
        """Get number of samples in the dataset."""
        return len(self.data_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing data fields
        """
        # Check if sample is in cache
        if idx in self.cache:
            return self.cache[idx]
        
        # Load sample from file
        file_path = self.data_files[idx]
        sample = self._load_sample(file_path)
        
        # Apply transforms if any
        if self.transform:
            sample = self.transform(sample)
        
        # Cache sample
        if len(self.cache) < self.max_cache_size:
            self.cache[idx] = sample
        
        return sample
    
    def _load_sample(self, file_path):
        """
        Load a sample from an HDF5 file.
        
        Args:
            file_path: Path to HDF5 file
            
        Returns:
            Dictionary with data fields
        """
        with h5py.File(file_path, 'r') as f:
            sample = {}
            
            # Load variables
            var_data = []
            for var in self.variables:
                if var.name in f:
                    data = paddle.to_tensor(f[var.name][()], dtype=paddle.float32)
                    var_data.append(data)
                else:
                    print(f"Warning: Variable {var.name} not found in {file_path}")
                    # Create zero tensor with appropriate shape
                    shape = list(f.attrs.get('grid_shape', [64, 64, 64]))
                    shape = [var.dims] + shape
                    var_data.append(paddle.zeros(shape, dtype=paddle.float32))
            
            # Combine variables into a single tensor
            if var_data:
                sample['x'] = paddle.concat(var_data, axis=0)
            
            # Load cell types if available
            if 'cell_type' in f:
                sample['cell_type'] = paddle.to_tensor(f['cell_type'][()], dtype=paddle.int64)
            
            # Load cell positions if available
            if 'cell_pos' in f:
                sample['cell_pos'] = paddle.to_tensor(f['cell_pos'][()], dtype=paddle.float32)
            
            # Load cell indices (for boundary conditions)
            if 'cell_idx' in f:
                sample['cell_idx'] = paddle.to_tensor(f['cell_idx'][()], dtype=paddle.int64)
            
            # Load metadata
            sample['metadata'] = {
                'filename': os.path.basename(file_path),
                'shape': sample['x'].shape[1:],
            }
            
            # Add any other attributes from the file
            for key, value in f.attrs.items():
                if isinstance(value, (int, float, str, bool, np.ndarray)):
                    sample['metadata'][key] = value
            
            return sample


class Normalization:
    """Handles normalization and denormalization of data."""
    
    def __init__(self, variables, mode='mean-std'):
        """
        Initialize normalization.
        
        Args:
            variables: List of variables to normalize
            mode: Normalization mode ('mean-std', 'min-max', 'none')
        """
        self.variables = variables
        self.mode = mode
    
    def normalize(self, x, stats):
        """
        Normalize data.
        
        Args:
            x: Input tensor [B, C, H, W, D]
            stats: Statistics dictionary
            
        Returns:
            Normalized tensor
        """
        if self.mode == 'none':
            return x
        
        # Start with a copy of the input
        normalized = paddle.clone(x)
        
        # Normalize each variable
        start_idx = 0
        for var in self.variables:
            if var.name in stats:
                end_idx = start_idx + var.dims
                
                if self.mode == 'mean-std':
                    mean = stats[var.name]['mean']
                    std = stats[var.name]['std']
                    
                    # Handle broadcasting for channel dimension
                    if mean.ndim == 1:
                        mean = mean.reshape([1, -1, 1, 1, 1])
                        std = std.reshape([1, -1, 1, 1, 1])
                    
                    normalized[:, start_idx:end_idx] = (x[:, start_idx:end_idx] - mean) / (std + 1e-8)
                
                elif self.mode == 'min-max':
                    min_val = stats[var.name]['min']
                    max_val = stats[var.name]['max']
                    
                    # Handle broadcasting for channel dimension
                    if min_val.ndim == 1:
                        min_val = min_val.reshape([1, -1, 1, 1, 1])
                        max_val = max_val.reshape([1, -1, 1, 1, 1])
                    
                    normalized[:, start_idx:end_idx] = 2.0 * (x[:, start_idx:end_idx] - min_val) / (max_val - min_val + 1e-8) - 1.0
            
            # Move to next variable
            start_idx += var.dims
        
        return normalized
    
    def denormalize(self, x, stats):
        """
        Denormalize data.
        
        Args:
            x: Normalized tensor [B, C, H, W, D]
            stats: Statistics dictionary
            
        Returns:
            Denormalized tensor
        """
        if self.mode == 'none':
            return x
        
        # Start with a copy of the input
        denormalized = paddle.clone(x)
        
        # Denormalize each variable
        start_idx = 0
        for var in self.variables:
            if var.name in stats:
                end_idx = start_idx + var.dims
                
                if self.mode == 'mean-std':
                    mean = stats[var.name]['mean']
                    std = stats[var.name]['std']
                    
                    # Handle broadcasting for channel dimension
                    if mean.ndim == 1:
                        mean = mean.reshape([1, -1, 1, 1, 1])
                        std = std.reshape([1, -1, 1, 1, 1])
                    
                    denormalized[:, start_idx:end_idx] = x[:, start_idx:end_idx] * (std + 1e-8) + mean
                
                elif self.mode == 'min-max':
                    min_val = stats[var.name]['min']
                    max_val = stats[var.name]['max']
                    
                    # Handle broadcasting for channel dimension
                    if min_val.ndim == 1:
                        min_val = min_val.reshape([1, -1, 1, 1, 1])
                        max_val = max_val.reshape([1, -1, 1, 1, 1])
                    
                    denormalized[:, start_idx:end_idx] = (x[:, start_idx:end_idx] + 1.0) * 0.5 * (max_val - min_val + 1e-8) + min_val
            
            # Move to next variable
            start_idx += var.dims
        
        return denormalized
    
    def normalize_batch(self, batch, stats):
        """Normalize a batch of data."""
        if 'x' in batch:
            batch['x_normalized'] = self.normalize(batch['x'], stats)
        return batch
    
    def denormalize_batch(self, batch, stats):
        """Denormalize a batch of data."""
        if 'x_normalized' in batch:
            batch['x'] = self.denormalize(batch['x_normalized'], stats)
        return batch


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def collate_fn(batch):
    """
    Custom collate function for batching samples.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched sample
    """
    # Extract fields present in all samples
    fields = batch[0].keys()
    
    result = {}
    for field in fields:
        if field == 'metadata':
            # Metadata is not tensors, just collect in a list
            result[field] = [sample[field] for sample in batch]
        else:
            # Stack tensors along the batch dimension
            result[field] = paddle.stack([sample[field] for sample in batch])
    
    return result
