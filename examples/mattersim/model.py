"""
Implementation of MatterSim model architecture in PaddlePaddle.

Based on the paper "MatterSim: A Deep Learning Atomistic Model Across Elements, 
Temperatures and Pressures" (https://arxiv.org/abs/2405.04967).
"""

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class GaussianBasis(nn.Layer):
    """Gaussian basis functions for atom distances."""
    
    def __init__(self, min_distance=0.0, max_distance=6.0, num_centers=128, width=0.2):
        """
        Initialize Gaussian basis.
        
        Args:
            min_distance: Minimum interatomic distance
            max_distance: Maximum interatomic distance
            num_centers: Number of Gaussian centers
            width: Width of Gaussian functions
        """
        super().__init__()
        
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.num_centers = num_centers
        self.width = width
        
        # Create centers for Gaussian functions
        centers = paddle.linspace(min_distance, max_distance, num_centers)
        self.centers = paddle.create_parameter(
            centers, 
            shape=centers.shape,
            dtype='float32',
            default_initializer=nn.initializer.Assign(centers)
        )
        self.centers.stop_gradient = True  # Make centers non-trainable
        
        # Initialize width
        self._width = paddle.to_tensor(width, dtype='float32')
    
    def forward(self, distances):
        """
        Convert interatomic distances to Gaussian basis representation.
        
        Args:
            distances: Tensor of interatomic distances [batch_size, num_edges]
            
        Returns:
            Tensor of basis function values [batch_size, num_edges, num_centers]
        """
        # Expand dimensions for broadcasting
        distances = distances.unsqueeze(-1)  # [batch, edges, 1]
        centers = self.centers.unsqueeze(0).unsqueeze(0)  # [1, 1, centers]
        
        # Compute Gaussian functions
        coeff = -0.5 / (self._width ** 2)
        basis = paddle.exp(coeff * ((distances - centers) ** 2))
        
        # Apply cutoff
        basis = basis * (distances <= self.max_distance).astype('float32')
        
        return basis


class ElementEmbedding(nn.Layer):
    """Embedding layer for chemical elements."""
    
    def __init__(self, embedding_dim=128, max_z=94):
        """
        Initialize element embedding.
        
        Args:
            embedding_dim: Dimension of element embeddings
            max_z: Maximum atomic number (number of elements to embed)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_z = max_z
        
        # Create embedding layer for elements
        self.element_embedding = nn.Embedding(max_z + 1, embedding_dim)
    
    def forward(self, atomic_numbers):
        """
        Get embeddings for atomic numbers.
        
        Args:
            atomic_numbers: Tensor of atomic numbers [batch_size, num_atoms]
            
        Returns:
            Tensor of element embeddings [batch_size, num_atoms, embedding_dim]
        """
        # Ensure atomic numbers are within valid range
        atomic_numbers = paddle.clip(atomic_numbers, min=0, max=self.max_z)
        return self.element_embedding(atomic_numbers)


class MessageBlock(nn.Layer):
    """Message passing block for graph neural network."""
    
    def __init__(self, node_dim=128, edge_dim=128, hidden_dim=256):
        """
        Initialize message block.
        
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # Node update network
        self.node_update = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Edge update network
        self.edge_update = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_dim)
        )
    
    def forward(self, node_features, edge_features, edge_index):
        """
        Perform message passing.
        
        Args:
            node_features: Tensor of node features [batch_size, num_nodes, node_dim]
            edge_features: Tensor of edge features [batch_size, num_edges, edge_dim]
            edge_index: Tensor of edge indices [2, num_edges]
            
        Returns:
            Updated node and edge features
        """
        batch_size = node_features.shape[0]
        num_edges = edge_features.shape[1]
        
        # Extract source and target node indices
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        
        # Gather source and target node features
        src_features = paddle.gather(node_features, src_idx, axis=1)
        dst_features = paddle.gather(node_features, dst_idx, axis=1)
        
        # Update edge features
        edge_inputs = paddle.concat([
            src_features, 
            dst_features, 
            edge_features
        ], axis=-1)
        new_edge_features = edge_features + self.edge_update(edge_inputs)
        
        # Aggregate messages for each node
        messages = new_edge_features
        
        # Use scatter_add to sum messages for each target node
        # This is a simplified approximation since paddle doesn't have direct scatter_add
        message_sum = paddle.zeros_like(node_features)
        for i in range(batch_size):
            for j in range(num_edges):
                target_idx = dst_idx[j]
                message_sum[i, target_idx] += messages[i, j]
        
        # Update node features
        node_inputs = paddle.concat([node_features, message_sum], axis=-1)
        new_node_features = node_features + self.node_update(node_inputs)
        
        return new_node_features, new_edge_features


class MatterSimModel(nn.Layer):
    """MatterSim model for atomistic simulations."""
    
    def __init__(
        self,
        embedding_dim=128,
        num_message_blocks=3,
        hidden_dim=256,
        num_radial_basis=128,
        max_z=94,
        cutoff_distance=6.0,
        with_temperature=True,
        with_pressure=True
    ):
        """
        Initialize MatterSim model.
        
        Args:
            embedding_dim: Dimension of element embeddings
            num_message_blocks: Number of message passing blocks
            hidden_dim: Dimension of hidden layers
            num_radial_basis: Number of radial basis functions
            max_z: Maximum atomic number
            cutoff_distance: Interatomic cutoff distance
            with_temperature: Whether to include temperature conditioning
            with_pressure: Whether to include pressure conditioning
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_message_blocks = num_message_blocks
        self.hidden_dim = hidden_dim
        self.cutoff_distance = cutoff_distance
        self.with_temperature = with_temperature
        self.with_pressure = with_pressure
        
        # Element embedding
        self.element_embedding = ElementEmbedding(embedding_dim, max_z)
        
        # Distance basis
        self.distance_basis = GaussianBasis(
            min_distance=0.0,
            max_distance=cutoff_distance,
            num_centers=num_radial_basis
        )
        
        # Edge embedding
        self.edge_embedding = nn.Sequential(
            nn.Linear(num_radial_basis, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Message passing blocks
        self.message_blocks = nn.LayerList([
            MessageBlock(embedding_dim, embedding_dim, hidden_dim)
            for _ in range(num_message_blocks)
        ])
        
        # Thermodynamic condition embedding
        if with_temperature or with_pressure:
            condition_dim = 0
            if with_temperature:
                condition_dim += 1
            if with_pressure:
                condition_dim += 1
                
            self.condition_embedding = nn.Sequential(
                nn.Linear(condition_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, embedding_dim)
            )
        
        # Output networks
        self.energy_network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, atomic_numbers, positions, edge_index, temperature=None, pressure=None):
        """
        Forward pass of MatterSim model.
        
        Args:
            atomic_numbers: Tensor of atomic numbers [batch_size, num_atoms]
            positions: Tensor of atomic positions [batch_size, num_atoms, 3]
            edge_index: Tensor of edge indices [2, num_edges]
            temperature: Optional tensor of temperatures [batch_size]
            pressure: Optional tensor of pressures [batch_size]
            
        Returns:
            Dictionary with energy, forces, and stress predictions
        """
        batch_size = positions.shape[0]
        num_atoms = positions.shape[1]
        
        # Get element embeddings
        node_features = self.element_embedding(atomic_numbers)
        
        # Calculate interatomic distances
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        
        # Get positions of source and target atoms
        src_pos = paddle.gather(positions, src_idx, axis=1)
        dst_pos = paddle.gather(positions, dst_idx, axis=1)
        
        # Calculate displacement vectors and distances
        displacement = dst_pos - src_pos
        distances = paddle.norm(displacement, axis=-1)
        
        # Apply distance basis
        distance_features = self.distance_basis(distances)
        
        # Get edge features
        edge_features = self.edge_embedding(distance_features)
        
        # Apply thermodynamic condition embedding if available
        if (self.with_temperature and temperature is not None) or \
           (self.with_pressure and pressure is not None):
            # Prepare condition tensor
            conditions = []
            if self.with_temperature and temperature is not None:
                # Normalize temperature (assuming range 0-5000K)
                temp_norm = temperature / 5000.0
                conditions.append(temp_norm.unsqueeze(-1))
            
            if self.with_pressure and pressure is not None:
                # Normalize pressure (assuming range 0-1000GPa)
                press_norm = pressure / 1000.0
                conditions.append(press_norm.unsqueeze(-1))
            
            # Combine conditions
            condition_tensor = paddle.concat(conditions, axis=-1)
            
            # Get condition embedding
            condition_embedding = self.condition_embedding(condition_tensor)
            
            # Add condition embedding to all node features
            condition_embedding = condition_embedding.unsqueeze(1).expand([-1, num_atoms, -1])
            node_features = node_features + condition_embedding
        
        # Apply message passing blocks
        for block in self.message_blocks:
            node_features, edge_features = block(node_features, edge_features, edge_index)
        
        # Calculate per-atom energies
        atom_energies = self.energy_network(node_features).squeeze(-1)
        
        # Sum energies to get total energy
        total_energy = paddle.sum(atom_energies, axis=1)
        
        # Return outputs
        outputs = {
            'energy': total_energy,
            'atom_energies': atom_energies
        }
        
        # Enable force calculation during training
        if self.training:
            # Forces are negative gradients of energy with respect to positions
            forces = -paddle.grad(
                outputs=total_energy.sum(),
                inputs=positions,
                create_graph=True,
                retain_graph=True
            )[0]
            
            outputs['forces'] = forces
            
            # Calculate stress tensor (simplified implementation)
            # True stress calculation would require additional terms
            stress = paddle.zeros([batch_size, 3, 3], dtype=positions.dtype)
            outputs['stress'] = stress
        
        return outputs


class MatterSimForceField:
    """MatterSim force field interface for molecular simulations."""
    
    def __init__(
        self,
        model=None,
        model_path=None,
        device='gpu',
        cutoff=6.0,
        with_temperature=True,
        with_pressure=True
    ):
        """
        Initialize MatterSim force field.
        
        Args:
            model: Pre-loaded MatterSim model
            model_path: Path to model checkpoint file
            device: Device to run model on ('gpu' or 'cpu')
            cutoff: Interatomic cutoff distance
            with_temperature: Whether to include temperature conditioning
            with_pressure: Whether to include pressure conditioning
        """
        self.device = device
        self.cutoff = cutoff
        self.with_temperature = with_temperature
        self.with_pressure = with_pressure
        
        # Initialize model
        if model is not None:
            self.model = model
        else:
            # Create default model
            self.model = MatterSimModel(
                cutoff_distance=cutoff,
                with_temperature=with_temperature,
                with_pressure=with_pressure
            )
            
            # Load checkpoint if provided
            if model_path is not None:
                state_dict = paddle.load(model_path)
                self.model.set_state_dict(state_dict)
        
        # Set device
        paddle.set_device(device)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def compute(self, atomic_numbers, positions, temperature=None, pressure=None):
        """
        Compute energy, forces, and stress for a molecular structure.
        
        Args:
            atomic_numbers: Numpy array of atomic numbers [num_atoms]
            positions: Numpy array of atomic positions [num_atoms, 3]
            temperature: Optional temperature in Kelvin
            pressure: Optional pressure in GPa
            
        Returns:
            Dictionary with energy, forces, and stress predictions
        """
        # Convert inputs to paddle tensors
        atomic_numbers_tensor = paddle.to_tensor(
            atomic_numbers, dtype='int64'
        ).unsqueeze(0)  # Add batch dimension
        
        positions_tensor = paddle.to_tensor(
            positions, dtype='float32'
        ).unsqueeze(0)  # Add batch dimension
        
        # Build edge index based on cutoff distance
        edge_index = self._build_edge_index(positions)
        edge_index_tensor = paddle.to_tensor(edge_index, dtype='int64')
        
        # Prepare temperature and pressure tensors if provided
        temp_tensor = None
        if temperature is not None and self.with_temperature:
            temp_tensor = paddle.to_tensor([temperature], dtype='float32')
        
        press_tensor = None
        if pressure is not None and self.with_pressure:
            press_tensor = paddle.to_tensor([pressure], dtype='float32')
        
        # Compute predictions
        with paddle.no_grad():
            outputs = self.model(
                atomic_numbers_tensor,
                positions_tensor,
                edge_index_tensor,
                temperature=temp_tensor,
                pressure=press_tensor
            )
        
        # Convert outputs to numpy arrays
        results = {
            'energy': outputs['energy'].numpy()[0],
            'atom_energies': outputs['atom_energies'].numpy()[0]
        }
        
        # Calculate forces using finite differences if not computed during forward pass
        if 'forces' not in outputs:
            forces = self._compute_forces_finite_diff(
                atomic_numbers, positions, temperature, pressure
            )
            results['forces'] = forces
        else:
            results['forces'] = outputs['forces'].numpy()[0]
        
        # Include stress if computed
        if 'stress' in outputs:
            results['stress'] = outputs['stress'].numpy()[0]
        
        return results
    
    def _build_edge_index(self, positions):
        """
        Build edge index based on cutoff distance.
        
        Args:
            positions: Numpy array of atomic positions [num_atoms, 3]
            
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
                    if dist <= self.cutoff:
                        sources.append(i)
                        targets.append(j)
        
        # Combine into edge index
        edge_index = np.array([sources, targets])
        
        return edge_index
    
    def _compute_forces_finite_diff(self, atomic_numbers, positions, temperature, pressure, delta=1e-4):
        """
        Compute forces using finite differences.
        
        Args:
            atomic_numbers: Numpy array of atomic numbers [num_atoms]
            positions: Numpy array of atomic positions [num_atoms, 3]
            temperature: Optional temperature in Kelvin
            pressure: Optional pressure in GPa
            delta: Displacement for finite differences
            
        Returns:
            Forces array [num_atoms, 3]
        """
        num_atoms = len(atomic_numbers)
        forces = np.zeros((num_atoms, 3))
        
        # Calculate baseline energy
        baseline = self.compute(atomic_numbers, positions, temperature, pressure)
        baseline_energy = baseline['energy']
        
        # Calculate forces using central differences
        for i in range(num_atoms):
            for j in range(3):
                # Forward displacement
                pos_plus = positions.copy()
                pos_plus[i, j] += delta
                energy_plus = self.compute(atomic_numbers, pos_plus, temperature, pressure)['energy']
                
                # Backward displacement
                pos_minus = positions.copy()
                pos_minus[i, j] -= delta
                energy_minus = self.compute(atomic_numbers, pos_minus, temperature, pressure)['energy']
                
                # Calculate force (negative gradient)
                forces[i, j] = -(energy_plus - energy_minus) / (2 * delta)
        
        return forces
