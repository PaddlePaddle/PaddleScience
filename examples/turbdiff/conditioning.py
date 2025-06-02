"""
Conditioning module for TurbDiff model.
"""

from enum import Enum, auto
import paddle
import paddle.nn as nn


class ConditioningType(Enum):
    """Types of conditioning supported by the model."""
    LOCAL = auto()  # Local conditioning like boundary conditions
    GLOBAL = auto()  # Global conditioning like domain parameters


class Conditioning:
    """Handles the conditioning for the TurbDiff model."""
    
    def __init__(self, cell_type_embedding=None, use_cell_pos=False):
        """
        Initialize the conditioning.
        
        Args:
            cell_type_embedding: Module to embed cell types
            use_cell_pos: Whether to use cell positions as features
        """
        self.cell_type_embedding = cell_type_embedding
        self.use_cell_pos = use_cell_pos
        
        # Calculate conditioning dimensions
        self.local_conditioning_dim = 0
        if cell_type_embedding is not None:
            self.local_conditioning_dim += cell_type_embedding.embedding_dim
        if use_cell_pos:
            self.local_conditioning_dim += 3  # x, y, z positions
            
        self.global_conditioning_dim = 0  # Will be set by specific implementations
        
    def prepare_local_conditioning(self, data):
        """
        Prepare local conditioning from data.
        
        Args:
            data: Input data containing cell types and possibly other information
            
        Returns:
            Tensor of local conditioning features
        """
        conditioning_elements = []
        
        # Add cell type embeddings if available
        if self.cell_type_embedding is not None and hasattr(data, 'cell_type'):
            cell_type_emb = self.cell_type_embedding(data.cell_type)
            conditioning_elements.append(cell_type_emb)
            
        # Add cell positions if requested
        if self.use_cell_pos and hasattr(data, 'cell_pos'):
            # Normalize cell positions to [-1, 1] range
            pos_min = paddle.min(data.cell_pos, axis=[0, 2, 3, 4], keepdim=True)
            pos_max = paddle.max(data.cell_pos, axis=[0, 2, 3, 4], keepdim=True)
            pos_norm = 2 * (data.cell_pos - pos_min) / (pos_max - pos_min + 1e-8) - 1
            conditioning_elements.append(pos_norm)
            
        # Combine all conditioning elements
        if conditioning_elements:
            return paddle.concat(conditioning_elements, axis=1)
        else:
            return None
    
    def prepare_global_conditioning(self, data):
        """
        Prepare global conditioning from data.
        
        Args:
            data: Input data containing global parameters
            
        Returns:
            Tensor of global conditioning features or None
        """
        # This should be implemented by specific model extensions
        # For base implementation, return None
        return None
    
    def prepare_conditioning(self, data):
        """
        Prepare all conditioning from data.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary of conditioning tensors by type
        """
        conditioning = {}
        
        local_cond = self.prepare_local_conditioning(data)
        if local_cond is not None:
            conditioning[ConditioningType.LOCAL] = local_cond
            
        global_cond = self.prepare_global_conditioning(data)
        if global_cond is not None:
            conditioning[ConditioningType.GLOBAL] = global_cond
            
        return conditioning


class CellTypeEmbedding(nn.Layer):
    """Embedding for different cell types (fluid, solid, boundary, etc.)."""
    
    def __init__(self, num_types, embedding_dim):
        """
        Initialize the cell type embedding.
        
        Args:
            num_types: Number of different cell types
            embedding_dim: Dimension of the embedding
        """
        super().__init__()
        self.embedding = nn.Embedding(num_types, embedding_dim)
        self.embedding_dim = embedding_dim
        
    def forward(self, cell_types):
        """
        Get embeddings for cell types.
        
        Args:
            cell_types: Tensor of cell type indices [B, 1, H, W, D]
            
        Returns:
            Embedded tensor [B, embedding_dim, H, W, D]
        """
        # Reshape for embedding lookup
        shape = cell_types.shape
        flat_types = paddle.reshape(cell_types, [shape[0], -1])
        
        # Lookup embeddings
        embeddings = self.embedding(flat_types)
        
        # Reshape back to original shape with embedding dimension
        embeddings = paddle.reshape(embeddings, 
                                   [shape[0], shape[2], shape[3], shape[4], self.embedding_dim])
        embeddings = paddle.transpose(embeddings, [0, 4, 1, 2, 3])
        
        return embeddings
    
    @classmethod
    def create(cls, embedding_type, embedding_dim, num_types=5):
        """
        Create a cell type embedding of the specified type.
        
        Args:
            embedding_type: Type of embedding ('learned', 'fixed', etc.)
            embedding_dim: Dimension of the embedding
            num_types: Number of different cell types
            
        Returns:
            CellTypeEmbedding instance
        """
        if embedding_type == 'learned':
            return cls(num_types, embedding_dim)
        elif embedding_type == 'fixed':
            embedding = cls(num_types, embedding_dim)
            # Initialize with fixed values and freeze parameters
            for param in embedding.parameters():
                param.stop_gradient = True
            return embedding
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
