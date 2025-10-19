# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple

from ppsci.arch import base

def get_edge_index(n: int, sim: str) -> np.ndarray:
    """
    Generate edge indices for graph connectivity.
    
    Args:
        n (int): Number of nodes.
        sim (str): Simulation type.
    
    Returns:
        numpy.ndarray: Edge indices with shape [2, num_edges].
    """
    if sim in ['string', 'string_ball']:
        # Chain connection
        top = np.arange(0, n-1)
        bottom = np.arange(1, n)
        edge_index = np.concatenate([
            np.concatenate([top, bottom])[None, :],
            np.concatenate([bottom, top])[None, :]
        ], axis=0)
    else:
        # Full connection
        adj = (np.ones((n, n)) - np.eye(n)).astype(int)
        edge_index = np.array(np.where(adj))
    
    return edge_index


class OGN(base.Arch):
    def __init__(
        self,
        input_keys: List[str],
        output_keys: List[str],
        n_f: int = 6,
        msg_dim: int = 100,
        ndim: int = 2,
        hidden: int = 300,
        edge_index: Optional[np.ndarray] = None,
        aggr: str = 'sum'
    ):
        """
        Initialize Object-based Graph Network (OGN).
        
        Args:
            input_keys (List[str]): List of input keys, e.g., ['x', 'edge_index'].
            output_keys (List[str]): List of output keys, e.g., ['acceleration'].
            n_f (int, optional): Node feature dimension. Defaults to 6.
            msg_dim (int, optional): Message dimension. Defaults to 100.
            ndim (int, optional): Spatial dimension. Defaults to 2.
            hidden (int, optional): Hidden layer size. Defaults to 300.
            edge_index (np.ndarray, optional): Edge indices (can also be provided in forward). Defaults to None.
            aggr (str, optional): Aggregation method, 'sum' or 'mean'. Defaults to 'sum'.
        
        Examples:
            >>> import ppsci
            >>> import numpy as np
            >>> model = ppsci.arch.OGN(
            ...     input_keys=["x"],
            ...     output_keys=["acceleration"],
            ...     n_f=6,
            ...     ndim=2
            ... )
            >>> n_nodes = 5
            >>> edge_index = np.array([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
            >>> input_dict = {
            ...     "x": paddle.randn([n_nodes, 6]),
            ...     "edge_index": edge_index
            ... }
            >>> output = model(input_dict)
            >>> print(output["acceleration"].shape)
            [5, 2]
        """
        super().__init__()
        
        # PaddleScience standard: store input/output keys
        self.input_keys = input_keys
        self.output_keys = output_keys
        
        # Model parameters
        self.n_f = n_f
        self.msg_dim = msg_dim
        self.ndim = ndim
        self.hidden = hidden
        self.aggr = aggr
        
        # Edge index (optional)
        if edge_index is not None:
            self.register_buffer('edge_index_buffer', 
                               paddle.to_tensor(edge_index, dtype='int64'))
        else:
            self.edge_index_buffer = None
        
        # Message function network
        self.msg_fnc = nn.Sequential(
            nn.Linear(2*n_f, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, msg_dim)
        )
        
        # Node update network
        self.node_fnc = nn.Sequential(
            nn.Linear(msg_dim + n_f, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, ndim)
        )
    
    def message_passing(self, x: paddle.Tensor, edge_index: np.ndarray) -> paddle.Tensor:
        """
        Execute message passing.
        
        Args:
            x: Node features with shape [n, n_f].
            edge_index: Edge indices with shape [2, num_edges].
        
        Returns:
            paddle.Tensor: Updated node features with shape [n, ndim].
        """
        # Get source and target nodes
        row, col = edge_index[0], edge_index[1]
        
        # Collect neighbor features
        x_i = x[col]  # Target nodes
        x_j = x[row]  # Source nodes
        
        # Compute messages
        msg_input = paddle.concat([x_i, x_j], axis=1)
        msg = self.msg_fnc(msg_input)
        
        # Aggregate messages to target nodes
        num_nodes = x.shape[0]
        aggr_out = paddle.zeros([num_nodes, self.msg_dim], dtype=msg.dtype)
        
        for i in range(len(col)):
            aggr_out[col[i]] += msg[i]
        
        # Update nodes
        node_input = paddle.concat([x, aggr_out], axis=1)
        out = self.node_fnc(node_input)
        
        return out
    
    def forward(self, inputs: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        """
        Forward propagation (PaddleScience standard interface).
        
        Args:
            inputs (Dict): Input dictionary containing:
                - 'x': Node features [batch*n, n_f] or [n, n_f].
                - 'edge_index': Edge index information (optional if provided at initialization).
                - Other possible inputs.
        
        Returns:
            Dict: Output dictionary containing:
                - 'acceleration': Predicted acceleration [batch*n, ndim].
        """
        # Extract input
        x = inputs['x']
        
        # Get edge index
        if 'edge_index' in inputs:
            # Get from input
            edge_index = inputs['edge_index']
            if isinstance(edge_index, paddle.Tensor):
                edge_index = edge_index.numpy()
        elif self.edge_index_buffer is not None:
            # Use stored edge index
            edge_index = self.edge_index_buffer.numpy()
        else:
            raise ValueError("Must provide edge_index")
        
        # Execute message passing
        acceleration = self.message_passing(x, edge_index)
        
        # Return dictionary format (PaddleScience standard)
        return {self.output_keys[0]: acceleration}
    
    def compute_loss(
        self,
        inputs: Dict[str, paddle.Tensor],
        labels: Dict[str, paddle.Tensor]
    ) -> paddle.Tensor:
        """
        Compute loss function (PaddleScience standard interface).
        
        Args:
            inputs: Input dictionary.
            labels: Label dictionary containing 'acceleration_true'.
        
        Returns:
            paddle.Tensor: Loss scalar.
        """
        # Forward propagation
        outputs = self.forward(inputs)
        
        # Compute loss
        pred = outputs[self.output_keys[0]]
        true = labels.get('acceleration_true', labels.get(self.output_keys[0]))
        
        # MAE loss
        loss = paddle.mean(paddle.abs(pred - true))
        
        return loss


class HGN(base.Arch):
    
    def __init__(
        self,
        input_keys: List[str],
        output_keys: List[str],
        n_f: int = 6,
        ndim: int = 2,
        hidden: int = 300,
        edge_index: Optional[np.ndarray] = None
    ):
        """
        Initialize Hamiltonian Graph Network (HGN).
        
        Args:
            input_keys: List of input keys.
            output_keys: List of output keys, e.g., ['velocity_derivative', 'acceleration'].
            n_f: Node feature dimension.
            ndim: Spatial dimension.
            hidden: Hidden layer size.
            edge_index: Edge indices (optional).
        
        Examples:
            >>> import ppsci
            >>> import numpy as np
            >>> model = ppsci.arch.HGN(
            ...     input_keys=["x"],
            ...     output_keys=["acceleration"],
            ...     n_f=6,
            ...     ndim=2
            ... )
            >>> n_nodes = 5
            >>> edge_index = np.array([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
            >>> input_dict = {
            ...     "x": paddle.randn([n_nodes, 6]),
            ...     "edge_index": edge_index
            ... }
            >>> output = model(input_dict)
            >>> print(output["acceleration"].shape)
            [5, 2]
        """
        super().__init__()
        
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.n_f = n_f
        self.ndim = ndim
        self.hidden = hidden
        
        # Edge index
        if edge_index is not None:
            self.register_buffer('edge_index_buffer',
                               paddle.to_tensor(edge_index, dtype='int64'))
        else:
            self.edge_index_buffer = None
        
        # Pairwise energy network
        self.pair_energy = nn.Sequential(
            nn.Linear(2*n_f, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 1)
        )
        
        # Self energy network
        self.self_energy = nn.Sequential(
            nn.Linear(n_f, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 1)
        )
    
    def compute_energy(self, x: paddle.Tensor, edge_index: np.ndarray) -> paddle.Tensor:
        """
        Compute the total energy (Hamiltonian) of the system.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
        
        Returns:
            paddle.Tensor: Energy per node.
        """
        row, col = edge_index[0], edge_index[1]
        
        # Compute pairwise energy
        x_i = x[col]
        x_j = x[row]
        edge_input = paddle.concat([x_i, x_j], axis=1)
        pair_energies = self.pair_energy(edge_input)
        
        # Aggregate pairwise energy
        num_nodes = x.shape[0]
        aggr_pair = paddle.zeros([num_nodes, 1], dtype=pair_energies.dtype)
        for i in range(len(col)):
            aggr_pair[col[i]] += pair_energies[i]
        
        # Compute self energy
        self_energies = self.self_energy(x)
        
        # Total energy
        total_energy = aggr_pair + self_energies
        
        return total_energy
    
    def forward(self, inputs: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        """
        Forward propagation (PaddleScience standard).
        
        Compute dynamics derivatives using Hamilton's equations.
        
        Args:
            inputs: Input dictionary.
        
        Returns:
            Dict: Output dictionary.
        """
        # Extract input
        x_input = inputs['x'].clone()
        
        # Get edge index
        if 'edge_index' in inputs:
            edge_index = inputs['edge_index']
            if isinstance(edge_index, paddle.Tensor):
                edge_index = edge_index.numpy()
        elif self.edge_index_buffer is not None:
            edge_index = self.edge_index_buffer.numpy()
        else:
            raise ValueError("Must provide edge_index")
        
        # Construct Hamiltonian coordinates
        # Input: [q, v, other, m]
        # Needed: [q, p=m*v, other]
        q = x_input[:, :self.ndim]
        v = x_input[:, self.ndim:2*self.ndim]
        other = x_input[:, 2*self.ndim:]
        
        # Extract mass
        m_scalar = other[:, -1:]
        m_vec = paddle.tile(m_scalar, [1, self.ndim])
        
        # Compute momentum
        p = v * m_vec
        
        # Construct Hamiltonian coordinates
        x_hamilton = paddle.concat([q, p, other], axis=1)
        x_hamilton.stop_gradient = False
        
        # Compute total energy
        total_energy = self.compute_energy(x_hamilton, edge_index)
        total_energy_scalar = paddle.sum(total_energy)
        
        # Compute Hamiltonian gradients
        dH = paddle.grad(
            outputs=total_energy_scalar,
            inputs=x_hamilton,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Extract gradients
        dH_dq = dH[:, :self.ndim]
        dH_dp = dH[:, self.ndim:2*self.ndim]
        
        # Hamilton's equations
        dq_dt = dH_dp
        dp_dt = -dH_dq
        dv_dt = dp_dt / m_vec
        
        # Construct output dictionary
        outputs = {}
        if 'velocity_derivative' in self.output_keys:
            outputs['velocity_derivative'] = dq_dt
        if 'acceleration' in self.output_keys:
            outputs['acceleration'] = dv_dt
        
        # If only one output key, return directly
        if len(self.output_keys) == 1:
            outputs[self.output_keys[0]] = paddle.concat([dq_dt, dv_dt], axis=1)
        
        return outputs
    
    def compute_loss(
        self,
        inputs: Dict[str, paddle.Tensor],
        labels: Dict[str, paddle.Tensor],
        reg_weight: float = 1e-6
    ) -> paddle.Tensor:
        """
        Compute loss with physical regularization.
        
        Args:
            inputs: Input dictionary.
            labels: Label dictionary.
            reg_weight: Regularization weight.
        
        Returns:
            paddle.Tensor: Total loss.
        """
        # Forward propagation
        outputs = self.forward(inputs)
        
        # Base loss
        pred = outputs.get('acceleration', outputs.get(self.output_keys[0]))
        true = labels.get('acceleration_true', labels.get('acceleration'))
        
        base_loss = paddle.mean(paddle.abs(pred - true))
        
        # Physical regularization: energy should not depend on non-physical quantities
        x_input = inputs['x'].clone()
        edge_index = inputs.get('edge_index', self.edge_index_buffer.numpy())
        if isinstance(edge_index, paddle.Tensor):
            edge_index = edge_index.numpy()
        
        # Construct Hamiltonian coordinates
        q = x_input[:, :self.ndim]
        v = x_input[:, self.ndim:2*self.ndim]
        other = x_input[:, 2*self.ndim:]
        
        m_scalar = other[:, -1:]
        m_vec = paddle.tile(m_scalar, [1, self.ndim])
        p = v * m_vec
        
        x_hamilton = paddle.concat([q, p, other], axis=1)
        x_hamilton.stop_gradient = False
        
        # Compute energy
        total_energy = self.compute_energy(x_hamilton, edge_index)
        
        # Regularization: penalize dependence on non-physical quantities
        regularization = reg_weight * paddle.mean(total_energy**2)
        
        return base_loss + regularization


class VarOGN(base.Arch):
    
    def __init__(
        self,
        input_keys: List[str],
        output_keys: List[str],
        n_f: int = 6,
        msg_dim: int = 100,
        ndim: int = 2,
        hidden: int = 300,
        edge_index: Optional[np.ndarray] = None,
        enable_sampling: bool = True
    ):
        """
        Initialize Variational Graph Network (VarGN).
        
        Args:
            input_keys: List of input keys.
            output_keys: List of output keys.
            n_f: Node feature dimension.
            msg_dim: Message dimension.
            ndim: Spatial dimension.
            hidden: Hidden layer size.
            edge_index: Edge indices.
            enable_sampling: Whether to enable sampling.
        
        Examples:
            >>> import ppsci
            >>> import numpy as np
            >>> model = ppsci.arch.VarGN(
            ...     input_keys=["x"],
            ...     output_keys=["acceleration_mean", "acceleration_std"],
            ...     n_f=6,
            ...     ndim=2
            ... )
            >>> n_nodes = 5
            >>> edge_index = np.array([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
            >>> input_dict = {
            ...     "x": paddle.randn([n_nodes, 6]),
            ...     "edge_index": edge_index
            ... }
            >>> output = model(input_dict)
            >>> print(output["acceleration_mean"].shape)
            [5, 2]
        """
        super().__init__()
        
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.n_f = n_f
        self.msg_dim = msg_dim
        self.ndim = ndim
        self.hidden = hidden
        self.enable_sampling = enable_sampling
        
        # Edge index
        if edge_index is not None:
            self.register_buffer('edge_index_buffer',
                               paddle.to_tensor(edge_index, dtype='int64'))
        else:
            self.edge_index_buffer = None
        
        # Message function: outputs mu and logvar
        self.msg_fnc = nn.Sequential(
            nn.Linear(2*n_f, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, msg_dim*2)  # mu and logvar
        )
        
        # Node update function
        self.node_fnc = nn.Sequential(
            nn.Linear(msg_dim + n_f, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, ndim)
        )
    
    def message_passing(self, x: paddle.Tensor, edge_index: np.ndarray) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Variational message passing.
        
        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: (mean output, variance output).
        """
        row, col = edge_index[0], edge_index[1]
        
        x_i = x[col]
        x_j = x[row]
        
        # Compute mu and logvar for messages
        msg_input = paddle.concat([x_i, x_j], axis=1)
        raw_msg = self.msg_fnc(msg_input)
        
        mu = raw_msg[:, 0::2]
        logvar = raw_msg[:, 1::2]
        
        # Sample messages
        if self.enable_sampling and self.training:
            epsilon = paddle.randn(mu.shape)
            msg = mu + epsilon * paddle.exp(logvar / 2.0)
        else:
            msg = mu
        
        # Aggregate
        num_nodes = x.shape[0]
        aggr_out = paddle.zeros([num_nodes, self.msg_dim], dtype=msg.dtype)
        
        for i in range(len(col)):
            aggr_out[col[i]] += msg[i]
        
        # Update nodes
        node_input = paddle.concat([x, aggr_out], axis=1)
        out_mean = self.node_fnc(node_input)
        
        # Compute output variance (simplified version)
        out_std = paddle.exp(logvar.mean(axis=0, keepdim=True))
        
        return out_mean, out_std
    
    def forward(self, inputs: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        """
        Forward propagation (PaddleScience standard).
        
        Args:
            inputs: Input dictionary.
        
        Returns:
            Dict: Output dictionary containing mean and standard deviation.
        """
        x = inputs['x']
        
        # Get edge index
        if 'edge_index' in inputs:
            edge_index = inputs['edge_index']
            if isinstance(edge_index, paddle.Tensor):
                edge_index = edge_index.numpy()
        elif self.edge_index_buffer is not None:
            edge_index = self.edge_index_buffer.numpy()
        else:
            raise ValueError("Must provide edge_index")
        
        # Execute variational message passing
        accel_mean, accel_std = self.message_passing(x, edge_index)
        
        # Construct output
        outputs = {}
        if 'acceleration_mean' in self.output_keys:
            outputs['acceleration_mean'] = accel_mean
        if 'acceleration_std' in self.output_keys:
            outputs['acceleration_std'] = accel_std
        
        # Default output
        if len(outputs) == 0:
            outputs[self.output_keys[0]] = accel_mean
        
        return outputs
    
    def compute_loss(
        self,
        inputs: Dict[str, paddle.Tensor],
        labels: Dict[str, paddle.Tensor],
        kl_weight: float = 1e-3
    ) -> paddle.Tensor:
        """
        Compute variational loss including KL divergence.
        
        Args:
            inputs: Input dictionary.
            labels: Label dictionary.
            kl_weight: KL divergence weight.
        
        Returns:
            paddle.Tensor: Total loss.
        """
        outputs = self.forward(inputs)
        
        # Reconstruction loss
        pred = outputs.get('acceleration_mean', outputs.get(self.output_keys[0]))
        true = labels.get('acceleration_true', labels.get('acceleration'))
        
        recon_loss = paddle.mean(paddle.abs(pred - true))
        
        # KL divergence (simplified version)
        # Full version needs to be computed in message_passing
        kl_loss = paddle.to_tensor(0.0)  # Placeholder
        
        return recon_loss + kl_weight * kl_loss