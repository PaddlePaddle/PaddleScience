
import copy
from typing import Callable, Dict, Tuple

import paddle
from paddle import Tensor
from paddle.nn import GRUCell, Linear

from paddle_geometric.nn.inits import zeros
from paddle_geometric.utils import scatter
from paddle_geometric.utils._scatter import scatter_argmax

TGNMessageStoreType = Dict[int, Tuple[Tensor, Tensor, Tensor, Tensor]]


class TGNMemory(paddle.nn.Layer):
    r"""The Temporal Graph Network (TGN) memory model from the
    `"Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    <https://arxiv.org/abs/2006.10637>`_ paper.

    .. note::

        For an example of using TGN, see `examples/tgn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        tgn.py>`_.

    Args:
        num_nodes (int): The number of nodes to save memories for.
        raw_msg_dim (int): The raw message dimensionality.
        memory_dim (int): The hidden memory dimensionality.
        time_dim (int): The time encoding dimensionality.
        message_module (paddle.nn.Layer): The message function which
            combines source and destination node memory embeddings, the raw
            message and the time encoding.
        aggregator_module (paddle.nn.Layer): The message aggregator function
            which aggregates messages to the same destination into a single
            representation.
    """
    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int,
                 time_dim: int, message_module: Callable,
                 aggregator_module: Callable):
        super(TGNMemory, self).__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)
        self.gru = GRUCell(message_module.out_channels, memory_dim)

        self.memory = self.create_parameter([num_nodes, memory_dim], dtype='float32', default_initializer=paddle.nn.initializer.Constant(0))
        last_update = self.create_parameter([self.num_nodes], dtype='int64', default_initializer=paddle.nn.initializer.Constant(0))
        self.last_update = last_update
        self._assoc = self.create_parameter([num_nodes], dtype='int64', default_initializer=paddle.nn.initializer.Constant(0))

        self.msg_s_store = {}
        self.msg_d_store = {}

        self.reset_parameters()

    @property
    def device(self) -> paddle.device:
        return self.time_enc.lin.weight.device

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if hasattr(self.msg_s_module, 'reset_parameters'):
            self.msg_s_module.reset_parameters()
        if hasattr(self.msg_d_module, 'reset_parameters'):
            self.msg_d_module.reset_parameters()
        if hasattr(self.aggr_module, 'reset_parameters'):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
        self.gru.reset_parameters()
        self.reset_state()

    def reset_state(self):
        """Resets the memory to its initial state."""
        zeros(self.memory)
        zeros(self.last_update)
        self._reset_message_store()

    def detach(self):
        """Detaches the memory from gradient computation."""
        self.memory.detach_()

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp.
        """
        if self.training:
            memory, last_update = self._get_updated_memory(n_id)
        else:
            memory, last_update = self.memory[n_id], self.last_update[n_id]

        return memory, last_update

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor,
                     raw_msg: Tensor):
        """Updates the memory with newly encountered interactions
        :obj:`(src, dst, t, raw_msg)`."""
        n_id = paddle.concat([src, dst]).unique()

        if self.training:
            self._update_memory(n_id)
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
        else:
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
            self._update_memory(n_id)

    def _reset_message_store(self):
        i = self.memory.new_empty((0, ), device=self.device, dtype='int64')
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

    def _update_memory(self, n_id: Tensor):
        memory, last_update = self._get_updated_memory(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def _get_updated_memory(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        self._assoc[n_id] = paddle.arange(n_id.size(0), device=n_id.device)

        # Compute messages (src -> dst).
        msg_s, t_s, src_s, dst_s = self._compute_msg(n_id, self.msg_s_store,
                                                     self.msg_s_module)

        # Compute messages (dst -> src).
        msg_d, t_d, src_d, dst_d = self._compute_msg(n_id, self.msg_d_store,
                                                     self.msg_d_module)

        # Aggregate messages.
        idx = paddle.concat([src_s, src_d], axis=0)
        msg = paddle.concat([msg_s, msg_d], axis=0)
        t = paddle.concat([t_s, t_d], axis=0)
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        # Get local copy of updated memory.
        memory = self.gru(aggr, self.memory[n_id])

        # Get local copy of updated `last_update`.
        dim_size = self.last_update.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce='max')[n_id]

        return memory, last_update

    def _update_msg_store(self, src: Tensor, dst: Tensor, t: Tensor,
                          raw_msg: Tensor, msg_store: TGNMessageStoreType):
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

    def _compute_msg(self, n_id: Tensor, msg_store: TGNMessageStoreType,
                     msg_module: Callable):
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg = list(zip(*data))
        src = paddle.concat(src, axis=0).to(self.device)
        dst = paddle.concat(dst, axis=0).to(self.device)
        t = paddle.concat(t, axis=0).to(self.device)
        raw_msg = [m for i, m in enumerate(raw_msg) if m.numel() > 0 or i == 0]
        raw_msg = paddle.concat(raw_msg, axis=0).to(self.device)
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)

        return msg, t, src, dst

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            self._update_memory(
                paddle.arange(self.num_nodes, device=self.memory.device))
            self._reset_message_store()
        super(TGNMemory, self).train(mode)


class IdentityMessage(paddle.nn.Layer):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super(IdentityMessage, self).__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim

    def forward(self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor,
                t_enc: Tensor):
        return paddle.concat([z_src, z_dst, raw_msg, t_enc], axis=-1)


class LastAggregator(paddle.nn.Layer):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        argmax = scatter_argmax(t, index, dim=0, dim_size=dim_size)
        out = msg.new_zeros((dim_size, msg.shape[-1]))
        mask = argmax < msg.shape[0]  # Filter items with at least one entry.
        out[mask] = msg[argmax[mask]]
        return out


class MeanAggregator(paddle.nn.Layer):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce='mean')


class TimeEncoder(paddle.nn.Layer):
    def __init__(self, out_channels: int):
        super(TimeEncoder, self).__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t: Tensor) -> Tensor:
        return self.lin(t.reshape((-1, 1))).cos()


class LastNeighborLoader:
    def __init__(self, num_nodes: int, size: int, device=None):
        self.size = size

        self.neighbors = paddle.empty([num_nodes, size], dtype='int64',
                                      device=device)
        self.e_id = paddle.empty([num_nodes, size], dtype='int64',
                                 device=device)
        self._assoc = paddle.empty([num_nodes], dtype='int64', device=device)

        self.reset_state()

    def __call__(self, n_id: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        neighbors = self.neighbors[n_id]
        nodes = n_id.view(-1, 1).repeat(1, self.size)
        e_id = self.e_id[n_id]

        mask = e_id >= 0
        neighbors, nodes, e_id = neighbors[mask], nodes[mask], e_id[mask]

        n_id = paddle.concat([n_id, neighbors]).unique()
        self._assoc[n_id] = paddle.arange(n_id.size(0), device=n_id.device)
        neighbors, nodes = self._assoc[neighbors], self._assoc[nodes]

        return n_id, paddle.stack([neighbors, nodes]), e_id

    def insert(self, src: Tensor, dst: Tensor):
        neighbors = paddle.concat([src, dst], axis=0)
        nodes = paddle.concat([dst, src], axis=0)
        e_id = paddle.arange(self.cur_e_id, self.cur_e_id + src.size(0),
                             device=src.device).repeat(2)
        self.cur_e_id += src.numel()

        nodes, perm = nodes.sort()
        neighbors, e_id = neighbors[perm], e_id[perm]

        n_id = nodes.unique()
        self._assoc[n_id] = paddle.arange(n_id.numel(), device=n_id.device)

        dense_id = paddle.arange(nodes.size(0), device=nodes.device) % self.size
        dense_id += self._assoc[nodes].mul_(self.size)

        dense_e_id = e_id.new_full([n_id.numel() * self.size], -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.reshape([-1, self.size])

        dense_neighbors = e_id.new_empty(n_id.numel() * self.size)
        dense_neighbors[dense_id] = neighbors
        dense_neighbors = dense_neighbors.reshape([-1, self.size])

        e_id = paddle.concat([self.e_id[n_id, :self.size], dense_e_id], axis=-1)
        neighbors = paddle.concat([self.neighbors[n_id, :self.size], dense_neighbors], axis=-1)

        e_id, perm = e_id.topk(self.size, axis=-1)
        self.e_id[n_id] = e_id
        self.neighbors[n_id] = paddle.gather(neighbors, 1, perm)

    def reset_state(self):
        self.cur_e_id = 0
        self.e_id.fill_(-1)
