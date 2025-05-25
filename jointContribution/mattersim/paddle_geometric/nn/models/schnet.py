import os
import os.path as osp
import warnings
from math import pi as PI
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Embedding, Linear, LayerList, Sequential

from paddle_geometric.data import Dataset, download_url, extract_zip
from paddle_geometric.io import fs
from paddle_geometric.nn import MessagePassing, SumAggregation, radius_graph
from paddle_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from paddle_geometric.typing import OptTensor

qm9_target_dict: Dict[int, str] = {
    0: 'dipole_moment',
    1: 'isotropic_polarizability',
    2: 'homo',
    3: 'lumo',
    4: 'gap',
    5: 'electronic_spatial_extent',
    6: 'zpve',
    7: 'energy_U0',
    8: 'energy_U',
    9: 'enthalpy_H',
    10: 'free_energy',
    11: 'heat_capacity',
}


class SchNet(paddle.nn.Layer):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        interaction_graph: Optional[Callable] = None,
        max_num_neighbors: int = 32,
        readout: str = 'add',
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: OptTensor = None,
    ):
        super(SchNet, self).__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.dipole = dipole
        self.sum_aggr = SumAggregation()
        self.readout = aggr_resolver('sum' if self.dipole else readout)
        self.mean = mean
        self.std = std
        self.scale = None

        if self.dipole:
            import ase

            atomic_mass = paddle.to_tensor(ase.data.atomic_masses)
            self.register_buffer('atomic_mass', atomic_mass)

        self.embedding = Embedding(100, hidden_channels, padding_idx=0)

        if interaction_graph is not None:
            self.interaction_graph = interaction_graph
        else:
            self.interaction_graph = RadiusInteractionGraph(cutoff, max_num_neighbors)

        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = LayerList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.set_value(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        paddle.nn.initializer.XavierUniform()(self.lin1.weight)
        self.lin1.bias.set_value(paddle.zeros_like(self.lin1.bias))
        paddle.nn.initializer.XavierUniform()(self.lin2.weight)
        self.lin2.bias.set_value(paddle.zeros_like(self.lin2.bias))
        if self.atomref is not None:
            self.atomref.weight.set_value(self.initial_atomref)

    @staticmethod
    def from_qm9_pretrained(
        root: str,
        dataset: Dataset,
        target: int,
    ) -> Tuple['SchNet', Dataset, Dataset, Dataset]:
        import ase
        import schnetpack as spk

        assert target >= 0 and target <= 12
        is_dipole = target == 0

        units = [1] * 12
        units[0] = ase.units.Debye
        units[1] = ase.units.Bohr**3
        units[5] = ase.units.Bohr**2

        root = osp.expanduser(osp.normpath(root))
        os.makedirs(root, exist_ok=True)
        folder = 'trained_schnet_models'
        if not osp.exists(osp.join(root, folder)):
            path = download_url(SchNet.url, root)
            extract_zip(path, root)
            os.unlink(path)

        name = f'qm9_{qm9_target_dict[target]}'
        path = osp.join(root, 'trained_schnet_models', name, 'split.npz')

        split = np.load(path)
        train_idx = split['train_idx']
        val_idx = split['val_idx']
        test_idx = split['test_idx']

        idx = dataset.data.idx
        assoc = idx.new_empty(idx.max().item() + 1)
        assoc[idx] = paddle.arange(idx.size(0))

        train_idx = assoc[train_idx[paddle.isin(train_idx, idx)]]
        val_idx = assoc[val_idx[paddle.isin(val_idx, idx)]]
        test_idx = assoc[test_idx[paddle.isin(test_idx, idx)]]

        path = osp.join(root, 'trained_schnet_models', name, 'best_model')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            state = fs.paddle_load(path, map_location='cpu')

        net = SchNet(
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0,
            dipole=is_dipole,
            atomref=dataset.atomref(target),
        )

        net.embedding.weight.set_value(state.representation.embedding.weight)

        for int1, int2 in zip(state.representation.interactions, net.interactions):
            int2.mlp[0].weight.set_value(int1.filter_network[0].weight)
            int2.mlp[0].bias.set_value(int1.filter_network[0].bias)
            int2.mlp[2].weight.set_value(int1.filter_network[1].weight)
            int2.mlp[2].bias.set_value(int1.filter_network[1].bias)
            int2.lin.weight.set_value(int1.dense.weight)
            int2.lin.bias.set_value(int1.dense.bias)

            int2.conv.lin1.weight.set_value(int1.cfconv.in2f.weight)
            int2.conv.lin2.weight.set_value(int1.cfconv.f2out.weight)
            int2.conv.lin2.bias.set_value(int1.cfconv.f2out.bias)

        net.lin1.weight.set_value(state.output_modules[0].out_net[1].out_net[0].weight)
        net.lin1.bias.set_value(state.output_modules[0].out_net[1].out_net[0].bias)
        net.lin2.weight.set_value(state.output_modules[0].out_net[1].out_net[1].weight)
        net.lin2.bias.set_value(state.output_modules[0].out_net[1].out_net[1].bias)

        mean = state.output_modules[0].atom_pool.average
        net.readout = aggr_resolver('mean' if mean else 'add')

        dipole = state.output_modules[0].__class__.__name__ == 'DipoleMoment'
        net.dipole = dipole

        net.mean = state.output_modules[0].standardize.mean.item()
        net.std = state.output_modules[0].standardize.stddev.item()

        if state.output_modules[0].atomref is not None:
            net.atomref.weight.set_value(state.output_modules[0].atomref.weight)
        else:
            net.atomref = None

        net.scale = 1.0 / units[target]

        return net, (dataset[train_idx], dataset[val_idx], dataset[test_idx])

    def forward(self, z: Tensor, pos: Tensor, batch: OptTensor = None) -> Tensor:
        batch = paddle.zeros_like(z) if batch is None else batch

        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            mass = self.atomic_mass[z].view(-1, 1)
            M = self.sum_aggr(mass, batch, dim=0)
            c = self.sum_aggr(mass * pos, batch, dim=0) / M
            h = h * (pos - c.index_select(0, batch))

        if not self.dipole and self.mean and self.std:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        out = self.readout(h, batch, dim=0)

        if self.dipole:
            out = paddle.norm(out, dim=-1, keepdim=True)

        if self.scale:
            out = self.scale * out

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')


class RadiusInteractionGraph(paddle.nn.Layer):
    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32):
        super(RadiusInteractionGraph, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(axis=-1)
        return edge_index, edge_weight


class InteractionBlock(paddle.nn.Layer):
    def __init__(self, hidden_channels: int, num_gaussians: int,
                 num_filters: int, cutoff: float):
        super(InteractionBlock, self).__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        paddle.nn.initializer.XavierUniform()(self.mlp[0].weight)
        self.mlp[0].bias.set_value(paddle.zeros_like(self.mlp[0].bias))
        paddle.nn.initializer.XavierUniform()(self.mlp[2].weight)
        self.mlp[2].bias.set_value(paddle.zeros_like(self.mlp[2].bias))
        self.conv.reset_parameters()
        paddle.nn.initializer.XavierUniform()(self.lin.weight)
        self.lin.bias.set_value(paddle.zeros_like(self.lin.bias))

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        nn: Sequential,
        cutoff: float,
    ):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        paddle.nn.initializer.XavierUniform()(self.lin1.weight)
        paddle.nn.initializer.XavierUniform()(self.lin2.weight)
        self.lin2.bias.set_value(paddle.zeros_like(self.lin2.bias))

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        C = 0.5 * (paddle.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class GaussianSmearing(paddle.nn.Layer):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ):
        super(GaussianSmearing, self).__init__()
        offset = paddle.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return paddle.exp(self.coeff * paddle.pow(dist, 2))


class ShiftedSoftplus(paddle.nn.Layer):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = paddle.log(paddle.to_tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift