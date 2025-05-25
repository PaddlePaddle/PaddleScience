import os
import os.path as osp
from functools import partial
from math import pi as PI
from math import sqrt
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import paddle
from paddle import nn
from paddle import Tensor

from paddle_geometric.data import Dataset, download_url
from paddle_geometric.nn import radius_graph
from paddle_geometric.nn.inits import glorot_orthogonal
from paddle_geometric.nn.resolver import activation_resolver
from paddle_geometric.typing import OptTensor, SparseTensor
from paddle_geometric.utils import scatter


qm9_target_dict: Dict[int, str] = {
    0: 'mu',
    1: 'alpha',
    2: 'homo',
    3: 'lumo',
    5: 'r2',
    6: 'zpve',
    7: 'U0',
    8: 'U',
    9: 'H',
    10: 'G',
    11: 'Cv',
}


class Envelope(nn.Layer):
    def __init__(self, exponent: int):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x: Tensor) -> Tensor:
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = paddle.pow(x, p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return (1.0 / x + a * x_pow_p0 + b * x_pow_p1 +
                c * x_pow_p2) * (x < 1.0).astype(x.dtype)


class BesselBasisLayer(nn.Layer):
    def __init__(self, num_radial: int, cutoff: float = 5.0,
                 envelope_exponent: int = 5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = self.create_parameter(
            shape=[num_radial], dtype='float32', is_bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        paddle.assign(paddle.arange(1, self.freq.shape[0] + 1, dtype='float32') * PI, self.freq)

    def forward(self, dist: Tensor) -> Tensor:
        dist = paddle.unsqueeze(dist, axis=-1) / self.cutoff
        return self.envelope(dist) * paddle.sin(self.freq * dist)


class SphericalBasisLayer(nn.Layer):
    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
    ):
        super().__init__()
        import sympy as sym

        from paddle_geometric.nn.models.dimenet_utils import (
            bessel_basis,
            real_sph_harm,
        )

        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': paddle.sin, 'cos': paddle.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(partial(self._sph_to_tensor, sph1))
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    @staticmethod
    def _sph_to_tensor(sph, x: Tensor) -> Tensor:
        return paddle.zeros_like(x) + sph

    def forward(self, dist: Tensor, angle: Tensor, idx_kj: Tensor) -> Tensor:
        dist = dist / self.cutoff
        rbf = paddle.stack([f(dist) for f in self.bessel_funcs], axis=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = paddle.stack([f(angle) for f in self.sph_funcs], axis=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].reshape([-1, n, k]) * cbf.reshape([-1, n, 1])).reshape([-1, n * k])
        return out


class EmbeddingBlock(nn.Layer):
    def __init__(self, num_radial: int, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act

        self.emb = nn.Embedding(95, hidden_channels)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels)
        self.lin = nn.Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.initializer.Uniform(-sqrt(3), sqrt(3))(self.emb.weight)
        self.lin_rbf.weight.set_value(paddle.randn(self.lin_rbf.weight.shape))
        self.lin.weight.set_value(paddle.randn(self.lin.weight.shape))

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor, j: Tensor) -> Tensor:
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(paddle.concat([x[i], x[j], rbf], axis=-1)))


class ResidualLayer(nn.Layer):
    def __init__(self, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.initializer.XavierUniform()(self.lin1.weight)
        self.lin1.bias.set_value(paddle.zeros_like(self.lin1.bias))
        nn.initializer.XavierUniform()(self.lin2.weight)
        self.lin2.bias.set_value(paddle.zeros_like(self.lin2.bias))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class InteractionBlock(nn.Layer):
    def __init__(
        self,
        hidden_channels: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        act: callable,
    ):
        super().__init__()
        self.act = act

        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias_attr=False)
        self.lin_sbf = nn.Linear(num_spherical * num_radial, num_bilinear, bias_attr=False)

        # Dense transformations of input messages.
        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.W = self.create_parameter(
            shape=[hidden_channels, num_bilinear, hidden_channels],
            default_initializer=nn.initializer.Normal(mean=0, std=2 / hidden_channels),
        )

        self.layers_before_skip = nn.LayerList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)]
        )
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = nn.LayerList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.initializer.XavierUniform()(self.lin_rbf.weight)
        nn.initializer.XavierUniform()(self.lin_sbf.weight)
        nn.initializer.XavierUniform()(self.lin_kj.weight)
        self.lin_kj.bias.set_value(paddle.zeros_like(self.lin_kj.bias))
        nn.initializer.XavierUniform()(self.lin_ji.weight)
        self.lin_ji.bias.set_value(paddle.zeros_like(self.lin_ji.bias))
        for layer in self.layers_before_skip:
            layer.reset_parameters()
        nn.initializer.XavierUniform()(self.lin.weight)
        self.lin.bias.set_value(paddle.zeros_like(self.lin.bias))
        for layer in self.layers_after_skip:
            layer.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, sbf: Tensor, idx_kj: Tensor, idx_ji: Tensor) -> Tensor:
        rbf = self.lin_rbf(rbf)
        sbf = self.lin_sbf(sbf)

        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))
        x_kj = x_kj * rbf
        x_kj = paddle.einsum('wj,wl,ijl->wi', sbf, x_kj[idx_kj], self.W)
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.shape[0], reduce="sum")

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class InteractionPPBlock(nn.Layer):
    def __init__(
        self,
        hidden_channels: int,
        int_emb_size: int,
        basis_emb_size: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        act: callable,
    ):
        super().__init__()
        self.act = act

        # Transformation of Bessel and spherical basis representations:
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias_attr=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias_attr=False)

        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size, bias_attr=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias_attr=False)

        # Hidden transformation of input message:
        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets:
        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias_attr=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias_attr=False)

        # Residual layers before and after skip connection:
        self.layers_before_skip = nn.LayerList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)]
        )
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = nn.LayerList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.initializer.XavierUniform()(self.lin_rbf1.weight)
        nn.initializer.XavierUniform()(self.lin_rbf2.weight)
        nn.initializer.XavierUniform()(self.lin_sbf1.weight)
        nn.initializer.XavierUniform()(self.lin_sbf2.weight)

        nn.initializer.XavierUniform()(self.lin_kj.weight)
        self.lin_kj.bias.set_value(paddle.zeros_like(self.lin_kj.bias))
        nn.initializer.XavierUniform()(self.lin_ji.weight)
        self.lin_ji.bias.set_value(paddle.zeros_like(self.lin_ji.bias))

        nn.initializer.XavierUniform()(self.lin_down.weight)
        nn.initializer.XavierUniform()(self.lin_up.weight)

        for layer in self.layers_before_skip:
            layer.reset_parameters()
        nn.initializer.XavierUniform()(self.lin.weight)
        self.lin.bias.set_value(paddle.zeros_like(self.lin.bias))
        for layer in self.layers_after_skip:
            layer.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, sbf: Tensor, idx_kj: Tensor, idx_ji: Tensor) -> Tensor:
        # Initial transformation:
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        # Transformation via Bessel basis:
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down project embedding and generating triple-interactions:
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis:
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings:
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.shape[0], reduce="sum")
        x_kj = self.act(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h

class OutputBlock(nn.Layer):
    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        act: callable,
        output_initializer: str = "zeros",
    ):
        assert output_initializer in {"zeros", "glorot_orthogonal"}

        super().__init__()

        self.act = act
        self.output_initializer = output_initializer

        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias_attr=False)
        self.lins = nn.LayerList([nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.lin = nn.Linear(hidden_channels, out_channels, bias_attr=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.initializer.XavierUniform()(self.lin_rbf.weight)
        for lin in self.lins:
            nn.initializer.XavierUniform()(lin.weight)
            lin.bias.set_value(paddle.zeros_like(lin.bias))
        if self.output_initializer == "zeros":
            self.lin.weight.set_value(paddle.zeros_like(self.lin.weight))
        elif self.output_initializer == "glorot_orthogonal":
            nn.initializer.XavierUniform()(self.lin.weight)

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor, num_nodes: Optional[int] = None) -> Tensor:
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce="sum")
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class OutputPPBlock(nn.Layer):
    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_emb_channels: int,
        out_channels: int,
        num_layers: int,
        act: callable,
        output_initializer: str = "zeros",
    ):
        assert output_initializer in {"zeros", "glorot_orthogonal"}

        super().__init__()

        self.act = act
        self.output_initializer = output_initializer

        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias_attr=False)
        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias_attr=False)
        self.lins = nn.LayerList([nn.Linear(out_emb_channels, out_emb_channels) for _ in range(num_layers)])
        self.lin = nn.Linear(out_emb_channels, out_channels, bias_attr=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.initializer.XavierUniform()(self.lin_rbf.weight)
        nn.initializer.XavierUniform()(self.lin_up.weight)
        for lin in self.lins:
            nn.initializer.XavierUniform()(lin.weight)
            lin.bias.set_value(paddle.zeros_like(lin.bias))
        if self.output_initializer == "zeros":
            self.lin.weight.set_value(paddle.zeros_like(self.lin.weight))
        elif self.output_initializer == "glorot_orthogonal":
            nn.initializer.XavierUniform()(self.lin.weight)

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor, num_nodes: Optional[int] = None) -> Tensor:
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce="sum")
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


def triplets(
    edge_index: Tensor,
    num_nodes: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    row, col = edge_index  # j->i

    value = paddle.arange(row.shape[0], dtype=row.dtype)
    adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(axis=1).astype("int64")

    # Node indices (k->j->i) for triplets.
    idx_i = paddle.repeat_interleave(col, num_triplets)
    idx_j = paddle.repeat_interleave(row, num_triplets)
    idx_k = adj_t_row.storage_col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage_value()[mask]
    idx_ji = adj_t_row.storage_row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

class DimeNet(nn.Layer):
    """
    The directional message passing neural network (DimeNet) from the
    paper: "Directional Message Passing for Molecular Graphs"
    (https://arxiv.org/abs/2003.03123).

    DimeNet transforms messages based on the angle between them in a
    rotation-equivariant fashion.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        num_bilinear (int): Size of the bilinear layer tensor.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: 5.0)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the `cutoff` distance. (default: 32)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: 5)
        num_before_skip (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: 1)
        num_after_skip (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: 2)
        num_output_layers (int, optional): Number of linear layers for the
            output blocks. (default: 3)
        act (str or Callable, optional): The activation function.
            (default: "swish")
        output_initializer (str, optional): The initialization method for the
            output layer ("zeros", "glorot_orthogonal"). (default: "zeros")
    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = "swish",
        output_initializer: str = "zeros",
    ):
        super().__init__()

        if num_spherical < 2:
            raise ValueError("'num_spherical' should be greater than 1")

        # Resolve the activation function
        act = self._activation_resolver(act)

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_blocks = num_blocks

        # Radial and spherical basis function layers
        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(
            num_spherical, num_radial, cutoff, envelope_exponent
        )

        # Embedding block
        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)

        # Output blocks
        self.output_blocks = nn.LayerList(
            [
                OutputBlock(
                    num_radial,
                    hidden_channels,
                    out_channels,
                    num_output_layers,
                    act,
                    output_initializer,
                )
                for _ in range(num_blocks + 1)
            ]
        )

        # Interaction blocks
        self.interaction_blocks = nn.LayerList(
            [
                InteractionBlock(
                    hidden_channels,
                    num_bilinear,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act,
                )
                for _ in range(num_blocks)
            ]
        )

    def reset_parameters(self):
        """
        Resets all learnable parameters of the module.
        """
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    @classmethod
    def from_qm9_pretrained(
            cls,
            root: str,
            dataset: Dataset,
            target: int,
    ) -> Tuple['DimeNet', Dataset, Dataset, Dataset]:  # pragma: no cover
        """
        Returns a pre-trained DimeNet model on the PaddlePaddle version of the QM9 dataset,
        trained on the specified target.

        Args:
            root (str): Path to save or load the pre-trained model.
            dataset (Dataset): The dataset object for QM9.
            target (int): The target property to train on.

        Returns:
            Tuple: A pre-trained DimeNet model and split datasets (train, val, test).
        """
        import paddle
        import numpy as np
        import os
        import os.path as osp
        from paddle.utils.download import get_weights_path_from_url

        assert 0 <= target <= 12 and target != 4, "Target index is invalid."

        root = osp.expanduser(osp.normpath(root))
        path = osp.join(root, 'pretrained_dimenet', qm9_target_dict[target])

        os.makedirs(path, exist_ok=True)
        url = f'{cls.url}/{qm9_target_dict[target]}'

        weights_files = [
            'checkpoint', 'ckpt.data-00000-of-00002', 'ckpt.data-00001-of-00002', 'ckpt.index'
        ]

        for file in weights_files:
            if not osp.exists(osp.join(path, file)):
                get_weights_path_from_url(f'{url}/{file}', osp.join(path, file))

        path = osp.join(path, 'ckpt')
        reader = paddle.static.load_program_state(path)

        model = cls(
            hidden_channels=128,
            out_channels=1,
            num_blocks=6,
            num_bilinear=8,
            num_spherical=7,
            num_radial=6,
            cutoff=5.0,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
        )

        def copy_(dst, name, transpose=False):
            """
            Copies weights from the TensorFlow checkpoint to the Paddle model.

            Args:
                dst (paddle.Tensor): Destination tensor in the Paddle model.
                name (str): Name of the source weight in the checkpoint.
                transpose (bool, optional): Whether to transpose the weight. Defaults to False.
            """
            init = np.array(reader[name])
            if transpose:
                init = init.T
            dst.set_value(paddle.to_tensor(init))

        copy_(model.rbf.freq, 'rbf_layer/frequencies')
        copy_(model.emb.emb.weight, 'emb_block/embeddings')
        copy_(model.emb.lin_rbf.weight, 'emb_block/dense_rbf/kernel')
        copy_(model.emb.lin_rbf.bias, 'emb_block/dense_rbf/bias')
        copy_(model.emb.lin.weight, 'emb_block/dense/kernel')
        copy_(model.emb.lin.bias, 'emb_block/dense/bias')

        for i, block in enumerate(model.output_blocks):
            copy_(block.lin_rbf.weight, f'output_blocks/{i}/dense_rbf/kernel')
            for j, lin in enumerate(block.lins):
                copy_(lin.weight, f'output_blocks/{i}/dense_layers/{j}/kernel')
                copy_(lin.bias, f'output_blocks/{i}/dense_layers/{j}/bias')
            copy_(block.lin.weight, f'output_blocks/{i}/dense_final/kernel')

        for i, block in enumerate(model.interaction_blocks):
            copy_(block.lin_rbf.weight, f'int_blocks/{i}/dense_rbf/kernel')
            copy_(block.lin_sbf.weight, f'int_blocks/{i}/dense_sbf/kernel')
            copy_(block.lin_kj.weight, f'int_blocks/{i}/dense_kj/kernel')
            copy_(block.lin_kj.bias, f'int_blocks/{i}/dense_kj/bias')
            copy_(block.lin_ji.weight, f'int_blocks/{i}/dense_ji/kernel')
            copy_(block.lin_ji.bias, f'int_blocks/{i}/dense_ji/bias')
            copy_(block.W, f'int_blocks/{i}/bilinear')
            for j, layer in enumerate(block.layers_before_skip):
                copy_(layer.lin1.weight,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_1/kernel')
                copy_(layer.lin1.bias,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_1/bias')
                copy_(layer.lin2.weight,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_2/kernel')
                copy_(layer.lin2.bias,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_2/bias')
            copy_(block.lin.weight, f'int_blocks/{i}/final_before_skip/kernel')
            copy_(block.lin.bias, f'int_blocks/{i}/final_before_skip/bias')
            for j, layer in enumerate(block.layers_after_skip):
                copy_(layer.lin1.weight,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_1/kernel')
                copy_(layer.lin1.bias,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_1/bias')
                copy_(layer.lin2.weight,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_2/kernel')
                copy_(layer.lin2.bias,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_2/bias')

        # Use the same random seed as the official DimeNet implementation.
        np.random.seed(42)
        perm = np.random.permutation(130831)
        train_idx = paddle.to_tensor(perm[:110000], dtype='int64')
        val_idx = paddle.to_tensor(perm[110000:120000], dtype='int64')
        test_idx = paddle.to_tensor(perm[120000:], dtype='int64')

        return model, (dataset[train_idx], dataset[val_idx], dataset[test_idx])

    def forward(
            self,
            z: Tensor,
            pos: Tensor,
            batch: OptTensor = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            z (paddle.Tensor): Atomic number of each atom with shape
                [num_atoms].
            pos (paddle.Tensor): Coordinates of each atom with shape
                [num_atoms, 3].
            batch (paddle.Tensor, optional): Batch indices assigning each atom
                to a separate molecule with shape [num_atoms].
                Defaults to None.
        """
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.shape[0])

        # Calculate distances.
        dist = paddle.sqrt(paddle.sum(paddle.square(pos[i] - pos[j]), axis=-1))

        # Calculate angles.
        if isinstance(self, DimeNetPlusPlus):
            pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
            a = paddle.sum(pos_ij * pos_jk, axis=-1)
            b = paddle.norm(paddle.cross(pos_ij, pos_jk, axis=1), axis=-1)
        elif isinstance(self, DimeNet):
            pos_ji, pos_ki = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_i]
            a = paddle.sum(pos_ji * pos_ki, axis=-1)
            b = paddle.norm(paddle.cross(pos_ji, pos_ki, axis=1), axis=-1)
        angle = paddle.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.shape[0])

        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=pos.shape[0])

        if batch is None:
            return P.sum(axis=0)
        else:
            return scatter(P, batch, axis=0, reduce='sum')

class DimeNetPlusPlus(DimeNet):
    """
    The DimeNet++ from the `"Fast and Uncertainty-Aware
    Directional Message Passing for Non-Equilibrium Molecules"
    <https://arxiv.org/abs/2011.14115>`_ paper.

    DimeNetPlusPlus is an upgrade to the DimeNet model with
    8x faster and 10% more accurate than DimeNet.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Size of embedding in the interaction block.
        basis_emb_size (int): Size of basis embedding in the interaction block.
        out_emb_channels (int): Size of embedding in the output block.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (str or Callable, optional): The activation function.
            (default: :obj:`"swish"`)
        output_initializer (str, optional): The initialization method for the
            output layer (:obj:`"zeros"`, :obj:`"glorot_orthogonal"`).
            (default: :obj:`"zeros"`)
    """

    url = ('https://raw.githubusercontent.com/gasteigerjo/dimenet/'
           'master/pretrained/dimenet_pp')

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        int_emb_size: int,
        basis_emb_size: int,
        out_emb_channels: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = 'swish',
        output_initializer: str = 'zeros',
    ):
        act = activation_resolver(act)

        super().__init__(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            num_bilinear=1,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act=act,
            output_initializer=output_initializer,
        )

        # Reuse RBF, SBF, and embedding layers from DimeNet.
        self.output_blocks = nn.LayerList([
            OutputPPBlock(
                num_radial,
                hidden_channels,
                out_emb_channels,
                out_channels,
                num_output_layers,
                act,
                output_initializer,
            ) for _ in range(num_blocks + 1)
        ])

        self.interaction_blocks = nn.LayerList([
            InteractionPPBlock(
                hidden_channels,
                int_emb_size,
                basis_emb_size,
                num_spherical,
                num_radial,
                num_before_skip,
                num_after_skip,
                act,
            ) for _ in range(num_blocks)
        ])

        self.reset_parameters()
    @classmethod
    def from_qm9_pretrained(
        cls,
        root: str,
        dataset: Dataset,
        target: int,
    ) -> Tuple['DimeNetPlusPlus', Dataset, Dataset, Dataset]:
        """
        Returns a pre-trained `DimeNetPlusPlus` model on the QM9 dataset, trained on
        the specified target `target`.
        """
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf

        assert target >= 0 and target <= 12 and target != 4

        root = osp.expanduser(osp.normpath(root))
        path = osp.join(root, 'pretrained_dimenet_pp', qm9_target_dict[target])

        os.makedirs(path, exist_ok=True)
        url = f'{cls.url}/{qm9_target_dict[target]}'

        if not osp.exists(osp.join(path, 'checkpoint')):
            download_url(f'{url}/checkpoint', path)
            download_url(f'{url}/ckpt.data-00000-of-00002', path)
            download_url(f'{url}/ckpt.data-00001-of-00002', path)
            download_url(f'{url}/ckpt.index', path)

        path = osp.join(path, 'ckpt')
        reader = tf.train.load_checkpoint(path)

        # Configuration from DimeNet++:
        model = cls(
            hidden_channels=128,
            out_channels=1,
            num_blocks=4,
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=256,
            num_spherical=7,
            num_radial=6,
            cutoff=5.0,
            max_num_neighbors=32,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
        )

        def copy_(src, name):
            init = reader.get_tensor(f'{name}/.ATTRIBUTES/VARIABLE_VALUE')
            init = paddle.to_tensor(init)
            if 'kernel' in name:
                init = init.transpose([1, 0])
            src.set_value(init)

        copy_(model.rbf.freq, 'rbf_layer/frequencies')
        copy_(model.emb.emb.weight, 'emb_block/embeddings')
        copy_(model.emb.lin_rbf.weight, 'emb_block/dense_rbf/kernel')
        copy_(model.emb.lin_rbf.bias, 'emb_block/dense_rbf/bias')
        copy_(model.emb.lin.weight, 'emb_block/dense/kernel')
        copy_(model.emb.lin.bias, 'emb_block/dense/bias')

        for i, block in enumerate(model.output_blocks):
            copy_(block.lin_rbf.weight, f'output_blocks/{i}/dense_rbf/kernel')
            copy_(block.lin_up.weight, f'output_blocks/{i}/up_projection/kernel')
            for j, lin in enumerate(block.lins):
                copy_(lin.weight, f'output_blocks/{i}/dense_layers/{j}/kernel')
                copy_(lin.bias, f'output_blocks/{i}/dense_layers/{j}/bias')
            copy_(block.lin.weight, f'output_blocks/{i}/dense_final/kernel')

        for i, block in enumerate(model.interaction_blocks):
            copy_(block.lin_rbf1.weight, f'int_blocks/{i}/dense_rbf1/kernel')
            copy_(block.lin_rbf2.weight, f'int_blocks/{i}/dense_rbf2/kernel')
            copy_(block.lin_sbf1.weight, f'int_blocks/{i}/dense_sbf1/kernel')
            copy_(block.lin_sbf2.weight, f'int_blocks/{i}/dense_sbf2/kernel')
            copy_(block.lin_ji.weight, f'int_blocks/{i}/dense_ji/kernel')
            copy_(block.lin_ji.bias, f'int_blocks/{i}/dense_ji/bias')
            copy_(block.lin_kj.weight, f'int_blocks/{i}/dense_kj/kernel')
            copy_(block.lin_kj.bias, f'int_blocks/{i}/dense_kj/bias')
            copy_(block.lin_down.weight, f'int_blocks/{i}/down_projection/kernel')
            copy_(block.lin_up.weight, f'int_blocks/{i}/up_projection/kernel')

            for j, layer in enumerate(block.layers_before_skip):
                copy_(layer.lin1.weight,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_1/kernel')
                copy_(layer.lin1.bias,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_1/bias')
                copy_(layer.lin2.weight,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_2/kernel')
                copy_(layer.lin2.bias,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_2/bias')

            copy_(block.lin.weight, f'int_blocks/{i}/final_before_skip/kernel')
            copy_(block.lin.bias, f'int_blocks/{i}/final_before_skip/bias')

            for j, layer in enumerate(block.layers_after_skip):
                copy_(layer.lin1.weight,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_1/kernel')
                copy_(layer.lin1.bias,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_1/bias')
                copy_(layer.lin2.weight,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_2/kernel')
                copy_(layer.lin2.bias,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_2/bias')

        random_state = np.random.RandomState(seed=42)
        perm = paddle.to_tensor(random_state.permutation(np.arange(130831)), dtype='int64')
        train_idx = perm[:110000]
        val_idx = perm[110000:120000]
        test_idx = perm[120000:]

        return model, (dataset[train_idx], dataset[val_idx], dataset[test_idx])