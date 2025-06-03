import paddle
from mattersim.utils.paddle_utils import scatter

from .layers import GatedMLP
from .layers import LinearLayer
from .layers import SigmoidLayer
from .layers import SwishLayer


def polynomial(r: paddle.Tensor, cutoff: float) -> paddle.Tensor:
    """
    Polynomial cutoff function
    Args:
        r (tf.Tensor): radius distance tensor
        cutoff (float): cutoff distance
    Returns: polynomial cutoff functions
    """
    ratio = paddle.divide(x=r, y=paddle.to_tensor(cutoff))
    result = (
        1
        - 6 * paddle.pow(x=ratio, y=5)
        + 15 * paddle.pow(x=ratio, y=4)
        - 10 * paddle.pow(x=ratio, y=3)
    )
    return paddle.clip(x=result, min=0.0)


class ThreeDInteraction(paddle.nn.Layer):
    def __init__(self, max_n, max_l, cutoff, units, spherecal_dim, threebody_cutoff):
        super().__init__()
        self.atom_mlp = SigmoidLayer(in_dim=units, out_dim=spherecal_dim)
        self.edge_gate_mlp = GatedMLP(
            in_dim=spherecal_dim, out_dims=[units], activation="swish", use_bias=False
        )
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff

    def forward(
        self,
        edge_attr,
        three_basis,
        atom_attr,
        edge_index,
        three_body_index,
        edge_length,
        num_edges,
        num_triple_ij,
    ):
        atom_mask = (
            self.atom_mlp(atom_attr)[edge_index[0][three_body_index[:, 1]]]
            * polynomial(edge_length[three_body_index[:, 0]], self.threebody_cutoff)
            * polynomial(edge_length[three_body_index[:, 1]], self.threebody_cutoff)
        )
        three_basis = three_basis * atom_mask
        index_map = paddle.arange(end=paddle.sum(x=num_edges).item()).to(
            edge_length.place
        )
        index_map = paddle.repeat_interleave(x=index_map, repeats=num_triple_ij).to(
            edge_length.place
        )
        e_ij_tuda = scatter(
            three_basis,
            index_map,
            dim=0,
            reduce="sum",
            dim_size=paddle.sum(x=num_edges).item(),
        )
        edge_attr_prime = edge_attr + self.edge_gate_mlp(e_ij_tuda)
        return edge_attr_prime


class AtomLayer(paddle.nn.Layer):
    """
    v_i'=v_i+sum(phi(v+i,v_j,e_ij',u)W*e_ij^0)
    """

    def __init__(self, atom_attr_dim, edge_attr_dim, spherecal_dim):
        super().__init__()
        self.gated_mlp = GatedMLP(
            in_dim=2 * atom_attr_dim + spherecal_dim, out_dims=[128, 64, atom_attr_dim]
        )
        self.edge_layer = LinearLayer(in_dim=edge_attr_dim, out_dim=1)

    def forward(self, atom_attr, edge_attr, edge_index, edge_attr_prime, num_atoms):
        feat = paddle.concat(
            x=[atom_attr[edge_index[0]], atom_attr[edge_index[1]], edge_attr_prime],
            axis=1,
        )
        atom_attr_prime = self.gated_mlp(feat) * self.edge_layer(edge_attr)
        atom_attr_prime = scatter(
            atom_attr_prime,
            edge_index[1],
            dim=0,
            dim_size=paddle.sum(x=num_atoms).item(),
        )
        return atom_attr_prime + atom_attr


class EdgeLayer(paddle.nn.Layer):
    """e_ij'=e_ij+phi(v_i,v_j,e_ij,u)W*e_ij^0"""

    def init(self, atom_attr_dim, edge_attr_dim, spherecal_dim):
        super().__init__()
        self.gated_mlp = GatedMLP(
            in_dim=2 * atom_attr_dim + spherecal_dim, out_dims=[128, 64, edge_attr_dim]
        )
        self.edge_layer = LinearLayer(in_dim=edge_attr_dim, out_dim=1)

    def forward(self, atom_attr, edge_attr, edge_index, edge_attr_prime):
        feat = paddle.concat(
            x=[atom_attr[edge_index[0]], atom_attr[edge_index[1]], edge_attr_prime],
            axis=1,
        )
        edge_attr_prime = self.gated_mlp(feat) * self.edge_layer(edge_attr)
        return edge_attr_prime + edge_attr


class MainBlock(paddle.nn.Layer):
    """
    MainBlock for Message Passing in M3GNet
    """

    def __init__(self, max_n, max_l, cutoff, units, spherical_dim, threebody_cutoff):
        super().__init__()
        self.gated_mlp_atom = GatedMLP(
            in_dim=2 * units + units, out_dims=[units, units], activation="swish"
        )
        self.edge_layer_atom = SwishLayer(
            in_dim=spherical_dim, out_dim=units, bias=False
        )
        self.gated_mlp_edge = GatedMLP(
            in_dim=2 * units + units, out_dims=[units, units], activation="swish"
        )
        self.edge_layer_edge = LinearLayer(
            in_dim=spherical_dim, out_dim=units, bias=False
        )
        self.three_body = ThreeDInteraction(
            max_n, max_l, cutoff, units, max_n * max_l, threebody_cutoff
        )

    def forward(
        self,
        atom_attr,
        edge_attr,
        edge_attr_zero,
        edge_index,
        three_basis,
        three_body_index,
        edge_length,
        num_edges,
        num_triple_ij,
        num_atoms,
    ):
        # threebody interaction
        edge_attr = self.three_body(
            edge_attr,
            three_basis,
            atom_attr,
            edge_index,
            three_body_index,
            edge_length,
            num_edges,
            num_triple_ij.view(-1),
        )
        # update bond feature
        feat = paddle.concat(
            x=[atom_attr[edge_index[0]], atom_attr[edge_index[1]], edge_attr], axis=1
        )
        edge_attr = edge_attr + self.gated_mlp_edge(feat) * self.edge_layer_edge(
            edge_attr_zero
        )

        # update atom feature
        feat = paddle.concat(
            x=[atom_attr[edge_index[0]], atom_attr[edge_index[1]], edge_attr], axis=1
        )
        atom_attr_prime = self.gated_mlp_atom(feat) * self.edge_layer_atom(
            edge_attr_zero
        )
        atom_attr = atom_attr + scatter(
            atom_attr_prime,
            edge_index[0],
            dim=0,
            dim_size=paddle.sum(x=num_atoms).item(),
        )

        return atom_attr, edge_attr
