import math
from typing import Optional, Tuple

import paddle
from paddle import Tensor
from paddle.nn import Embedding, LayerNorm, Linear, Layer, Silu

from paddle_geometric.nn import MessagePassing, radius_graph
from paddle_geometric.utils import scatter


class CosineCutoff(Layer):
    r"""Applies a cosine cutoff to the input distances.

    .. math::
        \text{cutoffs} =
        \begin{cases}
        0.5 * (\cos(\frac{\text{distances} * \pi}{\text{cutoff}}) + 1.0),
        & \text{if } \text{distances} < \text{cutoff} \\
        0, & \text{otherwise}
        \end{cases}

    Args:
        cutoff (float): A scalar that determines the point at which the cutoff
            is applied.
    """
    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances: Tensor) -> Tensor:
        r"""Applies a cosine cutoff to the input distances.

        Args:
            distances (paddle.Tensor): A tensor of distances.

        Returns:
            cutoffs (paddle.Tensor): A tensor where the cosine function
                has been applied to the distances,
                but any values that exceed the cutoff are set to 0.
        """
        cutoffs = 0.5 * (paddle.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).astype("float32")
        return cutoffs


class ExpNormalSmearing(Layer):
    r"""Applies exponential normal smearing to the input distances.

    .. math::
        \text{smeared\_dist} = \text{CosineCutoff}(\text{dist})
        * e^{-\beta * (e^{\alpha * (-\text{dist})} - \text{means})^2}

    Args:
        cutoff (float, optional): A scalar that determines the point at which
            the cutoff is applied. (default: :obj:`5.0`)
        num_rbf (int, optional): The number of radial basis functions.
            (default: :obj:`128`)
        trainable (bool, optional): If set to :obj:`False`, the means and betas
            of the RBFs will not be trained. (default: :obj:`True`)
    """
    def __init__(
        self,
        cutoff: float = 5.0,
        num_rbf: int = 128,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.add_parameter("means", self.create_parameter(shape=means.shape, dtype=means.dtype,
                                                              default_initializer=paddle.nn.initializer.Assign(means)))
            self.add_parameter("betas", self.create_parameter(shape=betas.shape, dtype=betas.dtype,
                                                              default_initializer=paddle.nn.initializer.Assign(betas)))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self) -> Tuple[Tensor, Tensor]:
        r"""Initializes the means and betas for the radial basis functions."""
        start_value = paddle.exp(paddle.to_tensor(-self.cutoff))
        means = paddle.linspace(start_value, 1, self.num_rbf)
        betas = paddle.full([self.num_rbf], (2 / self.num_rbf * (1 - start_value))**-2)
        return means, betas

    def reset_parameters(self):
        r"""Resets the means and betas to their initial values."""
        means, betas = self._initial_params()
        self.means.set_value(means)
        self.betas.set_value(betas)

    def forward(self, dist: Tensor) -> Tensor:
        r"""Applies the exponential normal smearing to the input distance.

        Args:
            dist (paddle.Tensor): A tensor of distances.
        """
        dist = paddle.unsqueeze(dist, axis=-1)
        smeared_dist = self.cutoff_fn(dist) * paddle.exp(
            -self.betas * (paddle.exp(self.alpha * (-dist)) - self.means)**2
        )
        return smeared_dist


class Sphere(Layer):
    r"""Computes spherical harmonics of the input data.

    This module computes the spherical harmonics up to a given degree
    :obj:`lmax` for the input tensor of 3D vectors.
    The vectors are assumed to be given in Cartesian coordinates.
    See `here <https://en.wikipedia.org/wiki/Table_of_spherical_harmonics>`_
    for mathematical details.

    Args:
        lmax (int, optional): The maximum degree of the spherical harmonics.
            (default: :obj:`2`)
    """
    def __init__(self, lmax: int = 2) -> None:
        super().__init__()
        self.lmax = lmax

    def forward(self, edge_vec: Tensor) -> Tensor:
        r"""Computes the spherical harmonics of the input tensor.

        Args:
            edge_vec (paddle.Tensor): A tensor of 3D vectors.
        """
        return self._spherical_harmonics(
            self.lmax,
            edge_vec[..., 0],
            edge_vec[..., 1],
            edge_vec[..., 2],
        )

    @staticmethod
    def _spherical_harmonics(
        lmax: int,
        x: Tensor,
        y: Tensor,
        z: Tensor,
    ) -> Tensor:
        r"""Computes the spherical harmonics up to degree :obj:`lmax` of the
        input vectors.

        Args:
            lmax (int): The maximum degree of the spherical harmonics.
            x (paddle.Tensor): The x coordinates of the vectors.
            y (paddle.Tensor): The y coordinates of the vectors.
            z (paddle.Tensor): The z coordinates of the vectors.
        """
        sh_1_0, sh_1_1, sh_1_2 = x, y, z

        if lmax == 1:
            return paddle.stack([sh_1_0, sh_1_1, sh_1_2], axis=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = paddle.pow(y, 2)
        x2z2 = paddle.pow(x, 2) + paddle.pow(z, 2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (paddle.pow(z, 2) - paddle.pow(x, 2))

        if lmax == 2:
            return paddle.stack([
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
            ], axis=-1)

        raise ValueError(f"'lmax' needs to be 1 or 2 (got {lmax})")

class VecLayerNorm(paddle.nn.Layer):
    r"""Applies layer normalization to the input data.

    This module applies a custom layer normalization to a tensor of vectors.
    The normalization can either be :obj:`"max_min"` normalization, or no
    normalization.

    Args:
        hidden_channels (int): The number of hidden channels in the input.
        trainable (bool): If set to :obj:`True`, the normalization weights are
            trainable parameters.
        norm_type (str, optional): The type of normalization to apply, one of
            :obj:`"max_min"` or :obj:`None`. (default: :obj:`"max_min"`)
    """
    def __init__(
        self,
        hidden_channels: int,
        trainable: bool,
        norm_type: Optional[str] = 'max_min',
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.eps = 1e-12

        weight = paddle.ones([self.hidden_channels])
        if trainable:
            self.weight = self.create_parameter(shape=weight.shape, default_initializer=paddle.nn.initializer.Constant(1.0))
        else:
            self.register_buffer('weight', weight)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the normalization weights to their initial values."""
        paddle.assign(paddle.ones_like(self.weight), self.weight)

    def max_min_norm(self, vec: Tensor) -> Tensor:
        r"""Applies max-min normalization to the input tensor.

        Args:
            vec (paddle.Tensor): The input tensor.
        """
        dist = paddle.norm(vec, axis=1, keepdim=True)

        if paddle.all(dist == 0):
            return paddle.zeros_like(vec)

        dist = paddle.clip(dist, min=self.eps)
        direct = vec / dist

        max_val = paddle.max(dist, axis=-1)
        min_val = paddle.min(dist, axis=-1)
        delta = max_val - min_val
        delta = paddle.where(delta == 0, paddle.ones_like(delta), delta)
        dist = (dist - paddle.unsqueeze(min_val, axis=(-1, -2))) / paddle.unsqueeze(delta, axis=(-1, -2))

        return paddle.nn.functional.relu(dist) * direct

    def forward(self, vec: Tensor) -> Tensor:
        r"""Applies the layer normalization to the input tensor.

        Args:
            vec (paddle.Tensor): The input tensor.
        """
        if vec.shape[1] == 3:
            if self.norm_type == 'max_min':
                vec = self.max_min_norm(vec)
            return vec * paddle.unsqueeze(paddle.unsqueeze(self.weight, axis=0), axis=0)
        elif vec.shape[1] == 8:
            vec1, vec2 = paddle.split(vec, [3, 5], axis=1)
            if self.norm_type == 'max_min':
                vec1 = self.max_min_norm(vec1)
                vec2 = self.max_min_norm(vec2)
            vec = paddle.concat([vec1, vec2], axis=1)
            return vec * paddle.unsqueeze(paddle.unsqueeze(self.weight, axis=0), axis=0)

        raise ValueError(f"'{self.__class__.__name__}' only supports 3 or 8 "
                         f"channels (got {vec.shape[1]})")


class Distance(paddle.nn.Layer):
    r"""Computes the pairwise distances between atoms in a molecule.

    This module computes the pairwise distances between atoms in a molecule,
    represented by their positions :obj:`pos`.
    The distances are computed only between points that are within a certain
    cutoff radius.

    Args:
        cutoff (float): The cutoff radius beyond
            which distances are not computed.
        max_num_neighbors (int, optional): The maximum number of neighbors
            considered for each point. (default: :obj:`32`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not
            include self-loops. (default: :obj:`True`)
    """
    def __init__(
        self,
        cutoff: float,
        max_num_neighbors: int = 32,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.add_self_loops = add_self_loops

    def forward(
        self,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Computes the pairwise distances between atoms in the molecule.

        Args:
            pos (paddle.Tensor): The positions of the atoms in the molecule.
            batch (paddle.Tensor): A batch vector, which assigns each node to a
                specific example.

        Returns:
            edge_index (paddle.Tensor): The indices of the edges in the graph.
            edge_weight (paddle.Tensor): The distances between connected nodes.
            edge_vec (paddle.Tensor): The vector differences between connected
                nodes.
        """
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            loop=self.add_self_loops,
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.add_self_loops:
            mask = edge_index[0] != edge_index[1]
            edge_weight = paddle.zeros([edge_vec.shape[0]], dtype=edge_vec.dtype)
            edge_weight[mask] = paddle.norm(edge_vec[mask], axis=-1)
        else:
            edge_weight = paddle.norm(edge_vec, axis=-1)

        return edge_index, edge_weight, edge_vec

class NeighborEmbedding(MessagePassing):
    r"""The :class:`NeighborEmbedding` module from the `"Enhancing Geometric
    Representations for Molecules with Equivariant Vector-Scalar Interactive
    Message Passing" <https://arxiv.org/abs/2210.16518>`_ paper.

    Args:
        hidden_channels (int): The number of hidden channels in the node
            embeddings.
        num_rbf (int): The number of radial basis functions.
        cutoff (float): The cutoff distance.
        max_z (int, optional): The maximum atomic numbers.
            (default: :obj:`100`)
    """

    def __init__(
        self,
        hidden_channels: int,
        num_rbf: int,
        cutoff: float,
        max_z: int = 100,
    ) -> None:
        super().__init__(aggr='add')
        self.embedding = Embedding(num_embeddings=max_z, embedding_dim=hidden_channels)
        self.distance_proj = Linear(num_rbf, hidden_channels)
        self.combine = Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        self.embedding.weight.set_value(
            paddle.nn.initializer.Normal()(self.embedding.weight.shape)
        )
        paddle.nn.initializer.XavierUniform()(self.distance_proj.weight)
        paddle.nn.initializer.XavierUniform()(self.combine.weight)
        if self.distance_proj.bias is not None:
            self.distance_proj.bias.set_value(paddle.zeros_like(self.distance_proj.bias))
        if self.combine.bias is not None:
            self.combine.bias.set_value(paddle.zeros_like(self.combine.bias))

    def forward(
        self,
        z: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        r"""Computes the neighborhood embedding of the nodes in the graph.

        Args:
            z (paddle.Tensor): The atomic numbers.
            x (paddle.Tensor): The node features.
            edge_index (paddle.Tensor): The indices of the edges.
            edge_weight (paddle.Tensor): The weights of the edges.
            edge_attr (paddle.Tensor): The edge features.

        Returns:
            x_neighbors (paddle.Tensor): The neighborhood embeddings of the
                nodes.
        """
        mask = edge_index[0] != edge_index[1]
        if not paddle.all(mask):
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.unsqueeze(-1)

        x_neighbors = self.embedding(z)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W)
        x_neighbors = self.combine(paddle.concat([x, x_neighbors], axis=1))
        return x_neighbors

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W

class EdgeEmbedding(Layer):
    r"""The :class:`EdgeEmbedding` module from the `"Enhancing Geometric
    Representations for Molecules with Equivariant Vector-Scalar Interactive
    Message Passing" <https://arxiv.org/abs/2210.16518>`_ paper.

    Args:
        num_rbf (int): The number of radial basis functions.
        hidden_channels (int): The number of hidden channels in the node
            embeddings.
    """

    def __init__(self, num_rbf: int, hidden_channels: int) -> None:
        super().__init__()
        self.edge_proj = Linear(num_rbf, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        paddle.nn.initializer.XavierUniform()(self.edge_proj.weight)
        if self.edge_proj.bias is not None:
            self.edge_proj.bias.set_value(paddle.zeros_like(self.edge_proj.bias))

    def forward(
        self,
        edge_index: Tensor,
        edge_attr: Tensor,
        x: Tensor,
    ) -> Tensor:
        r"""Computes the edge embeddings of the graph.

        Args:
            edge_index (paddle.Tensor): The indices of the edges.
            edge_attr (paddle.Tensor): The edge features.
            x (paddle.Tensor): The node features.

        Returns:
            out_edge_attr (paddle.Tensor): The edge embeddings.
        """
        x_j = paddle.gather(x, edge_index[0])
        x_i = paddle.gather(x, edge_index[1])
        return (x_i + x_j) * self.edge_proj(edge_attr)

class ViS_MP(MessagePassing):
    r"""The message passing module without vertex geometric features of the
    equivariant vector-scalar interactive graph neural network (ViSNet)
    from the `"Enhancing Geometric Representations for Molecules with
    Equivariant Vector-Scalar Interactive Message Passing"
    <https://arxiv.org/abs/2210.16518>`_ paper.

    Args:
        num_heads (int): The number of attention heads.
        hidden_channels (int): The number of hidden channels in the node
            embeddings.
        cutoff (float): The cutoff distance.
        vecnorm_type (str, optional): The type of normalization to apply to the
            vectors.
        trainable_vecnorm (bool): Whether the normalization weights are
            trainable.
        last_layer (bool, optional): Whether this is the last layer in the
            model. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_heads: int,
        hidden_channels: int,
        cutoff: float,
        vecnorm_type: Optional[str],
        trainable_vecnorm: bool,
        last_layer: bool = False,
    ) -> None:
        super().__init__(aggr='add', node_dim=0)

        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"The number of hidden channels (got {hidden_channels}) must "
                f"be evenly divisible by the number of attention heads "
                f"(got {num_heads})")

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer

        self.layernorm = LayerNorm(hidden_channels)
        self.vec_layernorm = VecLayerNorm(
            hidden_channels,
            trainable=trainable_vecnorm,
            norm_type=vecnorm_type,
        )

        self.act = Silu
        self.attn_activation = Silu

        self.cutoff = CosineCutoff(cutoff)

        self.vec_proj = Linear(hidden_channels, hidden_channels * 3, False)

        self.q_proj = Linear(hidden_channels, hidden_channels)
        self.k_proj = Linear(hidden_channels, hidden_channels)
        self.v_proj = Linear(hidden_channels, hidden_channels)
        self.dk_proj = Linear(hidden_channels, hidden_channels)
        self.dv_proj = Linear(hidden_channels, hidden_channels)

        self.s_proj = Linear(hidden_channels, hidden_channels * 2)
        if not self.last_layer:
            self.f_proj = Linear(hidden_channels, hidden_channels)
            self.w_src_proj = Linear(hidden_channels, hidden_channels, False)
            self.w_trg_proj = Linear(hidden_channels, hidden_channels, False)

        self.o_proj = Linear(hidden_channels, hidden_channels * 3)

        self.reset_parameters()

    @staticmethod
    def vector_rejection(vec: Tensor, d_ij: Tensor) -> Tensor:
        r"""Computes the component of :obj:`vec` orthogonal to :obj:`d_ij`.

        Args:
            vec (paddle.Tensor): The input vector.
            d_ij (paddle.Tensor): The reference vector.
        """
        vec_proj = paddle.sum(vec * d_ij.unsqueeze(2), axis=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        self.layernorm.reset_parameters()
        self.vec_layernorm.reset_parameters()
        paddle.nn.initializer.XavierUniform()(self.q_proj.weight)
        self.q_proj.bias.set_value(paddle.zeros_like(self.q_proj.bias))
        paddle.nn.initializer.XavierUniform()(self.k_proj.weight)
        self.k_proj.bias.set_value(paddle.zeros_like(self.k_proj.bias))
        paddle.nn.initializer.XavierUniform()(self.v_proj.weight)
        self.v_proj.bias.set_value(paddle.zeros_like(self.v_proj.bias))
        paddle.nn.initializer.XavierUniform()(self.o_proj.weight)
        self.o_proj.bias.set_value(paddle.zeros_like(self.o_proj.bias))
        paddle.nn.initializer.XavierUniform()(self.s_proj.weight)
        self.s_proj.bias.set_value(paddle.zeros_like(self.s_proj.bias))

        if not self.last_layer:
            paddle.nn.initializer.XavierUniform()(self.f_proj.weight)
            self.f_proj.bias.set_value(paddle.zeros_like(self.f_proj.bias))
            paddle.nn.initializer.XavierUniform()(self.w_src_proj.weight)
            paddle.nn.initializer.XavierUniform()(self.w_trg_proj.weight)

        paddle.nn.initializer.XavierUniform()(self.vec_proj.weight)
        paddle.nn.initializer.XavierUniform()(self.dk_proj.weight)
        self.dk_proj.bias.set_value(paddle.zeros_like(self.dk_proj.bias))
        paddle.nn.initializer.XavierUniform()(self.dv_proj.weight)
        self.dv_proj.bias.set_value(paddle.zeros_like(self.dv_proj.bias))

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        edge_index: Tensor,
        r_ij: Tensor,
        f_ij: Tensor,
        d_ij: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        r"""Computes the residual scalar and vector features of the nodes and
        scalar features of the edges.

        Args:
            x (paddle.Tensor): The scalar features of the nodes.
            vec (paddle.Tensor):The vector features of the nodes.
            edge_index (paddle.Tensor): The indices of the edges.
            r_ij (paddle.Tensor): The distances between connected nodes.
            f_ij (paddle.Tensor): The scalar features of the edges.
            d_ij (paddle.Tensor): The unit vectors of the edges.

        Returns:
            dx (paddle.Tensor): The residual scalar features of the nodes.
            dvec (paddle.Tensor): The residual vector features of the nodes.
            df_ij (paddle.Tensor, optional): The residual scalar features of the
                edges, or :obj:`None` if this is the last layer.
        """
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)

        q = self.q_proj(x).reshape([-1, self.num_heads, self.head_dim])
        k = self.k_proj(x).reshape([-1, self.num_heads, self.head_dim])
        v = self.v_proj(x).reshape([-1, self.num_heads, self.head_dim])
        dk = self.act(self.dk_proj(f_ij))
        dk = dk.reshape([-1, self.num_heads, self.head_dim])
        dv = self.act(self.dv_proj(f_ij))
        dv = dv.reshape([-1, self.num_heads, self.head_dim])

        vec1, vec2, vec3 = paddle.split(self.vec_proj(vec),
                                        self.hidden_channels, axis=-1)
        vec_dot = paddle.sum(vec1 * vec2, axis=1)

        x, vec_out = self.propagate(edge_index, q=q, k=k, v=v, dk=dk, dv=dv,
                                    vec=vec, r_ij=r_ij, d_ij=d_ij)

        o1, o2, o3 = paddle.split(self.o_proj(x), self.hidden_channels, axis=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if not self.last_layer:
            df_ij = self.edge_updater(edge_index, vec=vec, d_ij=d_ij,
                                      f_ij=f_ij)
            return dx, dvec, df_ij
        else:
            return dx, dvec, None

    def message(self, q_i: Tensor, k_j: Tensor, v_j: Tensor, vec_j: Tensor,
                dk: Tensor, dv: Tensor, r_ij: Tensor,
                d_ij: Tensor) -> Tuple[Tensor, Tensor]:
        attn = paddle.sum(q_i * k_j * dk, axis=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        v_j = v_j * dv
        v_j = (v_j * attn.unsqueeze(2)).reshape([-1, self.hidden_channels])

        s1, s2 = paddle.split(self.act(self.s_proj(v_j)), self.hidden_channels,
                              axis=1)
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)

        return v_j, vec_j

    def edge_update(self, vec_i: Tensor, vec_j: Tensor, d_ij: Tensor,
                    f_ij: Tensor) -> Tensor:
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = paddle.sum(w1 * w2, axis=1)
        df_ij = self.act(self.f_proj(f_ij)) * w_dot
        return df_ij

    def aggregate(
        self,
        features: Tuple[Tensor, Tensor],
        index: Tensor,
        ptr: Optional[Tensor],
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        x, vec = features
        x = paddle_geometric.utils.scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = paddle_geometric.utils.scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec
class ViS_MP_Vertex(ViS_MP):
    r"""
    The message passing module with vertex geometric features for the
    equivariant vector-scalar interactive graph neural network (ViSNet),
    introduced in the paper:
    "Enhancing Geometric Representations for Molecules with
    Equivariant Vector-Scalar Interactive Message Passing"
    (<https://arxiv.org/abs/2210.16518>).

    Args:
        num_heads (int): The number of attention heads.
        hidden_channels (int): The number of hidden channels in the node embeddings.
        cutoff (float): The cutoff distance.
        vecnorm_type (str, optional): The type of normalization applied to the vectors.
        trainable_vecnorm (bool): Whether the normalization weights are trainable.
        last_layer (bool, optional): If True, this is the last layer in the model. Defaults to False.
    """
    def __init__(
        self,
        num_heads: int,
        hidden_channels: int,
        cutoff: float,
        vecnorm_type: Optional[str],
        trainable_vecnorm: bool,
        last_layer: bool = False,
    ) -> None:
        super().__init__(num_heads, hidden_channels, cutoff, vecnorm_type,
                         trainable_vecnorm, last_layer)

        if not self.last_layer:
            self.f_proj = paddle.nn.Linear(hidden_channels, hidden_channels * 2)
            self.t_src_proj = paddle.nn.Linear(hidden_channels, hidden_channels, bias_attr=False)
            self.t_trg_proj = paddle.nn.Linear(hidden_channels, hidden_channels, bias_attr=False)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the parameters of the module.
        """
        super().reset_parameters()

        if not self.last_layer:
            if hasattr(self, 't_src_proj'):
                paddle.nn.initializer.XavierUniform()(self.t_src_proj.weight)
            if hasattr(self, 't_trg_proj'):
                paddle.nn.initializer.XavierUniform()(self.t_trg_proj.weight)

    def edge_update(self, vec_i: Tensor, vec_j: Tensor, d_ij: Tensor,
                    f_ij: Tensor) -> Tensor:
        """
        Updates the edge features.

        Args:
            vec_i (Tensor): Vector features of the source nodes.
            vec_j (Tensor): Vector features of the target nodes.
            d_ij (Tensor): Directional vectors between nodes.
            f_ij (Tensor): Scalar features of the edges.

        Returns:
            Tensor: Updated edge features.
        """
        # Compute the directional vector rejections.
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = paddle.sum(w1 * w2, axis=1)

        t1 = self.vector_rejection(self.t_trg_proj(vec_i), d_ij)
        t2 = self.vector_rejection(self.t_src_proj(vec_i), -d_ij)
        t_dot = paddle.sum(t1 * t2, axis=1)

        # Split the projected features into two components and compute the final edge features.
        f1, f2 = paddle.split(self.act(self.f_proj(f_ij)), self.hidden_channels, axis=-1)

        return f1 * w_dot + f2 * t_dot


class ViSNetBlock(paddle.nn.Layer):
    r"""
    The representation module of the equivariant vector-scalar
    interactive graph neural network (ViSNet) from the paper:
    "Enhancing Geometric Representations for Molecules with
    Equivariant Vector-Scalar Interactive Message Passing"
    (<https://arxiv.org/abs/2210.16518>).

    Args:
        lmax (int, optional): The maximum degree of the spherical harmonics.
            (default: :obj:`1`)
        vecnorm_type (str, optional): The type of normalization to apply to the
            vectors. (default: :obj:`None`)
        trainable_vecnorm (bool, optional): Whether the normalization weights
            are trainable. (default: :obj:`False`)
        num_heads (int, optional): The number of attention heads.
            (default: :obj:`8`)
        num_layers (int, optional): The number of layers in the network.
            (default: :obj:`6`)
        hidden_channels (int, optional): The number of hidden channels in the
            node embeddings. (default: :obj:`128`)
        num_rbf (int, optional): The number of radial basis functions.
            (default: :obj:`32`)
        trainable_rbf (bool, optional): Whether the radial basis function
            parameters are trainable. (default: :obj:`False`)
        max_z (int, optional): The maximum atomic numbers.
            (default: :obj:`100`)
        cutoff (float, optional): The cutoff distance. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors
            considered for each atom. (default: :obj:`32`)
        vertex (bool, optional): Whether to use vertex geometric features.
            (default: :obj:`False`)
    """
    def __init__(
        self,
        lmax: int = 1,
        vecnorm_type: Optional[str] = None,
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_channels: int = 128,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        vertex: bool = False,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.vecnorm_type = vecnorm_type
        self.trainable_vecnorm = trainable_vecnorm
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_rbf = num_rbf
        self.trainable_rbf = trainable_rbf
        self.max_z = max_z
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

        self.embedding = paddle.nn.Embedding(max_z, hidden_channels)
        self.distance = Distance(cutoff, max_num_neighbors=max_num_neighbors)
        self.sphere = Sphere(lmax=lmax)
        self.distance_expansion = ExpNormalSmearing(cutoff, num_rbf, trainable_rbf)
        self.neighbor_embedding = NeighborEmbedding(hidden_channels, num_rbf, cutoff, max_z)
        self.edge_embedding = EdgeEmbedding(num_rbf, hidden_channels)

        self.vis_mp_layers = paddle.nn.LayerList()
        vis_mp_kwargs = dict(
            num_heads=num_heads,
            hidden_channels=hidden_channels,
            cutoff=cutoff,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
        )
        vis_mp_class = ViS_MP if not vertex else ViS_MP_Vertex
        for _ in range(num_layers - 1):
            layer = vis_mp_class(last_layer=False, **vis_mp_kwargs)
            self.vis_mp_layers.append(layer)
        self.vis_mp_layers.append(vis_mp_class(last_layer=True, **vis_mp_kwargs))

        self.out_norm = paddle.nn.LayerNorm(hidden_channels)
        self.vec_out_norm = VecLayerNorm(
            hidden_channels,
            trainable=trainable_vecnorm,
            norm_type=vecnorm_type,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the parameters of the module.
        """
        self.embedding.weight.set_value(paddle.ones_like(self.embedding.weight))
        self.distance_expansion.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        for layer in self.vis_mp_layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()
        self.vec_out_norm.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Computes the scalar and vector features of the nodes.

        Args:
            z (paddle.Tensor): The atomic numbers.
            pos (paddle.Tensor): The coordinates of the atoms.
            batch (paddle.Tensor): A batch vector, which assigns each node to a
                specific example.

        Returns:
            x (paddle.Tensor): The scalar features of the nodes.
            vec (paddle.Tensor): The vector features of the nodes.
        """
        x = self.embedding(z)
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / paddle.norm(edge_vec[mask], axis=1, keepdim=True)
        edge_vec = self.sphere(edge_vec)
        x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)
        vec = paddle.zeros([x.shape[0], ((self.lmax + 1) ** 2) - 1, x.shape[1]],
                           dtype=x.dtype)
        edge_attr = self.edge_embedding(edge_index, edge_attr, x)

        for attn in self.vis_mp_layers[:-1]:
            dx, dvec, dedge_attr = attn(x, vec, edge_index, edge_weight, edge_attr, edge_vec)
            x = x + dx
            vec = vec + dvec
            edge_attr = edge_attr + dedge_attr

        dx, dvec, _ = self.vis_mp_layers[-1](x, vec, edge_index, edge_weight, edge_attr, edge_vec)
        x = x + dx
        vec = vec + dvec

        x = self.out_norm(x)
        vec = self.vec_out_norm(vec)

        return x, vec
class GatedEquivariantBlock(paddle.nn.Layer):
    r"""
    Applies a gated equivariant operation to scalar features and vector
    features from the paper:
    "Enhancing Geometric Representations for Molecules with Equivariant
    Vector-Scalar Interactive Message Passing"
    (<https://arxiv.org/abs/2210.16518>).

    Args:
        hidden_channels (int): The number of hidden channels in the node
            embeddings.
        out_channels (int): The number of output channels.
        intermediate_channels (int, optional): The number of channels in the
            intermediate layer, or :obj:`None` to use the same number as
            :obj:`hidden_channels`. (default: :obj:`None`)
        scalar_activation (bool, optional): Whether to apply a scalar
            activation function to the output node features.
            (default: obj:`False`)
    """
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        intermediate_channels: Optional[int] = None,
        scalar_activation: bool = False,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = paddle.nn.Linear(hidden_channels, hidden_channels, bias_attr=False)
        self.vec2_proj = paddle.nn.Linear(hidden_channels, out_channels, bias_attr=False)

        self.update_net = paddle.nn.Sequential(
            paddle.nn.Linear(hidden_channels * 2, intermediate_channels),
            paddle.nn.Silu(),
            paddle.nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = paddle.nn.Silu() if scalar_activation else None

        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters of the module."""
        paddle.nn.initializer.XavierUniform()(self.vec1_proj.weight)
        paddle.nn.initializer.XavierUniform()(self.vec2_proj.weight)
        paddle.nn.initializer.XavierUniform()(self.update_net[0].weight)
        paddle.nn.initializer.Constant(value=0.0)(self.update_net[0].bias)
        paddle.nn.initializer.XavierUniform()(self.update_net[2].weight)
        paddle.nn.initializer.Constant(value=0.0)(self.update_net[2].bias)

    def forward(self, x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Applies a gated equivariant operation to node features and vector
        features.

        Args:
            x (paddle.Tensor): The scalar features of the nodes.
            v (paddle.Tensor): The vector features of the nodes.
        """
        vec1 = paddle.norm(self.vec1_proj(v), axis=-2)
        vec2 = self.vec2_proj(v)

        x = paddle.concat([x, vec1], axis=-1)
        x, v = paddle.split(self.update_net(x), self.out_channels, axis=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)

        return x, v


class EquivariantScalar(paddle.nn.Layer):
    r"""
    Computes final scalar outputs based on node features and vector
    features.

    Args:
        hidden_channels (int): The number of hidden channels in the node
            embeddings.
    """
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()

        self.output_network = paddle.nn.LayerList([
            GatedEquivariantBlock(
                hidden_channels,
                hidden_channels // 2,
                scalar_activation=True,
            ),
            GatedEquivariantBlock(
                hidden_channels // 2,
                1,
                scalar_activation=False,
            ),
        ])

        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters of the module."""
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Computes the final scalar outputs.

        Args:
            x (paddle.Tensor): The scalar features of the nodes.
            v (paddle.Tensor): The vector features of the nodes.

        Returns:
            out (paddle.Tensor): The final scalar outputs of the nodes.
        """
        for layer in self.output_network:
            x, v = layer(x, v)

        return x + v.sum() * 0

class Atomref(paddle.nn.Layer):
    r"""
    Adds atom reference values to atomic energies.

    Args:
        atomref (paddle.Tensor, optional): A tensor of atom reference values,
            or :obj:`None` if not provided. (default: :obj:`None`)
        max_z (int, optional): The maximum atomic numbers.
            (default: :obj:`100`)
    """
    def __init__(
        self,
        atomref: Optional[Tensor] = None,
        max_z: int = 100,
    ) -> None:
        super().__init__()

        if atomref is None:
            atomref = paddle.zeros((max_z, 1))
        else:
            atomref = paddle.to_tensor(atomref, dtype=paddle.float32)

        if len(atomref.shape) == 1:
            atomref = paddle.unsqueeze(atomref, axis=-1)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = Embedding(len(atomref), 1)

        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters of the module."""
        self.atomref.weight.set_value(self.initial_atomref)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        """
        Adds atom reference values to atomic energies.

        Args:
            x (paddle.Tensor): The atomic energies.
            z (paddle.Tensor): The atomic numbers.
        """
        return x + self.atomref(z)


class ViSNet(paddle.nn.Layer):
    r"""
    Implements the equivariant vector-scalar interactive graph neural network
    (ViSNet).

    Args:
        lmax (int, optional): The maximum degree of the spherical harmonics.
            (default: :obj:`1`)
        vecnorm_type (str, optional): The type of normalization to apply to the
            vectors. (default: :obj:`None`)
        trainable_vecnorm (bool, optional): Whether the normalization weights
            are trainable. (default: :obj:`False`)
        num_heads (int, optional): The number of attention heads.
            (default: :obj:`8`)
        num_layers (int, optional): The number of layers in the network.
            (default: :obj:`6`)
        hidden_channels (int, optional): The number of hidden channels in the
            node embeddings. (default: :obj:`128`)
        num_rbf (int, optional): The number of radial basis functions.
            (default: :obj:`32`)
        trainable_rbf (bool, optional): Whether the radial basis function
            parameters are trainable. (default: :obj:`False`)
        max_z (int, optional): The maximum atomic numbers.
            (default: :obj:`100`)
        cutoff (float, optional): The cutoff distance. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors
            considered for each atom. (default: :obj:`32`)
        vertex (bool, optional): Whether to use vertex geometric features.
            (default: :obj:`False`)
        atomref (paddle.Tensor, optional): A tensor of atom reference values,
            or :obj:`None` if not provided. (default: :obj:`None`)
        reduce_op (str, optional): The type of reduction operation to apply
            (:obj:`"sum"`, :obj:`"mean"`). (default: :obj:`"sum"`)
        mean (float, optional): The mean of the output distribution.
            (default: :obj:`0.0`)
        std (float, optional): The standard deviation of the output
            distribution. (default: :obj:`1.0`)
        derivative (bool, optional): Whether to compute the derivative of the
            output with respect to the positions. (default: :obj:`False`)
    """
    def __init__(
        self,
        lmax: int = 1,
        vecnorm_type: Optional[str] = None,
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_channels: int = 128,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        vertex: bool = False,
        atomref: Optional[Tensor] = None,
        reduce_op: str = "sum",
        mean: float = 0.0,
        std: float = 1.0,
        derivative: bool = False,
    ) -> None:
        super().__init__()

        self.representation_model = ViSNetBlock(
            lmax=lmax,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            num_rbf=num_rbf,
            trainable_rbf=trainable_rbf,
            max_z=max_z,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            vertex=vertex,
        )

        self.output_model = EquivariantScalar(hidden_channels=hidden_channels)
        self.prior_model = Atomref(atomref=atomref, max_z=max_z)
        self.reduce_op = reduce_op
        self.derivative = derivative

        self.register_buffer('mean', paddle.to_tensor(mean, dtype=paddle.float32))
        self.register_buffer('std', paddle.to_tensor(std, dtype=paddle.float32))

        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters of the module."""
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Computes the energies or properties (forces) for a batch of molecules.

        Args:
            z (paddle.Tensor): The atomic numbers.
            pos (paddle.Tensor): The coordinates of the atoms.
            batch (paddle.Tensor): A batch vector, which assigns each node to
                a specific example.

        Returns:
            y (paddle.Tensor): The energies or properties for each molecule.
            dy (paddle.Tensor, optional): The negative derivative of energies.
        """
        if self.derivative:
            pos.stop_gradient = False

        x, v = self.representation_model(z, pos, batch)
        x = self.output_model.pre_reduce(x, v)
        x = x * self.std

        if self.prior_model is not None:
            x = self.prior_model(x, z)

        y = scatter(x, batch, dim=0, reduce=self.reduce_op)
        y = y + self.mean

        if self.derivative:
            dy = paddle.grad(
                outputs=[y],
                inputs=[pos],
                grad_outputs=paddle.ones_like(y),
                retain_graph=True,
                create_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError(
                    "Autograd returned None for the force prediction.")
            return y, -dy

        return y, None