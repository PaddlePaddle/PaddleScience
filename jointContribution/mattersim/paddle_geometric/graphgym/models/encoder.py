import paddle

from paddle_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
)


@register_node_encoder('Integer')
class IntegerFeatureEncoder(paddle.nn.Layer):
    r"""Provides an encoder for integer node features.

    Args:
        emb_dim (int): The output embedding dimension.
        num_classes (int): The number of classes/integers.

    Example:
        >>> encoder = IntegerFeatureEncoder(emb_dim=16, num_classes=10)
        >>> batch = paddle.randint(0, 10, (10, 2))
        >>> encoder(batch).size()
        paddle.Size([10, 16])
    """
    def __init__(self, emb_dim: int, num_classes: int):
        super().__init__()

        self.encoder = paddle.nn.Embedding(num_classes, emb_dim)
        paddle.nn.initializer.XavierUniform()(self.encoder.weight)

    def forward(self, batch):
        # Encode just the first dimension if more exist
        batch.x = self.encoder(batch.x[:, 0])

        return batch


@register_node_encoder('Atom')
class AtomEncoder(paddle.nn.Layer):
    r"""The atom encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): The output embedding dimension.

    Example:
        >>> encoder = AtomEncoder(emb_dim=16)
        >>> batch = paddle.randint(0, 10, (10, 3))
        >>> encoder(batch).size()
        paddle.Size([10, 16])
    """
    def __init__(self, emb_dim, *args, **kwargs):
        super().__init__()

        from ogb.utils.features import get_atom_feature_dims

        self.atom_embedding_list = paddle.nn.LayerList()

        for i, dim in enumerate(get_atom_feature_dims()):
            emb = paddle.nn.Embedding(dim, emb_dim)
            paddle.nn.initializer.XavierUniform()(emb.weight)
            self.atom_embedding_list.append(emb)

    def forward(self, batch):
        encoded_features = 0
        for i in range(batch.x.shape[1]):
            encoded_features += self.atom_embedding_list[i](batch.x[:, i])

        batch.x = encoded_features
        return batch


@register_edge_encoder('Bond')
class BondEncoder(paddle.nn.Layer):
    r"""The bond encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): The output embedding dimension.

    Example:
        >>> encoder = BondEncoder(emb_dim=16)
        >>> batch = paddle.randint(0, 10, (10, 3))
        >>> encoder(batch).size()
        paddle.Size([10, 16])
    """
    def __init__(self, emb_dim: int):
        super().__init__()

        from ogb.utils.features import get_bond_feature_dims

        self.bond_embedding_list = paddle.nn.LayerList()

        for i, dim in enumerate(get_bond_feature_dims()):
            emb = paddle.nn.Embedding(dim, emb_dim)
            paddle.nn.initializer.XavierUniform()(emb.weight)
            self.bond_embedding_list.append(emb)

    def forward(self, batch):
        bond_embedding = 0
        for i in range(batch.edge_attr.shape[1]):
            edge_attr = batch.edge_attr
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        batch.edge_attr = bond_embedding
        return batch
