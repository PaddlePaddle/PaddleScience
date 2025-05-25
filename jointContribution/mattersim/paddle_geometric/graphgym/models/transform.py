import paddle

from paddle_geometric.utils import negative_sampling


def create_link_label(pos_edge_index, neg_edge_index):
    """Create labels for link prediction, based on positive and negative edges.

    Args:
        pos_edge_index (paddle.Tensor): Positive edge index [2, num_edges]
        neg_edge_index (paddle.Tensor): Negative edge index [2, num_edges]

    Returns: Link label tensor, [num_positive_edges + num_negative_edges]
    """
    num_links = pos_edge_index.shape[1] + neg_edge_index.shape[1]
    link_labels = paddle.zeros([num_links], dtype='float32', place=pos_edge_index.place)
    link_labels[:pos_edge_index.shape[1]] = 1.0
    return link_labels


def neg_sampling_transform(data):
    """Perform negative sampling for link prediction tasks.

    Args:
        data (paddle_geometric.data.Data): Input data object

    Returns: Transformed data object with negative edges and link prediction labels.
    """
    train_neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.shape[1])

    data.train_edge_index = paddle.concat(
        [data.train_pos_edge_index, train_neg_edge_index], axis=-1)
    data.train_edge_label = create_link_label(data.train_pos_edge_index, train_neg_edge_index)

    return data
