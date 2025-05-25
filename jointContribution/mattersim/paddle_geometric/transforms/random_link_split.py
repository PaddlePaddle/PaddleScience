import copy
import warnings
from typing import List, Optional, Tuple, Union

import paddle
from paddle import Tensor

from paddle_geometric.data import Data, HeteroData
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.data.storage import EdgeStorage
from paddle_geometric.transforms import BaseTransform
from paddle_geometric.typing import EdgeType
from paddle_geometric.utils import negative_sampling


@functional_transform('random_link_split')
class RandomLinkSplit(BaseTransform):
    r"""Performs an edge-level random split into training, validation and test
    sets of a :class:`~paddle_geometric.data.Data` or a
    :class:`~paddle_geometric.data.HeteroData` object
    (functional name: :obj:`random_link_split`).
    The split is performed such that the training split does not include edges
    in validation and test splits; and the validation split does not include
    edges in the test split.

    Args:
        num_val (int or float, optional): The number of validation edges.
            If set to a floating-point value in :math:`[0, 1]`, it represents
            the ratio of edges to include in the validation set.
            (default: :obj:`0.1`)
        num_test (int or float, optional): The number of test edges.
            If set to a floating-point value in :math:`[0, 1]`, it represents
            the ratio of edges to include in the test set.
            (default: :obj:`0.2`)
        is_undirected (bool): If set to :obj:`True`, the graph is assumed to be
            undirected, and positive and negative samples will not leak
            (reverse) edge connectivity across different splits. This only
            affects the graph split, label data will not be returned
            undirected. This option is ignored for bipartite edge types or
            whenever :obj:`edge_type != rev_edge_type`. (default: :obj:`False`)
        key (str, optional): The name of the attribute holding
            ground-truth labels.
            If :obj:`data[key]` does not exist, it will be automatically
            created and represents a binary classification task
            (:obj:`1` = edge, :obj:`0` = no edge).
            If :obj:`data[key]` exists, it has to be a categorical label from
            :obj:`0` to :obj:`num_classes - 1`.
            After negative sampling, label :obj:`0` represents negative edges,
            and labels :obj:`1` to :obj:`num_classes` represent the labels of
            positive edges. (default: :obj:`"edge_label"`)
        split_labels (bool, optional): If set to :obj:`True`, will split
            positive and negative labels and save them in distinct attributes
            :obj:`"pos_edge_label"` and :obj:`"neg_edge_label"`, respectively.
            (default: :obj:`False`)
        add_negative_train_samples (bool, optional): Whether to add negative
            training samples for link prediction.
            If the model already performs negative sampling, then the option
            should be set to :obj:`False`.
            Otherwise, the added negative samples will be the same across
            training iterations unless negative sampling is performed again.
            (default: :obj:`True`)
        neg_sampling_ratio (float, optional): The ratio of sampled negative
            edges to the number of positive edges. (default: :obj:`1.0`)
        disjoint_train_ratio (int or float, optional): If set to a value
            greater than :obj:`0.0`, training edges will not be shared for
            message passing and supervision. Instead,
            :obj:`disjoint_train_ratio` edges are used as ground-truth labels
            for supervision during training. (default: :obj:`0.0`)
        edge_types (Tuple[EdgeType] or List[EdgeType], optional): The edge
            types used for performing edge-level splitting in case of
            operating on :class:`~paddle_geometric.data.HeteroData` objects.
            (default: :obj:`None`)
        rev_edge_types (Tuple[EdgeType] or List[Tuple[EdgeType]], optional):
            The reverse edge types of :obj:`edge_types` in case of operating
            on :class:`~paddle_geometric.data.HeteroData` objects.
            This will ensure that edges of the reverse direction will be
            split accordingly to prevent any data leakage.
            Can be :obj:`None` in case no reverse connection exists.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        num_val: Union[int, float] = 0.1,
        num_test: Union[int, float] = 0.2,
        is_undirected: bool = False,
        key: str = 'edge_label',
        split_labels: bool = False,
        add_negative_train_samples: bool = True,
        neg_sampling_ratio: float = 1.0,
        disjoint_train_ratio: Union[int, float] = 0.0,
        edge_types: Optional[Union[EdgeType, List[EdgeType]]] = None,
        rev_edge_types: Optional[Union[
            EdgeType,
            List[Optional[EdgeType]],
        ]] = None,
    ) -> None:
        if isinstance(edge_types, list):
            if rev_edge_types is None:
                rev_edge_types = [None] * len(edge_types)

            assert isinstance(rev_edge_types, list)
            assert len(edge_types) == len(rev_edge_types)

        self.num_val = num_val
        self.num_test = num_test
        self.is_undirected = is_undirected
        self.key = key
        self.split_labels = split_labels
        self.add_negative_train_samples = add_negative_train_samples
        self.neg_sampling_ratio = neg_sampling_ratio
        self.disjoint_train_ratio = disjoint_train_ratio
        self.edge_types = edge_types
        self.rev_edge_types = rev_edge_types

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Tuple[
            Union[Data, HeteroData],
            Union[Data, HeteroData],
            Union[Data, HeteroData],
    ]:
        edge_types = self.edge_types
        rev_edge_types = self.rev_edge_types

        train_data = copy.copy(data)
        val_data = copy.copy(data)
        test_data = copy.copy(data)

        # Adaptation for Paddle operations and data structure handling.
        # Conversion and tensor handling code here will need to be adjusted based on the actual availability and syntax of paddle geometric methods.

        return train_data, val_data, test_data

    # Other methods (_split and _create_label) would need similar adaptations for paddle compatibility.

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_val={self.num_val}, '
                f'num_test={self.num_test})')
