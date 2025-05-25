from collections import defaultdict
from typing import Dict, Optional, Union

import paddle
from paddle import Tensor

from paddle_geometric.explain import Explanation, HeteroExplanation
from paddle_geometric.explain.algorithm import ExplainerAlgorithm
from paddle_geometric.explain.config import MaskType
from paddle_geometric.typing import EdgeType, NodeType


class DummyExplainer(ExplainerAlgorithm):
    r"""A dummy explainer that returns random explanations (useful for testing
    purposes).
    """

    def forward(
            self,
            model: paddle.nn.Layer,
            x: Union[Tensor, Dict[NodeType, Tensor]],
            edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
            edge_attr: Optional[Union[Tensor, Dict[EdgeType, Tensor]]] = None,
            **kwargs,
    ) -> Union[Explanation, HeteroExplanation]:
        # Ensure input `x` is either a Tensor or a dictionary for heterogeneous graphs
        assert isinstance(x, (Tensor, dict))

        # Get the mask types for nodes and edges from the explainer configuration
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        if isinstance(x, Tensor):  # Case: Homogeneous graph
            assert isinstance(edge_index, Tensor)  # Ensure edge_index is a tensor

            # Initialize node mask based on mask type
            node_mask = None
            if node_mask_type == MaskType.object:
                node_mask = paddle.rand([x.shape[0], 1], dtype=x.dtype)
            elif node_mask_type == MaskType.common_attributes:
                node_mask = paddle.rand([1, x.shape[1]], dtype=x.dtype)
            elif node_mask_type == MaskType.attributes:
                node_mask = paddle.rand_like(x)

            # Initialize edge mask based on mask type
            edge_mask = None
            if edge_mask_type == MaskType.object:
                edge_mask = paddle.rand([edge_index.shape[1]], dtype=x.dtype)

            # Return an Explanation object with node and edge masks
            return Explanation(node_mask=node_mask, edge_mask=edge_mask)

        else:  # Case: Heterogeneous graph (x is a dictionary)
            assert isinstance(edge_index, dict)  # Ensure edge_index is a dictionary

            # Create random node masks for each node type
            node_dict = defaultdict(dict)
            for k, v in x.items():
                node_mask = None
                if node_mask_type == MaskType.object:
                    node_mask = paddle.rand([v.shape[0], 1], dtype=v.dtype)
                elif node_mask_type == MaskType.common_attributes:
                    node_mask = paddle.rand([1, v.shape[1]], dtype=v.dtype)
                elif node_mask_type == MaskType.attributes:
                    node_mask = paddle.rand_like(v)
                if node_mask is not None:
                    node_dict[k]['node_mask'] = node_mask

            # Create random edge masks for each edge type
            edge_dict = defaultdict(dict)
            for k, v in edge_index.items():
                edge_mask = None
                if edge_mask_type == MaskType.object:
                    edge_mask = paddle.rand([v.shape[1]], dtype=v.dtype)
                if edge_mask is not None:
                    edge_dict[k]['edge_mask'] = edge_mask

            # Return a HeteroExplanation with masks for each node and edge type
            return HeteroExplanation({**node_dict, **edge_dict})

    def supports(self) -> bool:
        # This explainer supports all configurations
        return True
