from enum import Enum
from typing import Dict, Optional, Tuple, Union

import paddle
from paddle import Tensor

from paddle_geometric.explain.algorithm.utils import (
    clear_masks,
    set_hetero_masks,
    set_masks,
)
from paddle_geometric.explain.config import (
    ModelConfig,
    ModelMode,
    ModelReturnType,
)
from paddle_geometric.typing import EdgeType, Metadata, NodeType


class MaskLevelType(Enum):
    node = 'node'
    edge = 'edge'
    node_and_edge = 'node_and_edge'

    @property
    def with_edge(self) -> bool:
        return self in [MaskLevelType.edge, MaskLevelType.node_and_edge]


class CaptumModel(paddle.nn.Layer):
    def __init__(
        self,
        model: paddle.nn.Layer,
        mask_type: Union[str, MaskLevelType],
        output_idx: Optional[Union[int, Tensor]] = None,
        model_config: Optional[ModelConfig] = None,
    ):
        super().__init__()

        self.mask_type = MaskLevelType(mask_type)
        self.model = model
        self.output_idx = output_idx
        self.model_config = model_config

    def forward(self, mask, *args):
        assert mask.shape[0] == 1, "Dimension 0 of input should be 1"
        if self.mask_type == MaskLevelType.edge:
            assert len(args) >= 2, "Expects at least x and edge_index as args."
        if self.mask_type == MaskLevelType.node:
            assert len(args) >= 1, "Expects at least edge_index as args."
        if self.mask_type == MaskLevelType.node_and_edge:
            assert args[0].shape[0] == 1, "Dimension 0 of input should be 1"
            assert len(args[1:]) >= 1, "Expects at least edge_index as args."

        if self.mask_type == MaskLevelType.edge:
            set_masks(self.model, mask.squeeze(0), args[1], apply_sigmoid=False)
        elif self.mask_type == MaskLevelType.node_and_edge:
            set_masks(self.model, args[0].squeeze(0), args[1], apply_sigmoid=False)
            args = args[1:]

        if self.mask_type == MaskLevelType.edge:
            x = self.model(*args)
        else:
            x = self.model(mask.squeeze(0), *args)

        return self.postprocess(x)

    def postprocess(self, x: Tensor) -> Tensor:
        if self.mask_type.with_edge:
            clear_masks(self.model)

        if self.output_idx is not None:
            x = x[self.output_idx]
            if isinstance(self.output_idx, int) or self.output_idx.ndim == 0:
                x = x.unsqueeze(0)

        if (self.model_config is not None
                and self.model_config.mode == ModelMode.binary_classification):
            assert self.model_config.return_type == ModelReturnType.probs
            x = x.reshape([-1, 1])
            x = paddle.concat([1 - x, x], axis=-1)

        return x


class CaptumHeteroModel(CaptumModel):
    def __init__(
        self,
        model: paddle.nn.Layer,
        mask_type: Union[str, MaskLevelType],
        output_idx: Optional[Union[int, Tensor]],
        metadata: Metadata,
        model_config: Optional[ModelConfig] = None,
    ):
        super().__init__(model, mask_type, output_idx, model_config)
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.num_node_types = len(self.node_types)
        self.num_edge_types = len(self.edge_types)

    def _captum_data_to_hetero_data(
        self, *args
    ) -> Tuple[Dict[NodeType, Tensor], Dict[EdgeType, Tensor], Optional[Dict[
            EdgeType, Tensor]]]:
        if self.mask_type == MaskLevelType.node:
            node_tensors = args[:self.num_node_types]
            node_tensors = [mask.squeeze(0) for mask in node_tensors]
            x_dict = dict(zip(self.node_types, node_tensors))
            edge_index_dict = args[self.num_node_types]
        elif self.mask_type == MaskLevelType.edge:
            edge_mask_tensors = args[:self.num_edge_types]
            x_dict = args[self.num_edge_types]
            edge_index_dict = args[self.num_edge_types + 1]
        else:
            node_tensors = args[:self.num_node_types]
            node_tensors = [mask.squeeze(0) for mask in node_tensors]
            x_dict = dict(zip(self.node_types, node_tensors))
            edge_mask_tensors = args[self.num_node_types:self.num_node_types +
                                     self.num_edge_types]
            edge_index_dict = args[self.num_node_types + self.num_edge_types]

        if self.mask_type.with_edge:
            edge_mask_tensors = [mask.squeeze(0) for mask in edge_mask_tensors]
            edge_mask_dict = dict(zip(self.edge_types, edge_mask_tensors))
        else:
            edge_mask_dict = None
        return x_dict, edge_index_dict, edge_mask_dict

    def forward(self, *args):
        if self.mask_type == MaskLevelType.node:
            assert len(args) >= self.num_node_types + 1
            len_remaining_args = len(args) - (self.num_node_types + 1)
        elif self.mask_type == MaskLevelType.edge:
            assert len(args) >= self.num_edge_types + 2
            len_remaining_args = len(args) - (self.num_edge_types + 2)
        else:
            assert len(args) >= self.num_node_types + self.num_edge_types + 1
            len_remaining_args = len(args) - (self.num_node_types +
                                              self.num_edge_types + 1)

        (x_dict, edge_index_dict,
         edge_mask_dict) = self._captum_data_to_hetero_data(*args)

        if self.mask_type.with_edge:
            set_hetero_masks(self.model, edge_mask_dict, edge_index_dict)

        if len_remaining_args > 0:
            x = self.model(x_dict, edge_index_dict,
                           *args[-len_remaining_args:])
        else:
            x = self.model(x_dict, edge_index_dict)

        return self.postprocess(x)


def _to_edge_mask(edge_index: Tensor) -> Tensor:
    num_edges = edge_index.shape[1]
    return paddle.ones([num_edges], dtype='float32', stop_gradient=False)


def to_captum_input(
    x: Union[Tensor, Dict[NodeType, Tensor]],
    edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
    mask_type: Union[str, MaskLevelType],
    *args,
) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]:
    mask_type = MaskLevelType(mask_type)

    additional_forward_args = []
    if isinstance(x, Tensor) and isinstance(edge_index, Tensor):
        if mask_type == MaskLevelType.node:
            inputs = [x.unsqueeze(0)]
        elif mask_type == MaskLevelType.edge:
            inputs = [_to_edge_mask(edge_index).unsqueeze(0)]
            additional_forward_args.append(x)
        else:
            inputs = [x.unsqueeze(0), _to_edge_mask(edge_index).unsqueeze(0)]
        additional_forward_args.append(edge_index)

    elif isinstance(x, Dict) and isinstance(edge_index, Dict):
        node_types = x.keys()
        edge_types = edge_index.keys()
        inputs = []
        if mask_type == MaskLevelType.node:
            for key in node_types:
                inputs.append(x[key].unsqueeze(0))
        elif mask_type == MaskLevelType.edge:
            for key in edge_types:
                inputs.append(_to_edge_mask(edge_index[key]).unsqueeze(0))
            additional_forward_args.append(x)
        else:
            for key in node_types:
                inputs.append(x[key].unsqueeze(0))
            for key in edge_types:
                inputs.append(_to_edge_mask(edge_index[key]).unsqueeze(0))
        additional_forward_args.append(edge_index)

    else:
        raise ValueError(
            "'x' and 'edge_index' need to be either"
            f"'Dict' or 'Tensor' got({type(x)}, {type(edge_index)})")

    additional_forward_args.extend(args)

    return tuple(inputs), tuple(additional_forward_args)


def captum_output_to_dicts(
    captum_attrs: Tuple[Tensor, ...],
    mask_type: Union[str, MaskLevelType],
    metadata: Metadata,
) -> Tuple[Optional[Dict[NodeType, Tensor]], Optional[Dict[EdgeType, Tensor]]]:
    mask_type = MaskLevelType(mask_type)
    node_types = metadata[0]
    edge_types = metadata[1]
    x_attr_dict, edge_attr_dict = None, None
    captum_attrs = [captum_attr.squeeze(0) for captum_attr in captum_attrs]
    if mask_type == MaskLevelType.node:
        assert len(node_types) == len(captum_attrs)
        x_attr_dict = dict(zip(node_types, captum_attrs))
    elif mask_type == MaskLevelType.edge:
        assert len(edge_types) == len(captum_attrs)
        edge_attr_dict = dict(zip(edge_types, captum_attrs))
    elif mask_type == MaskLevelType.node_and_edge:
        assert len(edge_types) + len(node_types) == len(captum_attrs)
        x_attr_dict = dict(zip(node_types, captum_attrs[:len(node_types)]))
        edge_attr_dict = dict(zip(edge_types, captum_attrs[len(node_types):]))
    return x_attr_dict, edge_attr_dict


def convert_captum_output(
    captum_attrs: Tuple[Tensor, ...],
    mask_type: Union[str, MaskLevelType],
    metadata: Optional[Metadata] = None,
):
    mask_type = MaskLevelType(mask_type)
    if metadata is not None:
        return captum_output_to_dicts(captum_attrs, mask_type, metadata)

    node_mask = edge_mask = None
    if mask_type == MaskLevelType.edge:
        edge_mask = captum_attrs[0].squeeze(0)
    elif mask_type == MaskLevelType.node:
        node_mask = captum_attrs[0].squeeze(0)
    else:
        node_mask = captum_attrs[0].squeeze(0)
        edge_mask = captum_attrs[1].squeeze(0)

    return node_mask, edge_mask
