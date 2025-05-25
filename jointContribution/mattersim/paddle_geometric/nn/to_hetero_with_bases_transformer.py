import copy
import warnings
from typing import Any, Dict, List, Optional, Union

import paddle
from paddle import Tensor
from paddle.nn import Layer, LayerList, LayerDict

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.dense import Linear
from paddle_geometric.nn.fx import Transformer
from paddle_geometric.typing import EdgeType, Metadata, NodeType, SparseTensor
from paddle_geometric.utils.hetero import get_unused_node_types

try:
    # from paddle.fx import Graph, GraphLayer, Node
    GraphLayer, Graph, Node = None, None, None  # 后期需要替换
except (ImportError, ModuleNotFoundError, AttributeError):
    GraphModule, Graph, Node = 'GraphModule', 'Graph', 'Node'


def to_hetero_with_bases(module: Layer, metadata: Metadata, num_bases: int,
                         in_channels: Optional[Dict[str, int]] = None,
                         input_map: Optional[Dict[str, str]] = None,
                         debug: bool = False) -> Any:
    """
    Converts a homogeneous GNN model into its heterogeneous equivalent
    via the basis-decomposition technique.
    """
    transformer = ToHeteroWithBasesTransformer(module, metadata, num_bases,
                                               in_channels, input_map, debug)
    return transformer.transform()


class ToHeteroWithBasesTransformer(Transformer):
    def __init__(
        self,
        module: Layer,
        metadata: Metadata,
        num_bases: int,
        in_channels: Optional[Dict[str, int]] = None,
        input_map: Optional[Dict[str, str]] = None,
        debug: bool = False,
    ):
        super().__init__(module, input_map, debug)

        self.metadata = metadata
        self.num_bases = num_bases
        self.in_channels = in_channels or {}
        assert len(metadata) == 2
        assert len(metadata[0]) > 0 and len(metadata[1]) > 0

        self.validate()

        # Compute IDs for each node and edge type:
        self.node_type2id = {k: i for i, k in enumerate(metadata[0])}
        self.edge_type2id = {k: i for i, k in enumerate(metadata[1])}

    def validate(self):
        unused_node_types = get_unused_node_types(*self.metadata)
        if len(unused_node_types) > 0:
            warnings.warn(
                f"There exist node types ({unused_node_types}) whose "
                f"representations do not get updated during message passing "
                f"as they do not occur as destination type in any edge type. "
                f"This may lead to unexpected behavior."
            )

        names = self.metadata[0] + [rel for _, rel, _ in self.metadata[1]]
        for name in names:
            if not name.isidentifier():
                warnings.warn(
                    f"The type '{name}' contains invalid characters which "
                    f"may lead to unexpected behavior. To avoid any issues, "
                    f"ensure that your types only contain letters, numbers "
                    f"and underscores."
                )

    def transform(self) -> Any:
        self._node_offset_dict_initialized = False
        self._edge_offset_dict_initialized = False
        self._edge_type_initialized = False
        out = super().transform()
        del self._node_offset_dict_initialized
        del self._edge_offset_dict_initialized
        del self._edge_type_initialized
        return out
    def placeholder(self, node, target: Any, name: str):
        if node.type is not None:
            Type = EdgeType if self.is_edge_level(node) else NodeType
            node.type = Dict[Type, node.type]

        out = node

        # Create `node_offset_dict` and `edge_offset_dict` dictionaries
        if self.is_edge_level(node) and not self._edge_offset_dict_initialized:
            self.graph.inserting_after(out)
            out = self.graph.create_node('call_function',
                                         target=get_edge_offset_dict,
                                         args=(node, self.edge_type2id),
                                         name='edge_offset_dict')
            self._edge_offset_dict_initialized = True

        elif not self._node_offset_dict_initialized:
            self.graph.inserting_after(out)
            out = self.graph.create_node('call_function',
                                         target=get_node_offset_dict,
                                         args=(node, self.node_type2id),
                                         name='node_offset_dict')
            self._node_offset_dict_initialized = True

        # Create a `edge_type` tensor used as input to `HeteroBasisConv`:
        if self.is_edge_level(node) and not self._edge_type_initialized:
            self.graph.inserting_after(out)
            out = self.graph.create_node('call_function', target=get_edge_type,
                                         args=(node, self.edge_type2id),
                                         name='edge_type')
            self._edge_type_initialized = True

        # Add `Linear` operation to align features to the same dimensionality:
        if name in self.in_channels:
            self.graph.inserting_after(out)
            out = self.graph.create_node('call_module',
                                         target=f'align_lin__{name}',
                                         args=(node, ),
                                         name=f'{name}__aligned')
            self._state[out.name] = self._state[name]

            lin = LinearAlign(self.metadata[int(self.is_edge_level(node))],
                              self.in_channels[name])
            setattr(self.module, f'align_lin__{name}', lin)

        # Perform grouping of type-wise values into a single tensor:
        if self.is_edge_level(node):
            self.graph.inserting_after(out)
            out = self.graph.create_node(
                'call_function', target=group_edge_placeholder,
                args=(out if name in self.in_channels else node,
                      self.edge_type2id,
                      self.find_by_name('node_offset_dict')),
                name=f'{name}__grouped')
            self._state[out.name] = 'edge'

        else:
            self.graph.inserting_after(out)
            out = self.graph.create_node(
                'call_function', target=group_node_placeholder,
                args=(out if name in self.in_channels else node,
                      self.node_type2id), name=f'{name}__grouped')
            self._state[out.name] = 'node'

        self.replace_all_uses_with(node, out)

    def call_message_passing_module(self, node, target: Any, name: str):
        # Call the `HeteroBasisConv` wrapper instead of a single
        # message passing layer. We need to inject the `edge_type` as the first
        # argument.
        node.args = (self.find_by_name('edge_type'), ) + node.args

    def output(self, node, target: Any, name: str):
        # Split the output into dictionaries holding either node type-wise or
        # edge type-wise data.
        def _recurse(value: Any) -> Any:
            if isinstance(value, Node) and self.is_edge_level(value):
                self.graph.inserting_before(node)
                return self.graph.create_node(
                    'call_function', target=split_output,
                    args=(value, self.find_by_name('edge_offset_dict')),
                    name=f'{value.name}__split')

            elif isinstance(value, Node):
                self.graph.inserting_before(node)
                return self.graph.create_node(
                    'call_function', target=split_output,
                    args=(value, self.find_by_name('node_offset_dict')),
                    name=f'{value.name}__split')

            elif isinstance(value, dict):
                return {k: _recurse(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_recurse(v) for v in value]
            elif isinstance(value, tuple):
                return tuple(_recurse(v) for v in value)
            else:
                return value

        if node.type is not None and isinstance(node.args[0], Node):
            output = node.args[0]
            Type = EdgeType if self.is_edge_level(output) else NodeType
            node.type = Dict[Type, node.type]
        else:
            node.type = None

        node.args = (_recurse(node.args[0]), )

    def init_submodule(self, module: Layer, target: str) -> Layer:
        if not isinstance(module, MessagePassing):
            return module

        # Replace each `MessagePassing` module with a `HeteroBasisConv` wrapper:
        return HeteroBasisConv(module, len(self.metadata[1]), self.num_bases)

###############################################################################

# Hook function to inject basis re-weighting for each edge type
def hook(module, inputs, output):
    assert isinstance(module._edge_type, Tensor)
    if module._edge_type.shape[0] != output.shape[-2]:
        raise ValueError(
            f"Number of messages ({output.shape[0]}) does not match "
            f"with the number of original edges "
            f"({module._edge_type.shape[0]}). Does your message "
            f"passing layer create additional self-loops? Try to "
            f"remove them via 'add_self_loops=False'")
    weight = module.edge_type_weight.reshape([-1])[module._edge_type]
    weight = weight.reshape([1] * (len(output.shape) - 2) + [-1, 1])
    return weight * output


class HeteroBasisConv(Layer):
    # A wrapper layer that applies the basis-decomposition technique to a
    # heterogeneous graph.
    def __init__(self, module: MessagePassing, num_relations: int,
                 num_bases: int):
        super().__init__()

        self.num_relations = num_relations
        self.num_bases = num_bases

        self.convs = LayerList()
        for _ in range(num_bases):
            conv = copy.deepcopy(module)
            conv.fuse = False  # Disable `message_and_aggregate` functionality.
            # We learn a single scalar weight for each individual edge type,
            # which is used to weight the output message based on edge type:
            conv.edge_type_weight = self.create_parameter(
                shape=[1, num_relations],
                default_initializer=paddle.nn.initializer.XavierUniform())
            conv.register_forward_post_hook(hook)
            self.convs.append(conv)

        if self.num_bases > 1:
            self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
            elif sum([p.numel() for p in conv.parameters()]) > 0:
                warnings.warn(
                    f"'{conv}' will be duplicated, but its parameters cannot "
                    f"be reset. To suppress this warning, add a "
                    f"'reset_parameters()' method to '{conv}'")

    def forward(self, edge_type: Tensor, *args, **kwargs) -> Tensor:
        out = None
        # Call message passing modules and perform aggregation:
        for conv in self.convs:
            conv._edge_type = edge_type
            res = conv(*args, **kwargs)
            del conv._edge_type
            out = res if out is None else out.add_(res)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_relations='
                f'{self.num_relations}, num_bases={self.num_bases})')


class LinearAlign(Layer):
    # Aligns representations to the same dimensionality.
    def __init__(self, keys: List[Union[NodeType, EdgeType]],
                 out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.lins = LayerDict()
        for key in keys:
            self.lins[key2str(key)] = Linear(-1, out_channels)

    def forward(
        self, x_dict: Dict[Union[NodeType, EdgeType], Tensor]
    ) -> Dict[Union[NodeType, EdgeType], Tensor]:
        return {key: self.lins[key2str(key)](x) for key, x in x_dict.items()}

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_relations={len(self.lins)}, '
                f'out_channels={self.out_channels})')


###############################################################################
# These methods are used to compute the cumulative sizes of input dictionaries
# for unified graph representation and splitting final output data.


def get_node_offset_dict(
    input_dict: Dict[NodeType, Union[Tensor, SparseTensor]],
    type2id: Dict[NodeType, int],
) -> Dict[NodeType, int]:
    cumsum = 0
    out: Dict[NodeType, int] = {}
    for key in type2id.keys():
        out[key] = cumsum
        cumsum += input_dict[key].shape[-2]
    return out

def get_edge_offset_dict(
    input_dict: Dict[EdgeType, Union[Tensor, SparseTensor]],
    type2id: Dict[EdgeType, int],
) -> Dict[EdgeType, int]:
    cumsum = 0
    out: Dict[EdgeType, int] = {}
    for key in type2id.keys():
        out[key] = cumsum
        value = input_dict[key]
        if isinstance(value, SparseTensor):
            cumsum += value.nnz()
        elif value.dtype == paddle.int64 and value.shape[0] == 2:
            cumsum += value.shape[-1]
        else:
            cumsum += value.shape[-2]
    return out


def get_edge_type(
    input_dict: Dict[EdgeType, Union[Tensor, SparseTensor]],
    type2id: Dict[EdgeType, int],
) -> Tensor:
    inputs = [input_dict[key] for key in type2id.keys()]
    outs = []

    for i, value in enumerate(inputs):
        if value.shape[0] == 2 and value.dtype == paddle.int64:  # edge_index
            out = paddle.full([value.shape[-1]], i, dtype=paddle.int64)
        elif isinstance(value, SparseTensor):
            out = paddle.full([value.nnz()], i, dtype=paddle.int64)
        else:
            out = paddle.full([value.shape[-2]], i, dtype=paddle.int64)
        outs.append(out)

    return outs[0] if len(outs) == 1 else paddle.concat(outs, axis=0)


def group_node_placeholder(input_dict: Dict[NodeType, Tensor],
                           type2id: Dict[NodeType, int]) -> Tensor:
    inputs = [input_dict[key] for key in type2id.keys()]
    return inputs[0] if len(inputs) == 1 else paddle.concat(inputs, axis=-2)


def group_edge_placeholder(
    input_dict: Dict[EdgeType, Union[Tensor, SparseTensor]],
    type2id: Dict[EdgeType, int],
    offset_dict: Dict[NodeType, int] = None,
) -> Union[Tensor, SparseTensor]:
    inputs = [input_dict[key] for key in type2id.keys()]

    if len(inputs) == 1:
        return inputs[0]

    if inputs[0].shape[0] == 2 and inputs[0].dtype == paddle.int64:  # edge_index
        if offset_dict is None:
            raise AttributeError(
                "Cannot infer node-level offsets. Please ensure that there "
                "exists a node-level argument before the 'edge_index' "
                "argument in your forward header.")

        outputs = []
        for value, (src_type, _, dst_type) in zip(inputs, type2id):
            value = value.clone()
            value[0, :] += offset_dict[src_type]
            value[1, :] += offset_dict[dst_type]
            outputs.append(value)

        return paddle.concat(outputs, axis=-1)

    elif isinstance(inputs[0], SparseTensor):
        if offset_dict is None:
            raise AttributeError(
                "Cannot infer node-level offsets. Please ensure that there "
                "exists a node-level argument before the 'SparseTensor' "
                "argument in your forward header.")

        rows, cols = [], []
        for value, (src_type, _, dst_type) in zip(inputs, type2id):
            col, row, _ = value.coo()
            rows.append(row + offset_dict[src_type])
            cols.append(col + offset_dict[dst_type])

        row = paddle.concat(rows, axis=0)
        col = paddle.concat(cols, axis=0)
        return paddle.stack([row, col], axis=0)

    else:
        return paddle.concat(inputs, axis=-2)


def split_output(
    output: Tensor,
    offset_dict: Union[Dict[NodeType, int], Dict[EdgeType, int]],
) -> Union[Dict[NodeType, Tensor], Dict[EdgeType, Tensor]]:
    cumsums = list(offset_dict.values()) + [output.shape[-2]]
    sizes = [cumsums[i + 1] - cumsums[i] for i in range(len(offset_dict))]
    outputs = paddle.split(output, sizes, axis=-2)
    return {key: output for key, output in zip(offset_dict, outputs)}


def key2str(key: Union[NodeType, EdgeType]) -> str:
    key = '__'.join(key) if isinstance(key, tuple) else key
    return key.replace(' ', '_').replace('-', '_').replace(':', '_')