import os.path as osp
import warnings
from abc import abstractmethod
from inspect import Parameter
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    OrderedDict,
    Set,
    Tuple,
    Union,
)

import paddle
from paddle import Tensor
import weakref
from paddle_geometric import EdgeIndex, is_compiling
from paddle_geometric.index import ptr2index
from paddle_geometric.inspector import Inspector, Signature
from paddle_geometric.nn.aggr import Aggregation
from paddle_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from paddle_geometric.template import module_from_template
from paddle_geometric.typing import Adj, Size, SparseTensor
from paddle_geometric.utils import (
    is_sparse,
    is_paddle_sparse_tensor,
    to_edge_index,
)

FUSE_AGGRS = {'add', 'sum', 'mean', 'min', 'max'}
HookDict = OrderedDict[int, Callable]

FUSE_AGGRS = {'add', 'sum', 'mean', 'min', 'max'}
HookDict = OrderedDict[int, Callable]


class RemovableHandle:
    from collections import OrderedDict
    r"""
    A handle which provides the capability to remove a hook.

    Args:
        hooks_dict (dict): A dictionary of hooks, indexed by hook ``id``.
        extra_dict (Union[dict, List[dict]]): An additional dictionary or list of
            dictionaries whose keys will be deleted when the same keys are
            removed from ``hooks_dict``.
    """

    id: int
    next_id: int = 0

    def __init__(self, hooks_dict: Any, *, extra_dict: Any = None) -> None:
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

        self.extra_dict_ref: Tuple = ()
        if isinstance(extra_dict, dict):
            self.extra_dict_ref = (weakref.ref(extra_dict),)
        elif isinstance(extra_dict, list):
            self.extra_dict_ref = tuple(weakref.ref(d) for d in extra_dict)

    def remove(self) -> None:
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

        for ref in self.extra_dict_ref:
            extra_dict = ref()
            if extra_dict is not None and self.id in extra_dict:
                del extra_dict[self.id]

    def __getstate__(self):
        if self.extra_dict_ref is None:
            return (self.hooks_dict_ref(), self.id)
        else:
            return (self.hooks_dict_ref(), self.id, tuple(ref() for ref in self.extra_dict_ref))

    def __setstate__(self, state) -> None:
        if state[0] is None:
            # create a dead reference
            self.hooks_dict_ref = weakref.ref(OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

        if len(state) < 3 or state[2] is None:
            self.extra_dict_ref = ()
        else:
            self.extra_dict_ref = tuple(weakref.ref(d) for d in state[2])

    def __enter__(self) -> "RemovableHandle":
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        self.remove()


class MessagePassing(paddle.nn.Layer):
    r"""Base class for creating message passing layers.

    Message passing layers follow the form

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \bigoplus_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) \right),

    where :math:`\bigoplus` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean, min, max or mul, and
    :math:`\gamma_{\mathbf{\Theta}}` and :math:`\phi_{\mathbf{\Theta}}` denote
    differentiable functions such as MLPs.

    Args:
        aggr (str or [str] or Aggregation, optional): The aggregation scheme
            to use, *e.g.*, :obj:`"sum"` :obj:`"mean"`, :obj:`"min"`,
            :obj:`"max"` or :obj:`"mul"`.
            In addition, can be any
            :class:`~pgl.nn.aggr.Aggregation` module (or any string
            that automatically resolves to it).
            If given as a list, will make use of multiple aggregations in which
            different outputs will get concatenated in the last dimension.
            If set to :obj:`None`, the :class:`MessagePassing` instantiation is
            expected to implement its own aggregation logic via
            :meth:`aggregate`. (default: :obj:`"add"`)
        aggr_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective aggregation function in case it gets automatically
            resolved. (default: :obj:`None`)
        flow (str, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`-2`)
        decomposed_layers (int, optional): The number of feature decomposition
            layers, as introduced in the `"Optimizing Memory Efficiency of
            Graph Neural Networks on Edge Computing Platforms"
            <https://arxiv.org/abs/2104.03058>`_ paper.
            Feature decomposition reduces the peak memory usage by slicing
            the feature dimensions into separated feature decomposition layers
            during GNN aggregation.
            (default: :obj:`1`)
    """

    special_args: Set[str] = {
        'edge_index', 'adj_t', 'edge_index_i', 'edge_index_j', 'size',
        'size_i', 'size_j', 'ptr', 'index', 'dim_size'
    }

    SUPPORTS_FUSED_EDGE_INDEX: Final[bool] = False

    def __init__(
        self,
        aggr: Optional[Union[str, List[str], Aggregation]] = 'sum',
        *,
        aggr_kwargs: Optional[Dict[str, Any]] = None,
        flow: str = "source_to_target",
        node_dim: int = -2,
        decomposed_layers: int = 1,
    ) -> None:
        super().__init__()

        if flow not in ['source_to_target', 'target_to_source']:
            raise ValueError(f"Expected 'flow' to be either 'source_to_target'"
                             f" or 'target_to_source' (got '{flow}')")

        # Cast `aggr` into a string representation for backward compatibility:
        self.aggr: Optional[Union[str, List[str]]]
        if aggr is None:
            self.aggr = None
        elif isinstance(aggr, (str, Aggregation)):
            self.aggr = str(aggr)
        elif isinstance(aggr, (tuple, list)):
            self.aggr = [str(x) for x in aggr]

        self.aggr_module = aggr_resolver(aggr, **(aggr_kwargs or {}))
        self.flow = flow
        self.node_dim = node_dim

        # Collect attribute names requested in message passing hooks:
        self.inspector = Inspector(self.__class__)
        self.inspector.inspect_signature(self.message)
        self.inspector.inspect_signature(self.aggregate, exclude=[0, 'aggr'])
        self.inspector.inspect_signature(self.message_and_aggregate, [0])
        self.inspector.inspect_signature(self.update, exclude=[0])
        self.inspector.inspect_signature(self.edge_update)

        self._user_args: List[str] = self.inspector.get_flat_param_names(
            ['message', 'aggregate', 'update'], exclude=self.special_args)
        self._fused_user_args: List[str] = self.inspector.get_flat_param_names(
            ['message_and_aggregate', 'update'], exclude=self.special_args)
        self._edge_user_args: List[str] = self.inspector.get_param_names(
            'edge_update', exclude=self.special_args)

        # Support for "fused" message passing:
        self.fuse = self.inspector.implements('message_and_aggregate')
        if self.aggr is not None:
            self.fuse &= isinstance(self.aggr, str) and self.aggr in FUSE_AGGRS

        # Hooks:
        self._propagate_forward_pre_hooks: HookDict = OrderedDict()
        self._propagate_forward_hooks: HookDict = OrderedDict()
        self._message_forward_pre_hooks: HookDict = OrderedDict()
        self._message_forward_hooks: HookDict = OrderedDict()
        self._aggregate_forward_pre_hooks: HookDict = OrderedDict()
        self._aggregate_forward_hooks: HookDict = OrderedDict()
        self._message_and_aggregate_forward_pre_hooks: HookDict = OrderedDict()
        self._message_and_aggregate_forward_hooks: HookDict = OrderedDict()
        self._edge_update_forward_pre_hooks: HookDict = OrderedDict()
        self._edge_update_forward_hooks: HookDict = OrderedDict()

        # Set jittable `propagate` and `edge_updater` function templates:
        self._set_jittable_templates()

        # Explainability:
        self._explain: Optional[bool] = None
        self._edge_mask: Optional[Tensor] = None
        self._loop_mask: Optional[Tensor] = None
        self._apply_sigmoid: bool = True

        # Inference Decomposition:
        self._decomposed_layers = 1
        self.decomposed_layers = decomposed_layers
    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
        if self.aggr_module is not None:
            self.aggr_module.reset_parameters()

    def __setstate__(self, data: Dict[str, Any]) -> None:
        self.inspector = data['inspector']
        self.fuse = data['fuse']
        self._set_jittable_templates()
        super().__setstate__(data)

    def __repr__(self) -> str:
        channels_repr = ''
        if hasattr(self, 'in_channels') and hasattr(self, 'out_channels'):
            channels_repr = f'{self.in_channels}, {self.out_channels}'
        elif hasattr(self, 'channels'):
            channels_repr = f'{self.channels}'
        return f'{self.__class__.__name__}({channels_repr})'

    def _check_input(
        self,
        edge_index: Union[Tensor, SparseTensor],
        size: Optional[Tuple[Optional[int], Optional[int]]],
    ) -> List[Optional[int]]:

        if is_sparse(edge_index):
            if self.flow == 'target_to_source':
                raise ValueError(
                    'Flow direction "target_to_source" is invalid for '
                    'message propagation via sparse tensors. Pass in the '
                    'transposed sparse tensor, e.g., `adj_t.t()`.')

            if isinstance(edge_index, SparseTensor):
                return [edge_index.shape[1], edge_index.shape[0]]

        elif isinstance(edge_index, Tensor):
            int_dtypes = (paddle.uint8, paddle.int8, paddle.int16, paddle.int32,
                          paddle.int64)

            if edge_index.dtype not in int_dtypes:
                raise ValueError(f"Expected 'edge_index' to be of integer "
                                 f"type (got '{edge_index.dtype}')")
            if edge_index.ndim != 2:
                raise ValueError(f"Expected 'edge_index' to be two-dimensional"
                                 f" (got {edge_index.ndim} dimensions)")
            if edge_index.shape[0] != 2:
                raise ValueError(f"Expected 'edge_index' to have size '2' in "
                                 f"the first dimension (got "
                                 f"'{edge_index.shape[0]}')")

            return list(size) if size is not None else [None, None]

        raise ValueError(
            '`MessagePassing.propagate` only supports integer tensors of '
            'shape `[2, num_messages]`, or `SparseTensor` for argument '
            '`edge_index`.')

    def _set_size(
        self,
        size: List[Optional[int]],
        dim: int,
        src: Tensor,
    ) -> None:
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.shape[self.node_dim]
        elif the_size != src.shape[self.node_dim]:
            raise ValueError(
                f'Encountered tensor with size {src.shape[self.node_dim]} in '
                f'dimension {self.node_dim}, but expected size {the_size}.')

    def _index_select(self, src: Tensor, index) -> Tensor:
        return paddle.index_select(src, index, axis=self.node_dim)

    def _lift(
        self,
        src: Tensor,
        edge_index: Union[Tensor, SparseTensor],
        dim: int,
    ) -> Tensor:
        if isinstance(edge_index, SparseTensor):
            row, col, _ = edge_index.coo()
            if dim == 0:
                return paddle.index_select(src, col, axis=self.node_dim)
            elif dim == 1:
                return paddle.index_select(src, row, axis=self.node_dim)

        elif isinstance(edge_index, Tensor):
            index = edge_index[dim]
            return paddle.index_select(src, index, axis=self.node_dim)

        raise ValueError(
            '`MessagePassing.propagate` only supports integer tensors of '
            'shape `[2, num_messages]`, or `SparseTensor` for argument '
            '`edge_index`.')
    def _collect(
        self,
        args: set,
        edge_index: Union[Tensor, SparseTensor],
        size: List[Optional[int]],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

        out = {}
        for arg in args:
            if arg[-2:] not in ['_i', '_j']:
                out[arg] = kwargs.get(arg, None)
            else:
                dim = j if arg[-2:] == '_j' else i
                data = kwargs.get(arg[:-2], None)

                if isinstance(data, (tuple, list)):
                    assert len(data) == 2
                    if isinstance(data[1 - dim], Tensor):
                        self._set_size(size, 1 - dim, data[1 - dim])
                    data = data[dim]

                if isinstance(data, Tensor):
                    self._set_size(size, dim, data)
                    data = self._lift(data, edge_index, dim)

                out[arg] = data

        if isinstance(edge_index, SparseTensor):
            row, col, value = edge_index.coo()
            out['adj_t'] = edge_index
            out['edge_index'] = None
            out['edge_index_i'] = row
            out['edge_index_j'] = col
            out['ptr'] = edge_index.row()  # Assuming CSR
            if out.get('edge_weight', None) is None:
                out['edge_weight'] = value
            if out.get('edge_attr', None) is None:
                out['edge_attr'] = value
            if out.get('edge_type', None) is None:
                out['edge_type'] = value

        elif isinstance(edge_index, Tensor):
            out['adj_t'] = None
            out['edge_index'] = edge_index
            out['edge_index_i'] = edge_index[i]
            out['edge_index_j'] = edge_index[j]

        out['index'] = out['edge_index_i']
        out['size'] = size
        out['size_i'] = size[i] if size[i] is not None else size[j]
        out['size_j'] = size[j] if size[j] is not None else size[i]
        out['dim_size'] = out['size_i']

        return out

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        r"""
        Runs the forward pass of the module.
        """

    def propagate(
        self,
        edge_index: Adj,
        size: Size = None,
        **kwargs: Any,
    ) -> Tensor:
        r"""
        The initial call to start propagating messages.
        """
        mutable_size = self._check_input(edge_index, size)

        if isinstance(edge_index, SparseTensor):
            coll_dict = self._collect(self._fused_user_args, edge_index, mutable_size, kwargs)

            msg_aggr_kwargs = self.inspector.collect_param_data(
                'message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.collect_param_data('update', coll_dict)
            out = self.update(out, **update_kwargs)

        else:
            coll_dict = self._collect(self._user_args, edge_index, mutable_size, kwargs)

            msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
            out = self.message(**msg_kwargs)

            aggr_kwargs = self.inspector.collect_param_data('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.collect_param_data('update', coll_dict)
            out = self.update(out, **update_kwargs)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        """
        Constructs messages from node `j` to node `i`.

        Args:
            x_j (Tensor): Node features of neighbors (source nodes).

        Returns:
            Tensor: Computed messages.
        """
        return x_j

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        """
        Aggregates messages from neighbors.

        Args:
            inputs (Tensor): Messages to be aggregated.
            index (Tensor): Indices for aggregation.
            ptr (Optional[Tensor], optional): Pointer tensor for segmented aggregation. Defaults to None.
            dim_size (Optional[int], optional): Size of the output dimension. Defaults to None.

        Returns:
            Tensor: Aggregated messages.
        """
        return self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size, axis=self.node_dim)

    @abstractmethod
    def message_and_aggregate(self, edge_index: Tensor) -> Tensor:
        """
        Combines `message` and `aggregate` computations into a single function.

        This optimization avoids materializing individual messages, improving efficiency.

        Args:
            edge_index (Tensor): Graph connectivity represented as edges.

        Returns:
            Tensor: Aggregated messages.
        """
        raise NotImplementedError

    def update(self, inputs: Tensor) -> Tensor:
        """
        Updates the node embeddings.

        Args:
            inputs (Tensor): Aggregated messages.

        Returns:
            Tensor: Updated node embeddings.
        """
        return inputs

    def edge_updater(
        self,
        edge_index: Tensor,
        size: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Computes or updates features for each edge in the graph.

        Args:
            edge_index (Tensor): Graph connectivity represented as edges.
            size (Optional[Tensor], optional): Size of the adjacency matrix. Defaults to None.
            **kwargs: Additional data required for edge updates.

        Returns:
            Tensor: Updated edge features.
        """
        for hook in self._edge_update_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        mutable_size = self._check_input(edge_index, size=None)

        coll_dict = self._collect(self._edge_user_args, edge_index, mutable_size, kwargs)

        edge_kwargs = self.inspector.collect_param_data('edge_update', coll_dict)
        out = self.edge_update(**edge_kwargs)

        for hook in self._edge_update_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out

    def edge_update(self) -> Tensor:
        """
        Computes or updates features for each edge in the graph.

        Returns:
            Tensor: Updated edge features.
        """
        raise NotImplementedError

    @property
    def decomposed_layers(self) -> int:
        """
        Returns the number of decomposed layers.
        """
        return self._decomposed_layers

    @decomposed_layers.setter
    def decomposed_layers(self, decomposed_layers: int) -> None:
        """
        Sets the number of decomposed layers for memory optimization.

        Args:
            decomposed_layers (int): Number of decomposed layers.
        """
        if decomposed_layers == self._decomposed_layers:
            return  # Skip if no change.

        self._decomposed_layers = decomposed_layers

    @property
    def explain(self) -> Optional[bool]:
        """
        Returns whether the layer is in explainability mode.
        """
        return self._explain

    @explain.setter
    def explain(self, explain: Optional[bool]) -> None:
        """
        Enables or disables explainability mode.

        Args:
            explain (Optional[bool]): Whether to enable explainability mode.
        """
        if explain == self._explain:
            return  # Skip if no change.

        self._explain = explain

    def explain_message(
        self,
        inputs: Tensor,
        dim_size: Optional[int],
    ) -> Tensor:
        """
        Customizes how messages are explained for interpretability.

        Args:
            inputs (Tensor): Messages to be explained.
            dim_size (Optional[int]): Size of the dimension for explanation.

        Returns:
            Tensor: Explained messages.
        """
        edge_mask = self._edge_mask

        if edge_mask is None:
            raise ValueError("No pre-defined 'edge_mask' found for explanation.")

        if self._apply_sigmoid:
            edge_mask = paddle.nn.functional.sigmoid(edge_mask)

        if inputs.shape[self.node_dim] != edge_mask.shape[0]:
            assert dim_size is not None
            edge_mask = edge_mask[self._loop_mask]
            loop = paddle.ones([dim_size], dtype=edge_mask.dtype)
            edge_mask = paddle.concat([edge_mask, loop], axis=0)
        assert inputs.shape[self.node_dim] == edge_mask.shape[0]

        size = [1] * len(inputs.shape)
        size[self.node_dim] = -1
        return inputs * edge_mask.reshape(size)

    def register_propagate_forward_pre_hook(self, hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._propagate_forward_pre_hooks)
        self._propagate_forward_pre_hooks[handle.id] = hook
        return handle

    def register_propagate_forward_hook(self, hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._propagate_forward_hooks)
        self._propagate_forward_hooks[handle.id] = hook
        return handle

    def register_message_forward_pre_hook(self, hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._message_forward_pre_hooks)
        self._message_forward_pre_hooks[handle.id] = hook
        return handle

    def register_message_forward_hook(self, hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._message_forward_hooks)
        self._message_forward_hooks[handle.id] = hook
        return handle

    def register_aggregate_forward_pre_hook(self, hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._aggregate_forward_pre_hooks)
        self._aggregate_forward_pre_hooks[handle.id] = hook
        return handle

    def register_aggregate_forward_hook(self, hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._aggregate_forward_hooks)
        self._aggregate_forward_hooks[handle.id] = hook
        return handle

    def register_message_and_aggregate_forward_pre_hook(self, hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._message_and_aggregate_forward_pre_hooks)
        self._message_and_aggregate_forward_pre_hooks[handle.id] = hook
        return handle

    def register_message_and_aggregate_forward_hook(self, hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._message_and_aggregate_forward_hooks)
        self._message_and_aggregate_forward_hooks[handle.id] = hook
        return handle

    def register_edge_update_forward_pre_hook(self, hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._edge_update_forward_pre_hooks)
        self._edge_update_forward_pre_hooks[handle.id] = hook
        return handle

    def register_edge_update_forward_hook(self, hook: Callable) -> RemovableHandle:
        handle = RemovableHandle(self._edge_update_forward_hooks)
        self._edge_update_forward_hooks[handle.id] = hook
        return handle

    def _set_jittable_templates(self, raise_on_error: bool = False) -> None:
        root_dir = osp.dirname(osp.realpath(__file__))
        jinja_prefix = f'{self.__module__}_{self.__class__.__name__}'

        # Optimize `propagate()` via templates:
        if not self.propagate.__module__.startswith(jinja_prefix):
            try:
                if ('propagate' in self.__class__.__dict__
                        and self.__class__.__dict__['propagate']
                        != MessagePassingLayer.propagate):
                    raise ValueError("Cannot compile custom 'propagate' method")

                # Placeholder for Jinja template compilation.
                # Add logic if Paddle needs Jinja-like behavior.
                self.__class__._orig_propagate = self.__class__.propagate
            except Exception as e:
                if raise_on_error:
                    raise e
                self.__class__._orig_propagate = self.__class__.propagate

        # Optimize `edge_updater()` via templates:
        if (hasattr(self, 'edge_update')
                and not self.edge_updater.__module__.startswith(jinja_prefix)):
            try:
                if ('edge_updater' in self.__class__.__dict__
                        and self.__class__.__dict__['edge_updater']
                        != MessagePassingLayer.edge_updater):
                    raise ValueError("Cannot compile custom 'edge_updater' method")

                # Placeholder for Jinja template compilation.
                self.__class__._orig_edge_updater = self.__class__.edge_updater
            except Exception as e:
                if raise_on_error:
                    raise e
                self.__class__._orig_edge_updater = self.__class__.edge_updater


    def _get_propagate_signature(self) -> Signature:
        """
        Gets the propagate method signature.

        Returns:
            A `Signature` object containing parameter details and return type.
        """
        param_dict = self.inspector.get_params_from_method_call(
            'propagate', exclude=[0, 'edge_index', 'size'])
        update_signature = self.inspector.get_signature('update')

        return Signature(
            param_dict=param_dict,
            return_type=update_signature.return_type,
            return_type_repr=update_signature.return_type_repr,
        )

    def _get_edge_updater_signature(self) -> Signature:
        """
        Gets the edge updater method signature.

        Returns:
            A `Signature` object containing parameter details and return type.
        """
        param_dict = self.inspector.get_params_from_method_call(
            'edge_updater', exclude=[0, 'edge_index', 'size'])
        edge_update_signature = self.inspector.get_signature('edge_update')

        return Signature(
            param_dict=param_dict,
            return_type=edge_update_signature.return_type,
            return_type_repr=edge_update_signature.return_type_repr,
        )

    def jittable(self, typing: Optional[str] = None) -> 'MessagePassingLayer':
        """
        Produces a new jittable module for compatibility.

        Note:
            This method is deprecated and a no-op in Paddle implementation.

        Args:
            typing (Optional[str]): Typing information (not used in Paddle).

        Returns:
            self: The current instance of the layer.
        """
        warnings.warn(f"'{self.__class__.__name__}.jittable' is deprecated "
                      f"and a no-op. Please remove its usage.")
        return self