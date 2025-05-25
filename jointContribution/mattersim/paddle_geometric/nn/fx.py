import copy
import warnings
from typing import Any, Dict, Optional

import paddle
from paddle.nn import Layer, LayerList, LayerDict, Sequential

try:
    from paddle.fx import Graph, GraphLayer, Node
except (ImportError, ModuleNotFoundError, AttributeError):
    GraphLayer, Graph, Node = 'GraphLayer', 'Graph', 'Node'


class Transformer:
    r"""
    A `Transformer` executes an FX graph node-by-node, applies transformations
    to each node, and produces a new `paddle.nn.Layer`. It exposes a `transform`
    method that returns the transformed `paddle.fx.GraphLayer`.

    Methods in the `Transformer` class can be overridden to customize the
    behavior of transformation.

    .. code-block:: none

        transform()
            +-- Iterate over each node in the graph
                +-- placeholder()
                +-- get_attr()
                +-- call_function()
                +-- call_method()
                +-- call_module()
                +-- call_message_passing_module()
                +-- call_global_pooling_module()
                +-- output()
            +-- Erase unused nodes in the graph
            +-- Iterate over each children module
                +-- init_submodule()

    Args:
        module (paddle.nn.Layer): The module to be transformed.
        input_map (Dict[str, str], optional): A dictionary holding information
            about the type of input arguments of `module.forward`.
            For example, if `arg` is a node-level argument, then
            `input_map['arg'] = 'node'`, and `input_map['arg'] = 'edge'` otherwise.
            If `input_map` is not specified, it will try to automatically
            determine the correct type of input arguments.
            (default: `None`)
        debug (bool, optional): If set to `True`, will perform transformation
            in debug mode. (default: `False`)
    """
    def __init__(
        self,
        module: Layer,
        input_map: Optional[Dict[str, str]] = None,
        debug: bool = False,
    ):
        self.module = module
        self.gm = symbolic_trace(module)
        self.input_map = input_map
        self.debug = debug

    # Methods to override #####################################################

    def placeholder(self, node: Node, target: Any, name: str):
        pass

    def get_attr(self, node: Node, target: Any, name: str):
        pass

    def call_message_passing_module(self, node: Node, target: Any, name: str):
        pass

    def call_global_pooling_module(self, node: Node, target: Any, name: str):
        pass

    def call_module(self, node: Node, target: Any, name: str):
        pass

    def call_method(self, node: Node, target: Any, name: str):
        pass

    def call_function(self, node: Node, target: Any, name: str):
        pass

    def output(self, node: Node, target: Any, name: str):
        pass

    def init_submodule(self, module: Layer, target: str) -> Layer:
        return module

    # Internal functionality ##################################################

    @property
    def graph(self) -> Graph:
        return self.gm.graph

    def transform(self) -> GraphLayer:
        """
        Transforms `self.module` and returns a transformed `paddle.fx.GraphLayer`.
        """
        if self.debug:
            self.graph.print_tabular()
            print()
            code = self.graph.python_code('self')
            print(code.src if hasattr(code, 'src') else code)

        # We create a private dictionary `self._state` which holds information
        # about whether a node returns node-level or edge-level information:
        # `self._state[node.name] in { 'node', 'edge' }`
        self._state = copy.copy(self.input_map or {})

        # We iterate over each node and determine its output level
        # (node-level, edge-level) by filling `self._state`:
        for node in list(self.graph.nodes):
            if node.op == 'call_function' and 'training' in node.kwargs:
                warnings.warn(f"Found function '{node.name}' with keyword "
                              f"argument 'training'. During FX tracing, this "
                              f"will likely be baked in as a constant value. "
                              f"Consider replacing this function by a module "
                              f"to properly encapsulate its training flag.")

            if node.op == 'placeholder':
                if node.name not in self._state:
                    if 'edge' in node.name or 'adj' in node.name:
                        self._state[node.name] = 'edge'
                    else:
                        self._state[node.name] = 'node'
            elif is_message_passing_op(self.module, node.op, node.target):
                self._state[node.name] = 'node'
            elif is_global_pooling_op(self.module, node.op, node.target):
                self._state[node.name] = 'graph'
            elif node.op in ['call_module', 'call_method', 'call_function']:
                if self.has_edge_level_arg(node):
                    self._state[node.name] = 'edge'
                elif self.has_node_level_arg(node):
                    self._state[node.name] = 'node'
                else:
                    self._state[node.name] = 'graph'

        # We iterate over each node and may transform it:
        for node in list(self.graph.nodes):
            # Call the corresponding `Transformer` method for each `node.op`,
            # e.g.: `call_module(...)`, `call_function(...)`, ...
            op = node.op
            if is_message_passing_op(self.module, op, node.target):
                op = 'call_message_passing_module'
            elif is_global_pooling_op(self.module, op, node.target):
                op = 'call_global_pooling_module'
            getattr(self, op)(node, node.target, node.name)

        # Remove all unused nodes in the computation graph, i.e., all nodes
        # which have been replaced by node type-wise or edge type-wise variants
        # but which are still present in the computation graph.
        # We do this by iterating over the computation graph in reversed order,
        # and try to remove every node. This does only succeed in case there
        # are no users of that node left in the computation graph.
        for node in reversed(list(self.graph.nodes)):
            try:
                if node.op not in ['placeholder', 'output']:
                    self.graph.erase_node(node)
            except RuntimeError:
                pass

        for target, submodule in dict(self.module._sub_layers).items():
            self.gm._sub_layers[target] = self._init_submodule(submodule, target)

        del self._state

        if self.debug:
            self.gm.graph.print_tabular()
            print()
            code = self.graph.python_code('self')
            print(code.src if hasattr(code, 'src') else code)

        self.gm.graph.lint()
        self.gm.recompile()

        return self.gm
    def _init_submodule(self, module: Layer, target: str) -> Layer:
        if isinstance(module, LayerList) or isinstance(module, Sequential):
            return LayerList([
                self._init_submodule(submodule, f'{target}.{i}')
                for i, submodule in enumerate(module)
            ])
        elif isinstance(module, LayerDict):
            return LayerDict({
                key:
                self._init_submodule(submodule, f'{target}.{key}')
                for key, submodule in module.items()
            })
        else:
            return self.init_submodule(module, target)

    def _is_level(self, node: Node, name: str) -> bool:
        return self._state[node.name] == name

    def _has_level_arg(self, node: Node, name: str) -> bool:
        def _recurse(value: Any) -> bool:
            if isinstance(value, Node):
                return getattr(self, f'is_{name}_level')(value)
            elif isinstance(value, dict):
                return any([_recurse(v) for v in value.values()])
            elif isinstance(value, (list, tuple)):
                return any([_recurse(v) for v in value])
            else:
                return False

        return (any([_recurse(value) for value in node.args])
                or any([_recurse(value) for value in node.kwargs.values()]))

    def is_node_level(self, node: Node) -> bool:
        return self._is_level(node, name='node')

    def is_edge_level(self, node: Node) -> bool:
        return self._is_level(node, name='edge')

    def is_graph_level(self, node: Node) -> bool:
        return self._is_level(node, name='graph')

    def has_node_level_arg(self, node: Node) -> bool:
        return self._has_level_arg(node, name='node')

    def has_edge_level_arg(self, node: Node) -> bool:
        return self._has_level_arg(node, name='edge')

    def has_graph_level_arg(self, node: Node) -> bool:
        return self._has_level_arg(node, name='graph')

    def find_by_name(self, name: str) -> Optional[Node]:
        for node in self.graph.nodes:
            if node.name == name:
                return node
        return None

    def find_by_target(self, target: Any) -> Optional[Node]:
        for node in self.graph.nodes:
            if node.target == target:
                return node
        return None

    def replace_all_uses_with(self, to_replace: Node, replace_with: Node):
        def maybe_replace_node(n: Node) -> Node:
            return replace_with if n == to_replace else n

        node = replace_with.next
        while node.op != 'root':
            node.args = paddle.fx.map_arg(node.args, maybe_replace_node)
            node.kwargs = paddle.fx.map_arg(node.kwargs, maybe_replace_node)
            node = node.next
def symbolic_trace(module: Layer, concrete_args: Optional[Dict[str, Any]] = None) -> GraphLayer:
    from paddle_geometric.nn import Aggregation

    class Tracer(paddle.fx.Tracer):
        def is_leaf_layer(self, module: Layer, *args, **kwargs) -> bool:
            return not isinstance(module, paddle.nn.Sequential)

        @staticmethod
        def trace(root: Any, concrete_args: Optional[Dict[str, Any]] = None) -> Graph:
            tracer = Tracer()
            tracer.root = root
            tracer.graph = Graph()
            tracer.tensor_attrs: Dict[Any, str] = {}

            def collect_tensor_attrs(m: Layer, prefix_atoms: list):
                for k, v in m.__dict__.items():
                    if isinstance(v, paddle.Tensor):
                        tracer.tensor_attrs[v] = '.'.join(prefix_atoms + [k])
                for k, v in m.named_children():
                    collect_tensor_attrs(v, prefix_atoms + [k])

            collect_tensor_attrs(root, [])

            fn, args = tracer.create_args_for_root(
                root.forward, isinstance(root, Layer), concrete_args
            )

            parameter_proxy_cache: Dict[str, Any] = {}

            def layer_getattr_wrapper(mod, attr):
                attr_val = getattr(mod, attr)
                return tracer.getattr(attr, attr_val, parameter_proxy_cache)

            def layer_call_wrapper(mod, *args, **kwargs):
                def forward(*args, **kwargs):
                    return mod.forward(*args, **kwargs)

                return tracer.call_layer(mod, forward, args, kwargs)

            with paddle.utils.PatchContext() as patcher:
                patcher.patch_method(Layer, "__getattr__", layer_getattr_wrapper)
                patcher.patch_method(Layer, "__call__", layer_call_wrapper)
                patcher.patch_method(Aggregation, "__call__", layer_call_wrapper)

                tracer.create_node(
                    'output', 'output', (tracer.create_arg(fn(*args)),), {}
                )

            return tracer.graph

    return GraphLayer(module, Tracer().trace(module, concrete_args))


def get_submodule(module: Layer, target: str) -> Layer:
    out = module
    for attr in target.split('.'):
        out = getattr(out, attr)
    return out


def is_message_passing_op(module: Layer, op: str, target: str) -> bool:
    from paddle_geometric.nn import MessagePassing
    if op == 'call_layer':
        return isinstance(get_submodule(module, target), MessagePassing)
    return False


def is_global_pooling_op(module: Layer, op: str, target: str) -> bool:
    from paddle_geometric.nn import Aggregation
    if op == 'call_layer':
        return isinstance(get_submodule(module, target), Aggregation)
    return False