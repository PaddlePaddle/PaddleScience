from collections import defaultdict
from typing import Any, List, Optional, Union

import paddle
from paddle import nn
from paddle.nn import Layer

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.dense.linear import is_uninitialized_parameter
from paddle_geometric.typing import SparseTensor


def summary(
    model: nn.Layer,
    *args,
    max_depth: int = 3,
    leaf_module: Optional[Union[Layer, List[Layer]]] = 'MessagePassing',
    **kwargs,
) -> str:
    r"""Summarizes a given :class:`paddle.nn.Layer`.
    The summarized information includes (1) layer names, (2) input and output
    shapes, and (3) the number of parameters.

    Args:
        model (paddle.nn.Layer): The model to summarize.
        *args: The arguments of the :obj:`model`.
        max_depth (int, optional): The depth of nested layers to display.
            Any layers deeper than this depth will not be displayed in the
            summary. (default: :obj:`3`)
        leaf_module (paddle.nn.Layer or [paddle.nn.Layer], optional): The
            modules to be treated as leaf modules, whose submodules are
            excluded from the summary.
            (default: :class:`~paddle_geometric.nn.conv.MessagePassing`)
        **kwargs: Additional arguments of the :obj:`model`.
    """
    if leaf_module == 'MessagePassing':
        leaf_module = MessagePassing

    def register_hook(info):
        def hook(module, inputs, output):
            info['input_shape'].append(get_shape(inputs))
            info['output_shape'].append(get_shape(output))

        return hook

    hooks = {}
    depth = 0
    stack = [(model.__class__.__name__, model, depth)]

    info_list = []
    input_shape = defaultdict(list)
    output_shape = defaultdict(list)
    while stack:
        name, module, depth = stack.pop()
        module_id = id(module)

        if name.startswith('(_'):  # Do not summarize private modules.
            continue

        if module_id in hooks:  # Avoid duplicated hooks.
            hooks[module_id].remove()

        info = {}
        info['name'] = name
        info['input_shape'] = input_shape[module_id]
        info['output_shape'] = output_shape[module_id]
        info['depth'] = depth
        if any([is_uninitialized_parameter(p) for p in module.parameters()]):  # Paddle has a similar check for uninitialized parameters
            info['#param'] = '-1'
        else:
            num_params = sum(p.numel() for p in module.parameters())
            info['#param'] = f'{num_params:,}' if num_params > 0 else '--'
        info_list.append(info)

        if not isinstance(module, nn.ScriptModule):
            hooks[module_id] = module.register_forward_hook(
                register_hook(info))

        if depth >= max_depth:
            continue

        if (leaf_module is not None and isinstance(module, leaf_module)):
            continue

        module_items = reversed(module._sub_layers.items())  # In Paddle, submodules are stored in `_sub_layers`
        stack += [(f"({name}){mod.__class__.__name__}", mod, depth + 1)
                  for name, mod in module_items if mod is not None]

    training = model.training
    model.eval()

    with paddle.no_grad():
        model(*args, **kwargs)

    model.train(training)

    for h in hooks.values():  # Remove hooks.
        h.remove()

    info_list = postprocess(info_list)
    return make_table(info_list, max_depth=max_depth)


def get_shape(inputs: Any) -> str:
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs, )

    out = []
    for x in inputs:
        if isinstance(x, SparseTensor):
            out.append(str(list(x.sizes())))
        elif hasattr(x, 'shape'):
            out.append(str(list(x.shape)))  # In Paddle, use `.shape` instead of `.size()`
    return ', '.join(out)


def postprocess(info_list: List[dict]) -> List[dict]:
    for idx, info in enumerate(info_list):
        depth = info['depth']
        if idx > 0:  # root module (0) is excluded
            if depth == 1:
                prefix = '├─'
            else:
                prefix = f"{'│    '*(depth-1)}└─"
            info['name'] = prefix + info['name']

        if info['input_shape']:
            info['input_shape'] = info['input_shape'].pop(0)
            info['output_shape'] = info['output_shape'].pop(0)
        else:
            info['input_shape'] = '--'
            info['output_shape'] = '--'
    return info_list


def make_table(info_list: List[dict], max_depth: int) -> str:
    from tabulate import tabulate
    content = [['Layer', 'Input Shape', 'Output Shape', '#Param']]
    for info in info_list:
        content.append([
            info['name'],
            info['input_shape'],
            info['output_shape'],
            info['#param'],
        ])
    return tabulate(content, headers='firstrow', tablefmt='psql')
