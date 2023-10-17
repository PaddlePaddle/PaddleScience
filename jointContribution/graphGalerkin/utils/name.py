from typing import Tuple, Optional, Union, List, Callable, Dict
import inspect
from paddle import Tensor
import re
from collections import OrderedDict
import pyparsing as pp
from sparsetensor import SparseTensor
# Types for accessing data ####################################################

# Node-types are denoted by a single string, e.g.: `data['paper']`:
NodeType = str

# Edge-types are denotes by a triplet of strings, e.g.:
# `data[('author', 'writes', 'paper')]
EdgeType = Tuple[str, str, str]

# There exist some short-cuts to query edge-types (given that the full triplet
# can be uniquely reconstructed, e.g.:
# * via str: `data['writes']`
# * via Tuple[str, str]: `data[('author', 'paper')]`
QueryType = Union[NodeType, EdgeType, str, Tuple[str, str]]

Metadata = Tuple[List[NodeType], List[EdgeType]]

# Types for message passing ###################################################

Adj = Union[Tensor, SparseTensor]
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Tensor]

# Types for sampling ##########################################################

InputNodes = Union[OptTensor, NodeType, Tuple[NodeType, OptTensor]]

Adj = Union[Tensor, SparseTensor]
Size = Optional[Tuple[int, int]]

def split_types_repr(types_repr: str) -> List[str]:
    out = []
    i = depth = 0
    for j, char in enumerate(types_repr):
        if char == '[':
            depth += 1
        elif char == ']':
            depth -= 1
        elif char == ',' and depth == 0:
            out.append(types_repr[i:j].strip())
            i = j + 1
    out.append(types_repr[i:].strip())
    return out

def sanitize(type_repr: str):
    type_repr = re.sub(r'<class \'(.*)\'>', r'\1', type_repr)
    type_repr = type_repr.replace('typing.', '')
    type_repr = type_repr.replace('torch_sparse.tensor.', '')
    type_repr = type_repr.replace('Adj', 'Union[Tensor, SparseTensor]')

    # Replace `Union[..., NoneType]` by `Optional[...]`.
    sexp = pp.nestedExpr(opener='[', closer=']')
    tree = sexp.parseString(f'[{type_repr.replace(",", " ")}]').asList()[0]

    def union_to_optional_(tree):
        for i in range(len(tree)):
            e, n = tree[i], tree[i + 1] if i + 1 < len(tree) else []
            if e == 'Union' and n[-1] == 'NoneType':
                tree[i] = 'Optional'
                tree[i + 1] = tree[i + 1][:-1]
            elif e == 'Union' and 'NoneType' in n:
                idx = n.index('NoneType')
                n[idx] = [n[idx - 1]]
                n[idx - 1] = 'Optional'
            elif isinstance(e, list):
                tree[i] = union_to_optional_(e)
        return tree

    tree = union_to_optional_(tree)
    type_repr = re.sub(r'\'|\"', '', str(tree)[1:-1]).replace(', [', '[')

    return type_repr

def param_type_repr(param) -> str:
    if param.annotation is inspect.Parameter.empty:
        return 'torch.Tensor'
    return sanitize(re.split(r':|='.strip(), str(param))[1])


def return_type_repr(signature) -> str:
    return_type = signature.return_annotation
    if return_type is inspect.Parameter.empty:
        return 'torch.Tensor'
    elif str(return_type)[:6] != '<class':
        return sanitize(str(return_type))
    elif return_type.__module__ == 'builtins':
        return return_type.__name__
    else:
        return f'{return_type.__module__}.{return_type.__name__}'
    
def parse_types(func: Callable) -> List[Tuple[Dict[str, str], str]]:
    source = inspect.getsource(func)
    signature = inspect.signature(func)

    # Parse `# type: (...) -> ...` annotation. Note that it is allowed to pass
    # multiple `# type:` annotations in `forward()`.
    iterator = re.finditer(r'#\s*type:\s*\((.*)\)\s*->\s*(.*)\s*\n', source)
    matches = list(iterator)

    if len(matches) > 0:
        out = []
        args = list(signature.parameters.keys())
        for match in matches:
            arg_types_repr, return_type = match.groups()
            arg_types = split_types_repr(arg_types_repr)
            arg_types = OrderedDict((k, v) for k, v in zip(args, arg_types))
            return_type = return_type.split('#')[0].strip()
            out.append((arg_types, return_type))
        return out

    # Alternatively, parse annotations using the inspected signature.
    else:
        ps = signature.parameters
        arg_types = OrderedDict((k, param_type_repr(v)) for k, v in ps.items())
        return [(arg_types, return_type_repr(signature))]