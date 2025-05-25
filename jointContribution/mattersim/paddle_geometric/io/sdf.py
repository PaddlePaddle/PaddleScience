import paddle

from paddle_geometric.data import Data
from paddle_geometric.io import parse_txt_array
from paddle_geometric.utils import coalesce, one_hot

elems = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}


def parse_sdf(src: str) -> Data:
    lines = src.split('\n')[3:]
    num_atoms, num_bonds = (int(item) for item in lines[0].split()[:2])

    atom_block = lines[1:num_atoms + 1]
    pos = parse_txt_array(atom_block, end=3)
    x = paddle.to_tensor([elems[item.split()[3]] for item in atom_block], dtype='int64')
    x = one_hot(x, num_classes=len(elems))

    bond_block = lines[1 + num_atoms:1 + num_atoms + num_bonds]
    row, col = parse_txt_array(bond_block, end=2, dtype='int64').transpose([1, 0]) - 1
    row, col = paddle.concat([row, col], axis=0), paddle.concat([col, row], axis=0)
    edge_index = paddle.stack([row, col], axis=0)
    edge_attr = parse_txt_array(bond_block, start=2, end=3) - 1
    edge_attr = paddle.concat([edge_attr, edge_attr], axis=0)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_atoms)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)


def read_sdf(path: str) -> Data:
    with open(path) as f:
        return parse_sdf(f.read())
