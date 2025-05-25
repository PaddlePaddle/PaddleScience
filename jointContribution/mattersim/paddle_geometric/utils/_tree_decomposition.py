from itertools import chain
from typing import Any, List, Literal, Tuple, Union, overload

import paddle
from paddle import Tensor
from paddle_geometric.utils import (
    from_scipy_sparse_matrix,
    to_scipy_sparse_matrix,
    to_undirected,
)

@overload
def tree_decomposition(mol: Any) -> Tuple[Tensor, Tensor, int]:
    pass

@overload
def tree_decomposition(
    mol: Any,
    return_vocab: Literal[False],
) -> Tuple[Tensor, Tensor, int]:
    pass

@overload
def tree_decomposition(
    mol: Any,
    return_vocab: Literal[True],
) -> Tuple[Tensor, Tensor, int, Tensor]:
    pass

def tree_decomposition(
    mol: Any,
    return_vocab: bool = False,
) -> Union[Tuple[Tensor, Tensor, int], Tuple[Tensor, Tensor, int, Tensor]]:
    r"""Tree decomposition algorithm for molecules based on the
    `"Junction Tree Variational Autoencoder for Molecular Graph Generation"
    <https://arxiv.org/abs/1802.04364>`_ paper.
    Returns the graph connectivity of the junction tree, the assignment
    mapping of each atom to the clique in the junction tree, and the number
    of cliques.

    Args:
        mol (rdkit.Chem.Mol): An :obj:`rdkit` molecule.
        return_vocab (bool, optional): If set to :obj:`True`, returns an
            identifier for each clique (ring, bond, bridged compounds, single).
            (default: :obj:`False`)

    :rtype: :obj:`(LongTensor, LongTensor, int)` if :obj:`return_vocab` is
        :obj:`False`, else :obj:`(LongTensor, LongTensor, int, LongTensor)`
    """
    import rdkit.Chem as Chem
    from scipy.sparse.csgraph import minimum_spanning_tree

    # Cliques are defined as rings and bonds.
    cliques: List[List[int]] = [list(x) for x in Chem.GetSymmSSSR(mol)]
    xs: List[int] = [0] * len(cliques)
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            cliques.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            xs.append(1)

    # Generate `atom2cliques` mappings.
    atom2cliques: List[List[int]] = [[] for _ in range(mol.GetNumAtoms())]
    for c in range(len(cliques)):
        for atom in cliques[c]:
            atom2cliques[atom].append(c)

    # Merge rings that share more than 2 atoms as bridged compounds.
    for c1 in range(len(cliques)):
        for atom in cliques[c1]:
            for c2 in atom2cliques[atom]:
                if c1 >= c2 or len(cliques[c1]) <= 2 or len(cliques[c2]) <= 2:
                    continue
                if len(set(cliques[c1]) & set(cliques[c2])) > 2:
                    cliques[c1] = list(set(cliques[c1]) | set(cliques[c2]))
                    xs[c1] = 2
                    cliques[c2] = []
                    xs[c2] = -1
    cliques = [c for c in cliques if len(c) > 0]
    xs = [x for x in xs if x >= 0]

    # Update `atom2cliques` mappings.
    atom2cliques = [[] for _ in range(mol.GetNumAtoms())]
    for c in range(len(cliques)):
        for atom in cliques[c]:
            atom2cliques[atom].append(c)

    # Add singleton cliques for more than 2 intersecting cliques and compute initial clique graph.
    edges = {}
    for atom in range(mol.GetNumAtoms()):
        cs = atom2cliques[atom]
        if len(cs) <= 1:
            continue

        bonds = [c for c in cs if len(cliques[c]) == 2]
        rings = [c for c in cs if len(cliques[c]) > 4]

        if len(bonds) > 2 or (len(bonds) == 2 and len(cs) > 2):
            cliques.append([atom])
            xs.append(3)
            c2 = len(cliques) - 1
            for c1 in cs:
                edges[(c1, c2)] = 1

        elif len(rings) > 2:
            cliques.append([atom])
            xs.append(3)
            c2 = len(cliques) - 1
            for c1 in cs:
                edges[(c1, c2)] = 99

        else:
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    c1, c2 = cs[i], cs[j]
                    count = len(set(cliques[c1]) & set(cliques[c2]))
                    edges[(c1, c2)] = min(count, edges.get((c1, c2), 99))

    # Update `atom2cliques` mappings.
    atom2cliques = [[] for _ in range(mol.GetNumAtoms())]
    for c in range(len(cliques)):
        for atom in cliques[c]:
            atom2cliques[atom].append(c)

    if len(edges) > 0:
        edge_index_T, weight = zip(*edges.items())
        edge_index = paddle.to_tensor(edge_index_T).t()
        inv_weight = 100 - paddle.to_tensor(weight)
        graph = to_scipy_sparse_matrix(edge_index, inv_weight, len(cliques))
        junc_tree = minimum_spanning_tree(graph)
        edge_index, _ = from_scipy_sparse_matrix(junc_tree)
        edge_index = to_undirected(edge_index, num_nodes=len(cliques))
    else:
        edge_index = paddle.empty((2, 0), dtype=paddle.int64)

    rows = [[i] * len(atom2cliques[i]) for i in range(mol.GetNumAtoms())]
    row = paddle.to_tensor(list(chain.from_iterable(rows)))
    col = paddle.to_tensor(list(chain.from_iterable(atom2cliques)))
    atom2clique = paddle.stack([row, col], axis=0).astype(paddle.int64)

    if return_vocab:
        vocab = paddle.to_tensor(xs, dtype=paddle.int64)
        return edge_index, atom2clique, len(cliques), vocab
    else:
        return edge_index, atom2clique, len(cliques)
