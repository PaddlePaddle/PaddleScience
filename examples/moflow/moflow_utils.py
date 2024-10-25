import json
import math
import os
import pickle as cPickle
import re

import networkx as nx
import numpy as np
import paddle
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
from rdkit.six import iteritems

from ppsci.utils import download
from ppsci.utils import logger

atom_decoder_m = {0: 6, 1: 7, 2: 8, 3: 9}
bond_decoder_m = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
}
ATOM_VALENCY = {
    6: 4,
    7: 3,
    8: 2,
    9: 1,
    15: 3,
    16: 2,
    17: 1,
    35: 1,
    53: 1,
}

_fscores = None


class Hyperparameters:
    def __init__(
        self,
        b_n_type=4,
        b_n_flow=-1,
        b_n_block=-1,
        b_n_squeeze=-1,
        b_hidden_ch=None,
        b_affine=True,
        b_conv_lu=2,
        a_n_node=-1,
        a_n_type=-1,
        a_hidden_gnn=None,
        a_hidden_lin=None,
        a_n_flow=-1,
        a_n_block=1,
        mask_row_size_list=None,
        mask_row_stride_list=None,
        a_affine=True,
        path=None,
        learn_dist=True,
        seed=1,
        noise_scale=0.6,
    ):
        """Model Hyperparameters
        Args:
            b_n_type (int, optional): Number of bond types/channels.
            b_n_flow (int, optional): Number of masked glow coupling layers per block for bond tensor.
            b_n_block (int, optional): Number of glow blocks for bond tensor.
            b_n_squeeze (int, optional):  Squeeze divisor, 3 for qm9, 2 for zinc250k.
            b_hidden_ch (list[int,...], optional): Hidden channel list for bonds tensor, delimited list input.
            b_affine (bool, optional): Using affine coupling layers for bonds glow.
            b_conv_lu (int, optional): Using L-U decomposition trick for 1-1 conv in bonds glow.
            a_n_node (int, optional): _Maximum number of atoms in a molecule.
            a_n_type (int, optional): _Number of atom types.
            a_hidden_gnn (object, optional): Hidden dimension list for graph convolution for atoms matrix, delimited list input.
            a_hidden_lin (object, optional): Hidden dimension list for linear transformation for atoms, delimited list input.
            a_n_flow (int, optional): _dNumber of masked flow coupling layers per block for atom matrix.
            a_n_block (int, optional): Number of flow blocks for atom matrix.
            mask_row_size_list (list[int,...], optional): Mask row list for atom matrix, delimited list input.
            mask_row_stride_list (list[int,...], optional): _Mask row stride  list for atom matrix, delimited list input.
            a_affine (bool, optional): Using affine coupling layers for atom conditional graph flow.
            path (str, optional): Hyperparameters save path.
            learn_dist (bool, optional): learn the distribution of feature matrix.
            seed (int, optional): Random seed to use.
            noise_scale (float, optional): x + torch.rand(x.shape) * noise_scale.

        """
        self.b_n_type = b_n_type
        self.b_n_flow = b_n_flow
        self.b_n_block = b_n_block
        self.b_n_squeeze = b_n_squeeze
        self.b_hidden_ch = b_hidden_ch
        self.b_affine = b_affine
        self.b_conv_lu = b_conv_lu
        self.a_n_node = a_n_node
        self.a_n_type = a_n_type
        self.a_hidden_gnn = a_hidden_gnn
        self.a_hidden_lin = a_hidden_lin
        self.a_n_flow = a_n_flow
        self.a_n_block = a_n_block
        self.mask_row_size_list = mask_row_size_list
        self.mask_row_stride_list = mask_row_stride_list
        self.a_affine = a_affine
        self.path = path
        self.learn_dist = learn_dist
        self.seed = seed
        self.noise_scale = noise_scale
        if path is not None:
            if os.path.exists(path) and os.path.isfile(path):
                with open(path, "r") as f:
                    obj = json.load(f)
                    for key, value in obj.items():
                        setattr(self, key, value)
            else:
                raise Exception("{} does not exist".format(path))

    def save(self, path):
        self.path = path
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True, cls=NumpyEncoder)

    def print(self):
        rows = []
        for key, value in self.__dict__.items():
            rows.append([key, value])
        return rows


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, paddle.Tensor):
            return obj.numpy().tolist()
        return json.JSONEncoder.default(self, obj)


def flatten_graph_data(adj, x):
    return paddle.concat(
        x=(adj.reshape([tuple(adj.shape)[0], -1]), x.reshape([tuple(x.shape)[0], -1])),
        axis=1,
    )


def split_channel(x):
    n = tuple(x.shape)[1] // 2
    return x[:, :n], x[:, n:]


def get_graph_data(x, num_nodes, num_relations, num_features):
    """Converts a vector of shape [b, num_nodes, m] to Adjacency matrix
    of shape [b, num_relations, num_nodes, num_nodes]
    and a feature matrix of shape [b, num_nodes, num_features].

    Args:
        x (paddle.Tensor): Adjacency.
        num_nodes (int): nodes number.
        num_relations (int): relations number.
        num_features (int): features number.

    Returns:
        Tuple[paddle.Tensor, ...]: Adjacency and A feature matrix.
    """
    adj = x[:, : num_nodes * num_nodes * num_relations].reshape(
        [-1, num_relations, num_nodes, num_nodes]
    )
    feat_mat = x[:, num_nodes * num_nodes * num_relations :].reshape(
        [-1, num_nodes, num_features]
    )
    return adj, feat_mat


def Tensor2Mol(A, x):
    mol = Chem.RWMol()
    atoms = np.argmax(x, 1)
    atoms_exist = atoms != 4
    atoms = atoms[atoms_exist]
    atoms += 6
    adj = np.argmax(A, 0)
    adj = np.array(adj)
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1
    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atom)))
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adj[start, end]])
    return mol


def construct_mol(x, A, atomic_num_list):
    """

    Args:
        x (paddle.Tensor): nodes.
        A (paddle.Tensor): Adjacency.
        atomic_num_list (list): atomic list number.
    Returns:
        rdkit mol object

    """
    mol = Chem.RWMol()
    atoms = np.argmax(x, axis=1)
    atoms_exist = atoms != len(atomic_num_list) - 1
    atoms = atoms[atoms_exist]
    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))
    adj = np.argmax(A, axis=0)
    adj = np.array(adj)
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adj[start, end]])
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and v - ATOM_VALENCY[an] == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


def construct_mol_with_validation(x, A, atomic_num_list):
    """
    Args:
        x (paddle.Tensor): nodes.
        A (paddle.Tensor): Adjacency.
        atomic_num_list (list): atomic list number.

    Returns:
        rdkit mol object

    """
    mol = Chem.RWMol()
    atoms = np.argmax(x, axis=1)
    atoms_exist = atoms != len(atomic_num_list) - 1
    atoms = atoms[atoms_exist]
    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))
    adj = np.argmax(A, axis=0)
    adj = np.array(adj)
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adj[start, end]])
            t = adj[start, end]
            while not valid_mol_can_with_seg(mol):
                mol.RemoveBond(int(start), int(end))
                t = t - 1
                if t >= 1:
                    mol.AddBond(int(start), int(end), bond_decoder_m[t])
    return mol


def valid_mol(x):
    s = (
        Chem.MolFromSmiles(Chem.MolToSmiles(x, isomericSmiles=True))
        if x is not None
        else None
    )
    if s is not None and "." not in Chem.MolToSmiles(s, isomericSmiles=True):
        return s
    return None


def valid_mol_can_with_seg(x, largest_connected_comp=True):
    if x is None:
        return None
    sm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = Chem.MolFromSmiles(sm)
    if largest_connected_comp and "." in sm:
        vsm = [(s, len(s)) for s in sm.split(".")]
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    return mol


def check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency.
    Args:
        mol (object): rdkit mol object.
    Returns:
        True if no valency issues, False otherwise.
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall("\\d+", e_sub)))
        return False, atomid_valence


def correct_mol(x):
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = x
    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            # v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append(
                    (
                        b.GetIdx(),
                        int(b.GetBondType()),
                        b.GetBeginAtomIdx(),
                        b.GetEndAtomIdx(),
                    )
                )
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder_m[t])
    return mol


def check_tensor(x):
    return valid_mol(Tensor2Mol(*x))


def adj_to_smiles(adj, x, atomic_num_list):
    valid = [
        Chem.MolToSmiles(
            construct_mol(x_elem, adj_elem, atomic_num_list), isomericSmiles=True
        )
        for x_elem, adj_elem in zip(x, adj)
    ]
    return valid


def check_validity(
    adj,
    x,
    atomic_num_list,
    return_unique=True,
    correct_validity=True,
    largest_connected_comp=True,
    debug=False,
):
    """

    Args:
        adj (paddle.Tensor): Adjacency.
        x (paddle.Tensor): nodes.
        atomic_num_list (list): atomic list number.
        return_unique (bool): if return unique
        correct_validity (bool): if apply validity correction after the generation.
        largest_connected_comp (bool): largest connected compare.
        debug (bool): To run with more information.

    """
    adj = _to_numpy_array(adj)
    x = _to_numpy_array(x)
    if correct_validity:
        valid = []
        for x_elem, adj_elem in zip(x, adj):
            mol = construct_mol(x_elem, adj_elem, atomic_num_list)
            cmol = correct_mol(mol)
            vcmol = valid_mol_can_with_seg(
                cmol, largest_connected_comp=largest_connected_comp
            )
            valid.append(vcmol)
    else:
        valid = [
            valid_mol(construct_mol(x_elem, adj_elem, atomic_num_list))
            for x_elem, adj_elem in zip(x, adj)
        ]
    valid = [mol for mol in valid if mol is not None]
    if debug:
        logger.info("valid molecules: {}/{}".format(len(valid), tuple(adj.shape)[0]))
        for i, mol in enumerate(valid):
            logger.info(
                "[{}] {}".format(i, Chem.MolToSmiles(mol, isomericSmiles=False))
            )
    n_mols = tuple(x.shape)[0]
    valid_ratio = len(valid) / n_mols
    valid_smiles = [Chem.MolToSmiles(mol, isomericSmiles=False) for mol in valid]
    unique_smiles = list(set(valid_smiles))
    unique_ratio = 0.0
    if len(valid) > 0:
        unique_ratio = len(unique_smiles) / len(valid)
    if return_unique:
        valid_smiles = unique_smiles
    valid_mols = [Chem.MolFromSmiles(s) for s in valid_smiles]
    abs_unique_ratio = len(unique_smiles) / n_mols
    if debug:
        logger.info(
            "valid: {:.3f}%, unique: {:.3f}%, abs unique: {:.3f}%".format(
                valid_ratio * 100, unique_ratio * 100, abs_unique_ratio * 100
            )
        )
    results = dict()
    results["valid_mols"] = valid_mols
    results["valid_smiles"] = valid_smiles
    results["valid_ratio"] = valid_ratio * 100
    results["unique_ratio"] = unique_ratio * 100
    results["abs_unique_ratio"] = abs_unique_ratio * 100
    return results


def check_novelty(gen_smiles, train_smiles, n_generated_mols):
    if len(gen_smiles) == 0:
        novel_ratio = 0.0
    else:
        duplicates = [(1) for mol in gen_smiles if mol in train_smiles]
        novel = len(gen_smiles) - sum(duplicates)
        novel_ratio = novel * 100.0 / len(gen_smiles)
        abs_novel_ratio = novel * 100.0 / n_generated_mols
    print("novelty: {:.3f}%, abs novelty: {:.3f}%".format(novel_ratio, abs_novel_ratio))
    return novel_ratio, abs_novel_ratio


def _to_numpy_array(a):
    if isinstance(a, paddle.Tensor):
        a = a.cpu().detach().numpy()
    elif isinstance(a, np.ndarray):
        pass
    else:
        raise TypeError("a ({}) is not a paddle.Tensor".format(type(a)))
    return a


def save_mol_png(mol, filepath, size=(600, 600)):
    Draw.MolToFile(mol, filepath, size=size)


def readFragmentScores(name="fpscores"):
    import gzip

    global _fscores
    if name == "fpscores":
        name = os.path.join(os.path.dirname(__file__), name)
    if not os.path.exists(name):
        download._download(
            "https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/fpscores.pkl.gz",
            "./",
        )
    _fscores = cPickle.load(gzip.open("%s.pkl.gz" % name))
    outDict = {}
    for i in _fscores:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateScore(m):
    if _fscores is None:
        readFragmentScores()
    fp = rdMolDescriptors.GetMorganFingerprint(m, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.0
    nf = 0
    for bitId, v in iteritems(fps):
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1
    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.0
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)
    score2 = (
        0.0
        - sizePenalty
        - stereoPenalty
        - spiroPenalty
        - bridgePenalty
        - macrocyclePenalty
    )
    score3 = 0.0
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * 0.5
    sascore = score1 + score2 + score3
    min = -4.0
    max = 2.5
    sascore = 11.0 - (sascore - min + 1) / (max - min) * 9.0
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0
    return sascore


def penalized_logp(mol):
    """Reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset.

    Args:
        mol (object): rdkit mol object.

    Returns:
        float: Scores are normalized based on the statistics.
    """
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455
    log_p = Chem.Descriptors.MolLogP(mol)
    SA = -calculateScore(mol)
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length
    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p + normalized_SA + normalized_cycle
