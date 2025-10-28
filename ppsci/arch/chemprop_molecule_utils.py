from __future__ import annotations

import json
import os
import pickle
from tempfile import TemporaryDirectory
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from warnings import warn

import numpy as np
import paddle
from packaging import version

try:
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
except ModuleNotFoundError:
    pass
import logging
import math
from itertools import zip_longest

try:
    from tap import Tap
except ModuleNotFoundError:

    class Tap:
        def __init__(self, *args, **kwargs):
            pass


from typing_extensions import Literal


# === featuriztion start ===
def make_mol(s: str, keep_h: bool, add_h: bool):
    """
    Builds an RDKit molecule from a SMILES string.

    :param s: SMILES string.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :return: RDKit molecule.
    """
    if keep_h:
        mol = Chem.MolFromSmiles(s, sanitize=False)
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
        )
    else:
        mol = Chem.MolFromSmiles(s)
    if add_h:
        mol = Chem.AddHs(mol)
    return mol


class Featurization_parameters:
    """
    A class holding molecule featurization parameters as attributes.
    """

    def __init__(self) -> None:
        self.MAX_ATOMIC_NUM = 100
        self.ATOM_FEATURES = {
            "atomic_num": list(range(self.MAX_ATOMIC_NUM)),
            "degree": [0, 1, 2, 3, 4, 5],
            "formal_charge": [-1, -2, 1, 2, 0],
            "chiral_tag": [0, 1, 2, 3],
            "num_Hs": [0, 1, 2, 3, 4],
            "hybridization": [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ],
        }
        self.PATH_DISTANCE_BINS = list(range(10))
        self.THREE_D_DISTANCE_MAX = 20
        self.THREE_D_DISTANCE_STEP = 1
        self.THREE_D_DISTANCE_BINS = list(
            range(0, self.THREE_D_DISTANCE_MAX + 1, self.THREE_D_DISTANCE_STEP)
        )
        self.ATOM_FDIM = (
            sum(len(choices) + 1 for choices in self.ATOM_FEATURES.values()) + 2
        )
        self.EXTRA_ATOM_FDIM = 0
        self.BOND_FDIM = 14
        self.EXTRA_BOND_FDIM = 0
        self.REACTION_MODE = None
        self.EXPLICIT_H = False
        self.REACTION = False
        self.ADDING_H = False


try:
    PARAMS = Featurization_parameters()
except NameError:
    pass  # not running this example


def reset_featurization_parameters(logger: logging.Logger = None) -> None:
    """
    Function resets feature parameter values to defaults by replacing the parameters instance.
    """
    if logger is not None:
        debug = logger.debug
    else:
        debug = print
    debug("Setting molecule featurization parameters to default.")
    global PARAMS
    PARAMS = Featurization_parameters()


def get_atom_fdim(
    overwrite_default_atom: bool = False, is_reaction: bool = False
) -> int:
    """
    Gets the dimensionality of the atom feature vector.

    :param overwrite_default_atom: Whether to overwrite the default atom descriptors
    :param is_reaction: Whether to add :code:`EXTRA_ATOM_FDIM` for reaction input when :code:`REACTION_MODE` is not None
    :return: The dimensionality of the atom feature vector.
    """
    if PARAMS.REACTION_MODE:
        return (
            not overwrite_default_atom
        ) * PARAMS.ATOM_FDIM + is_reaction * PARAMS.EXTRA_ATOM_FDIM
    else:
        return (not overwrite_default_atom) * PARAMS.ATOM_FDIM + PARAMS.EXTRA_ATOM_FDIM


def set_explicit_h(explicit_h: bool) -> None:
    """
    Sets whether RDKit molecules will be constructed with explicit Hs.

    :param explicit_h: Boolean whether to keep explicit Hs from input.
    """
    PARAMS.EXPLICIT_H = explicit_h


def set_adding_hs(adding_hs: bool) -> None:
    """
    Sets whether RDKit molecules will be constructed with adding the Hs to them.

    :param adding_hs: Boolean whether to add Hs to the molecule.
    """
    PARAMS.ADDING_H = adding_hs


def set_reaction(reaction: bool, mode: str) -> None:
    """
    Sets whether to use a reaction or molecule as input and adapts feature dimensions.

    :param reaction: Boolean whether to except reactions as input.
    :param mode: Reaction mode to construct atom and bond feature vectors.

    """
    PARAMS.REACTION = reaction
    if reaction:
        PARAMS.EXTRA_ATOM_FDIM = PARAMS.ATOM_FDIM - PARAMS.MAX_ATOMIC_NUM - 1
        PARAMS.EXTRA_BOND_FDIM = PARAMS.BOND_FDIM
        PARAMS.REACTION_MODE = mode


def is_explicit_h(is_mol: bool = True) -> bool:
    """Returns whether to retain explicit Hs (for reactions only)"""
    if not is_mol:
        return PARAMS.EXPLICIT_H
    return False


def is_adding_hs(is_mol: bool = True) -> bool:
    """Returns whether to add explicit Hs to the mol (not for reactions)"""
    if is_mol:
        return PARAMS.ADDING_H
    return False


def is_reaction(is_mol: bool = True) -> bool:
    """Returns whether to use reactions as input"""
    if is_mol:
        return False
    if PARAMS.REACTION:
        return True
    return False


def reaction_mode() -> str:
    """Returns the reaction mode"""
    return PARAMS.REACTION_MODE


def set_extra_atom_fdim(extra):
    """Change the dimensionality of the atom feature vector."""
    PARAMS.EXTRA_ATOM_FDIM = extra


def get_bond_fdim(
    atom_messages: bool = False,
    overwrite_default_bond: bool = False,
    overwrite_default_atom: bool = False,
    is_reaction: bool = False,
) -> int:
    """
    Gets the dimensionality of the bond feature vector.

    :param atom_messages: Whether atom messages are being used. If atom messages are used,
                          then the bond feature vector only contains bond features.
                          Otherwise it contains both atom and bond features.
    :param overwrite_default_bond: Whether to overwrite the default bond descriptors
    :param overwrite_default_atom: Whether to overwrite the default atom descriptors
    :param is_reaction: Whether to add :code:`EXTRA_BOND_FDIM` for reaction input when :code:`REACTION_MODE:` is not None
    :return: The dimensionality of the bond feature vector.
    """
    if PARAMS.REACTION_MODE:
        return (
            (not overwrite_default_bond) * PARAMS.BOND_FDIM
            + is_reaction * PARAMS.EXTRA_BOND_FDIM
            + (not atom_messages)
            * get_atom_fdim(
                overwrite_default_atom=overwrite_default_atom, is_reaction=is_reaction
            )
        )
    else:
        return (
            (not overwrite_default_bond) * PARAMS.BOND_FDIM
            + PARAMS.EXTRA_BOND_FDIM
            + (not atom_messages)
            * get_atom_fdim(
                overwrite_default_atom=overwrite_default_atom, is_reaction=is_reaction
            )
        )


def set_extra_bond_fdim(extra):
    """Change the dimensionality of the bond feature vector."""
    PARAMS.EXTRA_BOND_FDIM = extra


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def atom_features(
    atom: Chem.rdchem.Atom, functional_groups: List[int] = None
) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features = (
            onek_encoding_unk(
                atom.GetAtomicNum() - 1, PARAMS.ATOM_FEATURES["atomic_num"]
            )
            + onek_encoding_unk(atom.GetTotalDegree(), PARAMS.ATOM_FEATURES["degree"])
            + onek_encoding_unk(
                atom.GetFormalCharge(), PARAMS.ATOM_FEATURES["formal_charge"]
            )
            + onek_encoding_unk(
                int(atom.GetChiralTag()), PARAMS.ATOM_FEATURES["chiral_tag"]
            )
            + onek_encoding_unk(
                int(atom.GetTotalNumHs()), PARAMS.ATOM_FEATURES["num_Hs"]
            )
            + onek_encoding_unk(
                int(atom.GetHybridization()), PARAMS.ATOM_FEATURES["hybridization"]
            )
            + [1 if atom.GetIsAromatic() else 0]
            + [atom.GetMass() * 0.01]
        )
        if functional_groups is not None:
            features += functional_groups
    return features


def atom_features_zeros(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom containing only the atom number information.

    :param atom: An RDKit atom.
    :return: A list containing the atom features.
    """
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features = onek_encoding_unk(
            atom.GetAtomicNum() - 1, PARAMS.ATOM_FEATURES["atomic_num"]
        ) + [0] * (PARAMS.ATOM_FDIM - PARAMS.MAX_ATOMIC_NUM - 1)
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: An RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (PARAMS.BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated() if bt is not None else 0,
            bond.IsInRing() if bt is not None else 0,
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def map_reac_to_prod(mol_reac: Chem.Mol, mol_prod: Chem.Mol):
    """
    Build a dictionary of mapping atom indices in the reactants to the products.

    :param mol_reac: An RDKit molecule of the reactants.
    :param mol_prod: An RDKit molecule of the products.
    :return: A dictionary of corresponding reactant and product atom indices.
    """
    only_prod_ids = []
    prod_map_to_id = {}
    mapnos_reac = set([atom.GetAtomMapNum() for atom in mol_reac.GetAtoms()])
    for atom in mol_prod.GetAtoms():
        mapno = atom.GetAtomMapNum()
        if mapno > 0:
            prod_map_to_id[mapno] = atom.GetIdx()
            if mapno not in mapnos_reac:
                only_prod_ids.append(atom.GetIdx())
        else:
            only_prod_ids.append(atom.GetIdx())
    only_reac_ids = []
    reac_id_to_prod_id = {}
    for atom in mol_reac.GetAtoms():
        mapno = atom.GetAtomMapNum()
        if mapno > 0:
            try:
                reac_id_to_prod_id[atom.GetIdx()] = prod_map_to_id[mapno]
            except KeyError:
                only_reac_ids.append(atom.GetIdx())
        else:
            only_reac_ids.append(atom.GetIdx())
    return reac_id_to_prod_id, only_prod_ids, only_reac_ids


class MolGraph:
    """
    A :class:`MolGraph` represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:

    * :code:`n_atoms`: The number of atoms in the molecule.
    * :code:`n_bonds`: The number of bonds in the molecule.
    * :code:`f_atoms`: A mapping from an atom index to a list of atom features.
    * :code:`f_bonds`: A mapping from a bond index to a list of bond features.
    * :code:`a2b`: A mapping from an atom index to a list of incoming bond indices.
    * :code:`b2a`: A mapping from a bond index to the index of the atom the bond originates from.
    * :code:`b2revb`: A mapping from a bond index to the index of the reverse bond.
    * :code:`overwrite_default_atom_features`: A boolean to overwrite default atom descriptors.
    * :code:`overwrite_default_bond_features`: A boolean to overwrite default bond descriptors.
    * :code:`is_mol`: A boolean whether the input is a molecule.
    * :code:`is_reaction`: A boolean whether the molecule is a reaction.
    * :code:`is_explicit_h`: A boolean whether to retain explicit Hs (for reaction mode)
    * :code:`is_adding_hs`: A boolean whether to add explicit Hs (not for reaction mode)
    * :code:`reaction_mode`:  Reaction mode to construct atom and bond feature vectors
    """

    def __init__(
        self,
        mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]],
        atom_features_extra: np.ndarray = None,
        bond_features_extra: np.ndarray = None,
        overwrite_default_atom_features: bool = False,
        overwrite_default_bond_features: bool = False,
    ):
        """
        :param mol: A SMILES or an RDKit molecule.
        :param atom_features_extra: A list of 2D numpy array containing additional atom features to featurize the molecule
        :param bond_features_extra: A list of 2D numpy array containing additional bond features to featurize the molecule
        :param overwrite_default_atom_features: Boolean to overwrite default atom features by atom_features instead of concatenating
        :param overwrite_default_bond_features: Boolean to overwrite default bond features by bond_features instead of concatenating
        """
        self.is_mol = is_mol(mol)
        self.is_reaction = is_reaction(self.is_mol)
        self.is_explicit_h = is_explicit_h(self.is_mol)
        self.is_adding_hs = is_adding_hs(self.is_mol)
        self.reaction_mode = reaction_mode()
        if type(mol) == str:
            if self.is_reaction:
                mol = make_mol(
                    mol.split(">")[0], self.is_explicit_h, self.is_adding_hs
                ), make_mol(mol.split(">")[-1], self.is_explicit_h, self.is_adding_hs)
            else:
                mol = make_mol(mol, self.is_explicit_h, self.is_adding_hs)
        self.n_atoms = 0
        self.n_bonds = 0
        self.f_atoms = []
        self.f_bonds = []
        self.a2b = []
        self.b2a = []
        self.b2revb = []
        self.overwrite_default_atom_features = overwrite_default_atom_features
        self.overwrite_default_bond_features = overwrite_default_bond_features
        if not self.is_reaction:
            self.f_atoms = [atom_features(atom) for atom in mol.GetAtoms()]
            if atom_features_extra is not None:
                if overwrite_default_atom_features:
                    self.f_atoms = [descs.tolist() for descs in atom_features_extra]
                else:
                    self.f_atoms = [
                        (f_atoms + descs.tolist())
                        for f_atoms, descs in zip(self.f_atoms, atom_features_extra)
                    ]
            self.n_atoms = len(self.f_atoms)
            if (
                atom_features_extra is not None
                and len(atom_features_extra) != self.n_atoms
            ):
                raise ValueError(
                    f"The number of atoms in {Chem.MolToSmiles(mol)} is different from the length of the extra atom features"
                )
            for _ in range(self.n_atoms):
                self.a2b.append([])
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)
                    if bond is None:
                        continue
                    f_bond = bond_features(bond)
                    if bond_features_extra is not None:
                        descr = bond_features_extra[bond.GetIdx()].tolist()
                        if overwrite_default_bond_features:
                            f_bond = descr
                        else:
                            f_bond += descr
                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.n_bonds += 2
            if (
                bond_features_extra is not None
                and len(bond_features_extra) != self.n_bonds / 2
            ):
                raise ValueError(
                    f"The number of bonds in {Chem.MolToSmiles(mol)} is different from the length of the extra bond features"
                )
        else:
            if atom_features_extra is not None:
                raise NotImplementedError(
                    "Extra atom features are currently not supported for reactions"
                )
            if bond_features_extra is not None:
                raise NotImplementedError(
                    "Extra bond features are currently not supported for reactions"
                )
            mol_reac = mol[0]
            mol_prod = mol[1]
            ri2pi, pio, rio = map_reac_to_prod(mol_reac, mol_prod)
            if self.reaction_mode in ["reac_diff", "prod_diff", "reac_prod"]:
                f_atoms_reac = [atom_features(atom) for atom in mol_reac.GetAtoms()] + [
                    atom_features_zeros(mol_prod.GetAtomWithIdx(index)) for index in pio
                ]
                f_atoms_prod = [
                    (
                        atom_features(mol_prod.GetAtomWithIdx(ri2pi[atom.GetIdx()]))
                        if atom.GetIdx() not in rio
                        else atom_features_zeros(atom)
                    )
                    for atom in mol_reac.GetAtoms()
                ] + [atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio]
            else:
                f_atoms_reac = [atom_features(atom) for atom in mol_reac.GetAtoms()] + [
                    atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio
                ]
                f_atoms_prod = [
                    (
                        atom_features(mol_prod.GetAtomWithIdx(ri2pi[atom.GetIdx()]))
                        if atom.GetIdx() not in rio
                        else atom_features(atom)
                    )
                    for atom in mol_reac.GetAtoms()
                ] + [atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio]
            if self.reaction_mode in [
                "reac_diff",
                "prod_diff",
                "reac_diff_balance",
                "prod_diff_balance",
            ]:
                f_atoms_diff = [
                    list(map(lambda x, y: x - y, ii, jj))
                    for ii, jj in zip(f_atoms_prod, f_atoms_reac)
                ]
            if self.reaction_mode in ["reac_prod", "reac_prod_balance"]:
                self.f_atoms = [
                    (x + y[PARAMS.MAX_ATOMIC_NUM + 1 :])
                    for x, y in zip(f_atoms_reac, f_atoms_prod)
                ]
            elif self.reaction_mode in ["reac_diff", "reac_diff_balance"]:
                self.f_atoms = [
                    (x + y[PARAMS.MAX_ATOMIC_NUM + 1 :])
                    for x, y in zip(f_atoms_reac, f_atoms_diff)
                ]
            elif self.reaction_mode in ["prod_diff", "prod_diff_balance"]:
                self.f_atoms = [
                    (x + y[PARAMS.MAX_ATOMIC_NUM + 1 :])
                    for x, y in zip(f_atoms_prod, f_atoms_diff)
                ]
            self.n_atoms = len(self.f_atoms)
            n_atoms_reac = mol_reac.GetNumAtoms()
            for _ in range(self.n_atoms):
                self.a2b.append([])
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    if a1 >= n_atoms_reac and a2 >= n_atoms_reac:
                        bond_prod = mol_prod.GetBondBetweenAtoms(
                            pio[a1 - n_atoms_reac], pio[a2 - n_atoms_reac]
                        )
                        if self.reaction_mode in [
                            "reac_prod_balance",
                            "reac_diff_balance",
                            "prod_diff_balance",
                        ]:
                            bond_reac = bond_prod
                        else:
                            bond_reac = None
                    elif a1 < n_atoms_reac and a2 >= n_atoms_reac:
                        bond_reac = None
                        if a1 in ri2pi.keys():
                            bond_prod = mol_prod.GetBondBetweenAtoms(
                                ri2pi[a1], pio[a2 - n_atoms_reac]
                            )
                        else:
                            bond_prod = None
                    else:
                        bond_reac = mol_reac.GetBondBetweenAtoms(a1, a2)
                        if a1 in ri2pi.keys() and a2 in ri2pi.keys():
                            bond_prod = mol_prod.GetBondBetweenAtoms(
                                ri2pi[a1], ri2pi[a2]
                            )
                        elif self.reaction_mode in [
                            "reac_prod_balance",
                            "reac_diff_balance",
                            "prod_diff_balance",
                        ]:
                            if a1 in ri2pi.keys() or a2 in ri2pi.keys():
                                bond_prod = None
                            else:
                                bond_prod = bond_reac
                        else:
                            bond_prod = None
                    if bond_reac is None and bond_prod is None:
                        continue
                    f_bond_reac = bond_features(bond_reac)
                    f_bond_prod = bond_features(bond_prod)
                    if self.reaction_mode in [
                        "reac_diff",
                        "prod_diff",
                        "reac_diff_balance",
                        "prod_diff_balance",
                    ]:
                        f_bond_diff = [
                            (y - x) for x, y in zip(f_bond_reac, f_bond_prod)
                        ]
                    if self.reaction_mode in ["reac_prod", "reac_prod_balance"]:
                        f_bond = f_bond_reac + f_bond_prod
                    elif self.reaction_mode in ["reac_diff", "reac_diff_balance"]:
                        f_bond = f_bond_reac + f_bond_diff
                    elif self.reaction_mode in ["prod_diff", "prod_diff_balance"]:
                        f_bond = f_bond_prod + f_bond_diff
                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.n_bonds += 2


class BatchMolGraph:
    """
    A :class:`BatchMolGraph` represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a :class:`MolGraph` plus:

    * :code:`atom_fdim`: The dimensionality of the atom feature vector.
    * :code:`bond_fdim`: The dimensionality of the bond feature vector (technically the combined atom/bond features).
    * :code:`a_scope`: A list of tuples indicating the start and end atom indices for each molecule.
    * :code:`b_scope`: A list of tuples indicating the start and end bond indices for each molecule.
    * :code:`max_num_bonds`: The maximum number of bonds neighboring an atom in this batch.
    * :code:`b2b`: (Optional) A mapping from a bond index to incoming bond indices.
    * :code:`a2a`: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph]):
        """
        :param mol_graphs: A list of :class:`MolGraph`\\ s from which to construct the :class:`BatchMolGraph`.
        """
        self.overwrite_default_atom_features = mol_graphs[
            0
        ].overwrite_default_atom_features
        self.overwrite_default_bond_features = mol_graphs[
            0
        ].overwrite_default_bond_features
        self.is_reaction = mol_graphs[0].is_reaction
        self.atom_fdim = get_atom_fdim(
            overwrite_default_atom=self.overwrite_default_atom_features,
            is_reaction=self.is_reaction,
        )
        self.bond_fdim = get_bond_fdim(
            overwrite_default_bond=self.overwrite_default_bond_features,
            overwrite_default_atom=self.overwrite_default_atom_features,
            is_reaction=self.is_reaction,
        )
        self.n_atoms = 1
        self.n_bonds = 1
        self.a_scope = []
        self.b_scope = []
        f_atoms = [[0] * self.atom_fdim]
        f_bonds = [[0] * self.bond_fdim]
        a2b = [[]]
        b2a = [0]
        b2revb = [0]
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)
            for a in range(mol_graph.n_atoms):
                a2b.append([(b + self.n_bonds) for b in mol_graph.a2b[a]])
            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])
            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds
        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))
        self.f_atoms = paddle.to_tensor(data=f_atoms, dtype="float32")
        self.f_bonds = paddle.to_tensor(data=f_bonds, dtype="float32")
        self.a2b = paddle.to_tensor(
            data=[
                (a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])))
                for a in range(self.n_atoms)
            ],
            dtype="int64",
        )
        self.b2a = paddle.to_tensor(data=b2a, dtype="int64")
        self.b2revb = paddle.to_tensor(data=b2revb, dtype="int64")
        self.b2b = None
        self.a2a = None

    def get_components(
        self, atom_messages: bool = False
    ) -> Tuple[
        paddle.Tensor,
        paddle.Tensor,
        paddle.Tensor,
        paddle.Tensor,
        paddle.Tensor,
        List[Tuple[int, int]],
        List[Tuple[int, int]],
    ]:
        """
        Returns the components of the :class:`BatchMolGraph`.

        The returned components are, in order:

        * :code:`f_atoms`
        * :code:`f_bonds`
        * :code:`a2b`
        * :code:`b2a`
        * :code:`b2revb`
        * :code:`a_scope`
        * :code:`b_scope`

        :param atom_messages: Whether to use atom messages instead of bond messages. This changes the bond feature
                              vector to contain only bond features rather than both atom and bond features.
        :return: A tuple containing tensors with the atom features, bond features, graph structure,
                 and scope of the atoms and bonds (i.e., the indices of the molecules they belong to).
        """
        if atom_messages:
            f_bonds = self.f_bonds[
                :,
                -get_bond_fdim(
                    atom_messages=atom_messages,
                    overwrite_default_atom=self.overwrite_default_atom_features,
                    overwrite_default_bond=self.overwrite_default_bond_features,
                ) :,
            ]
        else:
            f_bonds = self.f_bonds
        return (
            self.f_atoms,
            f_bonds,
            self.a2b,
            self.b2a,
            self.b2revb,
            self.a_scope,
            self.b_scope,
        )

    def get_b2b(self) -> paddle.int64:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A tensor containing the mapping from each bond index to all the incoming bond indices.
        """
        if self.b2b is None:
            b2b = self.a2b[self.b2a]
            revmask = (
                b2b
                != self.b2revb.unsqueeze(axis=1).tile(repeat_times=[1, b2b.shape[1]])
            ).astype(dtype="int64")
            self.b2b = b2b * revmask
        return self.b2b

    def get_a2a(self) -> paddle.int64:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A tensor containing the mapping from each atom index to all the neighboring atom indices.
        """
        if self.a2a is None:
            self.a2a = self.b2a[self.a2b]
        return self.a2a


def mol2graph(
    mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]],
    atom_features_batch: List[np.array] = (None,),
    bond_features_batch: List[np.array] = (None,),
    overwrite_default_atom_features: bool = False,
    overwrite_default_bond_features: bool = False,
) -> BatchMolGraph:
    """
    Converts a list of SMILES or RDKit molecules to a :class:`BatchMolGraph` containing the batch of molecular graphs.

    :param mols: A list of SMILES or a list of RDKit molecules.
    :param atom_features_batch: A list of 2D numpy array containing additional atom features to featurize the molecule
    :param bond_features_batch: A list of 2D numpy array containing additional bond features to featurize the molecule
    :param overwrite_default_atom_features: Boolean to overwrite default atom descriptors by atom_descriptors instead of concatenating
    :param overwrite_default_bond_features: Boolean to overwrite default bond descriptors by bond_descriptors instead of concatenating
    :return: A :class:`BatchMolGraph` containing the combined molecular graph for the molecules.
    """
    return BatchMolGraph(
        [
            MolGraph(
                mol,
                af,
                bf,
                overwrite_default_atom_features=overwrite_default_atom_features,
                overwrite_default_bond_features=overwrite_default_bond_features,
            )
            for mol, af, bf in zip_longest(
                mols, atom_features_batch, bond_features_batch
            )
        ]
    )


def is_mol(mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]) -> bool:
    """Checks whether an input is a molecule or a reaction

    :param mol: str, RDKIT molecule or tuple of molecules
    :return: Whether the supplied input corresponds to a single molecule
    """
    if isinstance(mol, str) and ">" not in mol:
        return True
    elif isinstance(mol, Chem.Mol):
        return True
    return False


# === featuriztion end   ===

# === nn_util start ===
def compute_pnorm(model: paddle.nn.Layer) -> float:
    """
    Computes the norm of the parameters of a model.

    :param model: A model.
    :return: The norm of the parameters of the model.
    """
    return math.sqrt(sum([(p.norm().item() ** 2) for p in model.parameters()]))


def compute_gnorm(model: paddle.nn.Layer) -> float:
    """
    Computes the norm of the gradients of a model.

    :param model: A model.
    :return: The norm of the gradients of the model.
    """
    return math.sqrt(
        sum(
            [
                (p.grad.norm().item() ** 2)
                for p in model.parameters()
                if p.grad is not None
            ]
        )
    )


def param_count(model: paddle.nn.Layer) -> int:
    """
    Determines number of trainable parameters.

    :param model: A model.
    :return: The number of trainable parameters in the model.
    """
    return sum(param.size for param in model.parameters() if not param.stop_gradient)


def param_count_all(model: paddle.nn.Layer) -> int:
    """
    Determines number of trainable parameters.

    :param model: A model.
    :return: The number of trainable parameters in the model.
    """
    return sum(param.size for param in model.parameters())


def index_select_ND(source: paddle.Tensor, index: paddle.Tensor) -> paddle.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in :code:`index`.

    :param source: A tensor of shape :code:`(num_bonds, hidden_size)` containing message features.
    :param index: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds)` containing the atom or bond
                  indices to select from :code:`source`.
    :return: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds, hidden_size)` containing the message
             features corresponding to the atoms/bonds specified in index.
    """
    index_size = tuple(index.shape)
    suffix_dim = tuple(source.shape)[1:]
    final_size = index_size + suffix_dim
    # print("index", index)
    target = source.index_select(axis=0, index=index.reshape(-1))
    target = target.reshape(final_size)
    return target


def get_activation_function(activation: str) -> paddle.nn.Layer:
    """
    Gets an activation function module given the name of the activation.

    Supports:

    * :code:`ReLU`
    * :code:`LeakyReLU`
    * :code:`PReLU`
    * :code:`tanh`
    * :code:`SELU`
    * :code:`ELU`

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == "ReLU":
        return paddle.nn.ReLU()
    elif activation == "LeakyReLU":
        return paddle.nn.LeakyReLU(negative_slope=0.1)
    elif activation == "PReLU":
        return paddle.nn.PReLU()
    elif activation == "tanh":
        return paddle.nn.Tanh()
    elif activation == "SELU":
        return paddle.nn.SELU()
    elif activation == "ELU":
        return paddle.nn.ELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')


def initialize_weights(model: paddle.nn.Layer) -> None:
    """
    Initializes the weights of a model in place.

    :param model: A model.
    """
    for param in model.parameters():
        if param.dim() == 1:
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(param)
        else:
            init_XavierNormal = paddle.nn.initializer.XavierNormal()
            init_XavierNormal(param)


class NoamLR(paddle.optimizer.lr.LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where :code:`warmup_steps = warmup_epochs * steps_per_epoch`).
    Then the learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr` over the
    course of the remaining :code:`total_steps - warmup_steps` (where :code:`total_steps =
    total_epochs * steps_per_epoch`). This is roughly based on the learning rate
    schedule from `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_, section 5.3.
    """

    def __init__(
        self,
        optimizer: paddle.optimizer.Optimizer,
        warmup_epochs: List[Union[float, int]],
        total_epochs: List[int],
        steps_per_epoch: int,
        init_lr: List[float],
        max_lr: List[float],
        final_lr: List[float],
    ):
        """
        :param optimizer: A optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after :code:`warmup_epochs`).
        :param final_lr: The final learning rate (achieved after :code:`total_epochs`).
        """
        if (
            not len(optimizer._param_groups)
            == len(warmup_epochs)
            == len(total_epochs)
            == len(init_lr)
            == len(max_lr)
            == len(final_lr)
        ):
            raise ValueError(
                f"Number of param groups must match the number of epochs and learning rates! got: len(optimizer.param_groups)= {len(optimizer._param_groups)}, len(warmup_epochs)= {len(warmup_epochs)}, len(total_epochs)= {len(total_epochs)}, len(init_lr)= {len(init_lr)}, len(max_lr)= {len(max_lr)}, len(final_lr)= {len(final_lr)}"
            )
        self.num_lrs = len(optimizer._param_groups)
        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)
        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps
        self.exponential_gamma = (self.final_lr / self.max_lr) ** (
            1 / (self.total_steps - self.warmup_steps)
        )
        super(NoamLR, self).__init__(optimizer.get_lr())

    def get_lr(self) -> List[float]:
        """
        Gets a list of the current learning rates.

        :return: A list of the current learning rates.
        """
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
                             If None, :code:`current_step = self.current_step + 1`.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1
        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = (
                    self.init_lr[i] + self.current_step * self.linear_increment[i]
                )
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * self.exponential_gamma[i] ** (
                    self.current_step - self.warmup_steps[i]
                )
            else:
                self.lr[i] = self.final_lr[i]
            self.optimizer._param_groups[i]["learning_rate"] = self.lr[i]


def activate_dropout(module: paddle.nn.Layer, dropout_prob: float):
    """
    Set p of dropout layers and set to train mode during inference for uncertainty estimation.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param dropout_prob: A float on (0,1) indicating the dropout probability.
    """
    if isinstance(module, paddle.nn.Dropout):
        module.p = dropout_prob
        module.train()


# === nn_util end   ===

# === features_generators start ===
Molecule = Union[str, "Chem.Mol"]
FeaturesGenerator = Callable[[Molecule], np.ndarray]
FEATURES_GENERATOR_REGISTRY = {}


def register_features_generator(
    features_generator_name: str,
) -> Callable[[FeaturesGenerator], FeaturesGenerator]:
    """
    Creates a decorator which registers a features generator in a global dictionary to enable access by name.

    :param features_generator_name: The name to use to access the features generator.
    :return: A decorator which will add a features generator to the registry using the specified name.
    """

    def decorator(features_generator: FeaturesGenerator) -> FeaturesGenerator:
        FEATURES_GENERATOR_REGISTRY[features_generator_name] = features_generator
        return features_generator

    return decorator


def get_features_generator(features_generator_name: str) -> FeaturesGenerator:
    """
    Gets a registered features generator by name.

    :param features_generator_name: The name of the features generator.
    :return: The desired features generator.
    """
    if features_generator_name not in FEATURES_GENERATOR_REGISTRY:
        raise ValueError(
            f'Features generator "{features_generator_name}" could not be found. If this generator relies on rdkit features, you may need to install descriptastorus.'
        )
    return FEATURES_GENERATOR_REGISTRY[features_generator_name]


def get_available_features_generators() -> List[str]:
    """Returns a list of names of available features generators."""
    return list(FEATURES_GENERATOR_REGISTRY.keys())


MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048


@register_features_generator("morgan")
def morgan_binary_features_generator(
    mol: Molecule, radius: int = MORGAN_RADIUS, num_bits: int = MORGAN_NUM_BITS
) -> np.ndarray:
    """
    Generates a binary Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the binary Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    return features


@register_features_generator("morgan_count")
def morgan_counts_features_generator(
    mol: Molecule, radius: int = MORGAN_RADIUS, num_bits: int = MORGAN_NUM_BITS
) -> np.ndarray:
    """
    Generates a counts-based Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the counts-based Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    return features


try:
    from descriptastorus.descriptors import rdDescriptors
    from descriptastorus.descriptors import rdNormalizedDescriptors

    @register_features_generator("rdkit_2d")
    def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
        """
        Generates RDKit 2D features for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D features.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdDescriptors.RDKit2D()
        features = generator.process(smiles)[1:]
        return features

    @register_features_generator("rdkit_2d_normalized")
    def rdkit_2d_normalized_features_generator(mol: Molecule) -> np.ndarray:
        """
        Generates RDKit 2D normalized features for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D normalized features.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]
        return features

except ImportError:

    @register_features_generator("rdkit_2d")
    def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
        """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
        raise ImportError(
            "Failed to import descriptastorus. Please install descriptastorus (https://github.com/bp-kelley/descriptastorus) to use RDKit 2D features."
        )

    @register_features_generator("rdkit_2d_normalized")
    def rdkit_2d_normalized_features_generator(mol: Molecule) -> np.ndarray:
        """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
        raise ImportError(
            "Failed to import descriptastorus. Please install descriptastorus (https://github.com/bp-kelley/descriptastorus) to use RDKit 2D normalized features."
        )


"""
Custom features generator template.

Note: The name you use to register the features generator is the name
you will specify on the command line when using the --features_generator <name> flag.
Ex. python train.py ... --features_generator custom ...

@register_features_generator('custom')
def custom_features_generator(mol: Molecule) -> np.ndarray:
    # If you want to use the SMILES string
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol

    # If you want to use the RDKit molecule
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    # Replace this with code which generates features from the molecule
    features = np.array([0, 0, 1])

    return features
"""

# === features_generators end ===

# === args start ===
Metric = Literal[
    "auc",
    "prc-auc",
    "rmse",
    "mae",
    "mse",
    "r2",
    "accuracy",
    "cross_entropy",
    "binary_cross_entropy",
    "sid",
    "wasserstein",
    "f1",
    "mcc",
    "bounded_rmse",
    "bounded_mae",
    "bounded_mse",
]

SMILES_TO_GRAPH: Dict[str, MolGraph] = {}
CACHE_MOL = True
SMILES_TO_MOL: Dict[str, Union["Chem.Mol", Tuple["Chem.Mol", "Chem.Mol"]]] = {}


def empty_cache():
    """Empties the cache of :class:`~chemprop.features.MolGraph` and RDKit molecules."""
    SMILES_TO_GRAPH.clear()
    SMILES_TO_MOL.clear()


def set_cache_mol(cache_mol: bool) -> None:
    """Sets whether RDKit molecules will be cached."""
    global CACHE_MOL
    CACHE_MOL = cache_mol


def preprocess_smiles_columns(
    path: str,
    smiles_columns: Union[str, List[str]] = None,
    number_of_molecules: int = 1,
) -> List[str]:
    """
    Preprocesses the :code:`smiles_columns` variable to ensure that it is a list of column
    headings corresponding to the columns in the data file holding SMILES. Assumes file has a header.

    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param number_of_molecules: The number of molecules with associated SMILES for each
                           data point.
    :return: The preprocessed version of :code:`smiles_columns` which is guaranteed to be a list.
    """
    if smiles_columns is None:
        if os.path.isfile(path):
            columns = None  # get_header(path)
            smiles_columns = columns[:number_of_molecules]
        else:
            smiles_columns = [None] * number_of_molecules
    else:
        if not isinstance(smiles_columns, list):
            smiles_columns = [smiles_columns]
        if os.path.isfile(path):
            columns = None  # get_header(path)
            if len(smiles_columns) != number_of_molecules:
                raise ValueError(
                    "Length of smiles_columns must match number_of_molecules."
                )
            if any([(smiles not in columns) for smiles in smiles_columns]):
                raise ValueError(
                    "Provided smiles_columns do not match the header of data file."
                )
    return smiles_columns


def get_checkpoint_paths(
    checkpoint_path: Optional[str] = None,
    checkpoint_paths: Optional[List[str]] = None,
    checkpoint_dir: Optional[str] = None,
    ext: str = ".pt",
) -> Optional[List[str]]:
    """
    Gets a list of checkpoint paths either from a single checkpoint path or from a directory of checkpoints.

    If :code:`checkpoint_path` is provided, only collects that one checkpoint.
    If :code:`checkpoint_paths` is provided, collects all of the provided checkpoints.
    If :code:`checkpoint_dir` is provided, walks the directory and collects all checkpoints.
    A checkpoint is any file ending in the extension ext.

    :param checkpoint_path: Path to a checkpoint.
    :param checkpoint_paths: List of paths to checkpoints.
    :param checkpoint_dir: Path to a directory containing checkpoints.
    :param ext: The extension which defines a checkpoint file.
    :return: A list of paths to checkpoints or None if no checkpoint path(s)/dir are provided.
    """
    if (
        sum(
            var is not None
            for var in [checkpoint_dir, checkpoint_path, checkpoint_paths]
        )
        > 1
    ):
        raise ValueError(
            "Can only specify one of checkpoint_dir, checkpoint_path, and checkpoint_paths"
        )
    if checkpoint_path is not None:
        return [checkpoint_path]
    if checkpoint_paths is not None:
        return checkpoint_paths
    if checkpoint_dir is not None:
        checkpoint_paths = []
        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if fname.endswith(ext):
                    checkpoint_paths.append(os.path.join(root, fname))
        if len(checkpoint_paths) == 0:
            raise ValueError(
                f'Failed to find any checkpoints with extension "{ext}" in directory "{checkpoint_dir}"'
            )
        return checkpoint_paths
    return None


class CommonArgs(Tap):
    """:class:`CommonArgs` contains arguments that are used in both :class:`TrainArgs` and :class:`PredictArgs`."""

    smiles_columns: List[str] = None
    """List of names of the columns containing SMILES strings.
    By default, uses the first :code:`number_of_molecules` columns."""
    number_of_molecules: int = 1
    """Number of molecules in each input to the model.
    This must equal the length of :code:`smiles_columns` (if not :code:`None`)."""
    checkpoint_dir: str = None
    """Directory from which to load model checkpoints (walks directory and ensembles all models that are found)."""
    checkpoint_path: str = None
    """Path to model checkpoint (:code:`.pt` file)."""
    checkpoint_paths: List[str] = None
    """List of paths to model checkpoints (:code:`.pt` files)."""
    no_cuda: bool = False
    """Turn off cuda (i.e., use CPU instead of GPU)."""
    gpu: int = None
    """Which GPU to use."""
    features_generator: List[str] = None
    """Method(s) of generating additional features."""
    features_path: List[str] = None
    """Path(s) to features to use in FNN (instead of features_generator)."""
    phase_features_path: str = None
    """Path to features used to indicate the phase of the data in one-hot vector form. Used in spectra datatype."""
    no_features_scaling: bool = False
    """Turn off scaling of features."""
    max_data_size: int = None
    """Maximum number of data points to load."""
    num_workers: int = 8
    """Number of workers for the parallel data loading (0 means sequential)."""
    batch_size: int = 50
    """Batch size."""
    atom_descriptors: Literal["feature", "descriptor"] = None
    """
    Custom extra atom descriptors.
    :code:`feature`: used as atom features to featurize a given molecule.
    :code:`descriptor`: used as descriptor and concatenated to the machine learned atomic representation.
    """
    atom_descriptors_path: str = None
    """Path to the extra atom descriptors."""
    bond_features_path: str = None
    """Path to the extra bond descriptors that will be used as bond features to featurize a given molecule."""
    no_cache_mol: bool = False
    """
    Whether to not cache the RDKit molecule for each SMILES string to reduce memory usage (cached by default).
    """
    empty_cache: bool = False
    """
    Whether to empty all caches before training or predicting. This is necessary if multiple jobs are run within a single script and the atom or bond features change.
    """

    def __init__(self, *args, **kwargs):
        super(CommonArgs, self).__init__(*args, **kwargs)
        self._atom_features_size = 0
        self._bond_features_size = 0
        self._atom_descriptors_size = 0

    @property
    def device(self) -> (paddle.CPUPlace, paddle.CUDAPlace, str):
        """The :code:`paddle.Place` on which to load and process data and models."""
        if not self.cuda:
            # return paddle.CPUPlace()
            return "cpu"
        # return paddle.CUDAPlace(self.gpu)
        return "gpu:0"

    @device.setter
    def device(self, device: (paddle.CPUPlace, paddle.CUDAPlace, str)) -> None:
        self.cuda = device != "cpu"
        self.gpu = 0

    @property
    def cuda(self) -> bool:
        """Whether to use CUDA (i.e., GPUs) or not."""
        return not self.no_cuda and paddle.device.cuda.device_count() >= 1

    @cuda.setter
    def cuda(self, cuda: bool) -> None:
        self.no_cuda = not cuda

    @property
    def features_scaling(self) -> bool:
        """
        Whether to apply normalization with a :class:`~chemprop.data.scaler.StandardScaler`
        to the additional molecule-level features.
        """
        return not self.no_features_scaling

    @features_scaling.setter
    def features_scaling(self, features_scaling: bool) -> None:
        self.no_features_scaling = not features_scaling

    @property
    def atom_features_size(self) -> int:
        """The size of the atom features."""
        return self._atom_features_size

    @atom_features_size.setter
    def atom_features_size(self, atom_features_size: int) -> None:
        self._atom_features_size = atom_features_size

    @property
    def atom_descriptors_size(self) -> int:
        """The size of the atom descriptors."""
        return self._atom_descriptors_size

    @atom_descriptors_size.setter
    def atom_descriptors_size(self, atom_descriptors_size: int) -> None:
        self._atom_descriptors_size = atom_descriptors_size

    @property
    def bond_features_size(self) -> int:
        """The size of the atom features."""
        return self._bond_features_size

    @bond_features_size.setter
    def bond_features_size(self, bond_features_size: int) -> None:
        self._bond_features_size = bond_features_size

    def configure(self) -> None:
        self.add_argument(
            "--gpu", choices=list(range(paddle.device.cuda.device_count()))
        )
        self.add_argument(
            "--features_generator", choices=get_available_features_generators()
        )

    def process_args(self) -> None:
        self.checkpoint_paths = get_checkpoint_paths(
            checkpoint_path=self.checkpoint_path,
            checkpoint_paths=self.checkpoint_paths,
            checkpoint_dir=self.checkpoint_dir,
        )
        if (
            self.features_generator is not None
            and "rdkit_2d_normalized" in self.features_generator
            and self.features_scaling
        ):
            raise ValueError(
                "When using rdkit_2d_normalized features, --no_features_scaling must be specified."
            )
        if (self.atom_descriptors is None) != (self.atom_descriptors_path is None):
            raise ValueError(
                "If atom_descriptors is specified, then an atom_descriptors_path must be provided and vice versa."
            )
        if self.atom_descriptors is not None and self.number_of_molecules > 1:
            raise NotImplementedError(
                "Atom descriptors are currently only supported with one molecule per input (i.e., number_of_molecules = 1)."
            )
        if self.bond_features_path is not None and self.number_of_molecules > 1:
            raise NotImplementedError(
                "Bond descriptors are currently only supported with one molecule per input (i.e., number_of_molecules = 1)."
            )
        set_cache_mol(not self.no_cache_mol)
        if self.empty_cache:
            empty_cache()


class TrainArgs(CommonArgs):
    """:class:`TrainArgs` includes :class:`CommonArgs` along with additional arguments used for training a Chemprop model."""

    data_path: str
    """Path to data CSV file."""
    target_columns: List[str] = None
    """
    Name of the columns containing target values.
    By default, uses all columns except the SMILES column and the :code:`ignore_columns`.
    """
    ignore_columns: List[str] = None
    """Name of the columns to ignore when :code:`target_columns` is not provided."""
    dataset_type: Literal["regression", "classification", "multiclass", "spectra"]
    """Type of dataset. This determines the default loss function used during training."""
    loss_function: Literal[
        "mse",
        "bounded_mse",
        "binary_cross_entropy",
        "cross_entropy",
        "mcc",
        "sid",
        "wasserstein",
        "mve",
        "evidential",
        "dirichlet",
    ] = None
    """Choice of loss function. Loss functions are limited to compatible dataset types."""
    multiclass_num_classes: int = 3
    """Number of classes when running multiclass classification."""
    separate_val_path: str = None
    """Path to separate val set, optional."""
    separate_test_path: str = None
    """Path to separate test set, optional."""
    spectra_phase_mask_path: str = None
    """Path to a file containing a phase mask array, used for excluding particular regions in spectra predictions."""
    data_weights_path: str = None
    """Path to weights for each molecule in the training data, affecting the relative weight of molecules in the loss function"""
    target_weights: List[float] = None
    """Weights associated with each target, affecting the relative weight of targets in the loss function. Must match the number of target columns."""
    split_type: Literal[
        "random",
        "scaffold_balanced",
        "predetermined",
        "crossval",
        "cv",
        "cv-no-test",
        "index_predetermined",
        "random_with_repeated_smiles",
    ] = "random"
    """Method of splitting the data into train/val/test."""
    split_sizes: List[float] = None
    """Split proportions for train/validation/test sets."""
    split_key_molecule: int = 0
    """The index of the key molecule used for splitting when multiple molecules are present and constrained split_type is used, like scaffold_balanced or random_with_repeated_smiles.
       Note that this index begins with zero for the first molecule."""
    num_folds: int = 1
    """Number of folds when performing cross validation."""
    folds_file: str = None
    """Optional file of fold labels."""
    val_fold_index: int = None
    """Which fold to use as val for leave-one-out cross val."""
    test_fold_index: int = None
    """Which fold to use as test for leave-one-out cross val."""
    crossval_index_dir: str = None
    """Directory in which to find cross validation index files."""
    crossval_index_file: str = None
    """Indices of files to use as train/val/test. Overrides :code:`--num_folds` and :code:`--seed`."""
    seed: int = 0
    """
    Random seed to use when splitting data into train/val/test sets.
    When :code`num_folds > 1`, the first fold uses this seed and all subsequent folds add 1 to the seed.
    """
    paddle_seed: int = 0
    """Seed for Paddle randomness (e.g., random initial weights)."""
    metric: Metric = None
    """
    Metric to use during evaluation. It is also used with the validation set for early stopping.
    Defaults to "auc" for classification, "rmse" for regression, and "sid" for spectra.
    """
    extra_metrics: List[Metric] = []
    """Additional metrics to use to evaluate the model. Not used for early stopping."""
    save_dir: str = None
    """Directory where model checkpoints will be saved."""
    checkpoint_frzn: str = None
    """Path to model checkpoint file to be loaded for overwriting and freezing weights."""
    save_smiles_splits: bool = False
    """Save smiles for each train/val/test splits for prediction convenience later."""
    test: bool = False
    """Whether to skip training and only test the model."""
    quiet: bool = False
    """Skip non-essential print statements."""
    log_frequency: int = 10
    """The number of batches between each logging of the training loss."""
    show_individual_scores: bool = False
    """Show all scores for individual targets, not just average, at the end."""
    cache_cutoff: float = 10000
    """
    Maximum number of molecules in dataset to allow caching.
    Below this number, caching is used and data loading is sequential.
    Above this number, caching is not used and data loading is parallel.
    Use "inf" to always cache.
    """
    save_preds: bool = False
    """Whether to save test split predictions during training."""
    resume_experiment: bool = False
    """
    Whether to resume the experiment.
    Loads test results from any folds that have already been completed and skips training those folds.
    """
    bias: bool = False
    """Whether to add bias to linear layers."""
    hidden_size: int = 300
    """Dimensionality of hidden layers in MPN."""
    depth: int = 3
    """Number of message passing steps."""
    bias_solvent: bool = False
    """Whether to add bias to linear layers for solvent MPN if :code:`reaction_solvent` is True."""
    hidden_size_solvent: int = 300
    """Dimensionality of hidden layers in solvent MPN if :code:`reaction_solvent` is True."""
    depth_solvent: int = 3
    """Number of message passing steps for solvent if :code:`reaction_solvent` is True."""
    mpn_shared: bool = False
    """Whether to use the same message passing neural network for all input molecules
    Only relevant if :code:`number_of_molecules > 1`"""
    dropout: float = 0.0
    """Dropout probability."""
    activation: Literal["ReLU", "LeakyReLU", "PReLU", "tanh", "SELU", "ELU"] = "ReLU"
    """Activation function."""
    atom_messages: bool = False
    """Centers messages on atoms instead of on bonds."""
    undirected: bool = False
    """Undirected edges (always sum the two relevant bond vectors)."""
    ffn_hidden_size: int = None
    """Hidden dim for higher-capacity FFN (defaults to hidden_size)."""
    ffn_num_layers: int = 2
    """Number of layers in FFN after MPN encoding."""
    features_only: bool = False
    """Use only the additional features in an FFN, no graph network."""
    separate_val_features_path: List[str] = None
    """Path to file with features for separate val set."""
    separate_test_features_path: List[str] = None
    """Path to file with features for separate test set."""
    separate_val_phase_features_path: str = None
    """Path to file with phase features for separate val set."""
    separate_test_phase_features_path: str = None
    """Path to file with phase features for separate test set."""
    separate_val_atom_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate val set."""
    separate_test_atom_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate test set."""
    separate_val_bond_features_path: str = None
    """Path to file with extra atom descriptors for separate val set."""
    separate_test_bond_features_path: str = None
    """Path to file with extra atom descriptors for separate test set."""
    config_path: str = None
    """
    Path to a :code:`.json` file containing arguments. Any arguments present in the config file
    will override arguments specified via the command line or by the defaults.
    """
    ensemble_size: int = 1
    """Number of models in ensemble."""
    aggregation: Literal["mean", "sum", "norm"] = "mean"
    """Aggregation scheme for atomic vectors into molecular vectors"""
    aggregation_norm: int = 100
    """For norm aggregation, number by which to divide summed up atomic features"""
    reaction: bool = False
    """
    Whether to adjust MPNN layer to take reactions as input instead of molecules.
    """
    reaction_mode: Literal[
        "reac_prod",
        "reac_diff",
        "prod_diff",
        "reac_prod_balance",
        "reac_diff_balance",
        "prod_diff_balance",
    ] = "reac_diff"
    """
    Choices for construction of atom and bond features for reactions
    :code:`reac_prod`: concatenates the reactants feature with the products feature.
    :code:`reac_diff`: concatenates the reactants feature with the difference in features between reactants and products.
    :code:`prod_diff`: concatenates the products feature with the difference in features between reactants and products.
    :code:`reac_prod_balance`: concatenates the reactants feature with the products feature, balances imbalanced reactions.
    :code:`reac_diff_balance`: concatenates the reactants feature with the difference in features between reactants and products, balances imbalanced reactions.
    :code:`prod_diff_balance`: concatenates the products feature with the difference in features between reactants and products, balances imbalanced reactions.
    """
    reaction_solvent: bool = False
    """
    Whether to adjust the MPNN layer to take as input a reaction and a molecule, and to encode them with separate MPNNs.
    """
    explicit_h: bool = False
    """
    Whether H are explicitly specified in input (and should be kept this way). This option is intended to be used
    with the :code:`reaction` or :code:`reaction_solvent` options, and applies only to the reaction part.
    """
    adding_h: bool = False
    """
    Whether RDKit molecules will be constructed with adding the Hs to them. This option is intended to be used
    with Chemprop's default molecule or multi-molecule encoders, or in :code:`reaction_solvent` mode where it applies to the solvent only.
    """
    epochs: int = 30
    """Number of epochs to run."""
    warmup_epochs: float = 2.0
    """
    Number of epochs during which learning rate increases linearly from :code:`init_lr` to :code:`max_lr`.
    Afterwards, learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr`.
    """
    init_lr: float = 0.0001
    """Initial learning rate."""
    max_lr: float = 0.001
    """Maximum learning rate."""
    final_lr: float = 0.0001
    """Final learning rate."""
    grad_clip: float = None
    """Maximum magnitude of gradient during training."""
    class_balance: bool = False
    """Trains with an equal number of positives and negatives in each batch."""
    spectra_activation: Literal["exp", "softplus"] = "exp"
    """Indicates which function to use in dataset_type spectra training to constrain outputs to be positive."""
    spectra_target_floor: float = 1e-08
    """Values in targets for dataset type spectra are replaced with this value, intended to be a small positive number used to enforce positive values."""
    evidential_regularization: float = 0
    """Value used in regularization for evidential loss function. Value used in literature was 1."""
    overwrite_default_atom_features: bool = False
    """
    Overwrites the default atom descriptors with the new ones instead of concatenating them.
    Can only be used if atom_descriptors are used as a feature.
    """
    no_atom_descriptor_scaling: bool = False
    """Turn off atom feature scaling."""
    overwrite_default_bond_features: bool = False
    """Overwrites the default atom descriptors with the new ones instead of concatenating them"""
    no_bond_features_scaling: bool = False
    """Turn off atom feature scaling."""
    frzn_ffn_layers: int = 0
    """
    Overwrites weights for the first n layers of the ffn from checkpoint model (specified checkpoint_frzn),
    where n is specified in the input.
    Automatically also freezes mpnn weights.
    """
    freeze_first_only: bool = False
    """
    Determines whether or not to use checkpoint_frzn for just the first encoder.
    Default (False) is to use the checkpoint to freeze all encoders.
    (only relevant for number_of_molecules > 1, where checkpoint model has number_of_molecules = 1)
    """

    def __init__(self, *args, **kwargs) -> None:
        super(TrainArgs, self).__init__(*args, **kwargs)
        self._task_names = None
        self._crossval_index_sets = None
        self._task_names = None
        self._num_tasks = None
        self._features_size = None
        self._train_data_size = None

    @property
    def metrics(self) -> List[str]:
        """The list of metrics used for evaluation. Only the first is used for early stopping."""
        return [self.metric] + self.extra_metrics

    @property
    def minimize_score(self) -> bool:
        """Whether the model should try to minimize the score metric or maximize it."""
        return self.metric in {
            "rmse",
            "mae",
            "mse",
            "cross_entropy",
            "binary_cross_entropy",
            "sid",
            "wasserstein",
            "bounded_mse",
            "bounded_mae",
            "bounded_rmse",
        }

    @property
    def use_input_features(self) -> bool:
        """Whether the model is using additional molecule-level features."""
        return (
            self.features_generator is not None
            or self.features_path is not None
            or self.phase_features_path is not None
        )

    @property
    def num_lrs(self) -> int:
        """The number of learning rates to use (currently hard-coded to 1)."""
        return 1

    @property
    def crossval_index_sets(self) -> List[List[List[int]]]:
        """Index sets used for splitting data into train/validation/test during cross-validation"""
        return self._crossval_index_sets

    @property
    def task_names(self) -> List[str]:
        """A list of names of the tasks being trained on."""
        return self._task_names

    @task_names.setter
    def task_names(self, task_names: List[str]) -> None:
        self._task_names = task_names

    @property
    def num_tasks(self) -> int:
        """The number of tasks being trained on."""
        return len(self.task_names) if self.task_names is not None else 0

    @property
    def features_size(self) -> int:
        """The dimensionality of the additional molecule-level features."""
        return self._features_size

    @features_size.setter
    def features_size(self, features_size: int) -> None:
        self._features_size = features_size

    @property
    def train_data_size(self) -> int:
        """The size of the training data set."""
        return self._train_data_size

    @train_data_size.setter
    def train_data_size(self, train_data_size: int) -> None:
        self._train_data_size = train_data_size

    @property
    def atom_descriptor_scaling(self) -> bool:
        """
        Whether to apply normalization with a :class:`~chemprop.data.scaler.StandardScaler`
        to the additional atom features."
        """
        return not self.no_atom_descriptor_scaling

    @property
    def bond_feature_scaling(self) -> bool:
        """
        Whether to apply normalization with a :class:`~chemprop.data.scaler.StandardScaler`
        to the additional bond features."
        """
        return not self.no_bond_features_scaling

    def process_args(self) -> None:
        super(TrainArgs, self).process_args()
        global temp_save_dir
        if self.reaction_solvent is True and self.number_of_molecules != 2:
            raise ValueError(
                "In reaction_solvent mode, --number_of_molecules 2 must be specified."
            )
        self.smiles_columns = preprocess_smiles_columns(
            path=self.data_path,
            smiles_columns=self.smiles_columns,
            number_of_molecules=self.number_of_molecules,
        )
        if self.config_path is not None:
            with open(self.config_path) as f:
                config = json.load(f)
                for key, value in config.items():
                    setattr(self, key, value)
        if self.reaction_solvent is True and len(self.smiles_columns) != 2:
            raise ValueError(
                "In reaction_solvent mode, exactly two smiles column must be provided (one for reactions, and one for molecules)"
            )
        if self.reaction is True and self.reaction_solvent is True:
            raise ValueError(
                "Only reaction or reaction_solvent mode can be used, not both."
            )
        if self.save_dir is None:
            temp_save_dir = TemporaryDirectory()
            self.save_dir = temp_save_dir.name
        if self.checkpoint_paths is not None and len(self.checkpoint_paths) > 0:
            self.ensemble_size = len(self.checkpoint_paths)
        if self.metric is None:
            if self.dataset_type == "classification":
                self.metric = "auc"
            elif self.dataset_type == "multiclass":
                self.metric = "cross_entropy"
            elif self.dataset_type == "spectra":
                self.metric = "sid"
            elif (
                self.dataset_type == "regression"
                and self.loss_function == "bounded_mse"
            ):
                self.metric = "bounded_mse"
            elif self.dataset_type == "regression":
                self.metric = "rmse"
            else:
                raise ValueError(f"Dataset type {self.dataset_type} is not supported.")
        if self.metric in self.extra_metrics:
            raise ValueError(
                f"Metric {self.metric} is both the metric and is in extra_metrics. Please only include it once."
            )
        for metric in self.metrics:
            if not any(
                [
                    self.dataset_type == "classification"
                    and metric
                    in [
                        "auc",
                        "prc-auc",
                        "accuracy",
                        "binary_cross_entropy",
                        "f1",
                        "mcc",
                    ],
                    self.dataset_type == "regression"
                    and metric
                    in [
                        "rmse",
                        "mae",
                        "mse",
                        "r2",
                        "bounded_rmse",
                        "bounded_mae",
                        "bounded_mse",
                    ],
                    self.dataset_type == "multiclass"
                    and metric in ["cross_entropy", "accuracy", "f1", "mcc"],
                    self.dataset_type == "spectra" and metric in ["sid", "wasserstein"],
                ]
            ):
                raise ValueError(
                    f'Metric "{metric}" invalid for dataset type "{self.dataset_type}".'
                )
        if self.loss_function is None:
            if self.dataset_type == "classification":
                self.loss_function = "binary_cross_entropy"
            elif self.dataset_type == "multiclass":
                self.loss_function = "cross_entropy"
            elif self.dataset_type == "spectra":
                self.loss_function = "sid"
            elif self.dataset_type == "regression":
                self.loss_function = "mse"
            else:
                raise ValueError(
                    f"Default loss function not configured for dataset type {self.dataset_type}."
                )
        if self.loss_function != "bounded_mse" and any(
            metric in ["bounded_mse", "bounded_rmse", "bounded_mae"]
            for metric in self.metrics
        ):
            raise ValueError(
                "Bounded metrics can only be used in conjunction with the regression loss function bounded_mse."
            )
        if self.class_balance and self.dataset_type != "classification":
            raise ValueError(
                "Class balance can only be applied if the dataset type is classification."
            )
        if self.features_only and not (self.features_generator or self.features_path):
            raise ValueError(
                "When using features_only, a features_generator or features_path must be provided."
            )
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size
        if self.atom_messages and self.undirected:
            raise ValueError(
                "Undirected is unnecessary when using atom_messages since atom_messages are by their nature undirected."
            )
        if (
            not (self.split_type == "predetermined")
            == (self.folds_file is not None)
            == (self.test_fold_index is not None)
        ):
            raise ValueError(
                "When using predetermined split type, must provide folds_file and test_fold_index."
            )
        if not (self.split_type == "crossval") == (self.crossval_index_dir is not None):
            raise ValueError(
                "When using crossval split type, must provide crossval_index_dir."
            )
        if not (self.split_type in ["crossval", "index_predetermined"]) == (
            self.crossval_index_file is not None
        ):
            raise ValueError(
                "When using crossval or index_predetermined split type, must provide crossval_index_file."
            )
        if self.split_type in ["crossval", "index_predetermined"]:
            with open(self.crossval_index_file, "rb") as rf:
                self._crossval_index_sets = pickle.load(rf)
            self.num_folds = len(self.crossval_index_sets)
            self.seed = 0
        if self.split_sizes is None:
            if self.separate_val_path is None and self.separate_test_path is None:
                self.split_sizes = [0.8, 0.1, 0.1]
            elif self.separate_val_path is not None and self.separate_test_path is None:
                self.split_sizes = [0.8, 0.0, 0.2]
            elif self.separate_val_path is None and self.separate_test_path is not None:
                self.split_sizes = [0.8, 0.2, 0.0]
            else:
                self.split_sizes = [1.0, 0.0, 0.0]
        else:
            if not np.isclose(sum(self.split_sizes), 1):
                raise ValueError(
                    f"Provided split sizes of {self.split_sizes} do not sum to 1."
                )
            if any([(size < 0) for size in self.split_sizes]):
                raise ValueError(
                    f"Split sizes must be non-negative. Received split sizes: {self.split_sizes}"
                )
            if len(self.split_sizes) not in [2, 3]:
                raise ValueError(
                    f"Three values should be provided for train/val/test split sizes. Instead received {len(self.split_sizes)} value(s)."
                )
            if self.separate_val_path is None and self.separate_test_path is None:
                if len(self.split_sizes) != 3:
                    raise ValueError(
                        f"Three values should be provided for train/val/test split sizes. Instead received {len(self.split_sizes)} value(s)."
                    )
                if self.split_sizes[0] == 0.0:
                    raise ValueError(
                        f"Provided split size for train split must be nonzero. Received split size {self.split_sizes[0]}"
                    )
                if self.split_sizes[1] == 0.0:
                    raise ValueError(
                        f"Provided split size for validation split must be nonzero. Received split size {self.split_sizes[1]}"
                    )
            elif self.separate_val_path is not None and self.separate_test_path is None:
                if len(self.split_sizes) == 2:
                    self.split_sizes = [self.split_sizes[0], 0.0, self.split_sizes[1]]
                if self.split_sizes[0] == 0.0:
                    raise ValueError(
                        "Provided split size for train split must be nonzero."
                    )
                if self.split_sizes[1] != 0.0:
                    raise ValueError(
                        f"Provided split size for validation split must be 0 because validation set is provided separately. Received split size {self.split_sizes[1]}"
                    )
            elif self.separate_val_path is None and self.separate_test_path is not None:
                if len(self.split_sizes) == 2:
                    self.split_sizes = [self.split_sizes[0], self.split_sizes[1], 0.0]
                if self.split_sizes[0] == 0.0:
                    raise ValueError(
                        "Provided split size for train split must be nonzero."
                    )
                if self.split_sizes[1] == 0.0:
                    raise ValueError(
                        "Provided split size for validation split must be nonzero."
                    )
                if self.split_sizes[2] != 0.0:
                    raise ValueError(
                        f"Provided split size for test split must be 0 because test set is provided separately. Received split size {self.split_sizes[2]}"
                    )
            elif self.split_sizes != [1.0, 0.0, 0.0]:
                raise ValueError(
                    f"Separate data paths were provided for val and test splits. Split sizes should not also be provided. Received split sizes: {self.split_sizes}"
                )
        if self.test:
            self.epochs = 0
        for (
            features_argument,
            base_features_path,
            val_features_path,
            test_features_path,
        ) in [
            (
                "`--features_path`",
                self.features_path,
                self.separate_val_features_path,
                self.separate_test_features_path,
            ),
            (
                "`--phase_features_path`",
                self.phase_features_path,
                self.separate_val_phase_features_path,
                self.separate_test_phase_features_path,
            ),
            (
                "`--atom_descriptors_path`",
                self.atom_descriptors_path,
                self.separate_val_atom_descriptors_path,
                self.separate_test_atom_descriptors_path,
            ),
            (
                "`--bond_features_path`",
                self.bond_features_path,
                self.separate_val_bond_features_path,
                self.separate_test_bond_features_path,
            ),
        ]:
            if base_features_path is not None:
                if self.separate_val_path is not None and val_features_path is None:
                    raise ValueError(
                        f"Additional features were provided using the argument {features_argument}. The same kinds of features must be provided for the separate validation set."
                    )
                if self.separate_test_path is not None and test_features_path is None:
                    raise ValueError(
                        f"Additional features were provided using the argument {features_argument}. The same kinds of features must be provided for the separate test set."
                    )
        if self.overwrite_default_atom_features and self.atom_descriptors != "feature":
            raise NotImplementedError(
                "Overwriting of the default atom descriptors can only be used if theprovided atom descriptors are features."
            )
        if not self.atom_descriptor_scaling and self.atom_descriptors is None:
            raise ValueError(
                "Atom descriptor scaling is only possible if additional atom features are provided."
            )
        if self.overwrite_default_bond_features and self.bond_features_path is None:
            raise ValueError(
                "If you want to overwrite the default bond descriptors, a bond_descriptor_path must be provided."
            )
        if not self.bond_feature_scaling and self.bond_features_path is None:
            raise ValueError(
                "Bond descriptor scaling is only possible if additional bond features are provided."
            )
        if self.target_weights is not None:
            avg_weight = sum(self.target_weights) / len(self.target_weights)
            self.target_weights = [(w / avg_weight) for w in self.target_weights]
            if min(self.target_weights) < 0:
                raise ValueError("Provided target weights must be non-negative.")
        if self.split_key_molecule >= self.number_of_molecules:
            raise ValueError(
                "The index provided with the argument `--split_key_molecule` must be less than the number of molecules. Note that this index begins with 0 for the first molecule. "
            )


class PredictArgs(CommonArgs):
    """:class:`PredictArgs` includes :class:`CommonArgs` along with additional arguments used for predicting with a Chemprop model."""

    test_path: str
    """Path to CSV file containing testing data for which predictions will be made."""
    preds_path: str
    """Path to CSV file where predictions will be saved."""
    drop_extra_columns: bool = False
    """Whether to drop all columns from the test data file besides the SMILES columns and the new prediction columns."""
    ensemble_variance: bool = False
    """Deprecated. Whether to calculate the variance of ensembles as a measure of epistemic uncertainty. If True, the variance is saved as an additional column for each target in the preds_path."""
    individual_ensemble_predictions: bool = False
    """Whether to return the predictions made by each of the individual models rather than the average of the ensemble"""
    uncertainty_method: Literal[
        "mve",
        "ensemble",
        "evidential_epistemic",
        "evidential_aleatoric",
        "evidential_total",
        "classification",
        "dropout",
        "spectra_roundrobin",
    ] = None
    """The method of calculating uncertainty."""
    calibration_method: Literal[
        "zscaling",
        "tscaling",
        "zelikman_interval",
        "mve_weighting",
        "platt",
        "isotonic",
    ] = None
    """Methods used for calibrating the uncertainty calculated with uncertainty method."""
    evaluation_methods: List[str] = None
    """The methods used for evaluating the uncertainty performance if the test data provided includes targets.
    Available methods are [nll, miscalibration_area, ence, spearman] or any available classification or multiclass metric."""
    evaluation_scores_path: str = None
    """Location to save the results of uncertainty evaluations."""
    uncertainty_dropout_p: float = 0.1
    """The probability to use for Monte Carlo dropout uncertainty estimation."""
    dropout_sampling_size: int = 10
    """The number of samples to use for Monte Carlo dropout uncertainty estimation. Distinct from the dropout used during training."""
    calibration_interval_percentile: float = 95
    """Sets the percentile used in the calibration methods. Must be in the range (1,100)."""
    regression_calibrator_metric: Literal["stdev", "interval"] = None
    """Regression calibrators can output either a stdev or an inverval. """
    calibration_path: str = None
    """Path to data file to be used for uncertainty calibration."""
    calibration_features_path: str = None
    """Path to features data to be used with the uncertainty calibration dataset."""
    calibration_phase_features_path: str = None
    """ """
    calibration_atom_descriptors_path: str = None
    """Path to the extra atom descriptors."""
    calibration_bond_features_path: str = None
    """Path to the extra bond descriptors that will be used as bond features to featurize a given molecule."""

    @property
    def ensemble_size(self) -> int:
        """The number of models in the ensemble."""
        return len(self.checkpoint_paths)

    def process_args(self) -> None:
        super(PredictArgs, self).process_args()
        if self.regression_calibrator_metric is None:
            if self.calibration_method == "zelikman_interval":
                self.regression_calibrator_metric = "interval"
            else:
                self.regression_calibrator_metric = "stdev"
        if self.uncertainty_method == "dropout" and version.parse(
            paddle.__version__
        ) < version.parse("1.9.0"):
            raise ValueError(
                "Dropout uncertainty is only supported for versions >= 1.9.0"
            )
        self.smiles_columns = preprocess_smiles_columns(
            path=self.test_path,
            smiles_columns=self.smiles_columns,
            number_of_molecules=self.number_of_molecules,
        )
        if self.checkpoint_paths is None or len(self.checkpoint_paths) == 0:
            raise ValueError(
                "Found no checkpoints. Must specify --checkpoint_path <path> or --checkpoint_dir <dir> containing at least one checkpoint."
            )
        if self.ensemble_variance is True:
            if self.uncertainty_method in ["ensemble", None]:
                warn(
                    "The `--ensemble_variance` argument is deprecated and should                         be replaced with `--uncertainty_method ensemble`.",
                    DeprecationWarning,
                )
                self.uncertainty_method = "ensemble"
            else:
                raise ValueError(
                    f"Only one uncertainty method can be used at a time.                         The arguement `--ensemble_variance` was provided along                         with the uncertainty method {self.uncertainty_method}. The `--ensemble_variance`                         argument is deprecated and should be replaced with `--uncertainty_method ensemble`."
                )
        if (
            self.calibration_interval_percentile <= 1
            or self.calibration_interval_percentile >= 100
        ):
            raise ValueError(
                "The calibration interval must be a percentile value in the range (1,100)."
            )
        if self.uncertainty_dropout_p < 0 or self.uncertainty_dropout_p > 1:
            raise ValueError("The dropout probability must be in the range (0,1).")
        if self.dropout_sampling_size <= 1:
            raise ValueError(
                "The argument `--dropout_sampling_size` must be an integer greater than 1."
            )
        for features_argument, base_features_path, cal_features_path in [
            ("`--features_path`", self.features_path, self.calibration_features_path),
            (
                "`--phase_features_path`",
                self.phase_features_path,
                self.calibration_phase_features_path,
            ),
            (
                "`--atom_descriptors_path`",
                self.atom_descriptors_path,
                self.calibration_atom_descriptors_path,
            ),
            (
                "`--bond_features_path`",
                self.bond_features_path,
                self.calibration_bond_features_path,
            ),
        ]:
            if (
                base_features_path is not None
                and self.calibration_path is not None
                and cal_features_path is None
            ):
                raise ValueError(
                    f"Additional features were provided using the argument {features_argument}. The same kinds of features must be provided for the calibration dataset."
                )


class InterpretArgs(CommonArgs):
    """:class:`InterpretArgs` includes :class:`CommonArgs` along with additional arguments used for interpreting a trained Chemprop model."""

    data_path: str
    """Path to data CSV file."""
    batch_size: int = 500
    """Batch size."""
    property_id: int = 1
    """Index of the property of interest in the trained model."""
    rollout: int = 20
    """Number of rollout steps."""
    c_puct: float = 10.0
    """Constant factor in MCTS."""
    max_atoms: int = 20
    """Maximum number of atoms in rationale."""
    min_atoms: int = 8
    """Minimum number of atoms in rationale."""
    prop_delta: float = 0.5
    """Minimum score to count as positive."""

    def process_args(self) -> None:
        super(InterpretArgs, self).process_args()
        self.smiles_columns = preprocess_smiles_columns(
            path=self.data_path,
            smiles_columns=self.smiles_columns,
            number_of_molecules=self.number_of_molecules,
        )
        if self.features_path is not None:
            raise ValueError(
                "Cannot use --features_path <path> for interpretation since features need to be computed dynamically for molecular substructures. Please specify --features_generator <generator>."
            )
        if self.checkpoint_paths is None or len(self.checkpoint_paths) == 0:
            raise ValueError(
                "Found no checkpoints. Must specify --checkpoint_path <path> or --checkpoint_dir <dir> containing at least one checkpoint."
            )


class FingerprintArgs(PredictArgs):
    """:class:`FingerprintArgs` includes :class:`PredictArgs` with additional arguments for the generation of latent fingerprint vectors."""

    fingerprint_type: Literal["MPN", "last_FFN"] = "MPN"
    """Choice of which type of latent fingerprint vector to use. Default is the output of the MPNN, excluding molecular features"""


class HyperoptArgs(TrainArgs):
    """:class:`HyperoptArgs` includes :class:`TrainArgs` along with additional arguments used for optimizing Chemprop hyperparameters."""

    num_iters: int = 20
    """Number of hyperparameter choices to try."""
    config_save_path: str
    """Path to :code:`.json` file where best hyperparameter settings will be written."""
    log_dir: str = None
    """(Optional) Path to a directory where all results of the hyperparameter optimization will be written."""
    hyperopt_checkpoint_dir: str = None
    """Path to a directory where hyperopt completed trial data is stored. Hyperopt job will include these trials if restarted.
    Can also be used to run multiple instances in parallel if they share the same checkpoint directory."""
    startup_random_iters: int = None
    """The initial number of trials that will be randomly specified before TPE algorithm is used to select the rest.
    By default will be half the total number of trials."""
    manual_trial_dirs: List[str] = None
    """Paths to save directories for manually trained models in the same search space as the hyperparameter search.
    Results will be considered as part of the trial history of the hyperparameter search."""
    search_parameter_keywords: List[str] = ["basic"]
    """The model parameters over which to search for an optimal hyperparameter configuration.
    Some options are bundles of parameters or otherwise special parameter operations.

    Special keywords:
        basic - the default set of hyperparameters for search: depth, ffn_num_layers, dropout, and linked_hidden_size.
        linked_hidden_size - search for hidden_size and ffn_hidden_size, but constrained for them to have the same value.
            If either of the component words are entered in separately, both are searched independently.
        learning_rate - search for max_lr, init_lr, final_lr, and warmup_epochs. The search for init_lr and final_lr values
            are defined as fractions of the max_lr value. The search for warmup_epochs is as a fraction of the total epochs used.
        all - include search for all 13 inidividual keyword options

    Individual supported parameters:
        activation, aggregation, aggregation_norm, batch_size, depth,
        dropout, ffn_hidden_size, ffn_num_layers, final_lr, hidden_size,
        init_lr, max_lr, warmup_epochs
    """

    def process_args(self) -> None:
        super(HyperoptArgs, self).process_args()
        if self.log_dir is None:
            self.log_dir = self.save_dir
        if self.hyperopt_checkpoint_dir is None:
            self.hyperopt_checkpoint_dir = self.log_dir
        if self.startup_random_iters is None:
            self.startup_random_iters = self.num_iters // 2
        supported_keywords = [
            "basic",
            "learning_rate",
            "linked_hidden_size",
            "all",
            "activation",
            "aggregation",
            "aggregation_norm",
            "batch_size",
            "depth",
            "dropout",
            "ffn_hidden_size",
            "ffn_num_layers",
            "final_lr",
            "hidden_size",
            "init_lr",
            "max_lr",
            "warmup_epochs",
        ]
        supported_parameters = [
            "activation",
            "aggregation",
            "aggregation_norm",
            "batch_size",
            "depth",
            "dropout",
            "ffn_hidden_size",
            "ffn_num_layers",
            "final_lr_ratio",
            "hidden_size",
            "init_lr_ratio",
            "linked_hidden_size",
            "max_lr",
            "warmup_epochs",
        ]
        unsupported_keywords = set(self.search_parameter_keywords) - set(
            supported_keywords
        )
        if len(unsupported_keywords) != 0:
            raise NotImplementedError(
                f"Keywords for what hyperparameters to include in the search are designated                     with the argument `--search_parameter_keywords`. The following unsupported                    keywords were received: {unsupported_keywords}. The available supported                    keywords are: {supported_keywords}"
            )
        search_parameters = set()
        if "all" in self.search_parameter_keywords:
            search_parameters.update(supported_parameters)
        if "basic" in self.search_parameter_keywords:
            search_parameters.update(
                ["depth", "ffn_num_layers", "dropout", "linked_hidden_size"]
            )
        if "learning_rate" in self.search_parameter_keywords:
            search_parameters.update(
                ["max_lr", "init_lr_ratio", "final_lr_ratio", "warmup_epochs"]
            )
        for kw in self.search_parameter_keywords:
            if kw in supported_parameters:
                search_parameters.add(kw)
        if "init_lr" in self.search_parameter_keywords:
            search_parameters.add("init_lr_ratio")
        if "final_lr" in self.search_parameter_keywords:
            search_parameters.add("final_lr_ratio")
        if "linked_hidden_size" in search_parameters and (
            "hidden_size" in search_parameters or "ffn_hidden_size" in search_parameters
        ):
            search_parameters.remove("linked_hidden_size")
            search_parameters.update(["hidden_size", "ffn_hidden_size"])
        self.search_parameters = list(search_parameters)


class SklearnTrainArgs(TrainArgs):
    """:class:`SklearnTrainArgs` includes :class:`TrainArgs` along with additional arguments for training a scikit-learn model."""

    model_type: Literal["random_forest", "svm"]
    """scikit-learn model to use."""
    class_weight: Literal["balanced"] = None
    """How to weight classes (None means no class balance)."""
    single_task: bool = False
    """Whether to run each task separately (needed when dataset has null entries)."""
    radius: int = 2
    """Morgan fingerprint radius."""
    num_bits: int = 2048
    """Number of bits in morgan fingerprint."""
    num_trees: int = 500
    """Number of random forest trees."""
    impute_mode: Literal["single_task", "median", "mean", "linear", "frequent"] = None
    """How to impute missing data (None means no imputation)."""


class SklearnPredictArgs(Tap):
    """:class:`SklearnPredictArgs` contains arguments used for predicting with a trained scikit-learn model."""

    test_path: str
    """Path to CSV file containing testing data for which predictions will be made."""
    smiles_columns: List[str] = None
    """List of names of the columns containing SMILES strings.
    By default, uses the first :code:`number_of_molecules` columns."""
    number_of_molecules: int = 1
    """Number of molecules in each input to the model.
    This must equal the length of :code:`smiles_columns` (if not :code:`None`)."""
    preds_path: str
    """Path to CSV file where predictions will be saved."""
    checkpoint_dir: str = None
    """Path to directory containing model checkpoints (:code:`.pkl` file)"""
    checkpoint_path: str = None
    """Path to model checkpoint (:code:`.pkl` file)"""
    checkpoint_paths: List[str] = None
    """List of paths to model checkpoints (:code:`.pkl` files)"""

    def process_args(self) -> None:
        self.smiles_columns = preprocess_smiles_columns(
            path=self.test_path,
            smiles_columns=self.smiles_columns,
            number_of_molecules=self.number_of_molecules,
        )
        self.checkpoint_paths = get_checkpoint_paths(
            checkpoint_path=self.checkpoint_path,
            checkpoint_paths=self.checkpoint_paths,
            checkpoint_dir=self.checkpoint_dir,
            ext=".pkl",
        )


# === args end   ===
