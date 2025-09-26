# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2024 Kyle Swanson

from __future__ import annotations

import threading
from random import Random
from typing import Dict
from typing import List
from typing import Optional
from typing import OrderedDict
from typing import Tuple
from typing import Union

import numpy as np
import paddle
from paddle import io

try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

from ppsci.arch.chemprop_molecule_utils import atom_features_zeros
from ppsci.arch.chemprop_molecule_utils import get_bond_fdim
from ppsci.arch.chemprop_molecule_utils import get_features_generator
from ppsci.arch.chemprop_molecule_utils import map_reac_to_prod


class Featurization:
    """
    A class holding molecule featurization parameters as attributes.
    """

    def __init__(self):
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

    def is_mol(self, mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]):
        """Checks whether an input is a molecule or a reaction

        Args:
            mol: str, RDKIT molecule or tuple of molecules
        return:
            bool: Whether the supplied input corresponds to a single molecule
        """
        if isinstance(mol, str) and ">" not in mol:
            return True
        elif isinstance(mol, Chem.Mol):
            return True
        return False

    def is_explicit_h(self, is_mol=True):
        """Returns whether to retain explicit Hs (for reactions only)"""
        if not is_mol:
            return self.EXPLICIT_H
        return False

    def is_adding_hs(self, is_mol=True):
        """Returns whether to add explicit Hs to the mol (not for reactions)"""
        if is_mol:
            return self.ADDING_H
        return False

    def is_reaction(self, is_mol=True):
        """Returns whether to use reactions as input"""
        if is_mol:
            return False
        if self.REACTION:
            return True
        return False

    def reaction_mode(self):
        """Returns the reaction mode"""
        return self.REACTION_MODE

    def onek_encoding_unk(self, value: int, choices: List[int]) -> List[int]:
        """Creates a one-hot encoding with an extra category for uncommon values.
            If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.

        Args:
            value: The value for which the encoding should be one.
            choices: A list of possible values.
        :return:
            encoding: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
        """
        encoding = [0] * (len(choices) + 1)
        index = choices.index(value) if value in choices else -1
        encoding[index] = 1
        return encoding

    def atom_features(
        self, atom: Chem.rdchem.Atom, functional_groups: List[int] = None
    ) -> List[Union[bool, int, float]]:
        """Builds a feature vector for an atom.

        Args:
            atom: An RDKit atom.
            functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
        return:
            features: A list containing the atom features.
        """
        if atom is None:
            features = [0] * self.ATOM_FDIM
        else:
            features = (
                self.onek_encoding_unk(
                    atom.GetAtomicNum() - 1, self.ATOM_FEATURES["atomic_num"]
                )
                + self.onek_encoding_unk(
                    atom.GetTotalDegree(), self.ATOM_FEATURES["degree"]
                )
                + self.onek_encoding_unk(
                    atom.GetFormalCharge(), self.ATOM_FEATURES["formal_charge"]
                )
                + self.onek_encoding_unk(
                    int(atom.GetChiralTag()), self.ATOM_FEATURES["chiral_tag"]
                )
                + self.onek_encoding_unk(
                    int(atom.GetTotalNumHs()), self.ATOM_FEATURES["num_Hs"]
                )
                + self.onek_encoding_unk(
                    int(atom.GetHybridization()), self.ATOM_FEATURES["hybridization"]
                )
                + [1 if atom.GetIsAromatic() else 0]
                + [atom.GetMass() * 0.01]
            )
            if functional_groups is not None:
                features += functional_groups
        return features

    def bond_features(self, bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
        """Builds a feature vector for a bond.

        Args:
            bond: An RDKit bond.
        return:
            fbond: A list containing the bond features.
        """
        if bond is None:
            fbond = [1] + [0] * (self.BOND_FDIM - 1)
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
            fbond += self.onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
        return fbond

    def get_atom_fdim(
        self, overwrite_default_atom: bool = False, is_reaction: bool = False
    ) -> int:
        """Gets the dimensionality of the atom feature vector.

        Args:
            overwrite_default_atom: Whether to overwrite the default atom descriptors
            is_reaction: Whether to add :code:`EXTRA_ATOM_FDIM` for reaction input when :code:`REACTION_MODE` is not None
        return:
            The dimensionality of the atom feature vector.
        """
        if self.REACTION_MODE:
            return (
                not overwrite_default_atom
            ) * self.ATOM_FDIM + is_reaction * self.EXTRA_ATOM_FDIM
        else:
            return (not overwrite_default_atom) * self.ATOM_FDIM + self.EXTRA_ATOM_FDIM

    def get_bond_fdim(
        self,
        atom_messages: bool = False,
        overwrite_default_bond: bool = False,
        overwrite_default_atom: bool = False,
        is_reaction: bool = False,
    ) -> int:
        """Gets the dimensionality of the bond feature vector.

        Args:
            atom_messages: Whether atom messages are being used. If atom messages are used,
                            then the bond feature vector only contains bond features.
                            Otherwise it contains both atom and bond features.
            overwrite_default_bond: Whether to overwrite the default bond descriptors
            overwrite_default_atom: Whether to overwrite the default atom descriptors
            is_reaction: Whether to add :code:`EXTRA_BOND_FDIM` for reaction input when :code:`REACTION_MODE:` is not None
        return:
            The dimensionality of the bond feature vector.
        """
        if self.REACTION_MODE:
            return (
                (not overwrite_default_bond) * self.BOND_FDIM
                + is_reaction * self.EXTRA_BOND_FDIM
                + (not atom_messages)
                * self.get_atom_fdim(
                    overwrite_default_atom=overwrite_default_atom,
                    is_reaction=is_reaction,
                )
            )
        else:
            return (
                (not overwrite_default_bond) * self.BOND_FDIM
                + self.EXTRA_BOND_FDIM
                + (not atom_messages)
                * self.get_atom_fdim(
                    overwrite_default_atom=overwrite_default_atom,
                    is_reaction=is_reaction,
                )
            )

    def make_mol(self, s: str, keep_h: bool, add_h: bool):
        """
        Builds an RDKit molecule from a SMILES string.

        Args:
            s: SMILES string.
            keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
        return:
            RDKit molecule.
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


class MolGraph:
    """
    MolGraph represents the graph structure and featurization of a single molecule.
    A MolGraph computes the following attributes:

    Args:
        mol: A SMILES or an RDKit molecule.
        atom_features_extra: A list of 2D numpy array containing additional atom features to featurize the molecule
        bond_features_extra: A list of 2D numpy array containing additional bond features to featurize the molecule
        overwrite_default_atom_features: Boolean to overwrite default atom features by atom_features instead of concatenating
        overwrite_default_bond_features: Boolean to overwrite default bond features by bond_features instead of concatenating

    Returns:
        n_atoms: The number of atoms in the molecule.
        n_bonds: The number of bonds in the molecule.
        f_atoms: A mapping from an atom index to a list of atom features.
        f_bonds: A mapping from a bond index to a list of bond features.
        a2b: A mapping from an atom index to a list of incoming bond indices.
        b2a: A mapping from a bond index to the index of the atom the bond originates from.
        b2revb: A mapping from a bond index to the index of the reverse bond.
        overwrite_default_atom_features: A boolean to overwrite default atom descriptors.
        overwrite_default_bond_features: A boolean to overwrite default bond descriptors.
        is_mol: A boolean whether the input is a molecule.
        is_reaction: A boolean whether the molecule is a reaction.
        is_explicit_h: A boolean whether to retain explicit Hs (for reaction mode)
        is_adding_hs: A boolean whether to add explicit Hs (not for reaction mode)
        reaction_mode:  Reaction mode to construct atom and bond feature vectors
    """

    def __init__(
        self,
        mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]],
        atom_features_extra=None,
        bond_features_extra=None,
        overwrite_default_atom_features=False,
        overwrite_default_bond_features=False,
    ):
        self.Featuri = Featurization()
        self.is_mol = self.Featuri.is_mol(mol)
        self.is_explicit_h = self.Featuri.is_explicit_h(self.is_mol)
        self.is_adding_hs = self.Featuri.is_adding_hs(self.is_mol)
        self.is_reaction = self.Featuri.is_reaction(self.is_mol)
        self.reaction_mode = self.Featuri.reaction_mode()
        if type(mol) == str:
            if self.is_reaction:
                mol = self.Featuri.make_mol(
                    mol.split(">")[0], self.is_explicit_h, self.is_adding_hs
                ), self.Featuri.make_mol(
                    mol.split(">")[-1], self.is_explicit_h, self.is_adding_hs
                )
            else:
                mol = self.Featuri.make_mol(mol, self.is_explicit_h, self.is_adding_hs)
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
            self.f_atoms = [self.Featuri.atom_features(atom) for atom in mol.GetAtoms()]
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
                    f_bond = self.Featuri.bond_features(bond)
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
                f_atoms_reac = [
                    self.Featuri.atom_features(atom) for atom in mol_reac.GetAtoms()
                ] + [
                    atom_features_zeros(mol_prod.GetAtomWithIdx(index)) for index in pio
                ]
                f_atoms_prod = [
                    (
                        self.Featuri.atom_features(
                            mol_prod.GetAtomWithIdx(ri2pi[atom.GetIdx()])
                        )
                        if atom.GetIdx() not in rio
                        else atom_features_zeros(atom)
                    )
                    for atom in mol_reac.GetAtoms()
                ] + [
                    self.Featuri.atom_features(mol_prod.GetAtomWithIdx(index))
                    for index in pio
                ]
            else:
                f_atoms_reac = [
                    self.Featuri.atom_features(atom) for atom in mol_reac.GetAtoms()
                ] + [
                    self.Featuri.atom_features(mol_prod.GetAtomWithIdx(index))
                    for index in pio
                ]
                f_atoms_prod = [
                    (
                        self.Featuri.atom_features(
                            mol_prod.GetAtomWithIdx(ri2pi[atom.GetIdx()])
                        )
                        if atom.GetIdx() not in rio
                        else self.Featuri.atom_features(atom)
                    )
                    for atom in mol_reac.GetAtoms()
                ] + [
                    self.Featuri.atom_features(mol_prod.GetAtomWithIdx(index))
                    for index in pio
                ]
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
                    (x + y[self.Featuri.MAX_ATOMIC_NUM + 1 :])
                    for x, y in zip(f_atoms_reac, f_atoms_prod)
                ]
            elif self.reaction_mode in ["reac_diff", "reac_diff_balance"]:
                self.f_atoms = [
                    (x + y[self.Featuri.MAX_ATOMIC_NUM + 1 :])
                    for x, y in zip(f_atoms_reac, f_atoms_diff)
                ]
            elif self.reaction_mode in ["prod_diff", "prod_diff_balance"]:
                self.f_atoms = [
                    (x + y[self.Featuri.MAX_ATOMIC_NUM + 1 :])
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
                    f_bond_reac = self.Featuri.bond_features(bond_reac)
                    f_bond_prod = self.Featuri.bond_features(bond_prod)
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


CACHE_MOL = True
CACHE_GRAPH = True
SMILES_TO_MOL: Dict[str, Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]] = {}
SMILES_TO_GRAPH: Dict[str, MolGraph] = {}


def cache_graph():
    """
    Returns whether MolGraphs will be cached.
    """
    return CACHE_GRAPH


def empty_cache():
    """
    Empties the cache of MolGraph and RDKit molecules.
    """
    SMILES_TO_GRAPH.clear()
    SMILES_TO_MOL.clear()


def cache_mol() -> bool:
    """
    Returns whether RDKit molecules will be cached.
    """
    return CACHE_MOL


class BatchMolGraph:
    """
    BatchMolGraph represents the graph structure and featurization of a batch of molecules.
    A BatchMolGraph contains the attributes of a class `MolGraph` plus:

    Args:
        mol_graphs: A list of class `MolGraph`s from which to construct the class `BatchMolGraph`.

    Returns:
        atom_fdim: The dimensionality of the atom feature vector.
        bond_fdim: The dimensionality of the bond feature vector (technically the combined atom/bond features).
        a_scope: A list of tuples indicating the start and end atom indices for each molecule.
        b_scope: A list of tuples indicating the start and end bond indices for each molecule.
        max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
        b2b: (Optional) A mapping from a bond index to incoming bond indices.
        a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph]):
        self.Featuri = Featurization()
        self.overwrite_default_atom_features = mol_graphs[
            0
        ].overwrite_default_atom_features
        self.overwrite_default_bond_features = mol_graphs[
            0
        ].overwrite_default_bond_features
        self.is_reaction = mol_graphs[0].is_reaction
        self.atom_fdim = self.Featuri.get_atom_fdim(
            overwrite_default_atom=self.overwrite_default_atom_features,
            is_reaction=self.is_reaction,
        )
        self.bond_fdim = self.Featuri.get_bond_fdim(
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

    def get_components(self, atom_messages: bool = False):
        """
        A tuple containing Paddle tensors with the atom features, bond features, graph structure,
        Returns the components of the class `BatchMolGraph`.

        Args:
            atom_messages: Whether to use atom messages instead of bond messages. This changes the bond feature
                                vector to contain only bond features rather than both atom and bond features.
        :return: A tuple containing tensors with the atom features, bond features, graph structure,
                 and scope of the atoms and bonds (i.e., the indices of the molecules they belong to).
                 in order: f_atoms f_bonds a2b b2a b2revb a_scope b_scope
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

    def get_b2b(self):
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        return:
            A tensor containing the mapping from each bond index to all the incoming bond indices.
        """
        if self.b2b is None:
            b2b = self.a2b[self.b2a]
            revmask = (
                b2b
                != self.b2revb.unsqueeze(axis=1).tile(repeat_times=[1, b2b.shape[1]])
            ).astype(dtype="int64")
            self.b2b = b2b * revmask
        return self.b2b

    def get_a2a(self):
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        return:
            A tensor containing the mapping from each atom index to all the neighboring atom indices.
        """
        if self.a2a is None:
            self.a2a = self.b2a[self.a2b]
        return self.a2a


class StandardScaler:
    """A class normalizes the features of a dataset.
    When it is fit on a dataset, the class StandardScaler learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the class StandardScaler subtracts the means and divides by the standard deviations.

    Args:
        means: An optional 1D numpy array of precomputed means.
        stds: An optional 1D numpy array of precomputed standard deviations.
        replace_nan_token: A token to use to replace NaN entries in the features.
    """

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X: List[List[Optional[float]]]):
        """
        Learns means and standard deviations across the 0th axis of the data code X.

        Args:
            X: A list of lists of floats (or None).
        return:
            The fitted class StandardScaler (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(
            np.isnan(self.means), np.zeros(tuple(self.means.shape)), self.means
        )
        self.stds = np.where(
            np.isnan(self.stds), np.ones(tuple(self.stds.shape)), self.stds
        )
        self.stds = np.where(self.stds == 0, np.ones(tuple(self.stds.shape)), self.stds)
        return self

    def transform(self, X: List[List[Optional[float]]]):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        Args:
            X: A list of lists of floats (or None).
        return:
            The transformed data with NaNs replaced by code self.replace_nan_token.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan
        )
        return transformed_with_none

    def inverse_transform(self, X: List[List[Optional[float]]]):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        Args:
            X: A list of lists of floats.
        return:
            The inverse transformed data with NaNs replaced by code self.replace_nan_token.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan
        )
        return transformed_with_none


class MoleculeDatapoint:
    """
    MoleculeDatapoint contains a single molecule and its associated features and targets.

    Args:
        smiles: A list of the SMILES strings for the molecules.
        targets: A list of targets for the molecule (contains None for unknown target values).
        row: The raw CSV row containing the information for this molecule.
        data_weight: Weighting of the datapoint for the loss function.
        gt_targets: Indicates whether the targets are an inequality regression target of the form ">x".
        lt_targets: Indicates whether the targets are an inequality regression target of the form "<x".
        features: A numpy array containing additional features (e.g., Morgan fingerprint).
        features_generator: A list of features generators to use.
        phase_features: A one-hot vector indicating the phase of the data, as used in spectra data.
        atom_descriptors: A numpy array containing additional atom descriptors to featurize the molecule
        bond_features: A numpy array containing additional bond features to featurize the molecule
        overwrite_default_atom_features: Boolean to overwrite default atom features by atom_features
        overwrite_default_bond_features: Boolean to overwrite default bond features by bond_features
    """

    def __init__(
        self,
        smiles: List[str],
        targets: List[Optional[float]] = None,
        row: OrderedDict = None,
        data_weight: float = None,
        gt_targets: List[bool] = None,
        lt_targets: List[bool] = None,
        features: np.ndarray = None,
        features_generator: List[str] = None,
        phase_features: List[float] = None,
        atom_features: np.ndarray = None,
        atom_descriptors: np.ndarray = None,
        bond_features: np.ndarray = None,
        overwrite_default_atom_features: bool = False,
        overwrite_default_bond_features: bool = False,
    ):
        if features is not None and features_generator is not None:
            raise ValueError(
                "Cannot provide both loaded features and a features generator."
            )
        self.Featuri = Featurization()
        self.smiles = smiles
        self.targets = targets
        self.row = row
        self.features = features
        self.features_generator = features_generator
        self.phase_features = phase_features
        self.atom_descriptors = atom_descriptors
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.overwrite_default_atom_features = overwrite_default_atom_features
        self.overwrite_default_bond_features = overwrite_default_bond_features
        self.is_mol_list = [self.Featuri.is_mol(s) for s in smiles]
        self.is_reaction_list = [self.Featuri.is_reaction(x) for x in self.is_mol_list]
        self.is_reaction_list = [self.Featuri.is_reaction(x) for x in self.is_mol_list]
        self.is_explicit_h_list = [
            self.Featuri.is_explicit_h(x) for x in self.is_mol_list
        ]
        self.is_adding_hs_list = [
            self.Featuri.is_adding_hs(x) for x in self.is_mol_list
        ]
        if data_weight is not None:
            self.data_weight = data_weight
        if gt_targets is not None:
            self.gt_targets = gt_targets
        if lt_targets is not None:
            self.lt_targets = lt_targets
        if self.features_generator is not None:
            self.features = []
            for fg in self.features_generator:
                features_generator = get_features_generator(fg)
                for m, reaction in zip(self.mol, self.is_reaction_list):
                    if not reaction:
                        if m is not None and m.GetNumHeavyAtoms() > 0:
                            self.features.extend(features_generator(m))
                        elif m is not None and m.GetNumHeavyAtoms() == 0:
                            self.features.extend(
                                np.zeros(
                                    len(features_generator(Chem.MolFromSmiles("C")))
                                )
                            )
                    elif (
                        m[0] is not None
                        and m[1] is not None
                        and m[0].GetNumHeavyAtoms() > 0
                    ):
                        self.features.extend(features_generator(m[0]))
                    elif (
                        m[0] is not None
                        and m[1] is not None
                        and m[0].GetNumHeavyAtoms() == 0
                    ):
                        self.features.extend(
                            np.zeros(len(features_generator(Chem.MolFromSmiles("C"))))
                        )
            self.features = np.array(self.features)
        replace_token = 0
        if self.features is not None:
            self.features = np.where(
                np.isnan(self.features), replace_token, self.features
            )
        if self.atom_descriptors is not None:
            self.atom_descriptors = np.where(
                np.isnan(self.atom_descriptors), replace_token, self.atom_descriptors
            )
        if self.atom_features is not None:
            self.atom_features = np.where(
                np.isnan(self.atom_features), replace_token, self.atom_features
            )
        if self.bond_features is not None:
            self.bond_features = np.where(
                np.isnan(self.bond_features), replace_token, self.bond_features
            )
        self.raw_features, self.raw_targets = self.features, self.targets
        (self.raw_atom_descriptors, self.raw_atom_features, self.raw_bond_features) = (
            self.atom_descriptors,
            self.atom_features,
            self.bond_features,
        )

    @property
    def mol(self):
        """
        Gets the corresponding list of RDKit molecules for the corresponding SMILES list.
        """
        mol = make_mols(
            self.smiles,
            self.is_reaction_list,
            self.is_explicit_h_list,
            self.is_adding_hs_list,
        )
        if cache_mol():
            for s, m in zip(self.smiles, mol):
                SMILES_TO_MOL[s] = m
        return mol

    @property
    def number_of_molecules(self):
        """
        Gets the number of molecules in the :class:`MoleculeDatapoint`.

        return:
            The number of molecules.
        """
        return len(self.smiles)

    def set_features(self, features):
        """
        Sets the features of the molecule.

        Args:
            features: A 1D numpy array of features for the molecule.
        """
        self.features = features

    def set_atom_descriptors(self, atom_descriptors):
        """
        Sets the atom descriptors of the molecule.

        Args:
            atom_descriptors: A 1D numpy array of features for the molecule.
        """
        self.atom_descriptors = atom_descriptors

    def set_atom_features(self, atom_features):
        """
        Sets the atom features of the molecule.

        Args:
            atom_features: A 1D numpy array of features for the molecule.
        """
        self.atom_features = atom_features

    def set_bond_features(self, bond_features):
        """
        Sets the bond features of the molecule.

        Args:
            bond_features: A 1D numpy array of features for the molecule.
        """
        self.bond_features = bond_features

    def extend_features(self, features):
        """
        Extends the features of the molecule.

        Args:
            features: A 1D numpy array of extra features for the molecule.
        """
        self.features = (
            np.append(self.features, features)
            if self.features is not None
            else features
        )

    def num_tasks(self):
        """
        Returns the number of prediction tasks.

        return:
            The number of tasks.
        """
        return len(self.targets)

    def set_targets(self, targets):
        """
        Sets the targets of a molecule.

        Args:
            targets: A list of floats containing the targets.
        """
        self.targets = targets

    def reset_features_and_targets(self):
        """
        Resets the features (atom, bond, and molecule) and targets to their raw values.
        """
        self.features, self.targets = self.raw_features, self.raw_targets
        self.atom_descriptors, self.atom_features, self.bond_features = (
            self.raw_atom_descriptors,
            self.raw_atom_features,
            self.raw_bond_features,
        )


class MoleculeDataset(io.Dataset):
    """
    A class MoleculeDataset contains a list of class MoleculeDatapoints with access to their attributes.

    Args:
        data: A list of class MoleculeDatapoints
    """

    def __init__(self, data: List[MoleculeDatapoint]):
        self._data = data
        self._batch_graph = None
        self._random = Random()

    def smiles(self, flatten: bool = False):
        """
        Returns a list containing the SMILES list associated with each class MoleculeDatapoint.

        Args:
            flatten: Whether to flatten the returned SMILES to a list instead of a list of lists.
        return:
            A list of SMILES or a list of lists of SMILES, depending on :code:`flatten`.
        """
        if flatten:
            return [smiles for d in self._data for smiles in d.smiles]
        return [d.smiles for d in self._data]

    def mols(self, flatten: bool = False):
        """
        Returns a list of the RDKit molecules associated with each class MoleculeDatapoint.

        Args:
            flatten: Whether to flatten the returned RDKit molecules to a list instead of a list of lists.
        return:
            A list of SMILES or a list of lists of RDKit molecules, depending on :code:`flatten`.
        """
        if flatten:
            return [mol for d in self._data for mol in d.mol]
        return [d.mol for d in self._data]

    @property
    def number_of_molecules(self) -> int:
        """
        Gets the number of molecules in each class MoleculeDatapoint.

        return:
            The number of molecules.
        """
        return self._data[0].number_of_molecules if len(self._data) > 0 else None

    def batch_graph(self):
        """
        Constructs a class BatchMolGraph with the graph featurization of all the molecules.

        note:
           The class BatchMolGraph is cached in after the first time it is computed
           and is simply accessed upon subsequent calls to meth batch_graph. This means that if the underlying
           set of class MoleculeDatapoints changes, then the returned class BatchMolGraph
           will be incorrect for the underlying data.

        return: A list of BatchMolGraph containing the graph featurization of all the
                 molecules in each class MoleculeDatapoint.
        """
        if self._batch_graph is None:
            self._batch_graph = []
            mol_graphs = []
            for d in self._data:
                mol_graphs_list = []
                for s, m in zip(d.smiles, d.mol):
                    if s in SMILES_TO_GRAPH:
                        mol_graph = SMILES_TO_GRAPH[s]
                    else:
                        if len(d.smiles) > 1 and (
                            d.atom_features is not None or d.bond_features is not None
                        ):
                            raise NotImplementedError(
                                "Atom descriptors are currently only supported with one molecule per input (i.e., number_of_molecules = 1)."
                            )
                        mol_graph = MolGraph(
                            m,
                            d.atom_features,
                            d.bond_features,
                            overwrite_default_atom_features=d.overwrite_default_atom_features,
                            overwrite_default_bond_features=d.overwrite_default_bond_features,
                        )
                        if cache_graph():
                            SMILES_TO_GRAPH[s] = mol_graph
                    mol_graphs_list.append(mol_graph)
                mol_graphs.append(mol_graphs_list)
            self._batch_graph = [
                BatchMolGraph([g[i] for g in mol_graphs])
                for i in range(len(mol_graphs[0]))
            ]
        return self._batch_graph

    def features(self):
        """
        Returns the features associated with each molecule (if they exist).

        return:
            A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].features is None:
            return None
        return [d.features for d in self._data]

    def phase_features(self):
        """
        Returns the phase features associated with each molecule (if they exist).

        return:
            A list of 1D numpy arrays containing the phase features for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].phase_features is None:
            return None
        return [d.phase_features for d in self._data]

    def atom_features(self):
        """
        Returns the atom descriptors associated with each molecule (if they exit).

        return:
            A list of 2D numpy arrays containing the atom descriptors
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].atom_features is None:
            return None
        return [d.atom_features for d in self._data]

    def atom_descriptors(self):
        """
        Returns the atom descriptors associated with each molecule (if they exit).

        return:
            A list of 2D numpy arrays containing the atom descriptors
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].atom_descriptors is None:
            return None
        return [d.atom_descriptors for d in self._data]

    def bond_features(self):
        """
        Returns the bond features associated with each molecule (if they exit).

        return:
            A list of 2D numpy arrays containing the bond features
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].bond_features is None:
            return None
        return [d.bond_features for d in self._data]

    def data_weights(self):
        """
        Returns the loss weighting associated with each datapoint.
        """
        if not hasattr(self._data[0], "data_weight"):
            return [(1.0) for d in self._data]
        return [d.data_weight for d in self._data]

    def targets(self):
        """
        Returns the targets associated with each molecule.

        return:
            A list of lists of floats (or None) containing the targets.
        """
        return [d.targets for d in self._data]

    def mask(self):
        """
        Returns whether the targets associated with each molecule and task are present.

        return:
            A list of list of booleans associated with targets.
        """
        targets = self.targets()
        return [[(t is not None) for t in dt] for dt in targets]

    def gt_targets(self):
        """
        Returns indications of whether the targets associated with each molecule are greater-than inequalities.

        return:
            A list of lists of booleans indicating whether the targets in those positions are greater-than inequality targets.
        """
        if not hasattr(self._data[0], "gt_targets"):
            return None
        return [d.gt_targets for d in self._data]

    def lt_targets(self):
        """
        Returns indications of whether the targets associated with each molecule are less-than inequalities.

        return:
            A list of lists of booleans indicating whether the targets in those positions are less-than inequality targets.
        """
        if not hasattr(self._data[0], "lt_targets"):
            return None
        return [d.lt_targets for d in self._data]

    def num_tasks(self):
        """
        Returns the number of prediction tasks.

        return:
            The number of tasks.
        """
        return self._data[0].num_tasks() if len(self._data) > 0 else None

    def features_size(self):
        """
        Returns the size of the additional features vector associated with the molecules.

        return:
            The size of the additional features vector.
        """
        return (
            len(self._data[0].features)
            if len(self._data) > 0 and self._data[0].features is not None
            else None
        )

    def atom_descriptors_size(self):
        """
        Returns the size of custom additional atom descriptors vector associated with the molecules.

        return:
            The size of the additional atom descriptor vector.
        """
        return (
            len(self._data[0].atom_descriptors[0])
            if len(self._data) > 0 and self._data[0].atom_descriptors is not None
            else None
        )

    def atom_features_size(self):
        """
        Returns the size of custom additional atom features vector associated with the molecules.

        return:
            The size of the additional atom feature vector.
        """
        return (
            len(self._data[0].atom_features[0])
            if len(self._data) > 0 and self._data[0].atom_features is not None
            else None
        )

    def bond_features_size(self):
        """
        Returns the size of custom additional bond features vector associated with the molecules.

        return:
            The size of the additional bond feature vector.
        """
        return (
            len(self._data[0].bond_features[0])
            if len(self._data) > 0 and self._data[0].bond_features is not None
            else None
        )

    def normalize_features(
        self,
        scaler: StandardScaler = None,
        replace_nan_token: int = 0,
        scale_atom_descriptors: bool = False,
        scale_bond_features: bool = False,
    ):
        """
        Normalizes the features of the dataset using a class StandardScaler.
        The class StandardScaler subtracts the mean and divides by the standard deviation
        for each feature independently.
        If a StandardScaler is provided, it is used to perform the normalization.
        Otherwise, a class StandardScaler is first fit to the features in this dataset
        and is then used to perform the normalization.

        Args:
            scaler: A fitted class StandardScaler. If it is provided it is used,
                    otherwise a new class StandardScaler is first fitted to this
                    data and is then used.
            replace_nan_token: A token to use to replace NaN entries in the features.
            scale_atom_descriptors: If the features that need to be scaled are atom features rather than molecule.
            scale_bond_features: If the features that need to be scaled are bond descriptors rather than molecule.
        return:
            A fitted class StandardScaler. If a class StandardScaler is provided as a parameter,
            this is the same StandardScaler. Otherwise,this is a new class StandardScaler that has been fit on this dataset.
        """
        if (
            len(self._data) == 0
            or self._data[0].features is None
            and not scale_bond_features
            and not scale_atom_descriptors
        ):
            return None
        if scaler is None:
            if scale_atom_descriptors and self._data[0].atom_descriptors is not None:
                features = np.vstack([d.raw_atom_descriptors for d in self._data])
            elif scale_atom_descriptors and self._data[0].atom_features is not None:
                features = np.vstack([d.raw_atom_features for d in self._data])
            elif scale_bond_features:
                features = np.vstack([d.raw_bond_features for d in self._data])
            else:
                features = np.vstack([d.raw_features for d in self._data])
            scaler = StandardScaler(replace_nan_token=replace_nan_token)
            scaler.fit(features)
        if scale_atom_descriptors and self._data[0].atom_descriptors is not None:
            for d in self._data:
                d.set_atom_descriptors(scaler.transform(d.raw_atom_descriptors))
        elif scale_atom_descriptors and self._data[0].atom_features is not None:
            for d in self._data:
                d.set_atom_features(scaler.transform(d.raw_atom_features))
        elif scale_bond_features:
            for d in self._data:
                d.set_bond_features(scaler.transform(d.raw_bond_features))
        else:
            for d in self._data:
                d.set_features(scaler.transform(d.raw_features.reshape(1, -1))[0])
        return scaler

    def normalize_targets(self):
        """
        Normalizes the targets of the dataset using a class StandardScaler.
        The class StandardScaler subtracts the mean and divides by the standard deviation
        for each task independently.
        This should only be used for regression datasets.

        return:
            A class StandardScaler` fitted to the targets.
        """
        targets = [d.raw_targets for d in self._data]
        scaler = StandardScaler().fit(targets)
        scaled_targets = scaler.transform(targets).tolist()
        self.set_targets(scaled_targets)
        return scaler

    def set_targets(self, targets: List[List[Optional[float]]]):
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.

        Args:
            targets: A list of lists of floats (or None) containing targets for each molecule. This must be the
                        same length as the underlying dataset.
        """
        if not len(self._data) == len(targets):
            raise ValueError(
                f"number of molecules and targets must be of same length! num molecules: {len(self._data)}, num targets: {len(targets)}"
            )
        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i])

    def reset_features_and_targets(self):
        """
        Resets the features (atom, bond, and molecule) and targets to their raw values.
        """
        for d in self._data:
            d.reset_features_and_targets()

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e., the number of molecules).

        return:
            The length of the dataset.
        """
        return len(self._data)

    def __getitem__(self, item):
        """
        Gets one or more class MoleculeDatapoint s via an index or slice.

        Args:
            item: An index (int) or a slice object.
        return
            A class MoleculeDatapoint if an int is provided or a list of class MoleculeDatapoints
                 if a slice is provided.
        """
        return self._data[item]


class MoleculeSampler(io.Sampler):
    """
    A class MoleculeSampler samples data from a  class MoleculeDataset for a class MoleculeDataLoader.

    Args:
        class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                           and negative molecules). Set shuffle to True in order to get a random
                           subset of the larger class.
        shuffle: Whether to shuffle the data.
        seed: Random seed. Only needed if code shuffle is True.
    """

    def __init__(
        self,
        dataset: MoleculeDataset,
        class_balance: bool = False,
        shuffle: bool = False,
        seed: int = 0,
    ):
        super(io.Sampler, self).__init__()
        self.dataset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle
        self._random = Random(seed)
        if self.class_balance:
            indices = np.arange(len(dataset))
            has_active = np.array(
                [
                    any(target == 1 for target in datapoint.targets)
                    for datapoint in dataset
                ]
            )
            self.positive_indices = indices[has_active].tolist()
            self.negative_indices = indices[~has_active].tolist()
            self.length = 2 * min(
                len(self.positive_indices), len(self.negative_indices)
            )
        else:
            self.positive_indices = self.negative_indices = None
            self.length = len(self.dataset)

    def __iter__(self):
        """
        Creates an iterator over indices to sample.
        """
        if self.class_balance:
            if self.shuffle:
                self._random.shuffle(self.positive_indices)
                self._random.shuffle(self.negative_indices)
            indices = [
                index
                for pair in zip(self.positive_indices, self.negative_indices)
                for index in pair
            ]
        else:
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                self._random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        """
        Returns the number of indices that will be sampled.
        """
        return self.length


def construct_molecule_batch(data: List[MoleculeDatapoint]):
    """
    Constructs a :class:`MoleculeDataset` from a list of :class:`MoleculeDatapoint`\\ s.

    Additionally, precomputes the :class:`~chemprop.features.BatchMolGraph` for the constructed
    :class:`MoleculeDataset`.

    :param data: A list of :class:`MoleculeDatapoint`\\ s.
    :return: A :class:`MoleculeDataset` containing all the :class:`MoleculeDatapoint`\\ s.
    """

    data = MoleculeDataset(data)
    mol_graph = data.batch_graph()
    mol_graphs = [mol.get_components() for mol in mol_graph]
    features = data.features()
    targets = data.targets()
    if targets[0] is not None:
        masks = data.mask()
    else:
        masks = None
    atom_descriptors = data.atom_descriptors()
    atom_features = data.atom_features()
    bond_features = data.bond_features()
    data_weights = data.data_weights()
    lt_targets = data.lt_targets()
    gt_targets = data.gt_targets()
    return (
        mol_graphs,
        features,
        targets,
        masks,
        atom_descriptors,
        atom_features,
        bond_features,
        data_weights,
        lt_targets,
        gt_targets,
    )


class DataLoader(io.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
    ):

        if isinstance(dataset[0], (tuple, list)):
            return_list = True
        else:
            return_list = False

        super().__init__(
            dataset,
            feed_list=None,
            places=None,
            return_list=return_list,
            batch_sampler=batch_sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            num_workers=num_workers,
            use_buffer_reader=True,
            use_shared_memory=False,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
        if sampler is not None:
            self.batch_sampler.sampler = sampler


class MoleculeDataLoader(DataLoader):
    """
    A class MoleculeDataLoader  is a Paddle class DataLoader for loading a class MoleculeDataset.
    Args:
        dataset: The class MoleculeDataset containing the molecules to load.
        batch_size: Batch size.
        num_workers: Number of workers used to build batches.
        class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                       and negative molecules). Class balance is only available for single task
                       classification datasets. Set shuffle to True in order to get a random
                       subset of the larger class.
        shuffle: Whether to shuffle the data.
        seed: Random seed. Only needed if shuffle is True.
    """

    def __init__(
        self,
        dataset: MoleculeDataset,
        batch_size: int = 50,
        num_workers: int = 8,
        class_balance: bool = False,
        shuffle: bool = False,
        seed: int = 0,
    ):
        """ """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._class_balance = class_balance
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._context = "forkserver"
            self._timeout = 3600
        self._sampler = MoleculeSampler(
            dataset=self._dataset,
            class_balance=self._class_balance,
            shuffle=self._shuffle,
            seed=self._seed,
        )
        print("self._dataset:", len(dataset))
        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=construct_molecule_batch,
            multiprocessing_context=self._context,
            timeout=self._timeout,
        )

    @property
    def targets(self):
        """
        Returns the targets associated with each molecule.

        return:
            A list of lists of floats (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError(
                "Cannot safely extract targets when class balance or shuffle are enabled."
            )
        return [self._dataset[index].targets for index in self._sampler]

    @property
    def gt_targets(self):
        """
        Returns booleans for whether each target is an inequality rather than a value target, associated with each molecule.

        return:
            A list of lists of booleans (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError(
                "Cannot safely extract targets when class balance or shuffle are enabled."
            )
        if not hasattr(self._dataset[0], "gt_targets"):
            return None
        return [self._dataset[index].gt_targets for index in self._sampler]

    @property
    def lt_targets(self):
        """
        Returns booleans for whether each target is an inequality rather than a value target, associated with each molecule.

        return:
            A list of lists of booleans (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError(
                "Cannot safely extract targets when class balance or shuffle are enabled."
            )
        if not hasattr(self._dataset[0], "lt_targets"):
            return None
        return [self._dataset[index].lt_targets for index in self._sampler]

    @property
    def iter_size(self):
        """
        Returns the number of data points included in each full iteration through the class MoleculeDataLoader .
        """
        return len(self._sampler)

    def __iter__(self):
        """
        Creates an iterator which returns class MoleculeDatasets
        """
        return super(MoleculeDataLoader, self).__iter__()


def make_mols(
    smiles: List[str],
    reaction_list: List[bool],
    keep_h_list: List[bool],
    add_h_list: List[bool],
):
    """
    Builds a list of RDKit molecules (or a list of tuples of molecules if reaction is True) for a list of smiles.

    :param smiles: List of SMILES strings.
    :param reaction_list: List of booleans whether the SMILES strings are to be treated as a reaction.
    :param keep_h_list: List of booleans whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :param add_h_list: List of booleasn whether to add hydrogens to the input smiles.
    :return: List of RDKit molecules or list of tuple of molecules.
    """
    mol = []
    Featuri = Featurization()
    for s, reaction, keep_h, add_h in zip(
        smiles, reaction_list, keep_h_list, add_h_list
    ):
        if reaction:
            mol.append(
                SMILES_TO_MOL[s]
                if s in SMILES_TO_MOL
                else (
                    Featuri.make_mol(s.split(">")[0], keep_h, add_h),
                    Featuri.make_mol(s.split(">")[-1], keep_h, add_h),
                )
            )
        else:
            mol.append(
                SMILES_TO_MOL[s]
                if s in SMILES_TO_MOL
                else Featuri.make_mol(s, keep_h, add_h)
            )
    return mol


def chemprop_build_data_loader(
    smiles: list[str],
    fingerprints: np.ndarray = None,
    properties: list[int] = None,
    shuffle: bool = False,
    num_workers: int = 0,
) -> MoleculeDataLoader:
    """Builds a chemprop MoleculeDataLoader.

    :param smiles: A list of SMILES strings.
    :param fingerprints: A 2D array of molecular fingerprints (num_molecules, num_features).
    :param properties: A list of molecular properties (num_molecules,).
    :param shuffle: Whether to shuffle the data loader.
    :param num_workers: The number of workers for the data loader.
                        Zero workers needed for deterministic behavior and faster training/testing when CPU only.
    :return: A Chemprop data loader.
    """
    if fingerprints is None:
        fingerprints = [None] * len(smiles)

    if properties is None:
        properties = [None] * len(smiles)
    else:
        properties = [[float(prop)] for prop in properties]

    return MoleculeDataLoader(
        dataset=MoleculeDataset(
            [
                MoleculeDatapoint(
                    smiles=[smiles],
                    targets=prop,
                    features=fingerprint,
                )
                for smiles, fingerprint, prop in zip(smiles, fingerprints, properties)
            ]
        ),
        num_workers=num_workers,
        shuffle=shuffle,
    )


class MoleculeDatasetIter(io.IterableDataset):
    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        args,
        smiles: list[str],
        fingerprints: np.ndarray = None,
        properties: list[int] = None,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.args = args

        self.data_loader = chemprop_build_data_loader(
            smiles, fingerprints, properties, shuffle, num_workers
        )

    def __iter__(self):
        for batch in self.data_loader:
            (
                mol_batch,
                features_batch,
                target_batch,
                mask_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_features_batch,
                data_weights_batch,
                lt_target_batch,
                gt_target_batch,
            ) = batch

            mask = paddle.to_tensor(data=mask_batch, dtype="float32")
            targets = paddle.to_tensor(
                data=[[(0 if x is None else x) for x in tb] for tb in target_batch]
            )
            if self.args.target_weights is not None:
                target_weights = paddle.to_tensor(
                    data=self.args.target_weights
                ).unsqueeze(axis=0)
            else:
                target_weights = paddle.ones(shape=tuple(targets.shape)[1]).unsqueeze(
                    axis=0
                )
            data_weights = paddle.to_tensor(data=data_weights_batch).unsqueeze(axis=1)
            if self.args.loss_function == "bounded_mse":
                lt_target_batch = paddle.to_tensor(data=lt_target_batch)
                gt_target_batch = paddle.to_tensor(data=gt_target_batch)

            yield (
                {
                    self.input_keys[0]: mol_batch,
                    self.input_keys[1]: features_batch,
                    self.input_keys[2]: atom_descriptors_batch,
                    self.input_keys[3]: atom_features_batch,
                    self.input_keys[4]: bond_features_batch,
                },
                {
                    self.label_keys[0]: targets,
                    self.label_keys[1]: data_weights,
                    self.label_keys[2]: mask,
                    self.label_keys[3]: target_weights,
                },
                {},
            )
