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

# Copyright 2020 Chengxi Zang

from __future__ import annotations

import os
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from paddle import io
from tqdm import tqdm

from ppsci.utils import logger

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolops
except ModuleNotFoundError:
    pass


class MolGraph:
    """
    Args:
        max_atoms (int): Max number of atoms for each molecule, if the
            number of atoms is more than this value, this data is simply
            ignored.
            Setting negative value indicates no limit for max atoms.
        out_size (int): It specifies the size of array returned by
            `get_input_features`.
            If the number of atoms in the molecule is less than this value,
            the returned arrays is padded to have fixed size.
            Setting negative value indicates do not pad returned array.
        add_Hs (bool): If True, implicit Hs are added.
        kekulize (bool): If True, Kekulizes the molecule.
    """

    def __init__(self, max_atoms=-1, out_size=-1, add_Hs=False, kekulize=False):
        super(MolGraph, self).__init__()
        self.add_Hs = add_Hs
        self.kekulize = kekulize
        if max_atoms >= 0 and out_size >= 0 and max_atoms > out_size:
            raise ValueError(
                "max_atoms {} must be less or equal to out_size {}".format(
                    max_atoms, out_size
                )
            )
        self.max_atoms = max_atoms
        self.out_size = out_size

    def get_input_features(self, mol):
        """
        get input features
        Args:
            mol (Mol): mol instance

        Returns:
            (tuple): (`atom`, `adj`)

        """
        self.type_check_num_atoms(mol, self.max_atoms)
        atom_array = self.construct_atomic_number_array(mol, out_size=self.out_size)
        adj_array = self.construct_discrete_edge_matrix(mol, out_size=self.out_size)
        return atom_array, adj_array

    def prepare_smiles_and_mol(self, mol):
        """Prepare `smiles` and `mol` used in following preprocessing.
        This method is called before `get_input_features` is called, by parser
        class.
        This method may be overriden to support custom `smile`/`mol` extraction
        Args:
            mol (mol): mol instance

        Returns (tuple): (`smiles`, `mol`)
        """
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        mol = Chem.MolFromSmiles(canonical_smiles)
        if self.add_Hs:
            mol = Chem.AddHs(mol)
        if self.kekulize:
            Chem.Kekulize(mol)
        return canonical_smiles, mol

    def get_label(self, mol, label_names=None):
        """Extracts label information from a molecule.
        This method extracts properties whose keys are
        specified by ``label_names`` from a molecule ``mol``
        and returns these values as a list.
        The order of the values is same as that of ``label_names``.
        If the molecule does not have a
        property with some label, this function fills the corresponding
        index of the returned list with ``None``.

        Args:
            mol (rdkit.Chem.Mol): molecule whose features to be extracted
            label_names (None or iterable): list of label names.

        Returns:
            list of str: label information. Its length is equal to
            that of ``label_names``. If ``label_names`` is ``None``,
            this function returns an empty list.

        """
        if label_names is None:
            return []
        label_list = []
        for label_name in label_names:
            if mol.HasProp(label_name):
                label_list.append(mol.GetProp(label_name))
            else:
                label_list.append(None)
        return label_list

    def type_check_num_atoms(self, mol, num_max_atoms=-1):
        """Check number of atoms in `mol` does not exceed `num_max_atoms`
        If number of atoms in `mol` exceeds the number `num_max_atoms`, it will
        raise `MolGraphError` exception.

        Args:
            mol (Mol):
            num_max_atoms (int): If negative value is set, not check number of
                atoms.

        """
        num_atoms = mol.GetNumAtoms()
        if num_max_atoms >= 0 and num_atoms > num_max_atoms:
            raise MolGraphError(
                "Number of atoms in mol {} exceeds num_max_atoms {}".format(
                    num_atoms, num_max_atoms
                )
            )

    def construct_atomic_number_array(self, mol, out_size=-1):
        """Returns atomic numbers of atoms consisting a molecule.

        Args:
            mol (rdkit.Chem.Mol): Input molecule.
            out_size (int): The size of returned array.
                If this option is negative, it does not take any effect.
                Otherwise, it must be larger than the number of atoms
                in the input molecules. In that case, the tail of
                the array is padded with zeros.

        Returns:
            numpy.ndarray: an array consisting of atomic numbers
                of atoms in the molecule.
        """
        atom_list = [a.GetAtomicNum() for a in mol.GetAtoms()]
        n_atom = len(atom_list)
        if out_size < 0:
            return np.array(atom_list, dtype=np.int32)
        elif out_size >= n_atom:
            atom_array = np.zeros(out_size, dtype=np.int32)
            atom_array[:n_atom] = np.array(atom_list, dtype=np.int32)
            return atom_array
        else:
            raise ValueError(
                "`out_size` (={}) must be negative or larger than or equal to the number of atoms in the input molecules (={}).".format(
                    out_size, n_atom
                )
            )

    def construct_adj_matrix(self, mol, out_size=-1, self_connection=True):
        """Returns the adjacent matrix of the given molecule.

        This function returns the adjacent matrix of the given molecule.
        Contrary to the specification of
        :func:`rdkit.Chem.rdmolops.GetAdjacencyMatrix`,
        The diagonal entries of the returned matrix are all-one.

        Args:
            mol (rdkit.Chem.Mol): Input molecule.
            out_size (int): The size of the returned matrix.
                If this option is negative, it does not take any effect.
                Otherwise, it must be larger than the number of atoms
                in the input molecules. In that case, the adjacent
                matrix is expanded and zeros are padded to right
                columns and bottom rows.
            self_connection (bool): Add self connection or not.
                If True, diagonal element of adjacency matrix is filled with 1.

        Returns:
            adj_array (numpy.ndarray): The adjacent matrix of the input molecule.
                It is 2-dimensional array with shape (atoms1, atoms2), where
                atoms1 & atoms2 represent from and to of the edge respectively.
                If ``out_size`` is non-negative, the returned
                its size is equal to that value. Otherwise,
                it is equal to the number of atoms in the the molecule.
        """
        adj = rdmolops.GetAdjacencyMatrix(mol)
        s0, s1 = tuple(adj.shape)
        if s0 != s1:
            raise ValueError(
                "The adjacent matrix of the input moleculehas an invalid shape: ({}, {}). It must be square.".format(
                    s0, s1
                )
            )
        if self_connection:
            adj = adj + np.eye(s0)
        if out_size < 0:
            adj_array = adj.astype(np.float32)
        elif out_size >= s0:
            adj_array = np.zeros((out_size, out_size), dtype=np.float32)
            adj_array[:s0, :s1] = adj
        else:
            raise ValueError(
                "`out_size` (={}) must be negative or larger than or equal to the number of atoms in the input molecules (={}).".format(
                    out_size, s0
                )
            )
        return adj_array

    def construct_discrete_edge_matrix(self, mol, out_size=-1):
        """Returns the edge-type dependent adjacency matrix of the given molecule.

        Args:
            mol (rdkit.Chem.Mol): Input molecule.
            out_size (int): The size of the returned matrix.
                If this option is negative, it does not take any effect.
                Otherwise, it must be larger than the number of atoms
                in the input molecules. In that case, the adjacent
                matrix is expanded and zeros are padded to right
                columns and bottom rows.

        Returns:
            adj_array (numpy.ndarray): The adjacent matrix of the input molecule.
                It is 3-dimensional array with shape (edge_type, atoms1, atoms2),
                where edge_type represents the bond type,
                atoms1 & atoms2 represent from and to of the edge respectively.
                If ``out_size`` is non-negative, its size is equal to that value.
                Otherwise, it is equal to the number of atoms in the the molecule.
        """
        if mol is None:
            raise MolGraphError("mol is None")
        N = mol.GetNumAtoms()
        if out_size < 0:
            size = N
        elif out_size >= N:
            size = out_size
        else:
            raise ValueError(
                "out_size {} is smaller than number of atoms in mol {}".format(
                    out_size, N
                )
            )
        adjs = np.zeros((4, size, size), dtype=np.float32)
        bond_type_to_channel = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 1,
            Chem.BondType.TRIPLE: 2,
            Chem.BondType.AROMATIC: 3,
        }
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            ch = bond_type_to_channel[bond_type]
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adjs[ch, i, j] = 1.0
            adjs[ch, j, i] = 1.0
        return adjs


class MolGraphError(Exception):
    pass


class MOlFLOWDataset(io.Dataset):
    """Class for moflow qm9 and zinc250k Dataset of a tuple of datasets.

    It combines multiple datasets into one dataset. Each example is represented
    by a tuple whose ``i``-th item corresponds to the i-th dataset.
    And each ``i``-th dataset is expected to be an instance of numpy.ndarray.

    Args:
    file_path (str): Data set path.
    data_name (str): Data name,  "qm9" or "zinc250k"
    valid_idx (List[int, ...]): Data for validate
    mode (str): "train" or "eval", output Data
    input_keys (Tuple[str, ...]): Input keys, such as ("nodes","edges",).
    label_keys (Tuple[str, ...]): labels (str or list or None) .
    smiles_col (str): smiles column
    weight_dict (Optional[Dict[str, Union[Callable, float]]]): Define the weight of each constraint variable. Defaults to None.
    transform_fn: An optional function applied to an item bofre returning
    """

    # Whether support batch indexing for speeding up fetching process.
    batch_index: bool = True

    def __init__(
        self,
        file_path: str,
        data_name: str,
        valid_idx: List[int, ...],
        mode: str,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        smiles_col: str,
        weight_dict: Optional[Dict[str, float]] = None,
        transform_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.file_path = file_path
        self.data_name = data_name
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.smiles_col = smiles_col
        self.weight_dict = weight_dict

        if data_name == "qm9":
            max_atoms = 9
        elif data_name == "zinc250k":
            max_atoms = 38

        self.molgraph = MolGraph(out_size=max_atoms, kekulize=True)
        self.logger = logger
        # read and deal data from file
        inputs, labels = self.load_csv_file(file_path, data_name + ".csv")
        train_idx = [t for t in range(len(inputs[0])) if t not in valid_idx]
        self.train_idx = train_idx
        #  data train or test
        if mode == "train":
            inputs = [
                np.array(list(io.Subset(dataset=in_put, indices=train_idx)))
                for in_put in inputs
            ]
            labels = np.array(list(io.Subset(dataset=labels, indices=train_idx)))
        elif mode == "eval":
            inputs = [
                np.array(list(io.Subset(dataset=in_put, indices=valid_idx)))
                for in_put in inputs
            ]
            labels = np.array(list(io.Subset(dataset=labels, indices=valid_idx)))

        # fetch input data
        self.input = {key: inputs[i] for i, key in enumerate(self.input_keys)}
        # fetch label data
        self.label = {"label": labels}

        self.logger.message(
            "Dataload finished. MODE {}, inputs {}, labelS {}".format(
                mode,
                len(next(iter(self.input.values()))),
                len(next(iter(self.label.values()))),
            )
        )

        self._length = len(next(iter(self.input.values())))
        self.transform = transform_fn

    def __getitem__(self, index: int):
        input_item = {key: value[index] for key, value in self.input.items()}
        label_item = {key: value[index] for key, value in self.label.items()}

        if self.transform:
            input_item, label_item = self.transform_func(input_item, label_item)

        return (input_item, label_item, {})

    def __len__(self):
        return self._length

    def load_csv_file(self, path: str, name: str):
        """Parse DataFrame using `MolGraph` and prepare a dataset instance
        Labels are extracted from `labels` columns and input features are
        extracted from smiles information in `smiles` column.
        """
        file = os.path.join(path, name)
        df = pd.read_csv(file, index_col=0)
        all_nodes = []
        all_edges = []
        # inputs = []

        total_count = df.shape[0]
        fail_count = 0
        success_count = 0
        if isinstance(self.molgraph, MolGraph):
            for smiles in tqdm(df[self.smiles_col], total=df.shape[0]):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        fail_count += 1
                        continue
                    canonical_smiles, mol = self.molgraph.prepare_smiles_and_mol(mol)
                    nodes, edges = self.molgraph.get_input_features(mol)

                except MolGraphError as e:
                    fail_count += 1
                    self.logger.warning(
                        "parse(), type: {}, {}".format(type(e).__name__, e.args)
                    )
                    continue
                except Exception as e:
                    self.logger.warning(
                        "parse(), type: {}, {}".format(type(e).__name__, e.args)
                    )
                    fail_count += 1
                    continue
                # raw_data = misc.convert_to_dict(np.array([nodes, edges]), self.input_keys)

                all_nodes.append(nodes)
                all_edges.append(edges)
                # inputs.append(raw_data)

                success_count += 1

            labels = np.array(
                [*(df[label_col].values for label_col in self.label_keys)]
            ).T
            result = [np.array(all_nodes), np.array(all_edges)], labels
            self.logger.message(
                "Preprocess finished. FAIL {}, SUCCESS {}, TOTAL {}".format(
                    fail_count, success_count, total_count
                )
            )
        else:
            raise NotImplementedError

        return result

    def transform_func(self, data_dict, label_dict):
        items = []
        length = len(next(iter(data_dict.values())))
        for idx in range(length):
            input_item = [value[idx] for key, value in data_dict.items()]
            label_item = [value[idx] for key, value in label_dict.items()]
            item = input_item + label_item
            if self.transform:
                item = self.transform(item)
            items.append(item)
        items = np.array(items, dtype=object).T

        data_dict = {key: np.stack(items[i], axis=0) for i, key in enumerate(data_dict)}
        label_dict = {key: np.vstack(item[2]) for key in label_dict}

        return data_dict, label_dict
