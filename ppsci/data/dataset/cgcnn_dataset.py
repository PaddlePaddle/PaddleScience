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

import csv
import functools
import json
import os
import random
import warnings
from typing import Tuple

import numpy as np
import paddle
from paddle import io

try:
    from pymatgen.core.structure import Structure
except ModuleNotFoundError:
    pass


def collate_pool(dataset_list):

    """
    Collate a list of data and return a batch for predicting crystal properties.

    Args:
        dataset_list (list): A list of tuples for each data point containing:
            - atom_fea (paddle.Tensor): Shape (n_i, atom_fea_len).
            - nbr_fea (paddle.Tensor): Shape (n_i, M, nbr_fea_len).
            - nbr_fea_idx (paddle.Tensor): Shape (n_i, M).
            - target (paddle.Tensor): Shape (1,).
            - cif_id (str or int).

    Returns:
        tuple: Contains the following:
            - batch_atom_fea (paddle.Tensor): Shape (N, orig_atom_fea_len). Atom features from atom type.
            - batch_nbr_fea (paddle.Tensor): Shape (N, M, nbr_fea_len). Bond features of each atom's M neighbors.
            - batch_nbr_fea_idx (paddle.Tensor): Shape (N, M). Indices of M neighbors of each atom.
            - crystal_atom_idx (list): List of paddle.Tensor of length N0. Mapping from the crystal idx to atom idx.
            - target (paddle.Tensor): Shape (N, 1). Target value for prediction.
            - batch_cif_ids (list): List of CIF IDs.

    Notes:
        - N = sum(n_i); N0 = sum(i)
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, item in enumerate(dataset_list):
        input: Tuple[np.ndarray, np.ndarray, np.ndarray] = item[0]["i"]
        label = item[1]["l"]
        id = item[2]["c"]
        atom_fea, nbr_fea, nbr_fea_idx = input
        target = label
        cif_id = id
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = np.arange(n_i, dtype="int64") + int(base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    # Debugging: print shapes of the tensors to ensure they are consistent
    # print("Shapes of batch_atom_fea:", [x.shape for x in batch_atom_fea])
    # print("Shapes of batch_nbr_fea:", [x.shape for x in batch_nbr_fea])
    # print("Shapes of batch_nbr_fea_idx:", [x.shape for x in batch_nbr_fea_idx])
    # Ensure all tensors in the lists have consistent shapes before concatenation
    batch_atom_fea = np.concatenate(batch_atom_fea, axis=0)
    batch_nbr_fea = np.concatenate(batch_nbr_fea, axis=0)
    batch_nbr_fea_idx = np.concatenate(batch_nbr_fea_idx, axis=0)
    return (
        {
            "i": (
                np.array(batch_atom_fea, dtype="float32"),
                np.array(batch_nbr_fea, dtype="float32"),
                np.array(batch_nbr_fea_idx),
                [np.array(crys_idx) for crys_idx in crystal_atom_idx],
            )
        },
        {"l": np.array(np.stack(batch_target, axis=0))},
        {"c": batch_cif_ids},
    )


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Args:
        dmin (float): Minimum interatomic distance.
        dmax (float): Maximum interatomic distance.
        step (float): Step size for the Gaussian filter.
    """

    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian distance filter to a numpy distance array.

        Args:
            distance (np.array): n-dimensional distance matrix of any shape.

        Returns:
            np.array: Expanded distance matrix with the last dimension of length len(self.filter).
        """

        return np.exp(
            -((distances[..., np.newaxis] - self.filter) ** 2) / self.var**2
        )


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {
            idx: atom_type for atom_type, idx in self._embedding.items()
        }

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, "_decodedict"):
            self._decodedict = {
                idx: atom_type for atom_type, idx in self._embedding.items()
            }
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a Python dictionary mapping from element number to a list representing the feature vector of the element.

    Args:
        elem_embedding_file (str): The path to the .json file.
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(io.Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each  element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Args
        root_dir (str): The path to the root directory of the dataset
        max_num_nbr (int): The maximum number of neighbors while constructing the crystal graph
        radius (float): The cutoff radius for searching neighbors
        dmin (float): The minimum distance for constructing GaussianDistance
        step (float): The step size for constructing GaussianDistance
        random_seed (int): Random seed for shuffling the dataset


    Returns
        atom_fea (paddle.Tensor): Shape (n_i, atom_fea_len)
        nbr_fea (paddle.Tensor): Shape (n_i, M, nbr_fea_len)
        nbr_fea_idx (paddle.Tensor): Shape (n_i, M)
        target (paddle.Tensor): Shape (1, )
        cif_id (str or int)

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.CGCNNDataset(
        ...     "file_path": "/path/to/CGCNNDataset",
        ...     "input_keys": "i",
        ...     "label_keys": "l",
        ...     "id_keys": "c",
        ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        root_dir: str,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        id_keys: Tuple[str, ...],
        max_num_nbr: int = 12,
        radius: int = 8,
        dmin: int = 0,
        step: float = 0.2,
        random_seed: int = 123,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.id_keys = id_keys
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), "root_dir does not exist!"
        id_prop_file = os.path.join(self.root_dir, "id_prop.csv")
        assert os.path.exists(id_prop_file), "id_prop.csv does not exist!"
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, "atom_init.json")
        assert os.path.exists(atom_init_file), f"{atom_init_file} does not exist!"
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.raw_data = [self.get(i) for i in range(len(self))]

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        return (
            {self.input_keys[0]: self.raw_data[idx][0]},
            {self.label_keys[0]: self.raw_data[idx][1]},
            {self.id_keys[0]: self.raw_data[idx][2]},
        )

    def get(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id + ".cif"))
        atom_fea = np.vstack(
            [
                self.ari.get_atom_fea(crystal[i].specie.number)
                for i in range(len(crystal))
            ]
        )
        atom_fea = paddle.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    "{} not find enough neighbors to build graph. "
                    "If it happens frequently, consider increase "
                    "radius.".format(cif_id)
                )
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr))
                )
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr))
                    + [self.radius + 1.0] * (self.max_num_nbr - len(nbr))
                )
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[: self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[: self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = np.array(atom_fea)
        nbr_fea = np.array(nbr_fea)
        nbr_fea_idx = np.array(nbr_fea_idx, dtype="int64")
        target = np.array([float(target)], dtype="float32")
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id
