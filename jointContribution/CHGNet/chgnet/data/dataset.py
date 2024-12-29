from __future__ import annotations

import functools
import os
import random
import warnings
from typing import TYPE_CHECKING

import numpy as np
import paddle
from chgnet import utils
from chgnet.graph import CrystalGraph
from chgnet.graph import CrystalGraphConverter
from pymatgen.core.structure import Structure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chgnet import TrainTask
    from typing_extensions import Self
warnings.filterwarnings("ignore")
DTYPE = "float32"


class StructureData(paddle.io.Dataset):
    """A simple paddle Dataset of structures."""

    def __init__(
        self,
        structures: list[Structure],
        energies: list[float],
        forces: list[Sequence[Sequence[float]]],
        *,
        stresses: (list[Sequence[Sequence[float]]] | None) = None,
        magmoms: (list[Sequence[Sequence[float]]] | None) = None,
        structure_ids: (list | None) = None,
        graph_converter: (CrystalGraphConverter | None) = None,
        shuffle: bool = True,
    ) -> None:
        """Initialize the dataset.

        Args:
            structures (list[dict]): pymatgen Structure objects.
            energies (list[float]): [data_size, 1]
            forces (list[list[float]]): [data_size, n_atoms, 3]
            stresses (list[list[float]], optional): [data_size, 3, 3]
                Default = None
            magmoms (list[list[float]], optional): [data_size, n_atoms, 1]
                Default = None
            structure_ids (list, optional): a list of ids to track the structures
                Default = None
            graph_converter (CrystalGraphConverter, optional): Converts the structures
                to graphs. If None, it will be set to CHGNet 0.3.0 converter
                with AtomGraph cutoff = 6A.
            shuffle (bool): whether to shuffle the sequence of dataset
                Default = True

        Raises:
            RuntimeError: if the length of structures and labels (energies, forces,
                stresses, magmoms) are not equal.
        """
        for idx, struct in enumerate(structures):
            if not isinstance(struct, Structure):
                raise TypeError(f"{idx} is not a pymatgen Structure object: {struct}")
        for name in "energies forces stresses magmoms structure_ids".split():
            labels = locals()[name]
            if labels is not None and len(labels) != len(structures):
                raise RuntimeError(
                    f"Inconsistent number of structures and labels: len(structures)={len(structures)!r}, len({name})={len(labels)}"
                )
        self.structures = structures
        self.energies = energies
        self.forces = forces
        self.stresses = stresses
        self.magmoms = magmoms
        self.structure_ids = structure_ids
        self.keys = np.arange(len(structures))
        if shuffle:
            random.shuffle(self.keys)
        print(f"{type(self).__name__} imported {len(structures):,} structures")
        self.graph_converter = graph_converter or CrystalGraphConverter(
            atom_graph_cutoff=6, bond_graph_cutoff=3
        )
        self.failed_idx: list[int] = []
        self.failed_graph_id: dict[str, str] = {}

    @classmethod
    def from_vasp(
        cls,
        file_root: str,
        *,
        check_electronic_convergence: bool = True,
        save_path: (str | None) = None,
        graph_converter: (CrystalGraphConverter | None) = None,
        shuffle: bool = True,
    ) -> Self:
        """Parse VASP output files into structures and labels and feed into the dataset.

        Args:
            file_root (str): the directory of the VASP calculation outputs
            check_electronic_convergence (bool): if set to True, this function will
                raise Exception to VASP calculation that did not achieve
                electronic convergence.
                Default = True
            save_path (str): path to save the parsed VASP labels
                Default = None
            graph_converter (CrystalGraphConverter, optional): Converts the structures
                to graphs. If None, it will be set to CHGNet 0.3.0 converter
                with AtomGraph cutoff = 6A.
            shuffle (bool): whether to shuffle the sequence of dataset
                Default = True
        """
        result_dict = utils.parse_vasp_dir(
            base_dir=file_root,
            check_electronic_convergence=check_electronic_convergence,
            save_path=save_path,
        )
        return cls(
            structures=result_dict["structure"],
            energies=result_dict["energy_per_atom"],
            forces=result_dict["force"],
            stresses=None
            if result_dict["stress"] in [None, []]
            else result_dict["stress"],
            magmoms=None
            if result_dict["magmom"] in [None, []]
            else result_dict["magmom"],
            structure_ids=np.arange(len(result_dict["structure"])),
            graph_converter=graph_converter,
            shuffle=shuffle,
        )

    def __len__(self) -> int:
        """Get the number of structures in this dataset."""
        return len(self.keys)

    @functools.cache
    def __getitem__(self, idx: int) -> tuple[CrystalGraph, dict]:
        """Get one graph for a structure in this dataset.

        Args:
            idx (int): Index of the structure

        Returns:
            crystal_graph (CrystalGraph): graph of the crystal structure
            targets (dict): list of targets. i.e. energy, force, stress
        """
        if idx not in self.failed_idx:
            graph_id = self.keys[idx]
            try:
                struct = self.structures[graph_id]
                if self.structure_ids is not None:
                    mp_id = self.structure_ids[graph_id]
                else:
                    mp_id = graph_id
                crystal_graph = self.graph_converter(
                    struct, graph_id=graph_id, mp_id=mp_id
                )
                targets = {
                    "e": paddle.to_tensor(data=self.energies[graph_id], dtype=DTYPE),
                    "f": paddle.to_tensor(data=self.forces[graph_id], dtype=DTYPE),
                }
                if self.stresses is not None:
                    targets["s"] = (
                        paddle.to_tensor(data=self.stresses[graph_id], dtype=DTYPE)
                        * -0.1
                    )
                if self.magmoms is not None:
                    mag = self.magmoms[graph_id]
                    if mag is None:
                        targets["m"] = None
                    else:
                        targets["m"] = paddle.abs(
                            x=paddle.to_tensor(data=mag, dtype=DTYPE)
                        )
                return crystal_graph, targets
            except Exception:
                struct = self.structures[graph_id]
                self.failed_graph_id[graph_id] = struct.composition.formula
                self.failed_idx.append(idx)
                idx = random.randint(0, len(self) - 1)
                return self.__getitem__(idx)
        else:
            idx = random.randint(0, len(self) - 1)
            return self.__getitem__(idx)


class CIFData(paddle.io.Dataset):
    """A dataset from CIFs."""

    def __init__(
        self,
        cif_path: str,
        *,
        labels: (str | dict) = "labels.json",
        targets: TrainTask = "efsm",
        graph_converter: (CrystalGraphConverter | None) = None,
        energy_key: str = "energy_per_atom",
        force_key: str = "force",
        stress_key: str = "stress",
        magmom_key: str = "magmom",
        shuffle: bool = True,
    ) -> None:
        """Initialize the dataset from a directory containing CIFs.

        Args:
            cif_path (str): path that contain all the graphs, labels.json
            labels (str, dict): the path or dictionary of labels
            targets ("ef" | "efs" | "efm" | "efsm"): The training targets.
                Default = "efsm"
            graph_converter (CrystalGraphConverter, optional): Converts the structures
                to graphs. If None, it will be set to CHGNet 0.3.0 converter
                with AtomGraph cutoff = 6A.
            energy_key (str, optional): the key of energy in the labels.
                Default = "energy_per_atom"
            force_key (str, optional): the key of force in the labels.
                Default = "force"
            stress_key (str, optional): the key of stress in the labels.
                Default = "stress"
            magmom_key (str, optional): the key of magmom in the labels.
                Default = "magmom"
            shuffle (bool): whether to shuffle the sequence of dataset
                Default = True
        """
        self.data_dir = cif_path
        self.data = utils.read_json(os.path.join(cif_path, labels))
        self.cif_ids = list(self.data)
        if shuffle:
            random.shuffle(self.cif_ids)
        print(f"{cif_path}: {len(self.cif_ids):,} structures imported")
        self.graph_converter = graph_converter or CrystalGraphConverter(
            atom_graph_cutoff=6, bond_graph_cutoff=3
        )
        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key
        self.magmom_key = magmom_key
        self.targets = targets
        self.failed_idx: list[int] = []
        self.failed_graph_id: dict[str, str] = {}

    def __len__(self) -> int:
        """Get the number of structures in this dataset."""
        return len(self.cif_ids)

    @functools.cache
    def __getitem__(self, idx: int) -> tuple[CrystalGraph, dict[str, paddle.Tensor]]:
        """Get one item in the dataset.

        Returns:
            tuple[CrystalGraph, dict[str, Tensor]]: graph of the crystal structure
                and dict of targets i.e. energy, force, stress
        """
        if idx not in self.failed_idx:
            try:
                graph_id = self.cif_ids[idx]
                mp_id = self.data[graph_id].get("material_id", graph_id)
                structure = Structure.from_file(
                    os.path.join(self.data_dir, f"{graph_id}.cif")
                )
                crystal_graph = self.graph_converter(
                    structure, graph_id=graph_id, mp_id=mp_id
                )
                targets = {}
                for key in self.targets:
                    if key == "e":
                        energy = self.data[graph_id][self.energy_key]
                        targets["e"] = paddle.to_tensor(data=energy, dtype=DTYPE)
                    elif key == "f":
                        force = self.data[graph_id][self.force_key]
                        targets["f"] = paddle.to_tensor(data=force, dtype=DTYPE)
                    elif key == "s":
                        stress = self.data[graph_id][self.stress_key]
                        targets["s"] = paddle.to_tensor(data=stress, dtype=DTYPE) * -0.1
                    elif key == "m":
                        mag = self.data[graph_id][self.magmom_key]
                        targets["m"] = paddle.abs(
                            x=paddle.to_tensor(data=mag, dtype=DTYPE)
                        )
                return crystal_graph, targets
            except Exception:
                try:
                    graph_id = self.cif_ids[idx]
                except IndexError:
                    print(idx, len(self.cif_ids))
                structure = Structure.from_file(
                    os.path.join(self.data_dir, f"{graph_id}.cif")
                )
                self.failed_graph_id[graph_id] = structure.composition.formula
                self.failed_idx.append(idx)
                idx = random.randint(0, len(self) - 1)
                return self.__getitem__(idx)
        else:
            idx = random.randint(0, len(self) - 1)
            return self.__getitem__(idx)


class GraphData(paddle.io.Dataset):
    """A dataset of graphs. This is compatible with the graph.pt documents made by
    make_graphs.py. We recommend you to use the dataset to avoid graph conversion steps.
    """

    def __init__(
        self,
        graph_path: str,
        *,
        labels: (str | dict) = "labels.json",
        targets: TrainTask = "efsm",
        exclude: (str | list | None) = None,
        energy_key: str = "energy_per_atom",
        force_key: str = "force",
        stress_key: str = "stress",
        magmom_key: str = "magmom",
        shuffle: bool = True,
    ) -> None:
        """Initialize the dataset from a directory containing saved crystal graphs.

        Args:
            graph_path (str): path that contain all the graphs, labels.json
            labels (str, dict): the path or dictionary of labels.
                Default = "labels.json"
            targets ("ef" | "efs" | "efm" | "efsm"): The training targets.
                Default = "efsm"
            exclude (str, list | None): the path or list of excluded graphs.
                Default = None
            energy_key (str, optional): the key of energy in the labels.
                Default = "energy_per_atom"
            force_key (str, optional): the key of force in the labels.
                Default = "force"
            stress_key (str, optional): the key of stress in the labels.
                Default = "stress"
            magmom_key (str, optional): the key of magmom in the labels.
                Default = "magmom"
            shuffle (bool): whether to shuffle the sequence of dataset
                Default = True
        """
        self.graph_path = graph_path
        if isinstance(labels, str):
            labels = os.path.join(graph_path, labels)
            print(f"Importing: {labels}")
            self.labels = utils.read_json(labels)
        elif isinstance(labels, dict):
            self.labels = labels
        if isinstance(exclude, str):
            self.excluded_graph = utils.read_json(exclude)
        elif isinstance(exclude, list):
            self.excluded_graph = exclude
        else:
            self.excluded_graph = []
        self.keys = [
            (mp_id, graph_id) for mp_id, dic in self.labels.items() for graph_id in dic
        ]
        if shuffle:
            random.shuffle(self.keys)
        print(f"{len(self.labels)} mp_ids, {len(self)} frames imported")
        if self.excluded_graph is not None:
            print(f"{len(self.excluded_graph)} graphs are pre-excluded")
        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key
        self.magmom_key = magmom_key
        self.targets = targets
        self.failed_idx: list[int] = []
        self.failed_graph_id: list[str] = []

    def __len__(self) -> int:
        """Get the number of graphs in this dataset."""
        return len(self.keys)

    def __getitem__(self, idx) -> tuple[CrystalGraph, dict[str, paddle.Tensor]]:
        """Get one item in the dataset.

        Returns:
            crystal_graph (CrystalGraph): graph of the crystal structure
            targets (dict): dictionary of targets. i.e. energy, force, stress, magmom
        """
        if idx not in self.failed_idx:
            mp_id, graph_id = self.keys[idx]
            if [mp_id, graph_id] in self.excluded_graph:
                self.failed_graph_id.append(graph_id)
                self.failed_idx.append(idx)
                idx = random.randint(0, len(self) - 1)
                return self.__getitem__(idx)
            try:
                graph_path = os.path.join(self.graph_path, f"{graph_id}.pt")
                crystal_graph = CrystalGraph.from_file(graph_path)
                targets = {}
                for key in self.targets:
                    if key == "e":
                        energy = self.labels[mp_id][graph_id][self.energy_key]
                        targets["e"] = paddle.to_tensor(data=energy, dtype=DTYPE)
                    elif key == "f":
                        force = self.labels[mp_id][graph_id][self.force_key]
                        targets["f"] = paddle.to_tensor(data=force, dtype=DTYPE)
                    elif key == "s":
                        stress = self.labels[mp_id][graph_id][self.stress_key]
                        targets["s"] = paddle.to_tensor(data=stress, dtype=DTYPE) * -0.1
                    elif key == "m":
                        mag = self.labels[mp_id][graph_id][self.magmom_key]
                        if mag is None:
                            targets["m"] = None
                        else:
                            targets["m"] = paddle.abs(
                                x=paddle.to_tensor(data=mag, dtype=DTYPE)
                            )
                return crystal_graph, targets
            except Exception:
                self.failed_graph_id.append(graph_id)
                self.failed_idx.append(idx)
                idx = random.randint(0, len(self) - 1)
                return self.__getitem__(idx)
        else:
            idx = random.randint(0, len(self) - 1)
            return self.__getitem__(idx)

    def get_train_val_test_loader(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        *,
        train_key: (list[str] | None) = None,
        val_key: (list[str] | None) = None,
        test_key: (list[str] | None) = None,
        batch_size=32,
        num_workers=0,
        pin_memory=True,
    ) -> tuple[paddle.io.DataLoader, paddle.io.DataLoader, paddle.io.DataLoader]:
        """Partition the GraphData using materials id,
        randomly select the train_keys, val_keys, test_keys by train val test ratio,
        or use pre-defined train_keys, val_keys, and test_keys to create train, val,
        test loaders.

        Args:
            train_ratio (float): The ratio of the dataset to use for training
                Default = 0.8
            val_ratio (float): The ratio of the dataset to use for validation
                Default: 0.1
            train_key (List(str), optional): a list of mp_ids for train set
            val_key (List(str), optional): a list of mp_ids for val set
            test_key (List(str), optional): a list of mp_ids for test set
            batch_size (int): batch size
                Default = 32
            num_workers (int): The number of worker processes for loading the data
                see  Dataloader documentation for more info
                Default = 0
            pin_memory (bool): Whether to pin the memory of the data loaders
                Default: True

        Returns:
            train_loader, val_loader, test_loader
        """
        train_labels, val_labels, test_labels = {}, {}, {}
        if train_key is None:
            mp_ids = list(self.labels)
            random.shuffle(mp_ids)
            n_train = int(train_ratio * len(mp_ids))
            n_val = int(val_ratio * len(mp_ids))
            train_key = mp_ids[:n_train]
            val_key = mp_ids[n_train : n_train + n_val]
            test_key = mp_ids[n_train + n_val :]
        for mp_id in train_key:
            if mp_id in self.labels:
                train_labels[mp_id] = self.labels.pop(mp_id)
        train_dataset = GraphData(
            graph_path=self.graph_path,
            labels=train_labels,
            targets=self.targets,
            exclude=self.excluded_graph,
            energy_key=self.energy_key,
        )
        train_loader = paddle.io.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            shuffle=True,
            num_workers=num_workers,
        )
        for mp_id in val_key:
            if mp_id in self.labels:
                val_labels[mp_id] = self.labels.pop(mp_id)
        val_dataset = GraphData(
            graph_path=self.graph_path,
            labels=val_labels,
            targets=self.targets,
            exclude=self.excluded_graph,
            energy_key=self.energy_key,
        )
        val_loader = paddle.io.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            shuffle=True,
            num_workers=num_workers,
        )
        if test_key is not None:
            for mp_id in test_key:
                if mp_id in self.labels:
                    test_labels[mp_id] = self.labels.pop(mp_id)
            test_dataset = GraphData(
                graph_path=self.graph_path,
                labels=test_labels,
                targets=self.targets,
                exclude=self.excluded_graph,
                energy_key=self.energy_key,
            )
            test_loader = paddle.io.DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                collate_fn=collate_graphs,
                num_workers=num_workers,
            )
        else:
            test_loader = None
        return train_loader, val_loader, test_loader


class StructureJsonData(paddle.io.Dataset):
    """Read structure and targets from a JSON file.
    This class is used to load the MPtrj dataset.
    """

    def __init__(
        self,
        data: (str | dict),
        graph_converter: CrystalGraphConverter,
        *,
        targets: TrainTask = "efsm",
        energy_key: str = "energy_per_atom",
        force_key: str = "force",
        stress_key: str = "stress",
        magmom_key: str = "magmom",
        shuffle: bool = True,
    ) -> None:
        """Initialize the dataset by reading JSON files.

        Args:
            data (str | dict): file path or dir name that contain all the JSONs
            graph_converter (CrystalGraphConverter): Converts pymatgen.core.Structure
                to CrystalGraph object.
            targets ("ef" | "efs" | "efm" | "efsm"): The training targets.
                Default = "efsm"
            energy_key (str, optional): the key of energy in the labels.
                Default = "energy_per_atom"
            force_key (str, optional): the key of force in the labels.
                Default = "force"
            stress_key (str, optional): the key of stress in the labels.
                Default = "stress"
            magmom_key (str, optional): the key of magmom in the labels.
                Default = "magmom"
            shuffle (bool): whether to shuffle the sequence of dataset
                Default = True
        """
        if isinstance(data, str):
            self.data = {}
            if os.path.isdir(data):
                for json_path in os.listdir(data):
                    if json_path.endswith(".json"):
                        print(f"Importing: {json_path}")
                        self.data.update(utils.read_json(os.path.join(data, json_path)))
            else:
                print(f"Importing: {data}")
                self.data.update(utils.read_json(data))
        elif isinstance(data, dict):
            self.data = data
        else:
            raise TypeError(f"data must be JSON path or dictionary, got {type(data)}")
        self.keys = [
            (mp_id, graph_id) for mp_id, dct in self.data.items() for graph_id in dct
        ]
        if shuffle:
            random.shuffle(self.keys)
        print(f"{len(self.data)} MP IDs, {len(self)} structures imported")
        self.graph_converter = graph_converter
        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key
        self.magmom_key = magmom_key
        self.targets = targets
        self.failed_idx: list[int] = []
        self.failed_graph_id: dict[str, str] = {}

    def __len__(self) -> int:
        """Get the number of structures with targets in the dataset."""
        return len(self.keys)

    @functools.cache
    def __getitem__(self, idx: int) -> tuple[CrystalGraph, dict[str, paddle.Tensor]]:
        """Get one item in the dataset.

        Returns:
            crystal_graph (CrystalGraph): graph of the crystal structure
            targets (dict): dictionary of targets. i.e. energy, force, stress, magmom
        """
        if idx not in self.failed_idx:
            mp_id, graph_id = self.keys[idx]
            try:
                struct = Structure.from_dict(self.data[mp_id][graph_id]["structure"])
                crystal_graph = self.graph_converter(
                    struct, graph_id=graph_id, mp_id=mp_id
                )
                targets = {}
                for key in self.targets:
                    if key == "e":
                        energy = self.data[mp_id][graph_id][self.energy_key]
                        targets["e"] = paddle.to_tensor(data=energy, dtype=DTYPE)
                    elif key == "f":
                        force = self.data[mp_id][graph_id][self.force_key]
                        targets["f"] = paddle.to_tensor(data=force, dtype=DTYPE)
                    elif key == "s":
                        stress = self.data[mp_id][graph_id][self.stress_key]
                        targets["s"] = paddle.to_tensor(data=stress, dtype=DTYPE) * -0.1
                    elif key == "m":
                        mag = self.data[mp_id][graph_id][self.magmom_key]
                        if mag is None:
                            targets["m"] = None
                        else:
                            targets["m"] = paddle.abs(
                                x=paddle.to_tensor(data=mag, dtype=DTYPE)
                            )
                return crystal_graph, targets
            except Exception:
                structure = Structure.from_dict(self.data[mp_id][graph_id]["structure"])
                self.failed_graph_id[graph_id] = structure.composition.formula
                self.failed_idx.append(idx)
                idx = random.randint(0, len(self) - 1)
                return self.__getitem__(idx)
        else:
            idx = random.randint(0, len(self) - 1)
            return self.__getitem__(idx)

    def get_train_val_test_loader(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        *,
        train_key: (list[str] | None) = None,
        val_key: (list[str] | None) = None,
        test_key: (list[str] | None) = None,
        batch_size=32,
        num_workers=0,
        pin_memory=True,
    ) -> tuple[paddle.io.DataLoader, paddle.io.DataLoader, paddle.io.DataLoader]:
        """Partition the Dataset using materials id,
        randomly select the train_keys, val_keys, test_keys by train val test ratio,
        or use pre-defined train_keys, val_keys, and test_keys to create train, val,
        test loaders.

        Args:
            train_ratio (float): The ratio of the dataset to use for training
                Default = 0.8
            val_ratio (float): The ratio of the dataset to use for validation
                Default: 0.1
            train_key (List(str), optional): a list of mp_ids for train set
            val_key (List(str), optional): a list of mp_ids for val set
            test_key (List(str), optional): a list of mp_ids for test set
            batch_size (int): batch size
                Default = 32
            num_workers (int): The number of worker processes for loading the data
                see  Dataloader documentation for more info
                Default = 0
            pin_memory (bool): Whether to pin the memory of the data loaders
                Default: True

        Returns:
            train_loader, val_loader, test_loader
        """
        train_data, val_data, test_data = {}, {}, {}
        if train_key is None:
            mp_ids = list(self.data)
            random.shuffle(mp_ids)
            n_train = int(train_ratio * len(mp_ids))
            n_val = int(val_ratio * len(mp_ids))
            train_key = mp_ids[:n_train]
            val_key = mp_ids[n_train : n_train + n_val]
            test_key = mp_ids[n_train + n_val :]
        for mp_id in train_key:
            train_data[mp_id] = self.data.pop(mp_id)
        train_dataset = StructureJsonData(
            data=train_data,
            graph_converter=self.graph_converter,
            targets=self.targets,
            energy_key=self.energy_key,
        )
        train_loader = paddle.io.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            shuffle=True,
            num_workers=num_workers,
        )
        for mp_id in val_key:
            val_data[mp_id] = self.data.pop(mp_id)
        val_dataset = StructureJsonData(
            data=val_data,
            graph_converter=self.graph_converter,
            targets=self.targets,
            energy_key=self.energy_key,
        )
        val_loader = paddle.io.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            shuffle=True,
            num_workers=num_workers,
        )
        if test_key is not None:
            for mp_id in test_key:
                test_data[mp_id] = self.data.pop(mp_id)
            test_dataset = StructureJsonData(
                data=test_data,
                graph_converter=self.graph_converter,
                targets=self.targets,
                energy_key=self.energy_key,
            )
            test_loader = paddle.io.DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                collate_fn=collate_graphs,
                num_workers=num_workers,
            )
        else:
            test_loader = None
        return train_loader, val_loader, test_loader


def collate_graphs(
    batch_data: list,
) -> tuple[list[CrystalGraph], dict[str, paddle.Tensor]]:
    """Collate of list of (graph, target) into batch data.

    Args:
        batch_data (list): list of (graph, target(dict))

    Returns:
        graphs (List): a list of graphs
        targets (Dict): dictionary of targets, where key and values are:
            e (Tensor): energies of the structures [batch_size]
            f (Tensor): forces of the structures [n_batch_atoms, 3]
            s (Tensor): stresses of the structures [3*batch_size, 3]
            m (Tensor): magmom of the structures [n_batch_atoms]
    """
    graphs = [graph for graph, _ in batch_data]
    all_targets = {key: [] for key in batch_data[0][1]}
    all_targets["e"] = paddle.to_tensor(
        data=[targets["e"] for _, targets in batch_data], dtype=DTYPE
    )
    for _, targets in batch_data:
        for target, value in targets.items():
            if target != "e":
                all_targets[target].append(value)
    return graphs, all_targets


def get_train_val_test_loader(
    dataset: paddle.io.Dataset,
    *,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    return_test: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple[paddle.io.DataLoader, paddle.io.DataLoader, paddle.io.DataLoader]:
    """Randomly partition a dataset into train, val, test loaders.

    Args:
        dataset (Dataset): The dataset to partition.
        batch_size (int): The batch size for the data loaders
            Default = 64
        train_ratio (float): The ratio of the dataset to use for training
            Default = 0.8
        val_ratio (float): The ratio of the dataset to use for validation
            Default: 0.1
        return_test (bool): Whether to return a test data loader
            Default = True
        num_workers (int): The number of worker processes for loading the data
            see  Dataloader documentation for more info
            Default = 0
        pin_memory (bool): Whether to pin the memory of the data loaders
            Default: True

    Returns:
        train_loader, val_loader and optionally test_loader
    """
    total_size = len(dataset)
    indices = list(range(total_size))
    random.shuffle(indices)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    train_loader = paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        sampler=paddle.io.SubsetRandomSampler(indices=indices[0:train_size]),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        sampler=paddle.io.SubsetRandomSampler(
            indices=indices[train_size : train_size + val_size]
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if return_test:
        test_loader = paddle.io.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            sampler=paddle.io.SubsetRandomSampler(
                indices=indices[train_size + val_size :]
            ),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader
    return train_loader, val_loader


def get_loader(
    dataset, *, batch_size: int = 64, num_workers: int = 0, pin_memory: bool = True
) -> paddle.io.DataLoader:
    """Get a dataloader from a dataset.

    Args:
        dataset (Dataset): The dataset to partition.
        batch_size (int): The batch size for the data loaders
            Default = 64
        num_workers (int): The number of worker processes for loading the data
            see  Dataloader documentation for more info
            Default = 0
        pin_memory (bool): Whether to pin the memory of the data loaders
            Default: True

    Returns:
        data_loader
    """
    return paddle.io.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        shuffle=True,
        num_workers=num_workers,
    )
