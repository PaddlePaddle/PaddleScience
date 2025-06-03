from functools import lru_cache

import numpy as np
import paddle
from ase import Atoms
from paddle_geometric.data import Data


def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.shape[1] if len(tuple(x.shape)) > 1 else 1
    feature_offset = 1 + paddle.arange(
        start=0, end=feature_num * offset, step=offset, dtype="int64"
    )
    x = x + feature_offset
    return x


class AtomCalDataset:
    def __init__(
        self,
        atom_list: list[Atoms],
        energies: list[float] = None,
        forces: list[np.ndarray] = None,
        stresses: list[np.ndarray] = None,
        finetune_task_label: list = None,
    ):
        self.data = self._preprocess(
            atom_list, energies, forces, stresses, finetune_task_label
        )

    def _preprocess(
        self,
        atom_list,
        energies: list[float] = None,
        forces: list[np.ndarray] = None,
        stresses: list[np.ndarray] = None,
        finetune_task_label: list = None,
        use_ase_energy: bool = False,
        use_ase_force: bool = False,
        use_ase_stress: bool = False,
    ):
        data_list = []
        for i, (atom, energy, force, stress) in enumerate(
            zip(atom_list, energies, forces, stresses)
        ):
            item_dict = atom.todict()
            item_dict["info"] = {}
            if energy is None:
                energy = 0
            if force is None:
                force = np.zeros([len(atom), 3])
            if stress is None:
                stress = np.zeros([3, 3])
            try:
                energy = atom.get_total_energy() if use_ase_energy else energy
                force = (
                    atom.get_forces(apply_constraint=False) if use_ase_force else force
                )
                stress = atom.get_stress(voigt=False) if use_ase_stress else stress
            except Exception as e:
                RuntimeError(f"Error in {i}th data: {e}")
            if finetune_task_label is not None:
                item_dict["finetune_task_label"] = finetune_task_label[i]
            else:
                item_dict["finetune_task_label"] = 0
            item_dict["info"]["energy"] = energy
            item_dict["info"]["stress"] = stress
            item_dict["forces"] = force
            data_list.append(item_dict)
        return data_list

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.data[idx]
        return preprocess_atom_item(item, idx)

    def __len__(self):
        return len(self.data)


def preprocess_atom_item(item, idx):
    numbers = item["numbers"]
    item["x"] = paddle.to_tensor(data=numbers, dtype="int64").unsqueeze(axis=-1)
    positions = item["positions"]
    item["pos"] = paddle.to_tensor(data=positions, dtype="float64")
    item["cell"] = paddle.to_tensor(data=item["cell"], dtype="float64")
    item["pbc"] = paddle.to_tensor(data=item["pbc"], dtype="bool")
    item["idx"] = idx
    item["y"] = paddle.to_tensor(data=[item["finetune_task_label"]])
    item["total_energy"] = paddle.to_tensor(
        data=[item["info"]["energy"]], dtype="float64"
    )
    item["stress"] = paddle.to_tensor(data=item["info"]["stress"], dtype="float64")
    item["forces"] = paddle.to_tensor(data=item["forces"], dtype="float64")
    item = Data(**item)
    x = item.x
    item.x = convert_to_single_emb(x)
    return item
