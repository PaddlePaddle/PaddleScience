import time
import warnings

import numpy as np
import paddle
from ase import Atoms
from mattersim.datasets.utils.convertor import GraphConvertor
from mattersim.utils.paddle_utils import *  # noqa
from paddle_geometric.loader import DataLoader as DataLoader_pyg


def build_dataloader(
    atoms: list[Atoms] = None,
    energies: list[float] = None,
    forces: list[np.ndarray] = None,
    stresses: list[np.ndarray] = None,
    cutoff: float = 5.0,
    threebody_cutoff: float = 4.0,
    batch_size: int = 64,
    model_type: str = "m3gnet",
    shuffle=False,
    only_inference: bool = False,
    num_workers: int = 0,
    multiprocessing: int = 0,
    multithreading: int = 0,
    dataset=None,
    finetune_task_label: list = None,
    **kwargs,
):
    """
    Build a dataloader given a list of atoms
        - atoms : a list of atoms in ase format
        - energies, forces and stresses are necessary for training
            - energies : a list of energy (float) with unit eV
            - forces : a list of nx3 force matrix (np.ndarray) with unit eV/Å,
                where n is the number of atom in each structure.
            - stresses : a list of 3x3 stress matrix (np.ndarray) with unit GPa
        - only_inference : if True, energies, forces and stresses will be ignored
        - num_workers : number of workers for dataloader
        - dataset : the dataset object for the dataloader
                    only used for graphormer and geomformer
    """
    convertor = GraphConvertor(model_type, cutoff, True, threebody_cutoff)
    preprocessed_data = []
    if dataset is None:
        if not only_inference:
            assert (
                energies is not None
            ), "energies must be provided if only_inference is False"
        if stresses is not None:
            assert np.array(stresses[0]).shape == (
                3,
                3,
            ), "stresses must be a list of 3x3 matrices"
        length = len(atoms)
        if energies is None:
            energies = [None] * length
        if forces is None:
            forces = [None] * length
        if stresses is None:
            stresses = [None] * length
    if model_type == "m3gnet":
        if multiprocessing == 0 and multithreading == 0:
            for graph, energy, force, stress in zip(atoms, energies, forces, stresses):
                graph = convertor.convert(graph.copy(), energy, force, stress, **kwargs)
                if graph is not None:
                    preprocessed_data.append(graph)
        elif multithreading > 0 and multiprocessing == 0:
            from multiprocessing.pool import ThreadPool

            warnings.warn("multithreading is experimental")
            warnings.warn("it may not be faster than single thread due to GIL.")
            print("Using multithreading with {} threads".format(multithreading))
            start = time.time()
            pool = ThreadPool(processes=multithreading)
            preprocessed_data = pool.starmap(
                convertor.convert, zip(atoms, energies, forces, stresses)
            )
            pool.close()
            print("Time elapsed: {:.2f} s".format(time.time() - start))
        elif multiprocessing > 0 and multithreading == 0:
            import multiprocessing as mp

            warnings.warn("multiprocessing is experimental.")
            print("Using multiprocessing with {} workers".format(multiprocessing))
            start = time.time()
            pool = mp.Pool(multiprocessing)
            results = []
            for i in range(multiprocessing):
                left = int(i * length / multiprocessing)
                right = int((i + 1) * length / multiprocessing)
                results.append(
                    pool.apply_async(multiprocess_data, args=(atoms[left:right], 1))
                )
            pool.close()
            pool.join()
            for result in results:
                graph = result.get()
                if graph is not None:
                    preprocessed_data.extend(graph)
            print("Time for multiprocessing: {:.2f} s".format(time.time() - start))
        else:
            raise NotImplementedError

        class CustomizedDataset(paddle.io.Dataset):
            def __init__(self, data):
                super().__init__()
                self.data = data

            def __getitem__(self, idx):
                return idx, self.data[idx]

            def __len__(self):
                return len(self.data)

        return DataLoader_pyg(
            CustomizedDataset(preprocessed_data),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
    elif model_type == "graphormer" or model_type == "geomformer":
        raise NotImplementedError


def multiprocess_data(atoms: list[Atoms], number):
    convertor = GraphConvertor()
    result = []
    for graph in atoms:
        graph = convertor.convert(
            graph,
            graph.get_potential_energy(),
            graph.get_forces(),
            graph.get_stress(voigt=False) * 160.2,
        )
        if graph is not None:
            result.append(graph)
    return result


def pad_1d_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.shape[0]
    if xlen < padlen:
        new_x = paddle.zeros(shape=[padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(axis=0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1
    xlen, xdim = tuple(x.shape)
    if xlen < padlen:
        new_x = paddle.zeros(shape=[padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(axis=0)


def mask_after_k_persample(n_sample: int, n_len: int, persample_k: paddle.Tensor):
    assert tuple(persample_k.shape)[0] == n_sample
    assert persample_k.max_func() <= n_len
    mask = paddle.zeros(shape=[n_sample, n_len + 1])
    mask[paddle.arange(end=n_sample), persample_k] = 1
    mask = mask.cumsum(axis=1)[:, :-1]
    return mask.astype("bool")


def auto_cell(cell, cutoff=10.0):
    max_x = max(int(cutoff / paddle.min(x=paddle.abs(x=cell[:, 0, 0]))), 1)
    max_y = max(int(cutoff / paddle.min(x=paddle.abs(x=cell[:, 1, 1]))), 1)
    max_z = max(int(cutoff / paddle.min(x=paddle.abs(x=cell[:, 2, 2]))), 1)
    cells = []
    for i in range(-max_x, max_x + 1):
        for j in range(-max_y, max_y + 1):
            for k in range(-max_z, max_z + 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                cells.append([i, j, k])
    return cells


def cell_expand(pos, atoms, cell, cutoff=10.0):
    batch_size, max_num_atoms = tuple(pos.shape)[:2]
    cells = auto_cell(cell, cutoff)
    cell_tensor = (
        paddle.to_tensor(data=cells, place=pos.place)
        .to(cell.dtype)
        .unsqueeze(axis=0)
        .expand(shape=[batch_size, -1, -1])
    )
    offset = paddle.bmm(x=cell_tensor, y=cell)
    expand_pos = pos.unsqueeze(axis=1) + offset.unsqueeze(axis=2)
    expand_pos = expand_pos.view(batch_size, -1, 3)
    expand_dist = paddle.linalg.norm(
        x=pos.unsqueeze(axis=2) - expand_pos.unsqueeze(axis=1), p=2, axis=-1
    )
    expand_mask = expand_dist < cutoff
    expand_mask = paddle.masked_fill(
        x=expand_mask, mask=atoms.equal(y=0).unsqueeze(axis=-1), value=False
    )
    expand_mask = (paddle.sum(x=expand_mask, axis=1) > 0) & ~atoms.equal(y=0).tile(
        repeat_times=[1, len(cells)]
    )
    expand_len = paddle.sum(x=expand_mask, axis=-1)
    max_expand_len = paddle.max(x=expand_len)
    outcell_index = paddle.zeros(shape=[batch_size, max_expand_len], dtype="int64")
    expand_pos_compressed = paddle.zeros(
        shape=[batch_size, max_expand_len, 3], dtype=pos.dtype
    )
    outcell_all_index = paddle.arange(dtype="int64", end=max_num_atoms).tile(
        repeat_times=len(cells)
    )
    for i in range(batch_size):
        outcell_index[i, : expand_len[i]] = outcell_all_index[expand_mask[i]]
        expand_pos_compressed[i, : expand_len[i], :] = expand_pos[i, expand_mask[i], :]
    return (
        expand_pos_compressed,
        expand_len,
        outcell_index,
        mask_after_k_persample(batch_size, max_expand_len, expand_len),
    )


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.shape[0]
    if xlen < padlen:
        new_x = paddle.zeros(shape=[padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(axis=0)


def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.shape[1] if len(tuple(x.shape)) > 1 else 1
    feature_offset = 1 + paddle.arange(
        start=0, end=feature_num * offset, step=offset, dtype="int64"
    )
    x = x + feature_offset
    return x


class BatchedDataDataset(paddle.io.Dataset):
    def __init__(self, dataset, max_node=512, infer=False):
        super().__init__()
        self.dataset = dataset
        self.max_node = max_node
        self.infer = infer

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def __len__(self):
        return len(self.dataset)

    def collate(self, samples):
        return collator_ft(samples, max_node=self.max_node, use_pbc=True)


def pad_pos_unsqueeze(x, padlen):
    xlen, xdim = tuple(x.shape)
    if xlen < padlen:
        new_x = paddle.zeros(shape=[padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(axis=0)


def collator_ft(items, max_node=512, use_pbc=True):
    original_len = len(items)
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    filtered_len = len(items)
    if filtered_len < original_len:
        pass
    pos = None
    max_node_num = max(item.x.size(0) for item in items if item is not None)
    forces = None
    stress = None
    total_energy = None
    if hasattr(items[0], "pos") and items[0].pos is not None:
        poses = [(item.pos - item.pos.mean(dim=0, keepdim=True)) for item in items]
        pos = paddle.concat(x=[pad_pos_unsqueeze(i, max_node_num) for i in poses])
    if hasattr(items[0], "forces") and items[0].forces is not None:
        forcess = [item.forces for item in items]
        forces = paddle.concat(x=[pad_pos_unsqueeze(i, max_node_num) for i in forcess])
    if hasattr(items[0], "stress") and items[0].stress is not None:
        stress = paddle.concat(x=[item.stress.unsqueeze(0) for item in items], axis=0)
    if hasattr(items[0], "total_energy") and items[0].cell is not None:
        total_energy = paddle.concat(x=[item.total_energy for item in items])
    items = [
        (
            item.idx,
            item.x,
            item.y,
            (
                item.pbc
                if hasattr(item, "pbc")
                else paddle.to_tensor(data=[False, False, False])
            )
            if use_pbc
            else None,
            (item.cell if hasattr(item, "cell") else paddle.zeros(shape=[3, 3]))
            if use_pbc
            else None,
            int(item.num_atoms) if hasattr(item, "num_atoms") else item.x.size()[0],
        )
        for item in items
    ]
    idxs, xs, ys, pbcs, cells, natoms = zip(*items)
    y = paddle.concat(x=ys)
    x = paddle.concat(x=[pad_2d_unsqueeze(i, max_node_num) for i in xs])
    pbc = (
        paddle.concat(x=[i.unsqueeze(axis=0) for i in pbcs], axis=0)
        if use_pbc
        else None
    )
    cell = (
        paddle.concat(x=[i.unsqueeze(axis=0) for i in cells], axis=0)
        if use_pbc
        else None
    )
    natoms = paddle.to_tensor(data=natoms) if use_pbc else None
    node_type_edge = None
    return dict(
        idx=paddle.to_tensor(data=idxs, dtype="int64"),
        x=x,
        y=y,
        pos=pos + 1e-05,
        pbc=pbc,
        cell=cell,
        natoms=natoms,
        total_energy=total_energy,
        forces=forces,
        stress=stress,
        node_type_edge=node_type_edge,
    )
