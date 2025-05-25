import copy
import os
import os.path as osp
from dataclasses import dataclass
from functools import cached_property
from glob import glob
from pathlib import Path
from typing import Callable, List, MutableSequence, Optional, Tuple, Union

import numpy as np
import paddle
from paddle_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from paddle_geometric.data.data import BaseData


class HydroNet(InMemoryDataset):
    r"""The HydroNet dataset from the
    `"HydroNet: Benchmark Tasks for Preserving Intermolecular Interactions and
    Structural Motifs in Predictive and Generative Models for Molecular Data"
    <https://arxiv.org/abs/2012.00131>`_ paper, consisting of 5 million water
    clusters held together by hydrogen bonding networks. This dataset
    provides atomic coordinates and total energy in kcal/mol for the cluster.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str, optional): Name of the subset of the full dataset to use:
            :obj:`"small"` uses 500k graphs sampled from the :obj:`"medium"`
            dataset, :obj:`"medium"` uses 2.7m graphs with a maximum size of 75
            nodes.
            Mutually exclusive option with the clusters argument.
            (default :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`paddle_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        num_workers (int): Number of multiprocessing workers to use for
            pre-processing the dataset. (default :obj:`8`)
        clusters (int or List[int], optional): Select a subset of clusters
            from the full dataset. If set to :obj:`None`, will select all.
            (default :obj:`None`)
        use_processed (bool): Option to use a pre-processed version of the
            original :obj:`xyz` dataset. (default: :obj:`True`)
    """
    def __init__(
        self,
        root: str,
        name: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        num_workers: int = 8,
        clusters: Optional[Union[int, List[int]]] = None,
        use_processed: bool = True,
    ) -> None:
        self.name = name
        self.num_workers = num_workers
        self.use_processed = use_processed

        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)

        self.select_clusters(clusters)

    @property
    def raw_file_names(self) -> List[str]:
        return [f'W{c}_geoms_all.zip' for c in range(3, 31)]

    @property
    def processed_file_names(self) -> List[str]:
        return [f'W{c}_geoms_all.npz' for c in range(3, 31)]

    def download(self) -> None:
        token_file = Path(osp.join(self.raw_dir, 'use_processed'))
        if self.use_processed and token_file.exists():
            return

        file = RemoteFile.hydronet_splits()
        file.unpack_to(self.raw_dir)

        if self.use_processed:
            file = RemoteFile.processed_dataset()
            file.unpack_to(self.raw_dir)
            token_file.touch()
            return

        file = RemoteFile.raw_dataset()
        file.unpack_to(self.raw_dir)
        folder_name, _ = osp.splitext(file.name)
        files = glob(osp.join(self.raw_dir, folder_name, '*.zip'))

        for f in files:
            dst = osp.join(self.raw_dir, osp.basename(f))
            os.rename(f, dst)

        os.rmdir(osp.join(self.raw_dir, folder_name))

    def process(self) -> None:
        if self.use_processed:
            return self._unpack_processed()

        self._partitions = [
            self._create_partitions(f) for f in self.raw_paths
        ]

    def _unpack_processed(self) -> None:
        files = glob(osp.join(self.raw_dir, '*.npz'))
        for f in files:
            dst = osp.join(self.processed_dir, osp.basename(f))
            os.rename(f, dst)

    def _create_partitions(self, file: str) -> 'Partition':
        name = osp.basename(file)
        name, _ = osp.splitext(name)
        return Partition(self.root, name, self.transform, self.pre_transform)

    def select_clusters(
        self,
        clusters: Optional[Union[int, List[int]]],
    ) -> None:
        if self.name is not None:
            clusters = self._validate_name(clusters)

        self._partitions = [self._create_partitions(f) for f in self.raw_paths]

        if clusters is None:
            return

        clusters = [clusters] if isinstance(clusters, int) else clusters

        def is_valid_cluster(x: Union[int, List[int]]) -> bool:
            return isinstance(x, int) and x >= 3 and x <= 30

        if not all([is_valid_cluster(x) for x in clusters]):
            raise ValueError(
                "Selected clusters must be an integer in the range [3, 30]")

        self._partitions = [self._partitions[c - 3] for c in clusters]

    def _validate_name(
        self,
        clusters: Optional[Union[int, List[int]]],
    ) -> List[int]:
        if clusters is not None:
            raise ValueError("'name' and 'clusters' are mutually exclusive")

        if self.name not in ['small', 'medium']:
            raise ValueError(f"Invalid subset name '{self.name}'. "
                             f"Must be either 'small' or 'medium'")

        return list(range(3, 26))

    @cached_property
    def _dataset(self) -> List[Data]:
        dataset = []
        for partition in self._partitions:
            dataset.extend(partition)

        return dataset

    def len(self) -> int:
        return len(self._dataset)

    def get(self, idx: int) -> Data:
        return self._dataset[idx]


def get_num_clusters(filepath: str) -> int:
    name = osp.basename(filepath)
    return int(name[1:name.find('_')])


@dataclass
class RemoteFile:
    url: str
    name: str

    def unpack_to(self, dest_folder: str) -> None:
        file = download_url(self.url, dest_folder, filename=self.name)
        extract_zip(file, dest_folder)
        os.unlink(file)

    @staticmethod
    def raw_dataset() -> 'RemoteFile':
        return RemoteFile(
            url='https://figshare.com/ndownloader/files/38063847',
            name='W3-W30_all_geoms_TTM2.1-F.zip')

    @staticmethod
    def processed_dataset() -> 'RemoteFile':
        return RemoteFile(
            url='https://figshare.com/ndownloader/files/38075781',
            name='W3-W30_pyg_processed.zip')

    @staticmethod
    def hydronet_splits() -> 'RemoteFile':
        return RemoteFile(
            url="https://figshare.com/ndownloader/files/38075904",
            name="hydronet_splits.zip")


class Partition(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        self.name = name
        self.num_clusters = get_num_clusters(name)
        super().__init__(root, transform, pre_transform)

        self.is_loaded = False

    @property
    def raw_file_names(self) -> List[str]:
        return [self.name + ".zip"]

    @property
    def processed_file_names(self) -> List[str]:
        return [self.name + '.npz']

    def process(self) -> None:
        num_nodes = self.num_clusters * 3
        chunk_size = num_nodes + 2
        z, pos = read_atoms(self.raw_paths[0], chunk_size)
        y = read_energy(self.raw_paths[0], chunk_size)
        np.savez(self.processed_paths[0], z=z, pos=pos, y=y,
                 num_graphs=z.shape[0])

    def _load(self) -> None:
        if self.is_loaded:
            return

        with np.load(self.processed_paths[0]) as npzfile:
            self.z = npzfile['z']
            self.pos = npzfile['pos']
            self.y = npzfile['y']
            numel = int(npzfile['num_graphs'])

        self._data_list: MutableSequence[Optional[BaseData]] = [None] * numel
        self.is_loaded = True

    @cached_property
    def num_graphs(self) -> int:
        with np.load(self.processed_paths[0]) as npzfile:
            return int(npzfile['num_graphs'])

    def len(self) -> int:
        return self.num_graphs

    def get(self, idx: int) -> Data:
        self._load()

        if self._data_list[idx] is not None:
            cached_data = self._data_list[idx]
            assert isinstance(cached_data, Data)
            return copy.copy(cached_data)

        data = Data(
            z=paddle.to_tensor(self.z[idx, :]),
            pos=paddle.to_tensor(self.pos[idx, :, :]),
            y=paddle.to_tensor(self.y[idx]),
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self._data_list[idx] = copy.copy(data)
        return data
