import glob
import os.path as osp
from typing import Any, Callable, List, Optional

import numpy as np
import paddle

from paddle_geometric.data import Data, Dataset
from paddle_geometric.utils import index_sort, scatter


class TrackingData(Data):
    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == 'y_index':
            return paddle.to_tensor([value[0].max().item() + 1, self.num_nodes])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class TrackMLParticleTrackingDataset(Dataset):
    r"""The `TrackML Particle Tracking Challenge
    <https://www.kaggle.com/c/trackml-particle-identification>`_ dataset to
    reconstruct particle tracks from 3D points left in the silicon detectors.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """

    url = 'https://www.kaggle.com/c/trackml-particle-identification'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform)
        events = glob.glob(osp.join(self.raw_dir, 'event*-hits.csv'))
        events = [e.split(osp.sep)[-1].split('-')[0][5:] for e in events]
        self.events: List[str] = sorted(events)

    @property
    def raw_file_names(self) -> List[str]:
        event_indices = ['000001000']
        file_names = []
        file_names += [f'event{idx}-cells.csv' for idx in event_indices]
        file_names += [f'event{idx}-hits.csv' for idx in event_indices]
        file_names += [f'event{idx}-particles.csv' for idx in event_indices]
        file_names += [f'event{idx}-truth.csv' for idx in event_indices]
        return file_names

    def download(self) -> None:
        raise RuntimeError(
            f'Dataset not found. Please download it from {self.url} and move '
            f'all *.csv files to {self.raw_dir}')

    def len(self) -> int:
        return len(glob.glob(osp.join(self.raw_dir, 'event*-hits.csv')))

    def get(self, i: int) -> TrackingData:
        import pandas as pd

        idx = self.events[i]

        # Get hit positions.
        hits_path = osp.join(self.raw_dir, f'event{idx}-hits.csv')
        pos = pd.read_csv(hits_path, usecols=['x', 'y', 'z'], dtype=np.float32)
        pos = paddle.to_tensor(pos.values) / 1000.

        # Get hit features.
        cells_path = osp.join(self.raw_dir, f'event{idx}-cells.csv')
        cell = pd.read_csv(cells_path, usecols=['hit_id', 'value'])
        hit_id = paddle.to_tensor(cell['hit_id'].values).astype('int64') - 1
        value = paddle.to_tensor(cell['value'].values).astype('float32')
        ones = paddle.ones([hit_id.size], dtype='float32')
        num_cells = scatter(ones, hit_id, 0, pos.shape[0], reduce='sum') / 10.
        value = scatter(value, hit_id, 0, pos.shape[0], reduce='sum')
        x = paddle.stack([num_cells, value], axis=-1)

        # Get ground-truth hit assignments.
        truth_path = osp.join(self.raw_dir, f'event{idx}-truth.csv')
        y = pd.read_csv(truth_path,
                        usecols=['hit_id', 'particle_id', 'weight'])
        hit_id = paddle.to_tensor(y['hit_id'].values).astype('int64') - 1
        particle_id = paddle.to_tensor(y['particle_id'].values).astype('int64')
        particle_id = particle_id.unique(return_inverse=True)[1] - 1
        weight = paddle.to_tensor(y['weight'].values).astype('float32')

        # Sort.
        _, perm = index_sort(particle_id * hit_id.shape[0] + hit_id)
        hit_id = hit_id[perm]
        particle_id = particle_id[perm]
        weight = weight[perm]

        # Remove invalid particle ids.
        mask = particle_id >= 0
        hit_id = hit_id[mask]
        particle_id = particle_id[mask]
        weight = weight[mask]

        y_index = paddle.stack([particle_id, hit_id], axis=0)

        return TrackingData(x=x, pos=pos, y_index=y_index, y_weight=weight)
