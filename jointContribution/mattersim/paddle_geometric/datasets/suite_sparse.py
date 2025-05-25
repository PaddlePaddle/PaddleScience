import os.path as osp
from typing import Callable, Optional

import fsspec
import paddle

from paddle_geometric.data import Data, InMemoryDataset
from paddle_geometric.io import fs


class SuiteSparseMatrixCollection(InMemoryDataset):
    r"""A suite of sparse matrix benchmarks known as the `Suite Sparse Matrix
    Collection <https://sparse.tamu.edu>`_, collected from a wide range of
    applications.

    Args:
        root (str): Root directory where the dataset should be saved.
        group (str): The group of the sparse matrix.
        name (str): The name of the sparse matrix.
        transform (callable, optional): A function/transform that takes in a
            :obj:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            a :obj:`paddle_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    url = 'https://sparse.tamu.edu/mat/{}/{}.mat'

    def __init__(
        self,
        root: str,
        group: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        # Initialize with matrix group and name
        self.group = group
        self.name = name
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        # Directory for raw data
        return osp.join(self.root, self.group, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        # Directory for processed data
        return osp.join(self.root, self.group, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        # Raw file name pattern
        return f'{self.name}.mat'

    @property
    def processed_file_names(self) -> str:
        # Processed file name pattern
        return 'data.pt'

    def download(self) -> None:
        # Downloads the .mat file from the Suite Sparse Matrix Collection
        fs.cp(self.url.format(self.group, self.name), self.raw_dir)

    def process(self) -> None:
        # Process the .mat file into a graph format compatible with Paddle Geometric
        from scipy.io import loadmat

        with fsspec.open(self.raw_paths[0], 'rb') as f:
            mat = loadmat(f)['Problem'][0][0][2].tocsr().tocoo()

        row = paddle.to_tensor(mat.row, dtype='int64')
        col = paddle.to_tensor(mat.col, dtype='int64')
        edge_index = paddle.stack([row, col], axis=0)

        value = paddle.to_tensor(mat.data, dtype='float32')
        edge_attr = None if paddle.all(value == 1.0) else value

        size = mat.shape if mat.shape[0] != mat.shape[1] else None
        num_nodes = mat.shape[0]

        data = Data(edge_index=edge_index, edge_attr=edge_attr, size=size,
                    num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        # String representation for dataset object
        return (f'{self.__class__.__name__}(group={self.group}, '
                f'name={self.name})')
