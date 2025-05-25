import os
import os.path as osp
import glob
import pickle
from typing import Callable, List, Optional

import paddle
from paddle import Tensor

from paddle_geometric.data import Data, InMemoryDataset, download_url, extract_zip, extract_tar
from paddle_geometric.utils import one_hot, to_undirected


class GEDDataset(InMemoryDataset):
    r"""The GED datasets from the `"Graph Edit Distance Computation via Graph
    Neural Networks" <https://arxiv.org/abs/1808.05689>`_ paper.

    GEDs can be accessed via the global attributes :obj:`ged` and
    :obj:`norm_ged` for all train/train graph pairs and all train/test graph
    pairs:

    .. code-block:: python

        dataset = GEDDataset(root, name="LINUX")
        data1, data2 = dataset[0], dataset[1]
        ged = dataset.ged[data1.i, data2.i]  # GED between `data1` and `data2`.

    Note that GEDs are not available if both graphs are from the test set.
    For evaluation, it is recommended to pair up each graph from the test set
    with each graph in the training set.

    .. note::

        :obj:`ALKANE` is missing GEDs for train/test graph pairs since they are
        not provided in the `official datasets
        <https://github.com/yunshengb/SimGNN>`_.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (one of :obj:`"AIDS700nef"`,
            :obj:`"LINUX"`, :obj:`"ALKANE"`, :obj:`"IMDBMulti"`).
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`paddle_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`paddle_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #graphs
          - #nodes
          - #edges
          - #features
          - #classes
        * - AIDS700nef
          - 700
          - ~8.9
          - ~17.6
          - 29
          - 0
        * - LINUX
          - 1,000
          - ~7.6
          - ~13.9
          - 0
          - 0
        * - ALKANE
          - 150
          - ~8.9
          - ~15.8
          - 0
          - 0
        * - IMDBMulti
          - 1,500
          - ~13.0
          - ~131.9
          - 0
          - 0
    """
    datasets = {
        'AIDS700nef': {
            'id': '10czBPJDEzEDI2tq7Z7mkBjLhj55F-a2z',
            'extract': extract_zip,
            'pickle': '1OpV4bCHjBkdpqI6H5Mg0-BqlA2ee2eBW',
        },
        'LINUX': {
            'id': '1nw0RRVgyLpit4V4XFQyDy0pI6wUEXSOI',
            'extract': extract_tar,
            'pickle': '14FDm3NSnrBvB7eNpLeGy5Bz6FjuCSF5v',
        },
        'ALKANE': {
            'id': '1-LmxaWW3KulLh00YqscVEflbqr0g4cXt',
            'extract': extract_tar,
            'pickle': '15BpvMuHx77-yUGYgM27_sQett02HQNYu',
        },
        'IMDBMulti': {
            'id': '12QxZ7EhYA7pJiF4cO-HuE8szhSOWcfST',
            'extract': extract_zip,
            'pickle': '1wy9VbZvZodkixxVIOuRllC-Lp-0zdoYZ',
        },
    }

    # List of atoms contained in the AIDS700nef dataset:
    types = [
        'O', 'S', 'C', 'N', 'Cl', 'Br', 'B', 'Si', 'Hg', 'I', 'Bi', 'P', 'F',
        'Cu', 'Ho', 'Pd', 'Ru', 'Pt', 'Sn', 'Li', 'Ga', 'Tb', 'As', 'Co', 'Pb',
        'Sb', 'Se', 'Ni', 'Te'
    ]

    def __init__(
        self,
        root: str,
        name: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name
        assert self.name in self.datasets.keys()
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.load(path)
        path = osp.join(self.processed_dir, f'{self.name}_ged.pdparams')
        self.ged = paddle.load(path)
        path = osp.join(self.processed_dir, f'{self.name}_norm_ged.pdparams')
        self.norm_ged = paddle.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return [osp.join(self.name, s) for s in ['train', 'test']]

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{self.name}_{s}.pdparams' for s in ['training', 'test']]

    def download(self) -> None:
        id = self.datasets[self.name]['id']
        path = download_url(id, self.raw_dir)
        extract_fn = self.datasets[self.name]['extract']
        extract_fn(path, self.raw_dir)
        os.unlink(path)

        id = self.datasets[self.name]['pickle']
        path = download_url(id, self.raw_dir)

    def process(self) -> None:
        import networkx as nx

        ids, Ns = [], []
        for r_path, p_path in zip(self.raw_paths, self.processed_paths):
            names = glob.glob(osp.join(r_path, '*.gexf'))
            ids.append(sorted([int(osp.basename(i)[:-5]) for i in names]))

            data_list = []
            for idx in ids[-1]:
                G = nx.read_gexf(osp.join(r_path, f'{idx}.gexf'))
                mapping = {name: i for i, name in enumerate(G.nodes())}
                G = nx.relabel_nodes(G, mapping)
                Ns.append(len(G.nodes()))

                edge_index = paddle.to_tensor(list(G.edges)).T
                if edge_index.numel() == 0:
                    edge_index = paddle.empty([2, 0], dtype='int64')
                edge_index = to_undirected(edge_index, num_nodes=Ns[-1])

                data = Data(edge_index=edge_index)
                data.num_nodes = Ns[-1]

                if self.name == 'AIDS700nef':
                    x = paddle.zeros([data.num_nodes], dtype='int64')
                    for node, info in G.nodes(data=True):
                        x[int(node)] = self.types.index(info['type'])
                    data.x = one_hot(x, num_classes=len(self.types))

                if self.pre_filter and not self.pre_filter(data):
                    continue

                if self.pre_transform:
                    data = self.pre_transform(data)

                data_list.append(data)

            self.save(data_list, p_path)

        assoc = {idx: i for i, idx in enumerate(ids[0])}
        assoc.update({idx: i + len(ids[0]) for i, idx in enumerate(ids[1])})

        path = osp.join(self.raw_dir, self.name, 'ged.pickle')
        mat = paddle.full([len(assoc), len(assoc)], float('inf'))
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            for (_x, _y), g in obj.items():
                mat[assoc[_x], assoc[_y]] = g
                mat[assoc[_y], assoc[_x]] = g

        path = osp.join(self.processed_dir, f'{self.name}_ged.pdparams')
        paddle.save(mat, path)

        N = paddle.to_tensor(Ns, dtype='float32')
        norm_mat = mat / (0.5 * (N.unsqueeze(1) + N.unsqueeze(0)))

        path = osp.join(self.processed_dir, f'{self.name}_norm_ged.pdparams')
        paddle.save(norm_mat, path)

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
