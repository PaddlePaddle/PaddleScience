import os
import os.path as osp
from collections import Counter
from typing import Any, Callable, List, Optional

import pandas as pd
import paddle

from paddle_geometric.data import Data, HeteroData, InMemoryDataset, download_url
from paddle_geometric.utils import index_sort


class Entities(InMemoryDataset):
    r"""The relational entities networks :obj:`"AIFB"`, :obj:`"MUTAG"`,
    :obj:`"BGS"` and :obj:`"AM"` from the `"Modeling Relational Data with Graph
    Convolutional Networks" <https://arxiv.org/abs/1703.06103>`_ paper.
    Training and test splits are given by node indices.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"AIFB"`, :obj:`"MUTAG"`,
            :obj:`"BGS"`, :obj:`"AM"`).
        hetero (bool, optional): If set to :obj:`True`, will save the dataset
            as a :class:`~paddle_geometric.data.HeteroData` object.
            (default: :obj:`False`)
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
    """
    url = 'https://data.dgl.ai/dataset/{}.tgz'

    def __init__(
        self,
        root: str,
        name: str,
        hetero: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        self.hetero = hetero
        assert self.name in ['aifb', 'am', 'mutag', 'bgs']
        super().__init__(root, transform, pre_transform, force_reload)
        if hetero:
            self.load(self.processed_paths[0], data_cls=HeteroData)
        else:
            self.load(self.processed_paths[0], data_cls=Data)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            f'{self.name}_stripped.nt.gz',
            'completeDataset.tsv',
            'trainingSet.tsv',
            'testSet.tsv',
        ]

    @property
    def processed_file_names(self) -> str:
        return 'hetero_data.pdparams' if self.hetero else 'data.pdparams'

    def download(self) -> None:
        path = download_url(self.url.format(self.name), self.root)
        os.system(f"tar -xzf {path} -C {self.raw_dir}")
        os.unlink(path)

    def process(self) -> None:
        import gzip
        import rdflib as rdf

        graph_file, task_file, train_file, test_file = self.raw_paths

        with hide_stdout():
            g = rdf.Graph()
            with gzip.open(graph_file, 'rb') as f:
                g.parse(file=f, format='nt')

        freq = Counter(g.predicates())
        relations = sorted(set(g.predicates()), key=lambda p: -freq.get(p, 0))
        subjects = set(g.subjects())
        objects = set(g.objects())
        nodes = list(subjects.union(objects))

        N = len(nodes)
        R = 2 * len(relations)
        relations_dict = {rel: i for i, rel in enumerate(relations)}
        nodes_dict = {str(node): i for i, node in enumerate(nodes)}

        edges = []
        for s, p, o in g.triples((None, None, None)):
            src, dst = nodes_dict[str(s)], nodes_dict[str(o)]
            rel = relations_dict[p]
            edges.append([src, dst, 2 * rel])
            edges.append([dst, src, 2 * rel + 1])

        edge = paddle.to_tensor(edges, dtype='int64').t()
        _, perm = index_sort(N * R * edge[0] + R * edge[1] + edge[2])
        edge = edge[:, perm]

        edge_index, edge_type = edge[:2], edge[2]

        if self.name == 'am':
            label_header = 'label_cateogory'
            nodes_header = 'proxy'
        elif self.name == 'aifb':
            label_header = 'label_affiliation'
            nodes_header = 'person'
        elif self.name == 'mutag':
            label_header = 'label_mutagenic'
            nodes_header = 'bond'
        elif self.name == 'bgs':
            label_header = 'label_lithogenesis'
            nodes_header = 'rock'

        labels_df = pd.read_csv(task_file, sep='\t')
        labels_set = set(labels_df[label_header].values.tolist())
        labels_dict = {lab: i for i, lab in enumerate(list(labels_set))}

        train_labels_df = pd.read_csv(train_file, sep='\t')
        train_idx, train_y = [], []
        for nod, lab in zip(train_labels_df[nodes_header].values,
                            train_labels_df[label_header].values):
            train_idx.append(nodes_dict[nod])
            train_y.append(labels_dict[lab])

        train_idx = paddle.to_tensor(train_idx, dtype='int64')
        train_y = paddle.to_tensor(train_y, dtype='int64')

        test_labels_df = pd.read_csv(test_file, sep='\t')
        test_idx, test_y = [], []
        for nod, lab in zip(test_labels_df[nodes_header].values,
                            test_labels_df[label_header].values):
            test_idx.append(nodes_dict[nod])
            test_y.append(labels_dict[lab])

        test_idx = paddle.to_tensor(test_idx, dtype='int64')
        test_y = paddle.to_tensor(test_y, dtype='int64')

        data = Data(edge_index=edge_index, edge_type=edge_type,
                    train_idx=train_idx, train_y=train_y, test_idx=test_idx,
                    test_y=test_y, num_nodes=N)

        if self.hetero:
            data = data.to_heterogeneous(node_type_names=['v'])

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.upper()}{self.__class__.__name__}()'


class hide_stdout:
    def __enter__(self) -> None:
        self.level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)

    def __exit__(self, *args: Any) -> None:
        logging.getLogger().setLevel(self.level)
