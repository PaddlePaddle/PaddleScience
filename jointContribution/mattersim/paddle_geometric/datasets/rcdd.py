import os
from typing import Callable, List, Optional

import numpy as np
import paddle

from paddle_geometric.data import HeteroData, InMemoryDataset, download_url, extract_zip
from paddle_geometric.utils import index_to_mask


class RCDD(InMemoryDataset):
    r"""The risk commodity detection dataset (RCDD) from the
    `"Datasets and Interfaces for Benchmarking Heterogeneous Graph
    Neural Networks" <https://dl.acm.org/doi/10.1145/3583780.3615117>`_ paper.
    RCDD is an industrial-scale heterogeneous graph dataset based on a
    real risk detection scenario from Alibaba's e-commerce platform.
    It consists of 13,806,619 nodes and 157,814,864 edges across 7 node types
    and 7 edge types, respectively.
    """
    url = ('https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/'
           'openhgnn/AliRCD_ICDM.zip')

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'AliRCD_ICDM_nodes.csv',
            'AliRCD_ICDM_edges.csv',
            'AliRCD_ICDM_train_labels.csv',
            'AliRCD_ICDM_test_labels.csv',
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    @property
    def num_classes(self) -> int:
        return 2

    def process(self) -> None:
        import pandas as pd

        data = HeteroData()

        node_df = pd.read_csv(
            self.raw_paths[0],
            header=None,
            names=['node_id', 'node_type', 'node_feat'],
        )

        mapping = paddle.zeros((len(node_df),), dtype=paddle.int64)
        for node_type in node_df['node_type'].unique():
            mask = node_df['node_type'] == node_type
            node_id = paddle.to_tensor(node_df['node_id'][mask].values, dtype=paddle.int64)
            num_nodes = mask.sum()
            mapping[node_id] = paddle.arange(num_nodes, dtype=paddle.int64)
            data[node_type].num_nodes = num_nodes
            x = np.vstack([
                np.asarray(f.split(':'), dtype=np.float32)
                for f in node_df['node_feat'][mask]
            ])
            data[node_type].x = paddle.to_tensor(x)

        edge_df = pd.read_csv(
            self.raw_paths[1],
            header=None,
            names=['src_id', 'dst_id', 'src_type', 'dst_type', 'edge_type'],
        )
        for edge_type in edge_df['edge_type'].unique():
            edge_type_df = edge_df[edge_df['edge_type'] == edge_type]
            src_type = edge_type_df['src_type'].iloc[0]
            dst_type = edge_type_df['dst_type'].iloc[0]
            src = mapping[paddle.to_tensor(edge_type_df['src_id'].values, dtype=paddle.int64)]
            dst = mapping[paddle.to_tensor(edge_type_df['dst_id'].values, dtype=paddle.int64)]
            edge_index = paddle.stack([src, dst], axis=0)
            data[(src_type, edge_type, dst_type)].edge_index = edge_index

        train_df = pd.read_csv(
            self.raw_paths[2],
            header=None,
            names=['node_id', 'label'],
            dtype=int,
        )
        test_df = pd.read_csv(
            self.raw_paths[3],
            header=None,
            sep='\t',
            names=['node_id', 'label'],
            dtype=int,
        )

        train_idx = mapping[paddle.to_tensor(train_df['node_id'].values, dtype=paddle.int64)]
        test_idx = mapping[paddle.to_tensor(test_df['node_id'].values, dtype=paddle.int64)]

        y = paddle.full((data['item'].num_nodes,), -1, dtype=paddle.int64)
        y[train_idx] = paddle.to_tensor(train_df['label'].values, dtype=paddle.int64)
        y[test_idx] = paddle.to_tensor(test_df['label'].values, dtype=paddle.int64)

        train_mask = index_to_mask(train_idx, data['item'].num_nodes)
        test_mask = index_to_mask(test_idx, data['item'].num_nodes)

        data['item'].y = y
        data['item'].train_mask = train_mask
        data['item'].test_mask = test_mask

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
