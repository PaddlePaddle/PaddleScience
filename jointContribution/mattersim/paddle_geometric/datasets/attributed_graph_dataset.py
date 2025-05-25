import os
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import paddle

from paddle_geometric.data import Data, InMemoryDataset, download_google_url, extract_zip
from paddle_geometric.io import fs


class AttributedGraphDataset(InMemoryDataset):
    r"""A variety of attributed graph datasets from the
    `"Scaling Attributed Network Embedding to Massive Graphs"
    <https://arxiv.org/abs/2009.00826>`_ paper.

    Args:
        root: Root directory where the dataset should be saved.
        name: The name of the dataset (:obj:`"Wiki"`, :obj:`"Cora"`,
            :obj:`"CiteSeer"`, :obj:`"PubMed"`, :obj:`"BlogCatalog"`,
            :obj:`"PPI"`, :obj:`"Flickr"`, :obj:`"Facebook"`, :obj:`"Twitter"`,
            :obj:`"TWeibo"`, :obj:`"MAG"`).
        transform: A function/transform that takes in a
            :class:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
        pre_transform: A function/transform that takes in a
            :class:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before being saved to
            disk.
        force_reload: Whether to re-process the dataset.
    """
    datasets = {
        'wiki': '1EPhlbziZTQv19OsTrKrAJwsElbVPEbiV',
        'cora': '1FyVnpdsTT-lhkVPotUW8OVeuCi1vi3Ey',
        'citeseer': '1d3uQIpHiemWJPgLgTafi70RFYye7hoCp',
        'pubmed': '1DOK3FfslyJoGXUSCSrK5lzdyLfIwOz6k',
        'blogcatalog': '178PqGqh67RUYMMP6-SoRHDoIBh8ku5FS',
        'ppi': '1dvwRpPT4gGtOcNP_Q-G1TKl9NezYhtez',
        'flickr': '1tZp3EB20fAC27SYWwa-x66_8uGsuU62X',
        'facebook': '12aJWAGCM4IvdGI2fiydDNyWzViEOLZH8',
        'twitter': '1fUYggzZlDrt9JsLsSdRUHiEzQRW1kSA4',
        'tweibo': '1-2xHDPFCsuBuFdQN_7GLleWa8R_t50qU',
        'mag': '1ggraUMrQgdUyA3DjSRzzqMv0jFkU65V5',
    }

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        assert self.name in self.datasets.keys()
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ['attrs.npz', 'edgelist.txt', 'labels.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        dataset_id = self.datasets[self.name]
        path = download_google_url(dataset_id, self.raw_dir, 'data.zip')
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        dataset_path = osp.join(self.raw_dir, f'{self.name}.attr')
        if self.name == 'mag':
            dataset_path = osp.join(self.raw_dir, self.name)
        for name in self.raw_file_names:
            os.rename(osp.join(dataset_path, name), osp.join(self.raw_dir, name))
        fs.rm(dataset_path)

    def process(self) -> None:
        import pandas as pd
        import scipy.sparse as sp

        x = sp.load_npz(self.raw_paths[0]).tocsr()
        if x.shape[-1] > 10000 or self.name == 'mag':
            x = paddle.sparse.sparse_csr_tensor(
                crows=x.indptr.astype(np.int64),
                cols=x.indices.astype(np.int64),
                values=x.data.astype(np.float32),
                shape=x.shape,
            )
        else:
            x = paddle.to_tensor(x.todense(), dtype='float32')

        df = pd.read_csv(self.raw_paths[1], header=None, sep=None, engine='python')
        edge_index = paddle.to_tensor(df.values.T, dtype='int64')

        with open(self.raw_paths[2]) as f:
            rows = f.read().strip().split('\n')
            ys = [[int(y) - 1 for y in row.split()[1:]] for row in rows]
            multilabel = max(len(y) for y in ys) > 1

        if not multilabel:
            y = paddle.to_tensor(ys, dtype='int64').squeeze()
        else:
            num_classes = max(max(y) for y in ys) + 1
            y = paddle.zeros([len(ys), num_classes], dtype='float32')
            for i, row in enumerate(ys):
                for j in row:
                    y[i, j] = 1.0

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'
