import numpy as np
import paddle
import paddle.nn as nn
import pgl
from scipy.spatial.distance import cdist
import pandas
from paddle.io import Dataset, DataLoader
from paddle import io
from typing import Dict
from typing import Optional
from typing import Tuple

def gat_lstmcollate_fn(data):
    aq_train_data = []
    mete_train_data = []
    aq_g_list = []
    mete_g_list = []
    edge_index = []
    edge_attr = []
    pos = []
    label = []
    for unit in data:
        aq_train_data.append(unit[0]['aq_train_data'])
        mete_train_data.append(unit[0]['mete_train_data'])
        aq_g_list = aq_g_list + unit[0]['aq_g_list']
        mete_g_list = mete_g_list + unit[0]['mete_g_list']
        label.append(unit[1])
    label = paddle.stack(x=label)
    x = label
    perm_1 = list(range(x.ndim))
    perm_1[1] = 2
    perm_1[2] = 1
    label = paddle.transpose(x=x, perm=perm_1)
    label = paddle.flatten(x=label, start_axis=0, stop_axis=1)
    return {'aq_train_data': paddle.stack(x=aq_train_data),
        'mete_train_data': paddle.stack(x=mete_train_data), 
        'aq_G': pgl.graph.Graph.batch(aq_g_list), 
        'mete_G': pgl.graph.Graph.batch(mete_g_list)
        }, label


class pygmmdataLoader(DataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, args, data_dir, batch_size, shuffle=True,
        num_workers=1, training=True, T=24, t=12, collate_fn=gat_lstmcollate_fn
        ):
        self.T = T
        self.t = t
        self.dataset = STAFNetDataset(args=args, file_path=file_path)
        # if get_world_size() > 1:
        #     sampler = paddle.io.DistributedBatchSampler(dataset=self.
        #         dataset, shuffle=shuffle, batch_size=1)
        # else:
        #     sampler = None
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers = num_workers,
             collate_fn=collate_fn)

class STAFNetDataset(io.Dataset):

    def __init__(self, 
            file_path: str,
         input_keys: Optional[Tuple[str, ...]] = None,
         label_keys: Optional[Tuple[str, ...]] = None,
         seq_len: int = 72,
         pred_len: int = 48,
          use_edge_attr=True,):
        """
        root: 数据集保存的地方。
        会产生两个文件夹：
          raw_dir(downloaded dataset) 和 processed_dir(processed data)。
        """
        
        self.file_path = file_path
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.use_edge_attr = use_edge_attr

        self.seq_len = seq_len
        self.pred_len = pred_len
        
        
        super().__init__( )
        if file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                self.data = pandas.read_pickle(f)
        self.metedata = self.data['metedata']
        self.AQdata = self.data['AQdata']
        self.AQStation_imformation = self.data['AQStation_imformation']
        self.meteStation_imformation = self.data['meteStation_imformation']
        mete_coords = np.array(self.meteStation_imformation.loc[:, ['经度','纬度']]).astype('float64')
        AQ_coords = np.array(self.AQStation_imformation.iloc[:, -2:]).astype('float64')
        self.aq_edge_index, self.aq_edge_attr, self.aq_node_coords = (self.get_edge_attr(np.array(self.AQStation_imformation.iloc[:, -2:]).astype('float64')))
        (self.mete_edge_index, self.mete_edge_attr, self.mete_node_coords) = (self.get_edge_attr(np.array(self.meteStation_imformation.loc[:, ['经度', '纬度']]).astype('float64')))

        self.lut = self.find_nearest_point(AQ_coords, mete_coords)
        # self.AQdata = np.concatenate((self.AQdata, self.metedata[:, self.lut, -7:]), axis=2)


    def __len__(self):
        return len(self.AQdata) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        input_data = {}
        aq_train_data = paddle.to_tensor(data=self.AQdata[idx:idx + self.seq_len + self.pred_len]).astype(dtype='float32')
        mete_train_data = paddle.to_tensor(data=self.metedata[idx:idx + self.seq_len + self.pred_len]).astype(dtype='float32')
        aq_g_list = [pgl.Graph(num_nodes=s.shape[0],
            edges=self.aq_edge_index,
            node_feat={
                "feature": s,
                "pos": self.aq_node_coords.astype(dtype='float32')
            },
            edge_feat={
                "edge_feature": self.aq_edge_attr.astype(dtype='float32')
            })for s in aq_train_data[:self.seq_len]]
        
        mete_g_list = [pgl.Graph(num_nodes=s.shape[0],
            edges=self.mete_edge_index,
            node_feat={
                "feature": s,
                "pos": self.mete_node_coords.astype(dtype='float32')
            },
            edge_feat={
                "edge_feature": self.mete_edge_attr.astype(dtype='float32')
            })for s in mete_train_data[:self.seq_len]]
        
        # aq_g_list = [pgl.Graph(x=s, edge_index=self.aq_edge_index, edge_attr=self.aq_edge_attr.astype(dtype='float32'), pos=self.aq_node_coords.astype(dtype='float32')) for s in aq_train_data[:self.seq_len]]
        # mete_g_list = [pgl.Graph((x=s, edge_index=self.
        #     mete_edge_index, edge_attr=self.mete_edge_attr.astype(dtype=
        #     'float32'), pos=self.mete_node_coords.astype(dtype='float32')) for
        #     s in mete_train_data[:self.seq_len]]
        
        
        label = aq_train_data[-self.pred_len:, :, -7:]
        data = {'aq_train_data': aq_train_data, 'mete_train_data':
            mete_train_data, 'aq_g_list': aq_g_list, 'mete_g_list': mete_g_list
            }
        return data, label

    def get_edge_attr(self, node_coords, threshold=0.2):
        # node_coords = paddle.to_tensor(data=node_coords)
        dist_matrix = cdist(node_coords, node_coords)
        edge_index = np.where(dist_matrix < threshold)
        # edge_index = paddle.to_tensor(data=edge_index, dtype='int64')
        start_nodes, end_nodes = edge_index
        edge_lengths = dist_matrix[start_nodes, end_nodes]
        edge_directions = node_coords[end_nodes] - node_coords[start_nodes]
        edge_attr = paddle.to_tensor(data=np.concatenate((edge_lengths[:,
            np.newaxis], edge_directions), axis=1))
        node_coords = paddle.to_tensor(data=node_coords)
        return edge_index, edge_attr, node_coords

    def find_nearest_point(self, A, B):
        nearest_indices = []
        for a in A:
            distances = [np.linalg.norm(a - b) for b in B]
            nearest_indices.append(np.argmin(distances))
        return nearest_indices