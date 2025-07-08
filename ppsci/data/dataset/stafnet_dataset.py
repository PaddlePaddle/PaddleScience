from typing import Optional
from typing import Tuple

import numpy as np
import paddle
import pandas

try:
    import pgl
except ModuleNotFoundError:
    pass
from paddle import io
from paddle.io import DataLoader
from scipy.spatial.distance import cdist


def gat_lstmcollate_fn(data):
    aq_train_data = []
    mete_train_data = []
    aq_g_list = []
    mete_g_list = []
    label = []
    for unit in data:
        aq_train_data.append(unit[0]["aq_train_data"])
        mete_train_data.append(unit[0]["mete_train_data"])
        aq_g_list = aq_g_list + unit[0]["aq_g_list"]
        mete_g_list = mete_g_list + unit[0]["mete_g_list"]
        label.append(unit[1])
    label = paddle.stack(x=label)
    x = label
    perm_1 = list(range(x.ndim))
    perm_1[1] = 2
    perm_1[2] = 1
    label = paddle.transpose(x=x, perm=perm_1)
    label = paddle.flatten(x=label, start_axis=0, stop_axis=1)
    return {
        "aq_train_data": paddle.stack(x=aq_train_data),
        "mete_train_data": paddle.stack(x=mete_train_data),
        "aq_G": pgl.graph.Graph.batch(aq_g_list),
        "mete_G": pgl.graph.Graph.batch(mete_g_list),
    }, label


class pygmmdataLoader(DataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        args,
        data_dir,
        batch_size,
        shuffle=True,
        num_workers=1,
        training=True,
        T=24,
        t=12,
        collate_fn=gat_lstmcollate_fn,
    ):
        self.T = T
        self.t = t
        self.dataset = STAFNetDataset(args=args, file_path=data_dir)

        super().__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )


class STAFNetDataset(io.Dataset):
    """Dataset class for STAFNet data.

    Args:
        file_path (str): Path to the dataset file.
        input_keys (Optional[Tuple[str, ...]]): Tuple of input keys. Defaults to None.
        label_keys (Optional[Tuple[str, ...]]): Tuple of label keys. Defaults to None.
        seq_len (int): Sequence length. Defaults to 72.
        pred_len (int): Prediction length. Defaults to 48.
        use_edge_attr (bool): Whether to use edge attributes. Defaults to True.

    Examples:
        >>> from ppsci.data.dataset import STAFNetDataset

        >>> dataset = STAFNetDataset(file_path='example.pkl') # doctest: +SKIP

        >>> # get the length of the dataset
        >>> dataset_size = len(dataset) # doctest: +SKIP
        >>> # get the first sample of the data
        >>> first_sample = dataset[0] # doctest: +SKIP
        >>> print("First sample:", first_sample) # doctest: +SKIP
    """

    def __init__(
        self,
        file_path: str,
        input_keys: Optional[Tuple[str, ...]] = None,
        label_keys: Optional[Tuple[str, ...]] = None,
        seq_len: int = 72,
        pred_len: int = 48,
        use_edge_attr: bool = True,
    ):

        self.file_path = file_path
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.use_edge_attr = use_edge_attr

        self.seq_len = seq_len
        self.pred_len = pred_len

        super().__init__()
        if file_path.endswith(".pkl"):
            with open(file_path, "rb") as f:
                self.data = pandas.read_pickle(f)
        self.metedata = self.data["metedata"]
        self.AQdata = self.data["AQdata"]
        self.AQStation_imformation = self.data["AQStation_imformation"]
        self.meteStation_imformation = self.data["meteStation_imformation"]
        mete_coords = np.array(
            self.meteStation_imformation.loc[:, ["经度", "纬度"]]
        ).astype("float64")
        AQ_coords = np.array(self.AQStation_imformation.iloc[:, -2:]).astype("float64")
        self.aq_edge_index, self.aq_edge_attr, self.aq_node_coords = self.get_edge_attr(
            np.array(self.AQStation_imformation.iloc[:, -2:]).astype("float64")
        )
        (
            self.mete_edge_index,
            self.mete_edge_attr,
            self.mete_node_coords,
        ) = self.get_edge_attr(
            np.array(self.meteStation_imformation.loc[:, ["经度", "纬度"]]).astype(
                "float64"
            )
        )

        self.lut = self.find_nearest_point(AQ_coords, mete_coords)

    def __len__(self):
        return len(self.AQdata) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        aq_train_data = paddle.to_tensor(
            data=self.AQdata[idx : idx + self.seq_len + self.pred_len]
        ).astype(dtype="float32")
        mete_train_data = paddle.to_tensor(
            data=self.metedata[idx : idx + self.seq_len + self.pred_len]
        ).astype(dtype="float32")

        input_item = {
            "aq_train_data": aq_train_data,
            "mete_train_data": mete_train_data,
        }
        label_item = {self.label_keys[0]: aq_train_data[-self.pred_len :, :, -7:]}

        return input_item, label_item, {}

    def get_edge_attr(self, node_coords, threshold=0.2):
        # node_coords = paddle.to_tensor(data=node_coords)
        dist_matrix = cdist(node_coords, node_coords)
        edge_index = np.where(dist_matrix < threshold)
        # edge_index = paddle.to_tensor(data=edge_index, dtype='int64')
        start_nodes, end_nodes = edge_index
        edge_lengths = dist_matrix[start_nodes, end_nodes]
        edge_directions = node_coords[end_nodes] - node_coords[start_nodes]
        edge_attr = paddle.to_tensor(
            data=np.concatenate((edge_lengths[:, np.newaxis], edge_directions), axis=1)
        )
        node_coords = paddle.to_tensor(data=node_coords)
        return edge_index, edge_attr, node_coords

    def find_nearest_point(self, A, B):
        nearest_indices = []
        for a in A:
            distances = [np.linalg.norm(a - b) for b in B]
            nearest_indices.append(np.argmin(distances))
        return nearest_indices
