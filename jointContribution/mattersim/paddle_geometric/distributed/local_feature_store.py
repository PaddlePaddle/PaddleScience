import copy
import os.path as osp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import paddle
from paddle import Tensor

from paddle_geometric.data import FeatureStore, TensorAttr
from paddle_geometric.data.feature_store import _FieldStatus
from paddle_geometric.distributed.partition import load_partition_info
from paddle_geometric.distributed.rpc import (
    RPCCallBase,
    RPCRouter,
    rpc_async,
    rpc_register,
)
from paddle_geometric.io import fs
from paddle_geometric.typing import EdgeType, NodeOrEdgeType, NodeType


class RPCCallFeatureLookup(RPCCallBase):
    r"""A wrapper for RPC calls to the feature store."""
    def __init__(self, dist_feature: FeatureStore):
        super().__init__()
        self.dist_feature = dist_feature

    def rpc_async(self, *args, **kwargs):
        return self.dist_feature._rpc_local_feature_get(*args, **kwargs)

    def rpc_sync(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class LocalTensorAttr(TensorAttr):
    r"""Tensor attribute for storing features without :obj:`index`."""
    def __init__(
        self,
        group_name: Optional[Union[NodeType, EdgeType]] = _FieldStatus.UNSET,
        attr_name: Optional[str] = _FieldStatus.UNSET,
        index=None,
    ):
        super().__init__(group_name, attr_name, index)


class LocalFeatureStore(FeatureStore):
    r"""Implements the :class:`~paddle_geometric.data.FeatureStore` interface
    to act as a local feature store for distributed training.
    """
    def __init__(self):
        super().__init__(tensor_attr_cls=LocalTensorAttr)
        self._feat: Dict[Tuple[Union[NodeType, EdgeType], str], Tensor] = {}
        self._global_id: Dict[Union[NodeType, EdgeType], Tensor] = {}
        self._global_id_to_index: Dict[Union[NodeType, EdgeType], Tensor] = {}
        self.num_partitions: int = 1
        self.partition_idx: int = 0
        self.node_feat_pb: Union[Tensor, Dict[NodeType, Tensor]]
        self.edge_feat_pb: Union[Tensor, Dict[EdgeType, Tensor]]
        self.labels: Optional[Tensor] = None
        self.local_only: bool = False
        self.rpc_router: Optional[RPCRouter] = None
        self.meta: Optional[Dict] = None
        self.rpc_call_id: Optional[int] = None

    @staticmethod
    def key(attr: TensorAttr) -> Tuple[str, str]:
        return (attr.group_name, attr.attr_name)

    def put_global_id(
        self,
        global_id: Tensor,
        group_name: Union[NodeType, EdgeType],
    ) -> bool:
        self._global_id[group_name] = global_id
        self._set_global_id_to_index(group_name)
        return True

    def get_global_id(
        self,
        group_name: Union[NodeType, EdgeType],
    ) -> Optional[Tensor]:
        return self._global_id.get(group_name)

    def remove_global_id(self, group_name: Union[NodeType, EdgeType]) -> bool:
        return self._global_id.pop(group_name) is not None

    def _set_global_id_to_index(self, group_name: Union[NodeType, EdgeType]):
        global_id = self.get_global_id(group_name)

        if global_id is None:
            return

        global_id_to_index = paddle.full([int(global_id.max()) + 1], -1)
        global_id_to_index[global_id] = paddle.arange(global_id.shape[0])
        self._global_id_to_index[group_name] = global_id_to_index

    def _put_tensor(self, tensor: Tensor, attr: TensorAttr) -> bool:
        assert attr.index is None
        self._feat[self.key(attr)] = tensor
        return True

    def _get_tensor(self, attr: TensorAttr) -> Optional[Tensor]:
        tensor = self._feat.get(self.key(attr))
        if tensor is None:
            return None

        if attr.index is None:
            return tensor

        return tensor[attr.index]

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        assert attr.index is None
        return self._feat.pop(self.key(attr), None) is not None

    def get_tensor_from_global_id(self, *args, **kwargs) -> Optional[Tensor]:
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        assert attr.index is not None
        attr = copy.copy(attr)
        attr.index = self._global_id_to_index[attr.group_name][attr.index]
        return self.get_tensor(attr)

    def _get_tensor_size(self, attr: TensorAttr) -> Tuple[int, ...]:
        return self._get_tensor(attr).shape

    def get_all_tensor_attrs(self) -> List[LocalTensorAttr]:
        return [self._tensor_attr_cls.cast(*key) for key in self._feat.keys()]

    def set_rpc_router(self, rpc_router: RPCRouter):
        self.rpc_router = rpc_router
        if not self.local_only:
            if self.rpc_router is None:
                raise ValueError("An RPC router must be provided")
            rpc_call = RPCCallFeatureLookup(self)
            self.rpc_call_id = rpc_register(rpc_call)
        else:
            self.rpc_call_id = None

    def has_edge_attr(self) -> bool:
        has_edge_attr = False
        for k in [key for key in self._feat.keys() if 'edge_attr' in key]:
            try:
                self.get_tensor(k[0], 'edge_attr')
                has_edge_attr = True
            except KeyError:
                pass
        return has_edge_attr

    def lookup_features(
        self,
        index: Tensor,
        is_node_feat: bool = True,
        input_type: Optional[NodeOrEdgeType] = None,
    ) -> paddle.futures.Future:
        remote_fut = self._remote_lookup_features(index, is_node_feat,
                                                  input_type)
        local_feature = self._local_lookup_features(index, is_node_feat,
                                                    input_type)
        res_fut = paddle.futures.Future()

        def when_finish(*_):
            try:
                remote_feature_list = remote_fut.wait()
                result = paddle.zeros(
                    [index.shape[0], local_feature[0].shape[1]],
                    dtype=local_feature[0].dtype,
                )
                result[local_feature[1]] = local_feature[0]
                for remote in remote_feature_list:
                    result[remote[1]] = remote[0]
            except Exception as e:
                res_fut.set_exception(e)
            else:
                res_fut.set_result(result)

        remote_fut.add_done_callback(when_finish)
        return res_fut

    def _local_lookup_features(
        self,
        index: Tensor,
        is_node_feat: bool = True,
        input_type: Optional[Union[NodeType, EdgeType]] = None,
    ) -> Tuple[Tensor, Tensor]:
        pb = self.node_feat_pb if is_node_feat else self.edge_feat_pb
        input_order = paddle.arange(index.shape[0], dtype="int64")
        if self.meta['is_hetero']:
            partition_ids = pb[input_type][index]
        else:
            partition_ids = pb[index]

        local_mask = partition_ids == self.partition_idx
        local_ids = paddle.masked_select(index, local_mask)
        local_index = paddle.masked_select(input_order, local_mask)

        if self.meta['is_hetero']:
            kwargs = dict(group_name=input_type,
                          attr_name='x' if is_node_feat else 'edge_attr')
            ret_feat = self.get_tensor_from_global_id(
                index=local_ids, **kwargs)
        else:
            kwargs = dict(group_name=None,
                          attr_name='x' if is_node_feat else 'edge_attr')
            ret_feat = self.get_tensor_from_global_id(
                index=local_ids, **kwargs)

        return ret_feat, local_index

    def _remote_lookup_features(
        self,
        index: Tensor,
        is_node_feat: bool = True,
        input_type: Optional[Union[NodeType, EdgeType]] = None,
    ) -> paddle.futures.Future:
        pb = self.node_feat_pb if is_node_feat else self.edge_feat_pb
        input_order = paddle.arange(index.shape[0], dtype="int64")
        partition_ids = pb[input_type][index] if self.meta['is_hetero'] else pb[index]

        futs, indexes = [], []
        for pidx in range(self.num_partitions):
            if pidx == self.partition_idx:
                continue
            remote_mask = partition_ids == pidx
            remote_ids = index[remote_mask]
            if remote_ids.shape[0] > 0:
                to_worker = self.rpc_router.get_to_worker(pidx)
                futs.append(
                    rpc_async(
                        to_worker,
                        self.rpc_call_id,
                        args=(remote_ids.cpu(), is_node_feat, input_type),
                    ))
                indexes.append(paddle.masked_select(input_order, remote_mask))
        collect_fut = paddle.futures.collect_all(futs)
        res_fut = paddle.futures.Future()

        def when_finish(*_):
            try:
                fut_list = collect_fut.wait()
                result = [(fut.wait(), indexes[i]) for i, fut in enumerate(fut_list)]
            except Exception as e:
                res_fut.set_exception(e)
            else:
                res_fut.set_result(result)

        collect_fut.add_done_callback(when_finish)
        return res_fut

    def _rpc_local_feature_get(
        self,
        index: Tensor,
        is_node_feat: bool = True,
        input_type: Optional[Union[NodeType, EdgeType]] = None,
    ) -> Tensor:
        kwargs = dict(group_name=input_type if self.meta['is_hetero'] else None,
                      attr_name='x' if is_node_feat else 'edge_attr')
        return self.get_tensor_from_global_id(index=index, **kwargs)

    # Initialization ##########################################################

    @classmethod
    def from_data(
        cls,
        node_id: Tensor,
        x: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        edge_id: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
    ) -> 'LocalFeatureStore':
        feat_store = cls()
        feat_store.put_global_id(node_id, group_name=None)
        if x is not None:
            feat_store.put_tensor(x, group_name=None, attr_name='x')
        if y is not None:
            feat_store.put_tensor(y, group_name=None, attr_name='y')
        if edge_id is not None:
            feat_store.put_global_id(edge_id, group_name=(None, None))
        if edge_attr is not None:
            if edge_id is None:
                raise ValueError("'edge_id' needs to be present in case 'edge_attr' is passed")
            feat_store.put_tensor(edge_attr, group_name=(None, None), attr_name='edge_attr')
        return feat_store

    @classmethod
    def from_hetero_data(
        cls,
        node_id_dict: Dict[NodeType, Tensor],
        x_dict: Optional[Dict[NodeType, Tensor]] = None,
        y_dict: Optional[Dict[NodeType, Tensor]] = None,
        edge_id_dict: Optional[Dict[EdgeType, Tensor]] = None,
        edge_attr_dict: Optional[Dict[EdgeType, Tensor]] = None,
    ) -> 'LocalFeatureStore':
        feat_store = cls()
        for node_type, node_id in node_id_dict.items():
            feat_store.put_global_id(node_id, group_name=node_type)
        if x_dict is not None:
            for node_type, x in x_dict.items():
                feat_store.put_tensor(x, group_name=node_type, attr_name='x')
        if y_dict is not None:
            for node_type, y in y_dict.items():
                feat_store.put_tensor(y, group_name=node_type, attr_name='y')
        if edge_id_dict is not None:
            for edge_type, edge_id in edge_id_dict.items():
                feat_store.put_global_id(edge_id, group_name=edge_type)
        if edge_attr_dict is not None:
            for edge_type, edge_attr in edge_attr_dict.items():
                if edge_id_dict is None or edge_type not in edge_id_dict:
                    raise ValueError("'edge_id' needs to be present in case 'edge_attr' is passed")
                feat_store.put_tensor(edge_attr, group_name=edge_type, attr_name='edge_attr')
        return feat_store

    @classmethod
    def from_partition(cls, root: str, pid: int) -> 'LocalFeatureStore':
        part_dir = osp.join(root, f'part_{pid}')
        assert osp.exists(part_dir)
        feat_store = cls()
        meta, num_partitions, partition_idx, node_pb, edge_pb = load_partition_info(root, pid)
        feat_store.num_partitions = num_partitions
        feat_store.partition_idx = partition_idx
        feat_store.node_feat_pb = node_pb
        feat_store.edge_feat_pb = edge_pb
        feat_store.meta = meta

        node_feats = fs.paddle_load(osp.join(part_dir, 'node_feats.pdparams'))
        edge_feats = fs.paddle_load(osp.join(part_dir, 'edge_feats.pdparams'))

        if not meta['is_hetero'] and node_feats:
            feat_store.put_global_id(node_feats['global_id'], group_name=None)
            for key, value in node_feats['feats'].items():
                feat_store.put_tensor(value, group_name=None, attr_name=key)
            if 'time' in node_feats:
                feat_store.put_tensor(node_feats['time'], group_name=None, attr_name='time')

        if not meta['is_hetero'] and edge_feats:
            if 'global_id' in edge_feats:
                feat_store.put_global_id(edge_feats['global_id'], group_name=(None, None))
            if 'feats' in edge_feats:
                for key, value in edge_feats['feats'].items():
                    feat_store.put_tensor(value, group_name=(None, None), attr_name=key)
            if 'edge_time' in edge_feats:
                feat_store.put_tensor(edge_feats['edge_time'], group_name=(None, None), attr_name='edge_time')

        if meta['is_hetero'] and node_feats:
            for node_type, node_feat in node_feats.items():
                feat_store.put_global_id(node_feat['global_id'], group_name=node_type)
                for key, value in node_feat['feats'].items():
                    feat_store.put_tensor(value, group_name=node_type, attr_name=key)
                if 'time' in node_feat:
                    feat_store.put_tensor(node_feat['time'], group_name=node_type, attr_name='time')

        if meta['is_hetero'] and edge_feats:
            for edge_type, edge_feat in edge_feats.items():
                if 'global_id' in edge_feat:
                    feat_store.put_global_id(edge_feat['global_id'], group_name=edge_type)
                if 'feats' in edge_feat:
                    for key, value in edge_feat['feats'].items():
                        feat_store.put_tensor(value, group_name=edge_type, attr_name=key)
                if 'edge_time' in edge_feat:
                    feat_store.put_tensor(edge_feat['edge_time'], group_name=edge_type, attr_name='edge_time')

        return feat_store
