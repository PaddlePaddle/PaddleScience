import datetime
import os
from copy import deepcopy
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import h5py
import numpy as np
import paddle
import paddle.nn.functional as F
import pandas as pd
from paddle import io

# SEVIR Dataset constants
SEVIR_DATA_TYPES = ["vis", "ir069", "ir107", "vil", "lght"]
SEVIR_RAW_DTYPES = {
    "vis": np.int16,
    "ir069": np.int16,
    "ir107": np.int16,
    "vil": np.uint8,
    "lght": np.int16,
}
LIGHTING_FRAME_TIMES = np.arange(-120.0, 125.0, 5) * 60
SEVIR_DATA_SHAPE = {
    "lght": (48, 48),
}
PREPROCESS_SCALE_SEVIR = {
    "vis": 1,  # Not utilized in original paper
    "ir069": 1 / 1174.68,
    "ir107": 1 / 2562.43,
    "vil": 1 / 47.54,
    "lght": 1 / 0.60517,
}
PREPROCESS_OFFSET_SEVIR = {
    "vis": 0,  # Not utilized in original paper
    "ir069": 3683.58,
    "ir107": 1552.80,
    "vil": -33.44,
    "lght": -0.02990,
}
PREPROCESS_SCALE_01 = {
    "vis": 1,
    "ir069": 1,
    "ir107": 1,
    "vil": 1 / 255,  # currently the only one implemented
    "lght": 1,
}
PREPROCESS_OFFSET_01 = {
    "vis": 0,
    "ir069": 0,
    "ir107": 0,
    "vil": 0,  # currently the only one implemented
    "lght": 0,
}


def change_layout_np(data, in_layout="NHWT", out_layout="NHWT", ret_contiguous=False):
    # first convert to 'NHWT'
    if in_layout == "NHWT":
        pass
    elif in_layout == "NTHW":
        data = np.transpose(data, axes=(0, 2, 3, 1))
    elif in_layout == "NWHT":
        data = np.transpose(data, axes=(0, 2, 1, 3))
    elif in_layout == "NTCHW":
        data = data[:, :, 0, :, :]
        data = np.transpose(data, axes=(0, 2, 3, 1))
    elif in_layout == "NTHWC":
        data = data[:, :, :, :, 0]
        data = np.transpose(data, axes=(0, 2, 3, 1))
    elif in_layout == "NTWHC":
        data = data[:, :, :, :, 0]
        data = np.transpose(data, axes=(0, 3, 2, 1))
    elif in_layout == "TNHW":
        data = np.transpose(data, axes=(1, 2, 3, 0))
    elif in_layout == "TNCHW":
        data = data[:, :, 0, :, :]
        data = np.transpose(data, axes=(1, 2, 3, 0))
    else:
        raise NotImplementedError(f"{in_layout} is invalid.")

    if out_layout == "NHWT":
        pass
    elif out_layout == "NTHW":
        data = np.transpose(data, axes=(0, 3, 1, 2))
    elif out_layout == "NWHT":
        data = np.transpose(data, axes=(0, 2, 1, 3))
    elif out_layout == "NTCHW":
        data = np.transpose(data, axes=(0, 3, 1, 2))
        data = np.expand_dims(data, axis=2)
    elif out_layout == "NTHWC":
        data = np.transpose(data, axes=(0, 3, 1, 2))
        data = np.expand_dims(data, axis=-1)
    elif out_layout == "NTWHC":
        data = np.transpose(data, axes=(0, 3, 2, 1))
        data = np.expand_dims(data, axis=-1)
    elif out_layout == "TNHW":
        data = np.transpose(data, axes=(3, 0, 1, 2))
    elif out_layout == "TNCHW":
        data = np.transpose(data, axes=(3, 0, 1, 2))
        data = np.expand_dims(data, axis=2)
    else:
        raise NotImplementedError(f"{out_layout} is invalid.")
    if ret_contiguous:
        data = data.ascontiguousarray()
    return data


def change_layout_paddle(
    data, in_layout="NHWT", out_layout="NHWT", ret_contiguous=False
):
    # first convert to 'NHWT'
    if in_layout == "NHWT":
        pass
    elif in_layout == "NTHW":
        data = data.transpose(perm=[0, 2, 3, 1])
    elif in_layout == "NTCHW":
        data = data[:, :, 0, :, :]
        data = data.transpose(perm=[0, 2, 3, 1])
    elif in_layout == "NTHWC":
        data = data[:, :, :, :, 0]
        data = data.transpose(perm=[0, 2, 3, 1])
    elif in_layout == "TNHW":
        data = data.transpose(perm=[1, 2, 3, 0])
    elif in_layout == "TNCHW":
        data = data[:, :, 0, :, :]
        data = data.transpose(perm=[1, 2, 3, 0])
    else:
        raise NotImplementedError(f"{in_layout} is invalid.")

    if out_layout == "NHWT":
        pass
    elif out_layout == "NTHW":
        data = data.transpose(perm=[0, 3, 1, 2])
    elif out_layout == "NTCHW":
        data = data.transpose(perm=[0, 3, 1, 2])
        data = paddle.unsqueeze(data, axis=2)
    elif out_layout == "NTHWC":
        data = data.transpose(perm=[0, 3, 1, 2])
        data = paddle.unsqueeze(data, axis=-1)
    elif out_layout == "TNHW":
        data = data.transpose(perm=[3, 0, 1, 2])
    elif out_layout == "TNCHW":
        data = data.transpose(perm=[3, 0, 1, 2])
        data = paddle.unsqueeze(data, axis=2)
    else:
        raise NotImplementedError(f"{out_layout} is invalid.")
    return data


def path_splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


class SEVIRDataset(io.Dataset):
    """The Storm EVent ImagRy dataset.

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("input",).
        label_keys (Tuple[str, ...]): Name of label keys, such as ("output",).
        data_dir (str): The path of the dataset.
        weight_dict (Optional[Dict[str, Union[Callable, float]]]): Define the weight of each constraint variable. Defaults to None.
        data_types (Sequence[str], optional): A subset of SEVIR_DATA_TYPES. Defaults to [ "vil", ].
        seq_len (int, optional): The length of the data sequences. Should be smaller than the max length raw_seq_len. Defaults to 49.
        raw_seq_len (int, optional): The length of the raw data sequences. Defaults to 49.
        sample_mode (str, optional): The mode of sampling, eg.'random' or 'sequent'. Defaults to "sequent".
        stride (int, optional): Useful when sample_mode == 'sequent'
            stride must not be smaller than out_len to prevent data leakage in testing. Defaults to 12.
        batch_size (int, optional): The batch size. Defaults to 1.
        layout (str, optional): consists of batch_size 'N', seq_len 'T', channel 'C', height 'H', width 'W'
            The layout of sampled data. Raw data layout is 'NHWT'.
            valid layout: 'NHWT', 'NTHW', 'NTCHW', 'TNHW', 'TNCHW'. Defaults to "NHWT".
        in_len (int, optional): The length of input data. Defaults to 13.
        out_len (int, optional): The length of output data. Defaults to 12.
        num_shard (int, optional): Split the whole dataset into num_shard parts for distributed training. Defaults to 1.
        rank (int, optional): Rank of the current process within num_shard. Defaults to 0.
        split_mode (str, optional): if 'ceil', all `num_shard` dataloaders have the same length = ceil(total_len / num_shard).
            Different dataloaders may have some duplicated data batches, if the total size of datasets is not divided by num_shard.
            if 'floor', all `num_shard` dataloaders have the same length = floor(total_len / num_shard).
            The last several data batches may be wasted, if the total size of datasets is not divided by num_shard.
            if 'uneven', the last datasets has larger length when the total length is not divided by num_shard.
            The uneven split leads to synchronization error in dist.all_reduce() or dist.barrier().
            See related issue: https://github.com/pytorch/pytorch/issues/33148
            Notice: this also affects the behavior of `self.use_up`. Defaults to "uneven".
        start_date (datetime.datetime, optional): Start time of SEVIR samples to generate. Defaults to None.
        end_date (datetime.datetime, optional): End time of SEVIR samples to generate. Defaults to None.
        datetime_filter (function, optional): Mask function applied to time_utc column of catalog (return true to keep the row).
            Pass function of the form   lambda t : COND(t)
            Example:  lambda t: np.logical_and(t.dt.hour>=13,t.dt.hour<=21)  # Generate only day-time events. Defaults to None.
        catalog_filter (function, optional): function or None or 'default'
            Mask function applied to entire catalog dataframe (return true to keep row).
            Pass function of the form lambda catalog:  COND(catalog)
            Example:  lambda c:  [s[0]=='S' for s in c.id]   # Generate only the 'S' events
        shuffle (bool, optional): If True, data samples are shuffled before each epoch. Defaults to False.
        shuffle_seed (int, optional): Seed to use for shuffling. Defaults to 1.
        output_type (np.dtype, optional): The type of generated tensors. Defaults to np.float32.
        preprocess (bool, optional): If True, self.preprocess_data_dict(data_dict) is called before each sample generated. Defaults to True.
        rescale_method (str, optional): The method of rescale. Defaults to "01".
        downsample_dict (Dict[str, Sequence[int]], optional): downsample_dict.keys() == data_types. downsample_dict[key] is a Sequence of
            (t_factor, h_factor, w_factor),representing the downsampling factors of all dimensions. Defaults to None.
        verbose (bool, optional): Verbose when opening raw data files. Defaults to False.
        training (str, optional): Training pathse. Defaults to "train".
    """

    # Whether support batch indexing for speeding up fetching process.
    batch_index: bool = False

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        data_dir: str,
        weight_dict: Optional[Dict[str, float]] = None,
        data_types: Sequence[str] = [
            "vil",
        ],
        seq_len: int = 49,
        raw_seq_len: int = 49,
        sample_mode: str = "sequent",
        stride: int = 12,
        batch_size: int = 1,
        layout: str = "NHWT",
        in_len: int = 13,
        out_len: int = 12,
        num_shard: int = 1,
        rank: int = 0,
        split_mode: str = "uneven",
        start_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
        datetime_filter=None,
        catalog_filter="default",
        shuffle: bool = False,
        shuffle_seed: int = 1,
        output_type=np.float32,
        preprocess: bool = True,
        rescale_method: str = "01",
        downsample_dict: Dict[str, Sequence[int]] = None,
        verbose: bool = False,
        training="train",
    ):
        super(SEVIRDataset, self).__init__()
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.data_dir = data_dir
        self.weight_dict = {} if weight_dict is None else weight_dict
        if weight_dict is not None:
            self.weight_dict = {key: 1.0 for key in self.label_keys}
            self.weight_dict.update(weight_dict)

        # sevir
        SEVIR_ROOT_DIR = os.path.join(self.data_dir, "sevir")
        sevir_catalog = os.path.join(SEVIR_ROOT_DIR, "CATALOG.csv")
        sevir_data_dir = os.path.join(SEVIR_ROOT_DIR, "data")
        # sevir-lr
        # SEVIR_ROOT_DIR = os.path.join(self.data_dir, "sevir_lr")
        # SEVIR_CATALOG = os.path.join(SEVIR_ROOT_DIR, "CATALOG.csv")
        # SEVIR_DATA_DIR = os.path.join(SEVIR_ROOT_DIR, "data")

        if data_types is None:
            data_types = SEVIR_DATA_TYPES
        else:
            assert set(data_types).issubset(SEVIR_DATA_TYPES)

        # configs which should not be modified
        self._dtypes = SEVIR_RAW_DTYPES
        self.lght_frame_times = LIGHTING_FRAME_TIMES
        self.data_shape = SEVIR_DATA_SHAPE

        self.raw_seq_len = raw_seq_len
        self.seq_len = seq_len

        if seq_len > raw_seq_len:
            raise ValueError("seq_len must be small than raw_seq_len")

        if sample_mode not in ["random", "sequent"]:
            raise ValueError("sample_mode must be 'random' or 'sequent'.")

        self.sample_mode = sample_mode
        self.stride = stride
        self.batch_size = batch_size
        valid_layout = ("NHWT", "NTHW", "NTCHW", "NTHWC", "TNHW", "TNCHW")
        if layout not in valid_layout:
            raise ValueError(
                f"Invalid layout = {layout}! Must be one of {valid_layout}."
            )
        self.layout = layout
        self.in_len = in_len
        self.out_len = out_len

        self.num_shard = num_shard
        self.rank = rank
        valid_split_mode = ("ceil", "floor", "uneven")
        if split_mode not in valid_split_mode:
            raise ValueError(
                f"Invalid split_mode: {split_mode}! Must be one of {valid_split_mode}."
            )
        self.split_mode = split_mode
        self._samples = None
        self._hdf_files = {}
        self.data_types = data_types
        if isinstance(sevir_catalog, str):
            self.catalog = pd.read_csv(
                sevir_catalog, parse_dates=["time_utc"], low_memory=False
            )
        else:
            self.catalog = sevir_catalog
        self.sevir_data_dir = sevir_data_dir
        self.datetime_filter = datetime_filter
        self.catalog_filter = catalog_filter
        self.start_date = start_date
        self.end_date = end_date
        # train val test split
        self.start_date = (
            datetime.datetime(*start_date) if start_date is not None else None
        )
        self.end_date = datetime.datetime(*end_date) if end_date is not None else None

        self.shuffle = shuffle
        self.shuffle_seed = int(shuffle_seed)
        self.output_type = output_type
        self.preprocess = preprocess
        self.downsample_dict = downsample_dict
        self.rescale_method = rescale_method
        self.verbose = verbose

        if self.start_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc > self.start_date]
        if self.end_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc <= self.end_date]
        if self.datetime_filter:
            self.catalog = self.catalog[self.datetime_filter(self.catalog.time_utc)]

        if self.catalog_filter is not None:
            if self.catalog_filter == "default":
                self.catalog_filter = lambda c: c.pct_missing == 0
            self.catalog = self.catalog[self.catalog_filter(self.catalog)]

        self._compute_samples()
        self._open_files(verbose=self.verbose)

    def _compute_samples(self):
        """
        Computes the list of samples in catalog to be used. This sets self._samples
        """
        # locate all events containing colocated data_types
        imgt = self.data_types
        imgts = set(imgt)
        filtcat = self.catalog[
            np.logical_or.reduce([self.catalog.img_type == i for i in imgt])
        ]
        # remove rows missing one or more requested img_types
        filtcat = filtcat.groupby("id").filter(
            lambda x: imgts.issubset(set(x["img_type"]))
        )
        # If there are repeated IDs, remove them (this is a bug in SEVIR)
        # TODO: is it necessary to keep one of them instead of deleting them all
        filtcat = filtcat.groupby("id").filter(lambda x: x.shape[0] == len(imgt))
        self._samples = filtcat.groupby("id").apply(
            lambda df: self._df_to_series(df, imgt)
        )
        if self.shuffle:
            self.shuffle_samples()

    def shuffle_samples(self):
        self._samples = self._samples.sample(frac=1, random_state=self.shuffle_seed)

    def _df_to_series(self, df, imgt):
        d = {}
        df = df.set_index("img_type")
        for i in imgt:
            s = df.loc[i]
            idx = s.file_index if i != "lght" else s.id
            d.update({f"{i}_filename": [s.file_name], f"{i}_index": [idx]})

        return pd.DataFrame(d)

    def _open_files(self, verbose=True):
        """
        Opens HDF files
        """
        imgt = self.data_types
        hdf_filenames = []
        for t in imgt:
            hdf_filenames += list(np.unique(self._samples[f"{t}_filename"].values))
        self._hdf_files = {}
        for f in hdf_filenames:
            if verbose:
                print("Opening HDF5 file for reading", f)
            self._hdf_files[f] = h5py.File(self.sevir_data_dir + "/" + f, "r")

    def close(self):
        """
        Closes all open file handles
        """
        for f in self._hdf_files:
            self._hdf_files[f].close()
        self._hdf_files = {}

    @property
    def num_seq_per_event(self):
        return 1 + (self.raw_seq_len - self.seq_len) // self.stride

    @property
    def total_num_seq(self):
        """
        The total number of sequences within each shard.
        Notice that it is not the product of `self.num_seq_per_event` and `self.total_num_event`.
        """
        return int(self.num_seq_per_event * self.num_event)

    @property
    def total_num_event(self):
        """
        The total number of events in the whole dataset, before split into different shards.
        """
        return int(self._samples.shape[0])

    @property
    def start_event_idx(self):
        """
        The event idx used in certain rank should satisfy event_idx >= start_event_idx
        """
        return self.total_num_event // self.num_shard * self.rank

    @property
    def end_event_idx(self):
        """
        The event idx used in certain rank should satisfy event_idx < end_event_idx

        """
        if self.split_mode == "ceil":
            _last_start_event_idx = (
                self.total_num_event // self.num_shard * (self.num_shard - 1)
            )
            _num_event = self.total_num_event - _last_start_event_idx
            return self.start_event_idx + _num_event
        elif self.split_mode == "floor":
            return self.total_num_event // self.num_shard * (self.rank + 1)
        else:  # self.split_mode == 'uneven':
            if self.rank == self.num_shard - 1:  # the last process
                return self.total_num_event
            else:
                return self.total_num_event // self.num_shard * (self.rank + 1)

    @property
    def num_event(self):
        """
        The number of events split into each rank
        """
        return self.end_event_idx - self.start_event_idx

    def __len__(self):
        """
        Used only when self.sample_mode == 'sequent'
        """
        return self.total_num_seq // self.batch_size

    def _read_data(self, row, data):
        """
        Iteratively read data into data dict. Finally data[imgt] gets shape (batch_size, height, width, raw_seq_len).

        Args:
            row (Dict,optional): A series with fields IMGTYPE_filename, IMGTYPE_index, IMGTYPE_time_index.
            data (Dict,optional): , data[imgt] is a data tensor with shape = (tmp_batch_size, height, width, raw_seq_len).

        Returns:
            data (np.array): Updated data. Updated shape = (tmp_batch_size + 1, height, width, raw_seq_len).
        """

        imgtyps = np.unique([x.split("_")[0] for x in list(row.keys())])
        for t in imgtyps:
            fname = row[f"{t}_filename"]
            idx = row[f"{t}_index"]
            t_slice = slice(0, None)
            # Need to bin lght counts into grid
            if t == "lght":
                lght_data = self._hdf_files[fname][idx][:]
                data_i = self._lght_to_grid(lght_data, t_slice)
            else:
                data_i = self._hdf_files[fname][t][idx : idx + 1, :, :, t_slice]
            data[t] = (
                np.concatenate((data[t], data_i), axis=0) if (t in data) else data_i
            )
        return data

    def _lght_to_grid(self, data, t_slice=slice(0, None)):
        """
        Converts Nx5 lightning data matrix into a 2D grid of pixel counts
        """
        # out_size = (48,48,len(self.lght_frame_times)-1) if isinstance(t_slice,(slice,)) else (48,48)
        out_size = (
            (*self.data_shape["lght"], len(self.lght_frame_times))
            if t_slice.stop is None
            else (*self.data_shape["lght"], 1)
        )
        if data.shape[0] == 0:
            return np.zeros((1,) + out_size, dtype=np.float32)

        # filter out points outside the grid
        x, y = data[:, 3], data[:, 4]
        m = np.logical_and.reduce([x >= 0, x < out_size[0], y >= 0, y < out_size[1]])
        data = data[m, :]
        if data.shape[0] == 0:
            return np.zeros((1,) + out_size, dtype=np.float32)

        # Filter/separate times
        t = data[:, 0]
        if t_slice.stop is not None:  # select only one time bin
            if t_slice.stop > 0:
                if t_slice.stop < len(self.lght_frame_times):
                    tm = np.logical_and(
                        t >= self.lght_frame_times[t_slice.stop - 1],
                        t < self.lght_frame_times[t_slice.stop],
                    )
                else:
                    tm = t >= self.lght_frame_times[-1]
            else:  # special case:  frame 0 uses lght from frame 1
                tm = np.logical_and(
                    t >= self.lght_frame_times[0], t < self.lght_frame_times[1]
                )
            # tm=np.logical_and( (t>=FRAME_TIMES[t_slice],t<FRAME_TIMES[t_slice+1]) )

            data = data[tm, :]
            z = np.zeros(data.shape[0], dtype=np.int64)
        else:  # compute z coordinate based on bin location times
            z = np.digitize(t, self.lght_frame_times) - 1
            z[z == -1] = 0  # special case:  frame 0 uses lght from frame 1

        x = data[:, 3].astype(np.int64)
        y = data[:, 4].astype(np.int64)

        k = np.ravel_multi_index(np.array([y, x, z]), out_size)
        n = np.bincount(k, minlength=np.prod(out_size))
        return np.reshape(n, out_size).astype(np.int16)[np.newaxis, :]

    def _load_event_batch(self, event_idx, event_batch_size):
        """
        Loads a selected batch of events (not batch of sequences) into memory.

        Args:
            idx (int): The index of the event in the batch.
            event_batch_size (int): event_batch[i] = all_type_i_available_events[idx:idx + event_batch_size]
        Returns:
            event_batch (List[np.array,...]): list of event batches.
                event_batch[i] is the event batch of the i-th data type.
                Each event_batch[i] is a np.ndarray with shape = (event_batch_size, height, width, raw_seq_len)
        """
        event_idx_slice_end = event_idx + event_batch_size
        pad_size = 0
        if event_idx_slice_end > self.end_event_idx:
            pad_size = event_idx_slice_end - self.end_event_idx
            event_idx_slice_end = self.end_event_idx
        pd_batch = self._samples.iloc[event_idx:event_idx_slice_end]
        data = {}
        for index, row in pd_batch.iterrows():
            data = self._read_data(row, data)
        if pad_size > 0:
            event_batch = []
            for t in self.data_types:
                pad_shape = [
                    pad_size,
                ] + list(data[t].shape[1:])
                data_pad = np.concatenate(
                    (
                        data[t].astype(self.output_type),
                        np.zeros(pad_shape, dtype=self.output_type),
                    ),
                    axis=0,
                )
                event_batch.append(data_pad)
        else:
            event_batch = [data[t].astype(self.output_type) for t in self.data_types]
        return event_batch

    def __iter__(self):
        return self

    @staticmethod
    def preprocess_data_dict(data_dict, data_types=None, layout="NHWT", rescale="01"):
        """The preprocess of data dict.
        Args:
            data_dict (Dict[str, Union[np.ndarray, paddle.Tensor]]): The dict of data.
            data_types (Sequence[str]) : The data types that we want to rescale. This mainly excludes "mask" from preprocessing.
            layout (str) : consists of batch_size 'N', seq_len 'T', channel 'C', height 'H', width 'W'.
            rescale (str):
                'sevir': use the offsets and scale factors in original implementation.
                '01': scale all values to range 0 to 1, currently only supports 'vil'.
        Returns:
            data_dict (Dict[str, Union[np.ndarray, paddle.Tensor]]) : preprocessed data.
        """

        if rescale == "sevir":
            scale_dict = PREPROCESS_SCALE_SEVIR
            offset_dict = PREPROCESS_OFFSET_SEVIR
        elif rescale == "01":
            scale_dict = PREPROCESS_SCALE_01
            offset_dict = PREPROCESS_OFFSET_01
        else:
            raise ValueError(f"Invalid rescale option: {rescale}.")
        if data_types is None:
            data_types = data_dict.keys()
        for key, data in data_dict.items():
            if key in data_types:
                if isinstance(data, np.ndarray):
                    data = scale_dict[key] * (
                        data.astype(np.float32) + offset_dict[key]
                    )
                    data = change_layout_np(
                        data=data, in_layout="NHWT", out_layout=layout
                    )
                elif isinstance(data, paddle.Tensor):
                    data = scale_dict[key] * (data.astype("float32") + offset_dict[key])
                    data = change_layout_paddle(
                        data=data, in_layout="NHWT", out_layout=layout
                    )
                data_dict[key] = data
        return data_dict

    @staticmethod
    def process_data_dict_back(data_dict, data_types=None, rescale="01"):
        if rescale == "sevir":
            scale_dict = PREPROCESS_SCALE_SEVIR
            offset_dict = PREPROCESS_OFFSET_SEVIR
        elif rescale == "01":
            scale_dict = PREPROCESS_SCALE_01
            offset_dict = PREPROCESS_OFFSET_01
        else:
            raise ValueError(f"Invalid rescale option: {rescale}.")
        if data_types is None:
            data_types = data_dict.keys()
        for key in data_types:
            data = data_dict[key]
            data = data.astype("float32") / scale_dict[key] - offset_dict[key]
            data_dict[key] = data
        return data_dict

    @staticmethod
    def data_dict_to_tensor(data_dict, data_types=None):
        """
        Convert each element in data_dict to paddle.Tensor (copy without grad).
        """
        ret_dict = {}
        if data_types is None:
            data_types = data_dict.keys()
        for key, data in data_dict.items():
            if key in data_types:
                if isinstance(data, paddle.Tensor):
                    ret_dict[key] = data.detach().clone()
                elif isinstance(data, np.ndarray):
                    ret_dict[key] = paddle.to_tensor(data)
                else:
                    raise ValueError(
                        f"Invalid data type: {type(data)}. Should be paddle.Tensor or np.ndarray"
                    )
            else:  # key == "mask"
                ret_dict[key] = data
        return ret_dict

    @staticmethod
    def downsample_data_dict(
        data_dict, data_types=None, factors_dict=None, layout="NHWT"
    ):
        """The downsample of data.

        Args:
            data_dict (Dict[str, Union[np.array, paddle.Tensor]]): The dict of data.
            factors_dict ( Optional[Dict[str, Sequence[int]]]):each element `factors` is a Sequence of int, representing (t_factor,
                  h_factor, w_factor)

        Returns:
            downsampled_data_dict (Dict[str, paddle.Tensor]): Modify on a deep copy of data_dict instead of directly modifying the original
              data_dict
        """

        if factors_dict is None:
            factors_dict = {}
        if data_types is None:
            data_types = data_dict.keys()
        downsampled_data_dict = SEVIRDataset.data_dict_to_tensor(
            data_dict=data_dict, data_types=data_types
        )  # make a copy
        for key, data in data_dict.items():
            factors = factors_dict.get(key, None)
            if factors is not None:
                downsampled_data_dict[key] = change_layout_paddle(
                    data=downsampled_data_dict[key], in_layout=layout, out_layout="NTHW"
                )
                # downsample t dimension
                t_slice = [
                    slice(None, None),
                ] * 4
                t_slice[1] = slice(None, None, factors[0])
                downsampled_data_dict[key] = downsampled_data_dict[key][tuple(t_slice)]
                # downsample spatial dimensions
                downsampled_data_dict[key] = F.avg_pool2d(
                    input=downsampled_data_dict[key],
                    kernel_size=(factors[1], factors[2]),
                )

                downsampled_data_dict[key] = change_layout_paddle(
                    data=downsampled_data_dict[key], in_layout="NTHW", out_layout=layout
                )

        return downsampled_data_dict

    def layout_to_in_out_slice(
        self,
    ):
        t_axis = self.layout.find("T")
        num_axes = len(self.layout)
        in_slice = [
            slice(None, None),
        ] * num_axes
        out_slice = deepcopy(in_slice)
        in_slice[t_axis] = slice(None, self.in_len)
        if self.out_len is None:
            out_slice[t_axis] = slice(self.in_len, None)
        else:
            out_slice[t_axis] = slice(self.in_len, self.in_len + self.out_len)
        return in_slice, out_slice

    def __getitem__(self, index):
        event_idx = (index * self.batch_size) // self.num_seq_per_event
        seq_idx = (index * self.batch_size) % self.num_seq_per_event
        num_sampled = 0
        sampled_idx_list = []  # list of (event_idx, seq_idx) records
        while num_sampled < self.batch_size:
            sampled_idx_list.append({"event_idx": event_idx, "seq_idx": seq_idx})
            seq_idx += 1
            if seq_idx >= self.num_seq_per_event:
                event_idx += 1
                seq_idx = 0
            num_sampled += 1

        start_event_idx = sampled_idx_list[0]["event_idx"]
        event_batch_size = sampled_idx_list[-1]["event_idx"] - start_event_idx + 1

        event_batch = self._load_event_batch(
            event_idx=start_event_idx, event_batch_size=event_batch_size
        )
        ret_dict = {}
        for sampled_idx in sampled_idx_list:
            batch_slice = [
                sampled_idx["event_idx"] - start_event_idx,
            ]  # use [] to keepdim
            seq_slice = slice(
                sampled_idx["seq_idx"] * self.stride,
                sampled_idx["seq_idx"] * self.stride + self.seq_len,
            )
            for imgt_idx, imgt in enumerate(self.data_types):
                sampled_seq = event_batch[imgt_idx][batch_slice, :, :, seq_slice]
                if imgt in ret_dict:
                    ret_dict[imgt] = np.concatenate(
                        (ret_dict[imgt], sampled_seq), axis=0
                    )
                else:
                    ret_dict.update({imgt: sampled_seq})

        ret_dict = self.data_dict_to_tensor(
            data_dict=ret_dict, data_types=self.data_types
        )
        if self.preprocess:
            ret_dict = self.preprocess_data_dict(
                data_dict=ret_dict,
                data_types=self.data_types,
                layout=self.layout,
                rescale=self.rescale_method,
            )

        if self.downsample_dict is not None:
            ret_dict = self.downsample_data_dict(
                data_dict=ret_dict,
                data_types=self.data_types,
                factors_dict=self.downsample_dict,
                layout=self.layout,
            )
        in_slice, out_slice = self.layout_to_in_out_slice()
        data_seq = ret_dict["vil"]
        if isinstance(data_seq, paddle.Tensor):
            data_seq = data_seq.numpy()
        x = data_seq[in_slice[0], in_slice[1], in_slice[2], in_slice[3], in_slice[4]]
        y = data_seq[
            out_slice[0], out_slice[1], out_slice[2], out_slice[3], out_slice[4]
        ]

        weight_item = self.weight_dict
        input_item = {self.input_keys[0]: x}
        label_item = {
            self.label_keys[0]: y,
        }

        return input_item, label_item, weight_item
