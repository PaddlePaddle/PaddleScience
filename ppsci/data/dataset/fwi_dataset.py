import os
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
from paddle import io
from paddle import vision


class FWIDataset(io.Dataset):
    """Datasets for full waveform inversion tasks.
        For convenience, in this class, a batch refers to a npy file instead of the batch used during training.

    Args:
        input_keys (Tuple[str, ...]): List of input keys.
        label_keys (Tuple[str, ...]): List of label keys.
        weight: Define the weight dict for loss function.
        anno: Path to annotation file.
        preload: Whether to load the whole dataset into memory.
        sample_ratio: Downsample ratio for seismic data.
        file_size: Number of samples in each npy file.
        transform_data: Transformation applied to data.
        transform_label: Transformation applied to label.

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.FWIDataset(("input", ), ("label", ), "path/to/anno_file") # doctest: +SKIP
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        anno: str,
        weight: Optional[Dict[str, np.ndarray]] = None,
        preload: bool = True,
        sample_ratio: int = 1,
        file_size: int = 500,
        transform_data: Optional[vision.Compose] = None,
        transform_label: Optional[vision.Compose] = None,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.weight = {} if weight is None else weight
        if not os.path.exists(anno):
            print(f"Annotation file {anno} does not exists")
        self.preload = preload
        self.sample_ratio = sample_ratio
        self.file_size = file_size
        self.transform_data = transform_data
        self.transform_label = transform_label
        with open(anno, "r") as f:
            self.batches = f.readlines()
        if preload:
            self.data_list, self.label_list = [], []
            for batch in self.batches:
                data, label = self.load_every(batch)
                self.data_list.append(data)
                if label is not None:
                    self.label_list.append(label)

    def load_every(self, batch):
        batch = batch.split("\t")
        data_path = batch[0] if len(batch) > 1 else batch[0][:-1]
        data = np.load(data_path)[:, :, :: self.sample_ratio, :]
        data = data.astype("float32")
        if len(batch) > 1:
            label_path = batch[1][:-1]
            label = np.load(label_path)
            label = label.astype("float32")
        else:
            label = None

        return data, label

    def __getitem__(self, idx):
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size
        if self.preload:
            data = self.data_list[batch_idx][sample_idx]
            label = (
                self.label_list[batch_idx][sample_idx]
                if len(self.label_list) != 0
                else None
            )
        else:
            data, label = self.load_every(self.batches[batch_idx])
            data = data[sample_idx]
            label = label[sample_idx] if label is not None else None
        if self.transform_data:
            data = self.transform_data(data)
        if self.transform_label and label is not None:
            label = self.transform_label(label)

        input_item = {self.input_keys[0]: data}
        label_item = {self.label_keys[0]: label if label is not None else np.array([])}
        weight_item = self.weight

        return input_item, label_item, weight_item

    def __len__(self):
        return len(self.batches) * self.file_size
