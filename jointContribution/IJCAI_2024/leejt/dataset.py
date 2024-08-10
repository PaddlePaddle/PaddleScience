import paddle
from paddle.io import Dataset


class CustomDataset(Dataset):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class CustomDataLoader(paddle.io.DataLoader):
    def __init__(
        self, dataset, batch_size=1, shuffle=False, collate_fn=None, num_workers=0
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
        )
        self.dataset = dataset
        self.batch_size = batch_size
        self.index = 0

    def collate_fn(self, batch):
        return batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration

        batch_data = [
            self.dataset[i]
            for i in range(
                self.index, min(self.index + self.batch_size, len(self.dataset))
            )
        ]
        self.index += self.batch_size

        if len(batch_data) == 1:
            return batch_data[0]
        return batch_data
