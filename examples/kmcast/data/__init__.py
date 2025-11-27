import paddle
from data.LRHR_dataset import LRHRDataset as D


def create_dataloader(dataset, dataset_opt, mode):
    """create dataloader"""
    if mode == "train":
        return paddle.io.DataLoader(
            dataset=dataset,
            batch_size=dataset_opt.train.batch_size,
            shuffle=dataset_opt.train.use_shuffle,
            num_workers=dataset_opt.train.num_workers,
        )
    elif mode == "eval":
        return paddle.io.DataLoader(
            dataset=dataset,
            batch_size=dataset_opt.eval.batch_size,
            shuffle=False,
            num_workers=1,
        )
    else:
        raise NotImplementedError("Dataloader [{:s}] is not found.".format(mode))


def create_dataset(dataset_opt, mode):
    """create dataset"""
    if mode == "train":
        dataset = D(
            dataset_opt=dataset_opt, split=mode, data_len=dataset_opt.train.data_len
        )
    else:
        dataset = D(
            dataset_opt=dataset_opt, split=mode, data_len=dataset_opt.eval.data_len
        )
    return dataset
