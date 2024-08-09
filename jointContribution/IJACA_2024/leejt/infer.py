import os
import random

import numpy as np
import paddle
import utils.paddle_aux  # NOQA
from dataset import CustomDataLoader
from dataset import CustomDataset
from model import ConvBert
from tqdm import tqdm

root = "./"
mean, std = np.loadtxt(
    f"{root}/Dataset/Testset_track_B/Auxiliary/train_pressure_mean_std.txt"
)
area_min_bounds, area_max_bounds = np.loadtxt(
    f"{root}/Dataset/Testset_track_B/Auxiliary/area_bounds.txt"
)
bounds = np.loadtxt(f"{root}/Dataset/Testset_track_B/Auxiliary/global_bounds.txt")
global_min_bounds = paddle.to_tensor(data=bounds[0]).view(1, -1).astype(dtype="float32")
global_max_bounds = paddle.to_tensor(data=bounds[1]).view(1, -1).astype(dtype="float32")


def seed_everything(seed):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def read_npy_as_data(ids, training=True):
    data_list = []
    for file_id in tqdm(ids):
        press = None
        if training:
            pos = np.load(f"{root}/train_track_B/centroid_{file_id}.npy")
            press = np.load(f"{root}/train_track_B/press_{file_id}.npy")
            area = np.load(f"{root}/train_track_B/area_{file_id}.npy")
            pos = paddle.to_tensor(data=pos).to("float32")
            press = paddle.to_tensor(data=press).to("float32")
            area = paddle.to_tensor(data=area).to("float32")
            press = (press - mean) / std
        else:
            pos = np.load(
                f"{root}/Dataset/Testset_track_B/Inference/centroid_{file_id}.npy"
            )
            area = np.load(
                f"{root}/Dataset/Testset_track_B/Auxiliary/area_{file_id}.npy"
            )
            pos = paddle.to_tensor(data=pos).to("float32")
            area = paddle.to_tensor(data=area).to("float32")
        pos = (pos - global_min_bounds) / (global_max_bounds - global_min_bounds)
        pos = 2 * pos - 1
        area = (area - area_min_bounds) / (area_max_bounds - area_min_bounds)
        data = CustomDataset(pos=pos, y=press, area=area)
        data_list.append(data)
    return data_list


class LpLoss(paddle.nn.Layer):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x, y):
        a = paddle.linalg.norm(x=x - y, p=2, axis=self.dim)
        b = paddle.linalg.norm(x=y, p=2, axis=self.dim)
        return (a / b).mean()


def uneven_chunk(tensor, chunks):
    total_size = tensor.shape[0]
    chunk_size = total_size // chunks
    remainder = total_size % chunks
    chunks_indices = []
    start = 0
    for i in range(chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks_indices.append((start, end))
        start = end
    return [tensor[start:end] for start, end in chunks_indices]


@paddle.no_grad()
def get_preds(data, num_trials=10):
    pos = data.pos
    area = data.area
    final_outs = 0.0
    for i in range(num_trials):
        idx = paddle.randperm(n=pos.shape[0])
        outs = []
        with paddle.amp.auto_cast():
            for p, a in zip(pos[idx].chunk(chunks=100), area[idx].chunk(chunks=100)):
                data = CustomDataset(pos=p, area=a)
                out = model(data)
                outs.append(out)
        outs = paddle.concat(x=outs).astype(dtype="float32")
        tmp = paddle.zeros_like(x=outs)
        tmp[idx] = outs
        final_outs += tmp
    final_outs /= num_trials
    return final_outs


if __name__ == "__main__":
    seed_everything(2024)

    # load data
    test_ids = os.listdir(f"{root}/Dataset/Testset_track_B/Inference")
    test_ids = sorted(
        [i[i.find("_") + 1 : i.find(".")] for i in test_ids if "centroid_" in i]
    )
    test_ids = np.array(test_ids)
    print(f"Finish loading {len(test_ids)} test samples")
    test_data = read_npy_as_data(test_ids, training=False)
    test_loader = CustomDataLoader(test_data, batch_size=1, shuffle=False)

    # load model
    device = paddle.set_device("gpu")
    model = ConvBert().to(device)
    model.eval()
    model.set_state_dict(state_dict=paddle.load(path="model.pdparams"))

    track = "gen_answer_B"
    submit_path = f"results/{track}"
    os.makedirs(submit_path, exist_ok=True)
    for idx, data in enumerate(tqdm(test_loader)):
        out = get_preds(data).astype(dtype="float32")
        out = out.cpu().numpy() * std + mean
        file_id = int(test_ids[idx])
        np.save(f"{submit_path}/press_{file_id}.npy", out)
