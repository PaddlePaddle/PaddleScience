import argparse
import os
import random
import re
import time

import numpy as np
import paddle
from data_process import data_process
from dataset import B_load_train_val_fold
from dataset import GraphDataset
from dataset import get_samples
from model import Model


def set_seed(seed):
    np.random.seed(seed)
    paddle.seed(seed=seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="./Dataset/data_centroid_track_B_vtk",
    )
    parser.add_argument(
        "--test_data_dir",
        default="./Dataset/track_B_vtk",
    )
    parser.add_argument(
        "--save_dir",
        default="./Dataset/data_centroid_track_B_vtk_preprocessed_data",
    )
    parser.add_argument("--fold_id", default=1, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--cfd_model", default="Transolver", type=str)
    parser.add_argument("--cfd_mesh", default=True)
    parser.add_argument("--r", default=0.2, type=float)
    parser.add_argument("--val_iter", default=1, type=int)
    parser.add_argument("--lr", default=1e-05, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--weight", default=0.5, type=float)
    parser.add_argument("--nb_epochs", default=400, type=int)
    parser.add_argument("--preprocessed", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--train_split", default=2, type=int)
    parser.add_argument("--val_split", default=3, type=int)
    parser.add_argument("-f", help="a dummy argument to fool ipython", default="1")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    print(
        "Attention: Please run and only run `data_process()` at first time in `infer.py`. "
        "And change path in the file before run it."
    )
    data_process()

    # load setting
    set_seed(0)
    args = parse_args()
    print(args)
    hparams = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "nb_epochs": args.nb_epochs,
        "num_workers": args.num_workers,
    }
    n_gpu = paddle.device.cuda.device_count()
    use_cuda = 0 <= args.gpu < n_gpu and paddle.device.cuda.device_count() >= 1
    device = str(f"cuda:{args.gpu}" if use_cuda else "cpu").replace("cuda", "gpu")

    # load data
    _, _, coef_norm = B_load_train_val_fold(args, preprocessed=args.preprocessed)
    samples = get_samples(args.test_data_dir)
    total_samples = len(samples)
    np.random.shuffle(samples)
    testlst = samples[:50]
    test_ds = GraphDataset(
        testlst,
        use_cfd_mesh=args.cfd_mesh,
        r=args.r,
        root=args.test_data_dir,
        norm=True,
        coef_norm=coef_norm,
    )
    test_loader = paddle.io.DataLoader(
        test_ds, batch_size=1, collate_fn=test_ds.collate_fn
    )

    # load model
    if args.cfd_model == "Transolver":
        model = Model(
            n_hidden=256,
            n_layers=8,
            space_dim=6,
            fun_dim=0,
            n_head=8,
            act="gelu",
            mlp_ratio=2,
            out_dim=4,
            slice_num=32,
            unified_pos=False,
        ).to(device)
    else:
        print("inference model use Transolver, please set 'cfd_model' to 'Transolver'")

    model_path = "./pretrained_checkpoint.pdparams"
    model.set_state_dict(state_dict=paddle.load(path=model_path))
    model.to(device)

    if not os.path.exists("./results"):
        os.makedirs("./results")

    # infer
    with paddle.no_grad():
        model.eval()
        times = []
        index = 0
        for cfd_data, geom in test_loader:
            mesh_file = testlst[index]
            match = re.search("mesh_(\\d+)\\.vtk", mesh_file)
            if match:
                mesh_index = match.group(1)
                print(f"Processing mesh index: {mesh_index}")
            else:
                raise ValueError(f"Invalid mesh file format: {mesh_file}")
            tic = time.time()
            out = model((cfd_data, geom))
            toc = time.time()
            press_output = out[cfd_data.surf, -1]
            if coef_norm is not None:
                mean_out = paddle.to_tensor(data=coef_norm[2]).to(device)
                std_out = paddle.to_tensor(data=coef_norm[3]).to(device)
                press_output = press_output * std_out[-1] + mean_out[-1]
            press_output = press_output.detach().cpu().numpy()
            np.save(
                "./results/" + "press" + "_" + f"{mesh_index}.npy",
                press_output,
            )
            times.append(toc - tic)
            index += 1
        print("time:", np.mean(times))
