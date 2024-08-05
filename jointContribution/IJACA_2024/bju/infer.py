import argparse
import os

import numpy as np
import paddle
import yaml
from dataset import GraphDataset
from dataset import read_data
from geom.pc_encoder import load_geom_encoder
from model import MLP
from model import NN
from model import GeoCA3D
from paddle.io import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_dir",
        default="./Dataset/Trainset_track_B",
    )
    parser.add_argument(
        "--test_data_dir",
        default="./Dataset/Testset_track_B/Inference",
    )
    parser.add_argument(
        "--info_dir",
        default="./Dataset/Testset_track_B/Auxiliary",
    )
    parser.add_argument("--extra_data_dir", default=None)
    parser.add_argument("--fold_id", default=0, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--val_iter", default=10, type=int)
    parser.add_argument("--config_dir", default="params.yaml")
    parser.add_argument("--ulip_model", default="ULIP_PointBERT")
    parser.add_argument(
        "--ulip_ckpt",
        default="./geom/ckpt/checkpoint_pointbert.pdparams",
    )
    parser.add_argument("--frozen", default=True)
    parser.add_argument("--cfd_config_dir", default="cfd_params.yaml")
    parser.add_argument("--cfd_model", default="MLP")
    parser.add_argument("--cfd_mesh", default=True)
    parser.add_argument("--weight", default=0.5, type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # load setting
    args = parse_args()
    print(args)
    with open(args.cfd_config_dir, "r") as f:
        cfd_hparams = yaml.safe_load(f)[args.cfd_model]
    print(cfd_hparams)
    with open(args.config_dir, "r") as f:
        hparams = yaml.safe_load(f)["GeoCA3D"]
    n_gpu = paddle.device.cuda.device_count()
    use_cuda = 0 <= args.gpu < n_gpu and paddle.device.cuda.device_count() >= 1
    device = str(f"cuda:{args.gpu}" if use_cuda else "cpu").replace("cuda", "gpu")

    # load data
    train_data, val_data, test_data, coef_norm, test_index = read_data(args, norm=True)
    use_height = False
    r = cfd_hparams["r"] if "r" in cfd_hparams.keys() else None
    test_ds = GraphDataset(
        test_data, use_height=use_height, use_cfd_mesh=args.cfd_mesh, r=r
    )
    test_loader = DataLoader(test_ds, batch_size=1, collate_fn=test_ds.collate_fn)

    # load model
    if args.ulip_model == "none":
        print(
            "inference model use ULIP_PointBERT, please set 'ulip_model' to 'ULIP_PointBERT'"
        )
    else:
        g_encoder, g_proj = load_geom_encoder(
            args, pretrained=False, frozen=args.frozen
        )
        print(hparams)
    encoder = MLP(cfd_hparams["encoder"], batch_norm=False)
    decoder = MLP(cfd_hparams["decoder"], batch_norm=False)

    if args.cfd_model == "MLP":
        model = NN(cfd_hparams, encoder, decoder)
    else:
        print("inference model use mlp, please set 'cfd_model' to 'MLP'")

    model = GeoCA3D(model, geom_encoder=g_encoder, geom_proj=g_proj, **hparams).to(
        device
    )
    path = "./pretrained_checkpoint.pdparams"
    loaded_state_dict = paddle.load(path=path)
    model.set_state_dict(state_dict=loaded_state_dict)

    # infer
    model.eval()
    index = 0
    if not os.path.exists("./results"):
        os.makedirs("./results")

    with paddle.no_grad():
        for cfd_data, geom in test_loader:
            out = model((cfd_data[0], geom))
            if coef_norm is not None:
                mean = paddle.to_tensor(data=coef_norm[2]).to(device)
                std = paddle.to_tensor(data=coef_norm[3]).to(device)
                pred_press = out * std[-1] + mean[-1]
                np.save(
                    f"./results/press_{test_index[index]}.npy",
                    pred_press.detach().cpu().numpy().squeeze(),
                )
                print(f"Finish save sample {index}")
                index = index + 1
    npy_dir = "./results"
