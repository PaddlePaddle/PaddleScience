"""
This code generates the prediction on one instance.
Both the ground truth and the prediction are saved in a .pt file.
"""
import os
from argparse import ArgumentParser
from argparse import Namespace
from collections import OrderedDict

import numpy as np
import paddle
import paddle.distributed as dist
import yaml

# from utils.data_utils import get_data_loader
from data_utils.pois_helm_datasets import get_data_loader
from models.fno import build_fno
from pretrain_basic import l2_err
from scipy.stats import linregress
from tqdm import tqdm

# from utils.loss_utils import LossMSE
# from utils.YParams import YParams


@paddle.no_grad()
def get_pred(args):
    with open(args.config, "r") as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if args.ckpt_path:
        save_dir = os.path.join("/".join(args.ckpt_path.split("/")[:-1]), "results_icl")
    else:
        basedir = os.path.join("exp", config["log"]["logdir"])
        save_dir = os.path.join(basedir, "results_icl")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, "fno-prediction-demo%d.pt" % (args.num_demos if args.num_demos else 0)
    )
    if paddle.device.cuda.device_count() >= 1:
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")

    params = Namespace(**config["default"])
    if not hasattr(params, "n_demos"):
        params.n_demos = 0
    if "batch_size" in config:
        params.local_valid_batch_size = config["batch_size"]
    else:
        params.local_valid_batch_size = 1
    dataloader, dataset, sampler = get_data_loader(
        params, params.test_path, dist.is_initialized(), train=False
    )  # , pack=data_param.pack_data)
    if args.num_demos is not None and args.num_demos != 0:
        params.subsample = 1
        params.local_valid_batch_size = args.num_demos
        dataloader_icl, dataset_icl, _ = get_data_loader(
            params, params.train_path, dist.is_initialized(), train=False
        )  # , pack=data_param.pack_data)
        input_demos, target_demos = next(iter(dataloader_icl))
        input_demos = input_demos
        target_demos = target_demos

    # model_param = Namespace(**config['model'])
    # model_param.n_demos = params.n_demos
    model = build_fno(params)

    if args.ckpt_path:
        checkpoint = paddle.load(args.ckpt_path)
        try:
            model.set_state_dict(checkpoint["model_state"])
        except:  # noqa
            new_state_dict = OrderedDict()
            for key, val in checkpoint["model_state"].items():
                name = key
                if "module" in name:
                    name = name[7:]
                new_state_dict[name] = val
            state = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {
                k: v
                for k, v in new_state_dict.items()
                if k in state and state[k].size() == new_state_dict[k].size()
            }
            # 2. overwrite entries in the existing state dict
            state.update(pretrained_dict)
            # 3. load the new state dict
            # message = model.load_state_dict(state)
            # self.model.load_state_dict(new_state_dict)
            unload_keys = [k for k in new_state_dict.keys() if k not in pretrained_dict]
            if len(unload_keys) > 0:
                import warnings

                warnings.warn(
                    "Warning: unload keys during restoring checkpoint: %s"
                    % (str(unload_keys))
                )

    # mseloss = LossMSE(params, None)

    # metric
    # lploss = LpLoss(size_average=True)
    model.eval()
    truth_list = []
    pred_list = []
    losses = []
    losses_normalized = []
    pbar = tqdm(dataloader, total=len(dataloader))
    # if args.num_demos is not None and args.num_demos != 0:
    #     pbar = tqdm([next(iter(dataloader))], total=1)
    # for u, a_in in dataloader:
    for inputs, targets in pbar:
        # if len(pred_list) > len(dataloader) // 100: break
        inputs, targets = inputs, targets
        if args.num_demos is None or args.num_demos == 0:
            u = model(inputs)
        else:
            model.target = targets  # for debugging purpose
            u = model.forward_icl(inputs, input_demos, target_demos, use_tqdm=args.tqdm)
            # u = model.forward_icl_knn(inputs, input_demos, target_demos, use_tqdm=args.tqdm)
            # out = model.forward_icl(inputs, input_demos, target_demos, use_tqdm=False)
        # data_loss = lploss(out, u)
        # data_loss = mseloss.data(inputs, u, targets)
        data_loss = l2_err(u.detach(), targets.detach())
        losses.append(data_loss.item())
        data_loss_normalized = l2_err(
            u.detach() / paddle.abs(u).max(),
            targets.detach() / paddle.abs(targets).max(),
        )
        losses_normalized.append(data_loss_normalized.item())
        # print(data_loss.item())
        truth_list.append(targets.cpu())
        pred_list.append(u.cpu())

    # print(np.mean(losses))
    slope, intercept, r, p, se = linregress(
        paddle.concat(pred_list, axis=0).view([-1]).numpy(),
        paddle.concat(truth_list, axis=0).view([-1]).numpy(),
    )
    print(
        "RMSE:",
        np.mean(losses),
        "RMSE (normalized)",
        np.mean(losses_normalized),
        "R2:",
        r,
        "Slope:",
        slope,
    )
    truth_arr = paddle.concat(truth_list, axis=0)
    pred_arr = paddle.concat(pred_list, axis=0)
    paddle.save(
        {
            "truth": truth_arr,
            "pred": pred_arr,
            "rmse": np.mean(losses),
            "rmse_normalized": np.mean(losses_normalized),
            "r2": r,
            "slope": slope,
        },
        save_path,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/inference_helmholtz.yaml")
    # parser.add_argument('--ckpt_path', type=str, default='/pscratch/sd/p/puren93/neuralopt/expts/helm-64-o5_15_ft0/all_mask_m6/checkpoints/ckpt.tar')
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/pscratch/sd/j/jsong/deff_archive/neuraloperators-foundation_/expts/helm-64-o5_15_ft0/b012_m6/checkpoints/ckpt.tar",
    )
    # parser.add_argument('--ckpt_path', type=str, default='/pscratch/sd/j/jsong/deff_archive/neuraloperators-foundation_/expts/helm-64-o5_15_ft0/b01_m6/checkpoints/ckpt.tar')
    # parser.add_argument('--ckpt_path', type=str, default='/pscratch/sd/j/jsong/neuraloperators-foundation/expts/pois-64-e5_15_ft9/b01_m0/checkpoints/ckpt.tar') # [X]
    # parser.add_argument('--ckpt_path', type=str, default='/pscratch/sd/j/jsong/deff_archive/neuraloperators-foundation_/expts/pois-64-e5_15_ft9/b01_m0_/checkpoints/ckpt.tar')
    # parser.add_argument('--ckpt_path', type=str, default='/pscratch/sd/j/jsong/deff_archive/neuraloperators-foundation_/expts/pois-64-e5_15_ft9/b01_m0_r0/checkpoints/ckpt.tar')
    parser.add_argument("--num_demos", type=int, default=None)
    parser.add_argument(
        "--tqdm", action="store_true", default=False, help="Turn on the tqdm"
    )
    parser.add_argument(
        "--save_pred", action="store_true", default=False, help="Save predictions"
    )
    args = parser.parse_args()
    get_pred(args)
