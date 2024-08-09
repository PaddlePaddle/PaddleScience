import argparse
import json


def str2bool(v):
    """
    'boolean type variable' for add_argument
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected.")


def params(load=None):
    if load is not None:
        parser = argparse.ArgumentParser(
            description="train / test a paddle model to predict frames"
        )
        params = vars(parser.parse_args([]))
        with open(load + "/commandline_args.json", "rt") as f:
            params.update(json.load(f))
        for k, v in params.items():
            parser.add_argument("--" + k, default=v)
        args = parser.parse_args([])
        return args
    else:
        """
        return parameters for training / testing / plotting of models
        :return: parameter-Namespace
        """
        parser = argparse.ArgumentParser(
            description="train / test a paddle model to predict frames"
        )
        parser.add_argument(
            "--net",
            default="SAGE-Trans",
            type=str,
            help="network to train (default: GN-Cell)",
            choices=[
                "GN-Cell",
                "GN-Node",
                "SAGE-Trans",
                "GeoATT",
                "MultiGN",
                "MultiTrans",
                "Trans",
                "GM",
            ],
        )
        parser.add_argument(
            "--SAGE_MIXING_TYPE",
            default="FVGNAttUNet",
            type=str,
            help="SAGE & transolver features mixing type",
            choices=[
                "Origin",
                "AttMixing",
                "transUnet",
                "TransU-sep",
                "PureUnet",
                "FVGNUnet",
                "SDFUnet",
                "SageSDFUnet",
                "TransolverUnet",
                "TransAttUnet",
                "FVGNAttUNet",
            ],
        )
        parser.add_argument(
            "--GM_TYPE",
            default="AttuMMLP",
            type=str,
            help="SAGE & transolver features mixing type",
            choices=["FVGNAttUnet", "SageAttUnet", "transolunet", "AttuMMLP"],
        )
        parser.add_argument(
            "--n_epochs",
            default=150,
            type=int,
            help="number of epochs (after each epoch, the model gets saved)",
        )
        parser.add_argument(
            "--batch_size", default=4, type=int, help="batch size (default: 100)"
        )
        parser.add_argument(
            "--dataset_size", default=500, type=int, help="dataset size (default: 500)"
        )
        parser.add_argument(
            "--batch_size_for_attn",
            default=1,
            type=int,
            help="batch size (default: 100)",
        )
        parser.add_argument(
            "--lr",
            default=0.001,
            type=float,
            help="learning rate of optimizer (default: 0.0001)",
        )
        parser.add_argument(
            "--lr_scheduler",
            default="fixlr",
            type=str,
            help="choose learing rate scheduler (default: coslr)",
            choices=["coslr", "fix"],
        )
        parser.add_argument(
            "--log",
            default=True,
            type=str2bool,
            help="log models / metrics during training (turn off for debugging)",
        )
        parser.add_argument(
            "--on_gpu", default=0, type=int, help="set training on which gpu"
        )
        parser.add_argument(
            "--num_samples", default=5000, type=int, help="subsampling for trackB"
        )
        parser.add_argument(
            "--sample_khop", default=5, type=int, help="subsampling k-hop for trackB"
        )
        parser.add_argument(
            "--statistics_times",
            default=20,
            type=int,
            help="accumlate data statistics for normalization before backprapagation (default: 1)",
        )
        parser.add_argument(
            "--before_explr_decay_steps",
            default=500,
            type=int,
            help="steps before using exp lr decay technique (default:12000)",
        )
        parser.add_argument(
            "--loss",
            default="square",
            type=str,
            help="loss type to train network (default: square)",
            choices=["square"],
        )
        parser.add_argument(
            "--wgrad",
            default=0,
            type=float,
            help="weight of gradient loss (default: 0.1)",
        )
        parser.add_argument(
            "--wpress",
            default=1,
            type=float,
            help="weight of pressure loss (default: 0.9)",
        )
        parser.add_argument(
            "--load_date_time",
            default=None,
            type=str,
            help="date_time of run to load (default: None)",
        )
        parser.add_argument(
            "--load_index",
            default=None,
            type=int,
            help="index of run to load (default: None)",
        )
        parser.add_argument(
            "--load_optimizer",
            default=False,
            type=str2bool,
            help="load state of optimizer (default: True)",
        )
        parser.add_argument(
            "--load_latest",
            default=False,
            type=str2bool,
            help="load latest version for training (if True: leave load_date_time and load_index None. default: False)",
        )
        parser.add_argument(
            "--hidden_size",
            default=128,
            type=int,
            help="hidden size of network (default: 20)",
        )
        parser.add_argument(
            "--message_passing_num",
            default=8,
            type=int,
            help="message passing layer number (default:15)",
        )
        parser.add_argument(
            "--node_input_size",
            default=3,
            type=int,
            help="node encoder node_input_size (default: 2)",
        )
        parser.add_argument(
            "--edge_input_size",
            default=6,
            type=int,
            help="edge encoder edge_input_size, include edge center pos (x,y) (default: 3)",
        )
        parser.add_argument(
            "--cell_input_size",
            default=3,
            type=int,
            help="cell encoder cell_input_size, include uvp (default: 3)",
        )
        parser.add_argument(
            "--node_output_size",
            default=1,
            type=int,
            help="edge decoder edge_output_size uvp on edge center(default: 8)",
        )
        parser.add_argument(
            "--edge_output_size",
            default=1,
            type=int,
            help="edge decoder edge_output_size uvp on edge center(default: 8)",
        )
        parser.add_argument(
            "--cell_output_size",
            default=1,
            type=int,
            help="cell decoder cell_output_size uvp on cell center(default: 1)",
        )
        parser.add_argument(
            "--drop_out",
            default=False,
            type=str2bool,
            help="using dropout technique in message passing layer(default: True)",
        )
        parser.add_argument(
            "--attention",
            default=False,
            type=str2bool,
            help="using dropout technique in message passing layer(default: True)",
        )
        parser.add_argument(
            "--multihead",
            default=1,
            type=int,
            help="using dropout technique in message passing layer(default: True)",
        )
        parser.add_argument(
            "--dataset_type",
            default="h5",
            type=str,
            help="load latest version for training (if True: leave load_date_time and load_index None. default: False)",
        )
        parser.add_argument(
            "--dataset_dir",
            default="./Datasets",
            type=str,
            help="load latest version for training (if True: leave load_date_time and load_index None. default: False)",
        )
        parser.add_argument(
            "--git_branch",
            default="FVGN-pde-jtedu smaller tanh factor,test no prevent oversmooth still normalize,lr on bc=1e-2",
            type=str,
            help="current running code git branch",
        )
        parser.add_argument(
            "--git_commit_dates",
            default="March 14th, 2023 10:56 PM",
            type=str,
            help="current running code git commit date",
        )
        params = parser.parse_args([])
        git_info = {
            "git_branch": params.git_branch,
            "git_commit_dates": params.git_commit_dates,
        }
        return params, git_info


def get_hyperparam(params):
    return f"net {params.net}; hs {params.hidden_size};"
