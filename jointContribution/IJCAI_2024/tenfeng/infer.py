import argparse
import logging
import os

import numpy as np
import paddle
from Transolver import Model
from utils.utils import LpLoss


def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        filename=os.path.join(log_dir, "training.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def log_initial_configuration(args):
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")


def set_random_seed(seed):
    np.random.seed(seed)
    paddle.seed(seed=seed)


def set_device(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = str("cuda" if paddle.device.cuda.device_count() >= 1 else "cpu").replace(
        "cuda", "gpu"
    )
    return device


def load_tensor(file_path, mask=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    data = np.load(file_path)
    if mask is not None:
        data = data[mask]
    return paddle.to_tensor(data=data).astype(dtype="float32")


def custom_collate_fn(batch):
    return batch


class PointCloudDataset(paddle.io.Dataset):
    def __init__(
        self, root_dir, transform=None, train=True, translate=True, submit=False
    ):
        """
        Args:
            root_dir (string): Directory with all the point cloud files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.file_list = [
            f
            for f in os.listdir(os.path.join(root_dir, "centroid"))
            if f.endswith(".npy")
        ]
        self.transform = transform
        self.train = train
        self.translate = translate
        self.submit = submit

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # start_time = time.time()
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, "centroid", file_name)
        points = np.load(file_path)
        points_min = np.min(points, axis=0, keepdims=True)
        points_max = np.max(points, axis=0, keepdims=True)
        mean_std_dict = load_mean_std("mean_std.txt")
        if self.train:
            sample_rate = 0.1
        else:
            sample_rate = 0.4
        if self.submit:
            sampled_indices = np.arange(tuple(points.shape)[0])
        else:
            sampled_indices = np.random.choice(
                np.arange(tuple(points.shape)[0]),
                int(len(points) * sample_rate),
                replace=False,
            )
        sampled_points = points[sampled_indices].astype(np.float32)
        local_sampled_points = (sampled_points - points_min) / (
            points_max - points_min
        ).astype(np.float32)
        press_sample = np.load(
            os.path.join(self.root_dir, "press", file_name.replace("centroid", "press"))
        )[sampled_indices].astype(np.float32)
        if self.translate and self.train:
            translation_vector = np.random.rand(3) * 0.01 - 0.005
            sampled_points += translation_vector
        Normal = True
        if Normal:
            sampled_points = (
                sampled_points - mean_std_dict["centroid"][0]
            ) / mean_std_dict["centroid"][1]
        sample = {
            "centroid": sampled_points.astype(np.float32),
            "local_centroid": local_sampled_points.astype(np.float32),
            "press": press_sample.astype(np.float32),
            "file_name": file_name,
            "mean_std": mean_std_dict,
        }
        return sample


def load_mean_std(input_file):
    """
    Load mean and standard deviations from a text file.
    Args:
    input_file (str): The path to the text file containing the saved mean and std values.

    Returns:
    dict: A dictionary with keys as the data categories and values as tuples of mean and std.
    """
    results = {}
    with open(input_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split()
            category = parts[0]
            if category == "centroid":
                mean_val_1 = float(parts[3].strip(","))
                mean_val_2 = float(parts[4].strip(","))
                mean_val_3 = float(parts[5].strip(","))
                std_val_1 = float(parts[9].strip(","))
                std_val_2 = float(parts[10].strip(","))
                std_val_3 = float(parts[11].strip(","))
                mean_val = [mean_val_1, mean_val_2, mean_val_3]
                std_val = [std_val_1, std_val_2, std_val_3]
            else:
                mean_val = [float(parts[3].strip(","))]
                std_val = [float(parts[7])]
            results[category] = mean_val, std_val
    return results


class modified_log_transformed_l2_relative_error_loss(object):
    def __init__(self, epsison=1e-08):
        super(modified_log_transformed_l2_relative_error_loss, self).__init__()
        self.epsilon = epsison

    def rel(self, x, y):
        num_examples = tuple(x.shape)[0]
        sign_x = paddle.sign(x=x.reshape(num_examples, -1))
        sign_y = paddle.sign(x=y.reshape(num_examples, -1))
        log_abs_x = paddle.log(
            x=paddle.abs(x=x.reshape(num_examples, -1))
            + paddle.to_tensor(data=self.epsilon, dtype=x.dtype, place=x.place)
        )
        log_abs_y = paddle.log(
            x=paddle.abs(x=y.reshape(num_examples, -1))
            + paddle.to_tensor(data=self.epsilon, dtype=x.dtype, place=x.place)
        )
        signed_log_x = sign_x * log_abs_x
        signed_log_y = sign_y * log_abs_y
        diff = signed_log_x - signed_log_y
        diff_norm = paddle.linalg.norm(x=diff, p=2, axis=1)
        y_norm = paddle.linalg.norm(x=signed_log_y, p=2, axis=1)
        relative_error = diff_norm / y_norm
        return paddle.mean(x=relative_error)

    def __call__(self, x, y):
        return self.rel(x, y)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model for Automobile Aerodynamic Drag Prediction."
    )
    parser.add_argument(
        "--training_data_dir",
        type=str,
        default="./Dataset/train_track_B_e",
        help="Directory for the training data",
    )
    parser.add_argument(
        "--testing_data_dir",
        type=str,
        default="./Dataset/Testset_track_B_e",
        help="Directory for the testing data",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./results",
        help="Directory for saving logs and results",
    )
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument(
        "--num_segments", type=int, default=10, help="Number of segments to split"
    )
    parser.add_argument(
        "--segments_id", type=int, default=0, help="the id_th of segments to split"
    )
    parser.add_argument(
        "--overlap_ratio", type=float, default=0.5, help="Overlap ratio for segments"
    )
    parser.add_argument(
        "--global_normal",
        type=bool,
        default=True,
        help="wheter use global normal or not",
    )
    parser.add_argument(
        "--normalization", type=bool, default=True, help="Flag to normalize data or not"
    )
    parser.add_argument(
        "--translate", action="store_true", help="Translate data or not"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="rl2",
        choices=["l2", "rl2", "log_rl2", "huber"],
        help="The type of loss function to use.",
    )
    parser.add_argument(
        "--submit", action="store_true", help="if generate submitted data"
    )
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument(
        "--input_dim", type=int, default=6, help="Dimension of model input"
    )
    parser.add_argument(
        "--output_dim", type=int, default=1, help="Dimension of model output"
    )
    parser.add_argument("--depth", type=int, default=5, help="Depth of the model")
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Dimension of hidden features"
    )
    parser.add_argument(
        "--num_slices",
        type=int,
        default=32,
        help="Number of slices for slicing the input",
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--mlp_ratio",
        type=int,
        default=2,
        help="Ratio of mlp hidden dim to embedding dim",
    )
    parser.add_argument("--patch_size", type=int, default=20, help="Size of each patch")
    parser.add_argument(
        "--shift", type=int, default=4, help="Shift size for shifting the patches"
    )
    parser.add_argument("--n_layer", type=int, default=1, help="Number of layers")
    parser.add_argument(
        "--epochs", type=int, default=69, help="Number of epochs to train"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--scheduler_step",
        type=int,
        default=30,
        help="Number of steps after which to reduce learning rate",
    )
    parser.add_argument(
        "--milestones", nargs="+", type=int, default=[40, 50, 60, 65, 68]
    )
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.5,
        help="Gamma factor for reducing the learning rate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seed(123)
    args = parse_args()
    print(args)
    save_dir = args.log_dir
    setup_logging(save_dir)
    log_initial_configuration(args)
    device = set_device(args.gpu_id)
    model = Model(
        n_hidden=args.hidden_dim,
        n_layers=args.depth,
        space_dim=args.input_dim,
        fun_dim=0,
        n_head=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        out_dim=args.output_dim,
        slice_num=args.num_slices,
        n_iter=args.n_layer,
        unified_pos=0,
    ).to(device)
    L2_fn = LpLoss(reduction=False)
    if args.submit:
        submit_dataset = PointCloudDataset(
            root_dir=args.testing_data_dir, train=False, translate=False, submit=True
        )
        submitloader = paddle.io.DataLoader(
            dataset=submit_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        output_dirs = os.path.join(save_dir, "output")
        if os.path.exists(output_dirs) is False:
            os.makedirs(output_dirs)
        model = Model(
            n_hidden=args.hidden_dim,
            n_layers=args.depth,
            space_dim=args.input_dim,
            fun_dim=0,
            n_head=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            out_dim=args.output_dim,
            slice_num=args.num_slices,
            n_iter=args.n_layer,
            unified_pos=0,
        ).to(device)
        model.set_state_dict(
            state_dict=paddle.load(path=f"{save_dir}/checkpoint.pdparams")
        )
        y_list = []
        y_hat_list = []
        L2 = []
        model.eval()
        with paddle.no_grad():
            for batch in submitloader:
                for i in range(len(batch)):
                    x_centroid = batch[i]["centroid"].cuda().unsqueeze(axis=0)
                    x_local_centroid = (
                        batch[i]["local_centroid"].cuda().unsqueeze(axis=0)
                    )
                    y = batch[i]["press"].cuda().unsqueeze(axis=0)
                    features = paddle.concat(x=[x_centroid, x_local_centroid], axis=-1)
                    y_hat = model(features)
                    y_hat = (
                        y_hat * batch[i]["mean_std"]["press"][1][0]
                        + batch[i]["mean_std"]["press"][0][0]
                    ).cuda()
                    test_L2 = L2_fn(y_hat, y)
                    L2.append(test_L2.cpu())
                    np.save(
                        f"{output_dirs}/{batch[i]['file_name'].replace('centroid', 'press')}",
                        y_hat.cpu().numpy().squeeze(),
                    )
                    print(
                        f"{batch[i]['file_name'].replace('centroid', 'press')}  score: {test_L2.cpu().item():.5f}"
                    )
            L2 = paddle.mean(x=paddle.concat(x=L2, axis=0), axis=0)
            y_list.append(y.cpu().numpy().squeeze())
            y_hat_list.append(y_hat.cpu().numpy().squeeze())
        print(float(L2))
        print("##########################  submit sucessfully #################")
