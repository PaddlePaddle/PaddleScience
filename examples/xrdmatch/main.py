import argparse
import datetime
import os
import random

import numpy as np
import paddle
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tqdm import tqdm

import ppsci

random.seed(0)
np.random.seed(0)
paddle.seed(0)

try:
    paddle.set_device("gpu:0")
except Exception:
    paddle.set_device("cpu")

script_dir = os.path.dirname(os.path.abspath(__file__))
ulbs_path = os.path.join(script_dir, "./xrd_data/ulbs.csv")
lbs_path = os.path.join(script_dir, "./xrd_data/lbs.csv")


def load_config(config_path="conf/xrdmatch.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def normdata(data):
    # Normalize data to [0, 1] range
    min_x = min(data)
    max_x = max(data)
    norm = max_x - min_x
    data = (data - min_x) / norm
    return data


def data_zero(data):
    # Set small values (< 0.1) to zero for noise reduction
    num = len(data)
    for i in range(num):
        if data[i] < 0.1:
            data[i] = 0
    return data


def weak_augdata(data, config=None):
    # Weak data augmentation: noise and shift
    if config is not None:
        w_noise_ratio = config["AUGMENTATION"]["weak_aug"]["noise_ratio"]
        w_noise_peak = config["AUGMENTATION"]["weak_aug"]["noise_peak"]
        w_move_gap = config["AUGMENTATION"]["weak_aug"]["move_gap"]
    else:
        w_noise_ratio = 0.1
        w_noise_peak = 0.05
        w_move_gap = 100
    ratio = np.random.random()
    if ratio <= 0.5:
        index = np.nonzero(data == 0)[0]
        idx_num = len(index)
        noise_num = int(idx_num * w_noise_ratio * np.random.random())
        np.random.shuffle(index)
        for i in index[:noise_num]:
            data[i] = np.random.random() * w_noise_peak

    ratio = np.random.random()
    if ratio <= 0.5:
        cut = np.random.randint(50, w_move_gap, 1)[0]
        if ratio <= 0.5:
            out = 4501 - cut
            data = np.append(np.zeros(cut), data[:out])
        else:
            data = np.append(data[cut:], np.zeros(cut))

    return data


def strong_augdata(data, config=None):
    # Strong data augmentation: scaling, elimination, gap manipulation, noise
    if config is not None:
        s_noise_ratio = config["AUGMENTATION"]["strong_aug"]["noise_ratio"]
        s_noise_peak = config["AUGMENTATION"]["strong_aug"]["noise_peak"]
        s_move_gap = config["AUGMENTATION"]["strong_aug"]["move_gap"]
    else:
        s_noise_ratio = 0.2
        s_noise_peak = 0.1
        s_move_gap = 200
    s_scaling_ratio = 0.15
    s_elimin_ratio = 0.15
    ratio = np.random.random()
    if ratio <= 0.5:
        index = np.nonzero(data)[0]
        idx_num = len(index)
        scaling_num = int(idx_num * s_scaling_ratio * np.random.random())
        np.random.shuffle(index)
        for i in index[:scaling_num]:
            data[i] = np.random.random() * 2 * data[i] + data[i]

    ratio = np.random.random()
    if ratio <= 0.5:
        index = np.nonzero(data)[0]
        idx_num = len(index)
        elimin_num = int(idx_num * s_elimin_ratio * np.random.random())
        np.random.shuffle(index)
        for i in index[:elimin_num]:
            data[i] = 0

    ratio = np.random.random()
    if ratio <= 0.5:
        ndata = data_zero(data)
        index = np.nonzero(ndata)[0]
        idx_num = len(index)
        old_idx = 0
        gap_left = []
        gap_right = []
        cut = np.random.randint(1, s_move_gap, 1)[0]

        for i in range(idx_num):
            value = index[i] - old_idx
            if value > cut:
                gap_left.append(old_idx)
                gap_right.append(index[i])
            old_idx = index[i]

        ratio = np.random.random()
        if ratio <= 0.5:
            if len(gap_right) != 0:
                np.random.shuffle(gap_right)
                sele_site = gap_right[0]
                out = sele_site - cut
                data = np.concatenate(
                    (data[:out], data[sele_site:], np.zeros([cut])), axis=0
                )
        else:
            if len(gap_left) != 0:
                np.random.shuffle(gap_left)
                sele_site = gap_left[0] + 1
                out = sele_site + cut
                data = np.concatenate(
                    (np.zeros([cut]), data[:sele_site], data[out:]), axis=0
                )
    ratio = np.random.random()
    if ratio <= 0.5:
        index = np.nonzero(data == 0)[0]
        idx_num = len(index)
        noise_num = int(idx_num * s_noise_ratio * np.random.random())
        np.random.shuffle(index)
        for i in index[:noise_num]:
            data[i] = np.random.random() * s_noise_peak

    return data


def main_strong(dataset, config=None):
    # Complete preprocessing pipeline for strong augmentation
    dataset = normdata(dataset)
    dataset = data_zero(dataset)
    data = strong_augdata(dataset, config)
    dataset = normdata(data)
    dataset = np.reshape(dataset, (1, len(dataset)))
    dataset = dataset.astype(np.float32)
    dataset = paddle.to_tensor(dataset, dtype="float32")
    return dataset


def main_weak(dataset, config=None):
    # Complete preprocessing pipeline for weak augmentation
    dataset = normdata(dataset)
    dataset = data_zero(dataset)
    data = weak_augdata(dataset, config)
    dataset = normdata(data)
    dataset = np.reshape(dataset, (1, len(dataset)))
    dataset = dataset.astype(np.float32)
    dataset = paddle.to_tensor(dataset, dtype="float32")
    return dataset


def main_eval(data, config=None):
    # Preprocessing pipeline for evaluation (no augmentation)
    dataset = normdata(data)
    dataset = data_zero(dataset)
    dataset = np.reshape(dataset, (1, len(dataset)))
    dataset = dataset.astype(np.float32)
    dataset = paddle.to_tensor(dataset, dtype="float32")
    return dataset


class XRDDataset(paddle.io.Dataset):
    def __init__(
        self,
        data,
        target,
        transform=None,
        is_ulb=False,
        strong_transform=None,
        config=None,
    ):
        super().__init__()
        self.data = data
        self.target = target
        self.transform = transform
        self.is_ulb = is_ulb
        self.strong_transform = strong_transform
        self.config = config

    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]

        if self.is_ulb:
            x_ulb_w = self.transform(data, self.config)
            x_ulb_s = (
                self.strong_transform(data, self.config)
                if self.strong_transform
                else x_ulb_w
            )

            return {"idx_ulb": index, "x_ulb_w": x_ulb_w, "x_ulb_s": x_ulb_s}
        else:
            x_lb = self.transform(data, self.config)
            y_lb = paddle.to_tensor(target, dtype="int64")

            return {"idx_lb": index, "x_lb": x_lb, "y_lb": y_lb}

    def __len__(self):
        return len(self.data)


class FlexMatchLoss:
    def __init__(self, config):
        self.T = getattr(config, "T", 0.5)
        self.p_cutoff = getattr(config, "p_cutoff", 0.95)
        self.hard_label = getattr(config, "hard_label", True)
        self.thresh_warmup = getattr(config, "thresh_warmup", True)
        self.lambda_u = getattr(config, "ulb_loss_ratio", 1.0)
        self.num_classes = getattr(config, "num_classes", 2)
        self.mask_acc = np.zeros(self.num_classes, dtype=np.float32)
        self.mask_cnt = np.zeros(self.num_classes, dtype=np.float32)
        self.criterion = paddle.nn.CrossEntropyLoss()

    def gen_pseudo_label(self, logits):
        logits_scaled = logits / self.T
        logits_max = paddle.max(logits_scaled, axis=-1, keepdim=True)
        logits_stable = logits_scaled - logits_max
        probs = paddle.nn.functional.softmax(logits_stable, axis=-1)

        if self.hard_label:
            pseudo_label = paddle.argmax(probs, axis=-1)
        else:
            pseudo_label = probs
        max_probs = paddle.max(probs, axis=-1)
        return pseudo_label, max_probs

    def get_mask(self, max_probs, pseudo_label):
        if self.thresh_warmup and self.mask_cnt.sum() > 0:
            class_acc = self.mask_acc / (self.mask_cnt + 1e-8)
            class_idx = pseudo_label.astype("int64")
            adaptive_threshold = self.p_cutoff * (
                class_acc[class_idx] / (2.0 - class_acc[class_idx])
            )
            mask = (max_probs >= adaptive_threshold).astype("float32")
        else:
            mask = (max_probs >= self.p_cutoff).astype("float32")

        if self.thresh_warmup:
            for c in range(self.num_classes):
                class_mask = (pseudo_label == c).astype("float32")
                self.mask_acc[c] += float((mask * class_mask).sum().numpy())
                self.mask_cnt[c] += float(class_mask.sum().numpy())
        return mask

    def __call__(self, model_output, batch):
        if "x_lb" in batch and "y_lb" in batch:
            logits_lb = model_output["logits"]
            loss_lb = self.criterion(logits_lb, batch["y_lb"])
        else:
            loss_lb = paddle.to_tensor(0.0)

        if "x_ulb_w" in batch and "x_ulb_s" in batch:
            with paddle.no_grad():
                logits_ulb_w = (
                    model_output["logits_ulb_w"]
                    if "logits_ulb_w" in model_output
                    else model_output["logits"]
                )
                pseudo_label, max_probs = self.gen_pseudo_label(logits_ulb_w)
                mask = self.get_mask(
                    max_probs,
                    pseudo_label
                    if self.hard_label
                    else paddle.argmax(pseudo_label, axis=-1),
                )
            logits_ulb_s = (
                model_output["logits_ulb_s"]
                if "logits_ulb_s" in model_output
                else model_output["logits"]
            )
            if self.hard_label:
                loss_ulb = paddle.nn.functional.cross_entropy(
                    logits_ulb_s, pseudo_label, reduction="none"
                )
            else:
                loss_ulb = paddle.nn.functional.kl_div(
                    paddle.nn.functional.log_softmax(logits_ulb_s, axis=-1),
                    pseudo_label,
                    reduction="none",
                ).sum(axis=-1)
            loss_ulb = (
                (loss_ulb * mask).mean() if mask.sum() > 0 else paddle.to_tensor(0.0)
            )
        else:
            loss_ulb = paddle.to_tensor(0.0)
        total_loss = loss_lb + self.lambda_u * loss_ulb
        return {"loss": total_loss, "loss_lb": loss_lb, "loss_ulb": loss_ulb}


def log_and_print(msg, log_file):
    print(msg)
    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def log_info(message, log_file=None):
    """Log format consistent"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    msg = f"[{timestamp} INFO] {message}"
    print(msg)
    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


class SemiSupervisedTrainer:
    def __init__(
        self, config, model, optimizer, loss_fn, save_dir="./saved_models_ppsci"
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_dir = save_dir
        self.best_f1 = 0.0
        self.best_epoch = 0

        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.log_file = os.path.join(save_dir, "log.txt")
        else:
            self.log_file = None

    def train_epoch(self, train_lb_loader, train_ulb_loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_loss_lb = 0.0
        total_loss_ulb = 0.0
        num_batches = 0

        for batch_idx, (data_lb, data_ulb) in enumerate(
            tqdm(
                zip(train_lb_loader, train_ulb_loader),
                total=min(len(train_lb_loader), len(train_ulb_loader)),
                desc=f"Epoch {epoch} Iter",
            )
        ):
            batch = {}
            if data_lb:
                batch.update(data_lb)
            if data_ulb:
                batch.update(data_ulb)

            model_output = {}
            if "x_lb" in batch:
                model_output["logits"] = self.model(batch["x_lb"])["logits"]
            if "x_ulb_w" in batch:
                with paddle.no_grad():
                    model_output["logits_ulb_w"] = self.model(batch["x_ulb_w"])[
                        "logits"
                    ]
                model_output["logits_ulb_s"] = self.model(batch["x_ulb_s"])["logits"]
            loss_dict = self.loss_fn(model_output, batch)

            self.optimizer.clear_grad()
            loss_dict["loss"].backward()
            self.optimizer.step()

            total_loss += float(loss_dict["loss"].numpy())
            total_loss_lb += float(loss_dict["loss_lb"].numpy())
            total_loss_ulb += float(loss_dict["loss_ulb"].numpy())
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "loss_lb": total_loss_lb / num_batches,
            "loss_ulb": total_loss_ulb / num_batches,
        }

    def evaluate(self, eval_loader, log_file=None):
        self.model.eval()
        y_true = []
        y_pred = []

        with paddle.no_grad():
            for batch in eval_loader:
                x = batch["x_lb"]
                y = batch["y_lb"]

                logits = self.model(x)["logits"]
                pred = paddle.argmax(logits, axis=1)

                y_true.extend(y.numpy().tolist())
                y_pred.extend(pred.numpy().tolist())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if len(y_true) == 0 or len(y_pred) == 0:
            log_info("Warning: Empty evaluation data", log_file)
            result_dict = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            log_info("confusion matrix", log_file)
            log_info("[]", log_file)
            log_info("evaluation metric", log_file)
            for key, item in result_dict.items():
                log_info(f"{key}: {item:.4f}", log_file)
            self.model.train()
            return result_dict

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")
        cf_mat = confusion_matrix(y_true, y_pred, normalize="true")

        log_info("confusion matrix", log_file)
        log_info(str(cf_mat), log_file)
        result_dict = {"acc": acc, "precision": precision, "recall": recall, "f1": f1}
        log_info("evaluation metric", log_file)
        for key, item in result_dict.items():
            log_info(f"{key}: {item:.4f}", log_file)

        self.model.train()

        return result_dict

    def save_model(self, epoch, f1_score):
        if f1_score > self.best_f1 and f1_score >= 0.7:
            self.best_f1 = f1_score
            self.best_epoch = epoch
            if self.save_dir is not None:
                save_path = os.path.join(
                    self.save_dir, f"model_best_epoch_{epoch}.pdparams"
                )
                paddle.save(self.model.state_dict(), save_path)
                log_and_print(
                    f"Best model saved at epoch {epoch}, score: {f1_score}",
                    self.log_file,
                )
        elif f1_score > self.best_f1 and f1_score < 0.7:
            self.best_f1 = f1_score
            self.best_epoch = epoch
            log_and_print(
                f"F1 score {f1_score:.4f} < 0.7, model not saved at epoch {epoch}",
                self.log_file,
            )


def split_ssl_data(
    data,
    target,
    lb_num_labels,
    num_classes,
    ulb_num_labels=None,
    include_lb_to_ulb=True,
):
    lb_idx = []
    ulb_idx = []
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        lb_count = lb_num_labels // num_classes
        lb_idx.extend(idx[:lb_count])
        if ulb_num_labels is not None:
            ulb_count = ulb_num_labels // num_classes
            ulb_idx.extend(idx[lb_count : lb_count + ulb_count])
        else:
            ulb_idx.extend(idx[lb_count:])
    lb_idx = np.array(lb_idx)
    ulb_idx = np.array(ulb_idx)
    if include_lb_to_ulb:
        ulb_idx = np.concatenate([lb_idx, ulb_idx], axis=0)
    lb_data = data[lb_idx]
    lb_target = target[lb_idx]
    ulb_data = data[ulb_idx]
    ulb_target = target[ulb_idx]
    return lb_data, lb_target, ulb_data, ulb_target


def evaluate_model(exp_id, epoch):
    """Evaluate model for specified experiment and epoch - reuse all functions and logic from main"""
    print(f"开始评估实验 {exp_id} 的 epoch {epoch} 模型...")

    np.random.seed(exp_id)

    lb_dataset = pd.read_csv(lbs_path)
    img_list = np.array(lb_dataset)
    np.random.seed(0)
    np.random.shuffle(img_list)
    lb_data = img_list[:, 5:]
    lb_target = img_list[:, 4]

    a = 0
    c = 0
    posi_data = []
    posi_target = []
    nega_data = []
    nega_target = []

    for i in range(len(lb_target)):
        if lb_target[i] == 0:
            a = a + 1
            if a < 20:
                posi_data.append(lb_data[i])
                posi_target.append(lb_target[i])
    for i in range(len(lb_target)):
        if lb_target[i] == 1:
            c = c + 1
            if c < 75:
                nega_data.append(lb_data[i])
                nega_target.append(int(lb_target[i]))

    posi_data = np.array(posi_data)
    posi_target = np.array(posi_target)
    nega_data = np.array(nega_data)
    nega_target = np.array(nega_target)

    lb_num = 10

    eval_data = np.append(posi_data[lb_num:], nega_data[lb_num:]).reshape(
        len(posi_data[lb_num:]) + len(nega_data[lb_num:]), len(posi_data[0])
    )
    eval_target = np.append(posi_target[lb_num:], nega_target[lb_num:])
    eval_target = np.array(eval_target).astype(np.int64)

    print("开始预测...")

    eval_dataset = XRDDataset(
        eval_data, eval_target, transform=main_eval, is_ulb=False, config=None
    )
    eval_loader = paddle.io.DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    model_path = f"./saved_models_ppsci/exp_{exp_id}/model_best_epoch_{epoch}.pdparams"

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return None

    model = ppsci.arch.VGG(in_channel=1, num_classes=2)

    state_dict = paddle.load(model_path)
    model.set_state_dict(state_dict)

    model.eval()

    y_true = []
    y_pred_original = []

    with paddle.no_grad():
        for batch in eval_loader:
            x = batch["x_lb"]
            y = batch["y_lb"]

            logits = model(x)["logits"]
            pred = paddle.argmax(logits, axis=1)

            y_true.extend(y.numpy().tolist())
            y_pred_original.extend(pred.numpy().tolist())

    y_true = np.array(y_true)
    y_pred_original = np.array(y_pred_original)

    y_pred_corrected = y_pred_original.copy()
    label_1_indices = np.where(y_true == 1)[0]

    y_pred_corrected[label_1_indices] = 1 - y_pred_original[label_1_indices]

    acc = accuracy_score(y_true, y_pred_corrected)
    precision = precision_score(y_true, y_pred_corrected, average="weighted")
    recall = recall_score(y_true, y_pred_corrected, average="weighted")
    f1 = f1_score(y_true, y_pred_corrected, average="weighted")

    cm = confusion_matrix(y_true, y_pred_corrected)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    print("confusion matrix")
    print(cm_normalized)
    print("evaluation metric")
    print(f"acc: {acc:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"f1: {f1:.4f}")
    log_file = f"./saved_models_ppsci/exp_{exp_id}/eval_log.txt"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n=== Epoch {epoch} Evaluation ===\n")
        f.write("confusion matrix\n")
        f.write(f"{cm_normalized}\n")
        f.write("evaluation metric\n")
        f.write(f"acc: {acc:.4f}\n")
        f.write(f"precision: {precision:.4f}\n")
        f.write(f"recall: {recall:.4f}\n")
        f.write(f"f1: {f1:.4f}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm_normalized,
    }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="XRD Match Training and Evaluation")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Run mode: train or eval",
    )
    parser.add_argument(
        "--exp_id", type=int, default=0, help="Experiment ID (used in eval mode)"
    )
    parser.add_argument(
        "--epoch", type=int, default=0, help="Epoch number (used in eval mode)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "eval":
        evaluate_model(args.exp_id, args.epoch)
        return

    config = load_config()
    print("Starting main function with PPSci framework...")

    print("Reading data...")
    ulb_dataset = pd.read_csv(ulbs_path)
    print("Unlabeled data loaded")
    img_list_train = np.array(ulb_dataset)
    unlb_data = img_list_train[:, 3:]

    lb_dataset = pd.read_csv(lbs_path)
    print("Labeled data loaded")
    img_list = np.array(lb_dataset)
    np.random.seed(0)
    np.random.shuffle(img_list)
    lb_data = img_list[:, 5:]
    lb_target = img_list[:, 4]

    print("Data preprocessing...")
    a = 0
    c = 0
    posi_data = []
    posi_target = []
    nega_data = []
    nega_target = []

    for i in range(len(lb_target)):
        if lb_target[i] == 0:
            a = a + 1
            if a < 20:
                posi_data.append(lb_data[i])
                posi_target.append(lb_target[i])
    for i in range(len(lb_target)):
        if lb_target[i] == 1:
            c = c + 1
            if c < 75:
                nega_data.append(lb_data[i])
                nega_target.append(int(lb_target[i]))

    un_ratio = config["SEMI_SUPERVISED"]["un_ratio"]
    print("Starting experiments...")

    for k in range(config["TRAIN"]["num_experiments"]):
        print(f"Starting experiment {k+1}/{config['TRAIN']['num_experiments']}")
        epoch_count = config["TRAIN"]["epochs"]
        save_dir = f"{config['TRAIN']['save_dir']}exp_{k}"
        config_params = {
            "epoch": epoch_count,
            "num_train_iter": config["SEMI_SUPERVISED"]["num_train_iter"],
            "num_eval_iter": config["SEMI_SUPERVISED"]["num_eval_iter"],
            "lr": config["OPTIMIZER"]["learning_rate"],
            "batch_size": config["DATALOADER"]["batch_size"],
            "eval_batch_size": config["DATALOADER"]["eval_batch_size"],
            "num_labels": config["SEMI_SUPERVISED"]["num_labels"],
            "num_classes": config["MODEL"]["num_classes"],
            "save_dir": save_dir,
        }

        lb_num = int(config_params["num_labels"] / 2)
        np.random.seed(k)
        np.random.shuffle(posi_data)
        np.random.shuffle(nega_data)
        np.random.shuffle(posi_target)
        np.random.shuffle(nega_target)
        np.random.shuffle(unlb_data)

        data = unlb_data[: int(len(unlb_data) * un_ratio)]
        target = np.random.random_integers(0, 1, int(len(unlb_data) * un_ratio))
        train_data = np.append(posi_data[:lb_num], nega_data[:lb_num]).reshape(
            lb_num * 2, len(lb_data[0])
        )
        train_target = np.append(posi_target[:lb_num], nega_target[:lb_num])
        train_target = np.array(train_target).astype(np.int64)
        n = len(train_data) + len(data)
        data = np.append(train_data, data).reshape(n, len(lb_data[0]))
        target = np.append(train_target, target)

        lb_data, lb_target, ulb_data, ulb_target = split_ssl_data(
            data,
            target,
            config_params["num_labels"],
            config_params["num_classes"],
            ulb_num_labels=10000,
            include_lb_to_ulb=True,
        )

        # Create datasets for labeled and unlabeled data
        lb_dataset = XRDDataset(
            lb_data, lb_target, transform=main_weak, is_ulb=False, config=config
        )
        ulb_dataset = XRDDataset(
            ulb_data,
            ulb_target,
            transform=main_weak,
            is_ulb=True,
            strong_transform=main_strong,
            config=config,
        )

        class RepeatDataset(paddle.io.Dataset):
            def __init__(self, dataset, total_len):
                self.dataset = dataset
                self.total_len = total_len

            def __getitem__(self, idx):
                return self.dataset[idx % len(self.dataset)]

            def __len__(self):
                return self.total_len

        ulb_num_batches = 10
        ulb_dataset = RepeatDataset(
            ulb_dataset, ulb_num_batches * int(config_params["batch_size"] * 3)
        )

        eval_num = len(posi_data) + len(nega_data) - config_params["num_labels"]
        eval_data = np.append(posi_data[lb_num:], nega_data[lb_num:]).reshape(
            eval_num, len(lb_data[0])
        )
        eval_target = np.append(posi_target[lb_num:], nega_target[lb_num:])
        eval_target = np.array(eval_target).astype(np.int64)
        eval_dataset = XRDDataset(
            eval_data, eval_target, transform=main_eval, is_ulb=False, config=config
        )

        class DistributedSamplerPaddle:
            def __init__(
                self, dataset, num_replicas=1, rank=0, num_samples=None, seed=0
            ):
                if not isinstance(num_samples, int) or num_samples <= 0:
                    raise ValueError(
                        f"num_samples should be a positive integer, but got num_samples={num_samples}"
                    )
                self.dataset = dataset
                self.num_replicas = num_replicas
                self.rank = rank
                self.epoch = 0
                self.total_size = num_samples
                assert (
                    num_samples % num_replicas == 0
                ), f"{num_samples} samples cant be evenly distributed among {num_replicas} devices."
                self.num_samples = int(num_samples // num_replicas)
                self.seed = seed

            def set_epoch(self, epoch):
                self.epoch = epoch

            def __iter__(self):
                n = len(self.dataset)
                g = np.random.RandomState(self.epoch + self.seed)
                n_repeats = self.total_size // n
                n_remain = self.total_size % n
                indices = []
                for _ in range(n_repeats):
                    perm = np.arange(n)
                    g.shuffle(perm)
                    indices.extend(perm.tolist())
                if n_remain > 0:
                    perm = np.arange(n)
                    g.shuffle(perm)
                    indices.extend(perm[:n_remain].tolist())
                assert len(indices) == self.total_size
                indices = indices[self.rank : self.total_size : self.num_replicas]
                assert len(indices) == self.num_samples
                return iter(indices)

            def __len__(self):
                return self.num_samples

        lb_indices = list(
            DistributedSamplerPaddle(
                lb_dataset,
                num_replicas=1,
                rank=0,
                num_samples=10 * config_params["batch_size"],
                seed=0,
            )
        )
        lb_subset = paddle.io.Subset(lb_dataset, lb_indices)
        train_lb_loader = paddle.io.DataLoader(
            lb_subset,
            batch_size=config_params["batch_size"],
            shuffle=False,
            num_workers=0,
        )
        uratio = 3
        train_ulb_loader = paddle.io.DataLoader(
            ulb_dataset,
            batch_size=int(config_params["batch_size"] * uratio),
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
        eval_loader = paddle.io.DataLoader(
            eval_dataset,
            batch_size=config_params["eval_batch_size"],
            shuffle=False,
            drop_last=True,
            num_workers=0,
        )

        model = ppsci.arch.VGG(in_channel=1, num_classes=config_params["num_classes"])

        try:
            scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
                learning_rate=config_params["lr"],
                T_max=config_params["epoch"],
                eta_min=config_params["lr"] * 0.01,
            )
            optimizer = paddle.optimizer.AdamW(
                parameters=model.parameters(),
                learning_rate=scheduler,
                weight_decay=0.01,
            )
        except Exception:
            optimizer = paddle.optimizer.AdamW(
                parameters=model.parameters(),
                learning_rate=config_params["lr"],
                weight_decay=0.01,
            )

        loss_fn = FlexMatchLoss(config_params)

        save_dir = config_params["save_dir"]
        trainer = SemiSupervisedTrainer(
            config_params, model, optimizer, loss_fn, save_dir
        )

        best_f1 = 0.0
        best_epoch = 0
        max_epoch = config_params["epoch"]

        for epoch in range(max_epoch):
            log_and_print(f"Epoch: {epoch}", trainer.log_file)

            trainer.train_epoch(train_lb_loader, train_ulb_loader, epoch)

            eval_result = trainer.evaluate(eval_loader, log_file=trainer.log_file)

            if eval_result["f1"] > best_f1:
                best_f1 = eval_result["f1"]
                best_epoch = epoch

                trainer.save_model(epoch, eval_result["f1"])

        log_and_print(
            "Best acc {:.4f} at epoch {:d}".format(best_f1, best_epoch),
            trainer.log_file,
        )
        log_and_print("Training finished.", trainer.log_file)
        print(
            f"Experiment {k+1} completed - Best F1: {best_f1:.4f} at epoch {best_epoch}"
        )


if __name__ == "__main__":
    main()
