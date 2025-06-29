import paddle
from geoseg.datasets.potsdam_dataset import *
from geoseg.losses import *
from geoseg.models.DCSwin import dcswin_small
from tools.utils import Lookahead, process_model_params

max_epoch = 30
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 4
lr = 0.001
weight_decay = 0.00025
backbone_lr = 0.0001
backbone_weight_decay = 0.00025
num_classes = len(CLASSES)
classes = CLASSES
weights_name = "dcswin-small-1024-ms-512crop-e30"
weights_path = "model_weights/potsdam/{}".format(weights_name)
test_weights_name = "dcswin-small-1024-ms-512crop-e30"
log_name = "potsdam/{}".format(weights_name)
monitor = "val_F1"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = "auto"
resume_ckpt_path = None
net = dcswin_small(num_classes=num_classes)
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)
use_aux_loss = False


def get_training_transform():
    train_transform = [albu.RandomRotate90(p=0.5), albu.Normalize()]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose(
        [
            RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode="value"),
            SmartCropV1(
                crop_size=512, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False
            ),
        ]
    )
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug["image"], aug["mask"]
    return img, mask


def get_val_transform():
    val_transform = [albu.Normalize()]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug["image"], aug["mask"]
    return img, mask


train_dataset = PotsdamDataset(
    data_root="data/potsdam/train", mode="train", mosaic_ratio=0.25, transform=train_aug
)
val_dataset = PotsdamDataset(transform=val_aug)
test_dataset = PotsdamDataset(data_root="data/potsdam/test", transform=val_aug)
train_loader = paddle.io.DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    shuffle=True,
    drop_last=True,
)
val_loader = paddle.io.DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    drop_last=False,
)
layerwise_params = {
    "backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)
}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = paddle.optimizer.AdamW(
    parameters=net_params, learning_rate=lr, weight_decay=weight_decay
)
optimizer = Lookahead(base_optimizer)
tmp_lr = paddle.optimizer.lr.CosineAnnealingWarmRestarts(
    T_0=10, T_mult=2, learning_rate=optimizer.get_lr()
)
optimizer.set_lr_scheduler(tmp_lr)
lr_scheduler = tmp_lr
