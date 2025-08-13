import os

import paddle
from geoseg.datasets.vaihingen_dataset import CLASSES
from geoseg.datasets.vaihingen_dataset import VaihingenDataset
from geoseg.datasets.vaihingen_dataset import train_aug
from geoseg.datasets.vaihingen_dataset import val_aug
from geoseg.losses.useful_loss import UnetFormerLoss
from geoseg.models.UNetFormer import UNetFormer
from tools.utils import process_model_params

max_epoch = 105
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 8
lr = 0.0006
weight_decay = 0.01
backbone_lr = 6e-05
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES
weights_name = "unetformer-r18-512-crop-ms-e105"
weights_path = "model_weights/vaihingen/{}".format(weights_name)
test_weights_name = "unetformer-r18-512-crop-ms-e105_epoch0_best"
log_name = "vaihingen/{}".format(weights_name)
monitor = "val_F1"
monitor_mode = "max"
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = "auto"
resume_ckpt_path = None
net = UNetFormer(num_classes=num_classes)
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True
os.makedirs("data/vaihingen/train/images_1024", exist_ok=True)
os.makedirs("data/vaihingen/train/masks_1024", exist_ok=True)
if len(os.listdir("data/vaihingen/train/images_1024")) == 0:
    pass
else:
    train_dataset = VaihingenDataset(
        data_root="data/vaihingen/train",
        mode="train",
        mosaic_ratio=0.25,
        transform=train_aug,
    )
    train_loader = paddle.io.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
    )
val_dataset = VaihingenDataset(transform=val_aug)
test_dataset = VaihingenDataset(data_root="data/vaihingen/test", transform=val_aug)

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
optimizer = paddle.optimizer.AdamW(
    parameters=net_params, learning_rate=lr, weight_decay=weight_decay
)
tmp_lr = paddle.optimizer.lr.CosineAnnealingWarmRestarts(
    T_0=15, T_mult=2, learning_rate=optimizer.get_lr()
)
optimizer.set_lr_scheduler(tmp_lr)
lr_scheduler = tmp_lr
