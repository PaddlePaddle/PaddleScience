import paddle

"""
UnetFormer for uavid datasets with supervision training
Libo Wang, 2022.02.22
"""
from geoseg.datasets.uavid_dataset import *
from geoseg.losses import *
from geoseg.models.UNetFormer import UNetFormer
from tools.utils import Lookahead, process_model_params

max_epoch = 40
ignore_index = 255
train_batch_size = 8
val_batch_size = 8
lr = 0.0006
weight_decay = 0.01
backbone_lr = 6e-05
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES
weights_name = "unetformer-r18-1024-768crop-e40"
weights_path = "model_weights/uavid/{}".format(weights_name)
test_weights_name = "last"
log_name = "uavid/{}".format(weights_name)
monitor = "val_mIoU"
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
train_dataset = UAVIDDataset(
    data_root="data/uavid/train_val",
    img_dir="images",
    mask_dir="masks",
    mode="train",
    mosaic_ratio=0.25,
    transform=train_aug,
    img_size=(1024, 1024),
)
val_dataset = UAVIDDataset(
    data_root="data/uavid/val",
    img_dir="images",
    mask_dir="masks",
    mode="val",
    mosaic_ratio=0.0,
    transform=val_aug,
    img_size=(1024, 1024),
)
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
tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(
    T_max=max_epoch, learning_rate=optimizer.get_lr()
)
optimizer.set_lr_scheduler(tmp_lr)
lr_scheduler = tmp_lr
