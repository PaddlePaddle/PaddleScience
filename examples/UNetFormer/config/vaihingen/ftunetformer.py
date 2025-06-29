import paddle
from geoseg.datasets.vaihingen_dataset import *
from geoseg.losses import *
from geoseg.models.FTUNetFormer import ft_unetformer
from tools.utils import Lookahead, process_model_params

max_epoch = 45
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 4
lr = 0.0006
weight_decay = 0.00025
backbone_lr = 6e-05
backbone_weight_decay = 0.00025
num_classes = len(CLASSES)
classes = CLASSES
weights_name = "ftunetformer-512-ms-crop"
weights_path = "model_weights/vaihingen/{}".format(weights_name)
test_weights_name = "ftunetformer-512-ms-crop"
log_name = "vaihingen/{}".format(weights_name)
monitor = "val_F1"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = "auto"
resume_ckpt_path = None
net = ft_unetformer(num_classes=num_classes, decoder_channels=256)
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)
use_aux_loss = False
train_dataset = VaihingenDataset(
    data_root="data/vaihingen/train",
    mode="train",
    mosaic_ratio=0.25,
    transform=train_aug,
)
val_dataset = VaihingenDataset(transform=val_aug)
test_dataset = VaihingenDataset(data_root="data/vaihingen/test", transform=val_aug)
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
    T_0=15, T_mult=2, learning_rate=optimizer.get_lr()
)
optimizer.set_lr_scheduler(tmp_lr)
lr_scheduler = tmp_lr
