import paddle
from geoseg.datasets.potsdam_dataset import *
from geoseg.losses import *
from geoseg.models.UNetFormer import UNetFormer
from tools.utils import Lookahead, process_model_params

max_epoch = 45
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 8
lr = 0.0006
weight_decay = 0.01
backbone_lr = 6e-05
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES
weights_name = "unetformer-r18-768crop-ms-e45"
weights_path = "model_weights/potsdam/{}".format(weights_name)
test_weights_name = "unetformer-r18-768crop-ms-e45"
log_name = "potsdam/{}".format(weights_name)
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
    T_0=15, T_mult=2, learning_rate=optimizer.get_lr()
)
optimizer.set_lr_scheduler(tmp_lr)
lr_scheduler = tmp_lr
