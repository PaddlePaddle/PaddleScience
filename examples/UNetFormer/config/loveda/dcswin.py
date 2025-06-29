import paddle
from geoseg.datasets.loveda_dataset import *
from geoseg.losses import *
from geoseg.models.DCSwin import dcswin_small
from tools.utils import Lookahead, process_model_params

max_epoch = 30
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 8
lr = 0.0006
weight_decay = 0.01
backbone_lr = 6e-05
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES
weights_name = "dcswin-small-512crop-ms-epoch30"
weights_path = "model_weights/loveda/{}".format(weights_name)
test_weights_name = "dcswin-small-512crop-ms-epoch30"
log_name = "loveda/{}".format(weights_name)
monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = "auto"
resume_ckpt_path = None
net = dcswin_small(
    num_classes=num_classes,
    pretrained=True,
    weight_path="pretrain_weights/stseg_small.pth",
)
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)
use_aux_loss = False
train_dataset = LoveDATrainDataset(transform=train_aug, data_root="data/LoveDA/Train")
val_dataset = loveda_val_dataset
test_dataset = LoveDATestDataset()
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
    T_max=max_epoch, eta_min=1e-06, learning_rate=optimizer.get_lr()
)
optimizer.set_lr_scheduler(tmp_lr)
lr_scheduler = tmp_lr
