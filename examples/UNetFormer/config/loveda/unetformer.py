import paddle
from geoseg.datasets.loveda_dataset import *
from geoseg.losses import *
from geoseg.models.UNetFormer import UNetFormer
from tools.utils import Lookahead, process_model_params

max_epoch = 30
ignore_index = len(CLASSES)
train_batch_size = 16
val_batch_size = 16
lr = 0.0006
weight_decay = 0.01
backbone_lr = 6e-05
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES
weights_name = "unetformer-r18-512crop-ms-epoch30-rep"
weights_path = "model_weights/loveda/{}".format(weights_name)
test_weights_name = "last"
log_name = "loveda/{}".format(weights_name)
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


def get_training_transform():
    train_transform = [albu.HorizontalFlip(p=0.5), albu.Normalize()]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose(
        [
            RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode="value"),
            SmartCropV1(
                crop_size=512, max_ratio=0.75, ignore_index=ignore_index, nopad=False
            ),
        ]
    )
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug["image"], aug["mask"]
    return img, mask


train_dataset = LoveDATrainDataset(
    transform=train_aug, data_root="data/LoveDA/train_val"
)
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
