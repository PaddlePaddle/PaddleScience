import os
import os.path as osp
from itertools import chain
from typing import Callable, Dict, List, Optional
from xml.dom import minidom

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.io import DataLoader

from paddle_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
)
from paddle_geometric.io import fs


class PascalVOCKeypoints(InMemoryDataset):
    image_url = ('http://host.robots.ox.ac.uk/pascal/VOC/voc2011/'
                 'VOCtrainval_25-May-2011.tar')
    annotation_url = ('https://www2.eecs.berkeley.edu/Research/Projects/CS/'
                      'vision/shape/poselets/voc2011_keypoints_Feb2012.tgz')
    split_url = ('https://github.com/Thinklab-SJTU/PCA-GM/raw/master/data/'
                 'PascalVOC/voc2011_pairs.npz')

    categories = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    batch_size = 32

    def __init__(
        self,
        root: str,
        category: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = 'gpu' if paddle.device.is_compiled_with_cuda() else 'cpu'

        self.category = category.lower()
        assert self.category in self.categories
        self.device = device
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.load(path)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.category.capitalize(), 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ['images', 'annotations', 'splits.npz']

    @property
    def processed_file_names(self) -> List[str]:
        return ['training.pt', 'test.pt']

    def download(self) -> None:
        path = download_url(self.image_url, self.raw_dir)
        extract_tar(path, self.raw_dir, mode='r')
        os.unlink(path)
        image_path = osp.join(self.raw_dir, 'TrainVal', 'VOCdevkit', 'VOC2011')
        os.rename(image_path, osp.join(self.raw_dir, 'images'))
        fs.rm(osp.join(self.raw_dir, 'TrainVal'))

        path = download_url(self.annotation_url, self.raw_dir)
        extract_tar(path, self.raw_dir, mode='r')
        os.unlink(path)

        path = download_url(self.split_url, self.raw_dir)
        os.rename(path, osp.join(self.raw_dir, 'splits.npz'))

    def process(self) -> None:
        import paddle.vision.models as models
        import paddle.vision.transforms as T
        from PIL import Image

        splits = np.load(osp.join(self.raw_dir, 'splits.npz'),
                         allow_pickle=True)
        category_idx = self.categories.index(self.category)
        train_split = list(splits['train'])[category_idx]
        test_split = list(splits['test'])[category_idx]

        image_path = osp.join(self.raw_dir, 'images', 'JPEGImages')
        info_path = osp.join(self.raw_dir, 'images', 'Annotations')
        annotation_path = osp.join(self.raw_dir, 'annotations')

        labels: Dict[str, int] = {}

        vgg16_outputs = []

        def hook(layer, input, output):
            vgg16_outputs.append(output)

        vgg16 = models.vgg16(pretrained=True)
        vgg16.eval()
        vgg16.features[20].register_forward_post_hook(hook)  # relu4_2
        vgg16.features[25].register_forward_post_hook(hook)  # relu5_1

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_set, test_set = [], []
        for i, name in enumerate(chain(train_split, test_split)):
            filename = '_'.join(name.split('/')[1].split('_')[:-1])
            file_idx = int(name.split('_')[-1].split('.')[0]) - 1

            path = osp.join(info_path, f'{filename}.xml')
            obj = minidom.parse(path).getElementsByTagName('object')[file_idx]

            child = obj.getElementsByTagName('truncated')[0].firstChild
            trunc = child.data

            elements = obj.getElementsByTagName('occluded')
            occ = elements[0].firstChild.data if elements else '0'

            diff = obj.getElementsByTagName('difficult')[0].firstChild.data

            if bool(int(trunc)) or bool(int(occ)) or bool(int(diff)):
                continue

            if self.category == 'person' and int(filename[:4]) > 2008:
                continue

            xmin = int(obj.getElementsByTagName('xmin')[0].firstChild.data)
            xmax = int(obj.getElementsByTagName('xmax')[0].firstChild.data)
            ymin = int(obj.getElementsByTagName('ymin')[0].firstChild.data)
            ymax = int(obj.getElementsByTagName('ymax')[0].firstChild.data)
            box = (xmin, ymin, xmax, ymax)

            dom = minidom.parse(osp.join(annotation_path, name))
            keypoints = dom.getElementsByTagName('keypoint')
            poss, ys = [], []
            for keypoint in keypoints:
                label = keypoint.attributes['name'].value
                if label not in labels:
                    labels[label] = len(labels)
                ys.append(labels[label])
                _x = float(keypoint.attributes['x'].value)
                _y = float(keypoint.attributes['y'].value)
                poss += [_x, _y]
            y = paddle.to_tensor(ys, dtype='int64')
            pos = paddle.to_tensor(poss, dtype='float32').reshape([-1, 2])

            if pos.numel() == 0:
                continue

            box = (
                min(int(pos[:, 0].min().floor()), box[0]) - 16,
                min(int(pos[:, 1].min().floor()), box[1]) - 16,
                max(int(pos[:, 0].max().ceil()), box[2]) + 16,
                max(int(pos[:, 1].max().ceil()), box[3]) + 16,
            )

            pos[:, 0] = (pos[:, 0] - box[0]) * 256.0 / (box[2] - box[0])
            pos[:, 1] = (pos[:, 1] - box[1]) * 256.0 / (box[3] - box[1])

            path = osp.join(image_path, f'{filename}.jpg')
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB').crop(box)
                img = img.resize((256, 256), resample=Image.Resampling.BICUBIC)

            img = transform(img)
            data = Data(img=img, pos=pos, y=y, name=filename)

            if i < len(train_split):
                train_set.append(data)
            else:
                test_set.append(data)

        data_list = list(chain(train_set, test_set))
        imgs = [data.img for data in data_list]
        loader: DataLoader = DataLoader(
            dataset=imgs,
            batch_size=self.batch_size,
            shuffle=False,
        )
        for i, batch_img in enumerate(loader):
            vgg16_outputs.clear()
            with paddle.no_grad():
                vgg16(batch_img)

            out1 = F.interpolate(vgg16_outputs[0], (256, 256), mode='bilinear')
            out2 = F.interpolate(vgg16_outputs[1], (256, 256), mode='bilinear')

            for j in range(out1.shape[0]):
                data = data_list[i * self.batch_size + j]
                idx = paddle.clip(data.pos.round().astype('int64'), 0, 255)
                x_1 = out1[j, :, idx[:, 1], idx[:, 0]].cpu()
                x_2 = out2[j, :, idx[:, 1], idx[:, 0]].cpu()
                data.img = None
                data.x = paddle.concat([x_1.transpose([1, 0]), x_2.transpose([1, 0])], axis=-1)

        if self.pre_filter:
            train_set = [data for data in train_set if self.pre_filter(data)]
            test_set = [data for data in test_set if self.pre_filter(data)]

        if self.pre_transform:
            train_set = [self.pre_transform(data) for data in train_set]
            test_set = [self.pre_transform(data) for data in test_set]

        self.save(train_set, self.processed_paths[0])
        self.save(test_set, self.processed_paths[1])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'category={self.category})')
