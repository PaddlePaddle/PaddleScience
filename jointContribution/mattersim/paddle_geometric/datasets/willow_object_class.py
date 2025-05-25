import glob
import os
import os.path as osp
from typing import Callable, List, Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.io import DataLoader

from paddle_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from paddle_geometric.io import fs


class WILLOWObjectClass(InMemoryDataset):
    r"""The WILLOW-ObjectClass dataset from the `"Learning Graphs to Match"
    <https://www.di.ens.fr/willow/pdfscurrent/cho2013.pdf>`_ paper,
    containing 10 equal keypoints of at least 40 images in each category.
    The keypoints contain interpolated features from a pre-trained VGG16 model
    on ImageNet (:obj:`relu4_2` and :obj:`relu5_1`).

    Args:
        root (str): Root directory where the dataset should be saved.
        category (str): The category of the images (one of :obj:`"Car"`,
            :obj:`"Duck"`, :obj:`"Face"`, :obj:`"Motorbike"`,
            :obj:`"Winebottle"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`paddle_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`paddle_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        device (str or paddle.CUDAPlace, optional): The device to use for
            processing the raw data. If set to :obj:`None`, will utilize
            GPU-processing if available. (default: :obj:`None`)
    """
    url = ('http://www.di.ens.fr/willow/research/graphlearning/'
           'WILLOW-ObjectClass_dataset.zip')

    categories = ['face', 'motorbike', 'car', 'duck', 'winebottle']

    batch_size = 32

    def __init__(
        self,
        root: str,
        category: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')

        assert category.lower() in self.categories
        self.category = category
        self.device = device
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.category.capitalize(), 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return [category.capitalize() for category in self.categories]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        os.unlink(osp.join(self.root, 'README'))
        os.unlink(osp.join(self.root, 'demo_showAnno.m'))
        fs.rm(self.raw_dir)
        os.rename(osp.join(self.root, 'WILLOW-ObjectClass'), self.raw_dir)

    def process(self) -> None:
        from paddle.vision import models
        from paddle.vision.transforms import Compose, Normalize, ToTensor
        from PIL import Image
        from scipy.io import loadmat

        category = self.category.capitalize()
        names = glob.glob(osp.join(self.raw_dir, category, '*.png'))
        names = sorted([name[:-4] for name in names])

        vgg16_outputs = []

        def hook(layer: paddle.nn.Layer, x: Tensor, y: Tensor) -> None:
            vgg16_outputs.append(y.cpu())

        vgg16 = models.vgg16(pretrained=True)
        vgg16.eval()
        vgg16.features[20].register_forward_post_hook(hook)  # relu4_2
        vgg16.features[25].register_forward_post_hook(hook)  # relu5_1

        transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        data_list = []
        for name in names:
            pos = loadmat(f'{name}.mat')['pts_coord']
            x, y = paddle.to_tensor(pos[:, 0]), paddle.to_tensor(pos[:, 1])
            pos = paddle.stack([x, y], axis=1)

            # The "face" category contains a single image with less than 10
            # keypoints, so we need to skip it.
            if pos.shape[0] != 10:
                continue

            with open(f'{name}.png', 'rb') as f:
                img = Image.open(f).convert('RGB')

            # Rescale keypoints.
            pos[:, 0] = pos[:, 0] * 256.0 / (img.size[0])
            pos[:, 1] = pos[:, 1] * 256.0 / (img.size[1])

            img = img.resize((256, 256), resample=Image.Resampling.BICUBIC)
            img = transform(img)

            data = Data(img=img, pos=pos, name=name)
            data_list.append(data)

        imgs = [data.img for data in data_list]
        loader = DataLoader(
            dataset=imgs,
            batch_size=self.batch_size,
            shuffle=False,
        )
        for i, batch_img in enumerate(loader):
            vgg16_outputs.clear()

            with paddle.no_grad():
                vgg16(batch_img)

            out1 = F.interpolate(vgg16_outputs[0], size=(256, 256), mode='bilinear',
                                 align_corners=False)
            out2 = F.interpolate(vgg16_outputs[1], size=(256, 256), mode='bilinear',
                                 align_corners=False)

            for j in range(out1.shape[0]):
                data = data_list[i * self.batch_size + j]
                idx = paddle.to_tensor(data.pos.round().astype('int64').clip(0, 255))
                x_1 = out1[j, :, idx[:, 1], idx[:, 0]]
                x_2 = out2[j, :, idx[:, 1], idx[:, 0]]
                data.img = None
                data.x = paddle.concat([x_1.t(), x_2.t()], axis=-1)
            del out1
            del out2

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'category={self.category})')
