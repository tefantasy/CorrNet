import os
import os.path as osp
import torch
import torch.utils.data
import torchvision
from torchvision import datasets, transforms

import corr_net.utils.logging as logging

from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Imagenet(torch.utils.data.Dataset):
    """
    ILSVRC2012 ImageNet Dataset. 
    """
    def __init__(self, cfg, mode):
        assert mode in [
            "train",
            "val",
        ], "Split '{}' not supported for ImageNet".format(mode)

        self.mode = mode
        self.cfg = cfg

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        data_dir = osp.join(cfg.DATA.PATH_TO_DATA_DIR, mode)

        if mode == "train":
            data_transforms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif mode == "val":
            data_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        self.dataset = datasets.ImageFolder(data_dir, data_transforms)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
