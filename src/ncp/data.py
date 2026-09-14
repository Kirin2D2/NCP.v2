"""
data.py

Dataset constructor for the basketball CNP experiment, used by
PruningFineTuner.setup_dataloaders() when args.data_type == 'basketball_imagenet'.

The carton/dugong and crate/packet experiments build their own watermark datasets in
experiments/run_watermark_pruning_experiment.py.

Adapted from https://github.com/seulkiyeom/LRP_pruning/blob/master/modules/data.py
"""

from pathlib import Path

import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_basketball_imagenet(root_dir, transform=None, train_frac=0.8, seed=42):
    """Binary basketball ImageFolder (classes 'basketball' / 'not_basketball'), split 80/20.

    Args:
        root_dir:   ImageFolder root, e.g. data/images/imagenet_430_binary
        transform:  image transform; defaults to Resize(256) + CenterCrop(224) + ImageNet normalization
        train_frac: fraction of images used for training
        seed:       generator seed for the random split

    Returns:
        (train, test) torch.utils.data.Subset objects yielding (x, y)
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    dataset = datasets.ImageFolder(str(Path(root_dir)), transform=transform)
    print(dataset.class_to_idx)  # expected: {'basketball': 0, 'not_basketball': 1}

    train_size = int(train_frac * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size],
                        generator=torch.Generator().manual_seed(seed))
