"""
data.py

Dataset constructor for the basketball CNP experiment, used by
PruningFineTuner.setup_dataloaders() when args.data_type == 'basketball_imagenet',
and DataLoader worker settings shared by all experiments.

The carton/dugong and crate/packet experiments build their own watermark datasets in
experiments/run_watermark_pruning_experiment.py.

Adapted from https://github.com/seulkiyeom/LRP_pruning/blob/master/modules/data.py
"""

import os
from pathlib import Path

import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_num_workers(cuda: bool) -> int:
    """DataLoader worker processes per loader: none on CPU, else CPU count - 1, capped at 3.

    The pruning loop keeps several persistent loaders alive at once (train, rank, eval), so a
    2-vCPU machine such as Colab gets 1 worker per loader instead of oversubscribing its CPUs.
    """
    if not cuda:
        return 0
    return min(3, max(1, (os.cpu_count() or 2) - 1))


def loader_kwargs(cuda: bool, num_workers: int) -> dict:
    """DataLoader keyword arguments for `num_workers` persistent worker processes.

    Workers use the spawn start method: forked workers can deadlock once the parent process has
    initialized CUDA. Pinned memory is used only when training on a GPU.
    """
    if num_workers <= 0:
        return {}
    return {
        'num_workers': num_workers,
        'pin_memory': bool(cuda),
        'multiprocessing_context': 'spawn',
        'persistent_workers': True,
    }


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
