"""
src/data.py
===========
CIFAR-10 data loaders with standard normalisation and augmentation.
"""

import logging
from pathlib import Path

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

# ImageNet-style stats computed on CIFAR-10 training split
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_loaders(
    data_dir: str = "./data",
    batch_size: int = 256,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """
    Download CIFAR-10 (if necessary) and return (train_loader, test_loader).

    Training augmentations: random horizontal flip + random 32×32 crop
    with padding=4  (standard CIFAR-10 recipe).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        str(data_dir), train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(
        str(data_dir), train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,  batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    logger.info(
        "CIFAR-10 loaded | train=%d  test=%d  batch_size=%d",
        len(train_set), len(test_set), batch_size,
    )
    return train_loader, test_loader
