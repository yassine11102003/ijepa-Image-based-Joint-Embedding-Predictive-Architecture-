"""
CIFAR-10 data loading.

JEPA pretraining only needs the raw images (self-supervised, no labels used).
Labels are only needed later, for the linear-probe evaluation.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)


def _transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def get_pretrain_loaders(root="./data", batch_size=64, val_fraction=0.1, num_workers=0, seed=42):
    """
    Train/val DataLoaders for JEPA pretraining, split out of CIFAR-10's train set.
    Both still yield (image, label) pairs (that's what CIFAR10 returns), but the
    training loop ignores the label — this split exists only to monitor the JEPA
    pretext loss itself, so it stays independent of CIFAR-10's official test set
    (which is reserved entirely for the downstream linear-probe evaluation).
    """
    full_train = datasets.CIFAR10(root=root, train=True, download=True, transform=_transform())

    n_val   = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(full_train, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=num_workers)
    return train_loader, val_loader


def get_probe_loaders(root="./data", batch_size=128, num_workers=0):
    """Train/test loaders WITH labels, used only for the linear-probe evaluation."""
    train_ds = datasets.CIFAR10(root=root, train=True,  download=True, transform=_transform())
    test_ds  = datasets.CIFAR10(root=root, train=False, download=True, transform=_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
