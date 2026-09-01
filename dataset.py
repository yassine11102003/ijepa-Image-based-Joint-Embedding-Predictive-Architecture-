"""
CIFAR-10 data loading.

JEPA pretraining only needs the raw images (self-supervised, no labels used).
Labels are only needed later, for the linear-probe evaluation.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)


def _transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def get_pretrain_loader(root="./data", batch_size=64, train=True, num_workers=0):
    """
    DataLoader for JEPA pretraining. Still yields (image, label) pairs because
    that's what CIFAR10 returns, but the training loop simply ignores the label.
    """
    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=_transform())
    return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                      drop_last=train, num_workers=num_workers)


def get_probe_loaders(root="./data", batch_size=128, num_workers=0):
    """Train/test loaders WITH labels, used only for the linear-probe evaluation."""
    train_ds = datasets.CIFAR10(root=root, train=True,  download=True, transform=_transform())
    test_ds  = datasets.CIFAR10(root=root, train=False, download=True, transform=_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
