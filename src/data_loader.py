import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from . import utils


def train_eval_loaders_cifar10(data_dir,
                               batch_size,
                               augment=False,
                               angle=0,
                               eval_size=0.2,
                               num_workers=4):
    normalize = transforms.Normalize(utils.cifar10_mean, utils.cifar10_std)

    if augment:
        transform = transforms.Compose([
            transforms.RandomAffine(angle),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    eval_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    data_size = len(train_dataset)
    indices = list(range(data_size))
    train_size = int(np.floor(eval_size * data_size))

    train_idx, eval_idx = indices[train_size:], indices[:train_size]
    train_sampler = SubsetRandomSampler(train_idx)
    eval_sampler = SubsetRandomSampler(eval_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, drop_last=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, sampler=eval_sampler,
        num_workers=num_workers, drop_last=True
    )

    return train_loader, eval_loader


def test_loader(data_dir,
                batch_size,
                augment=False,
                angle=0,
                num_workers=4):

    normalize = transforms.Normalize(utils.cifar10_mean, utils.cifar10_std)

    if augment:
        transform = transforms.Compose([
            transforms.RandomAffine(angle),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        num_workers=num_workers, drop_last=True
    )

    return data_loader