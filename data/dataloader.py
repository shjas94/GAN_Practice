import torch
from torch.utils.data import DataLoader
from .dataset import *


def get_train_valid_loader(cfg):
    dataset_name = cfg.dataset_name
    img_size = cfg.img_size
    batch_size = cfg.batch_size
    train_size = cfg.train_size
    if dataset_name == 'mnist':
        x, y = load_mnist(is_train=True)
    elif dataset_name == "cifar-10":
        x, y = load_cifar10(is_train=True)
    elif dataset_name == "cifar-100:":
        x, y = load_cifar100(is_train=True)
    elif dataset_name == "imagenet":
        x, y = load_imagenet(is_train=True)
    else:
        raise ValueError("Use another set or implement for custom dataset")
    train_cnt = int(x.shape[0] * train_size)
    valid_cnt = x.shape[0] - train_cnt
    indices = torch.randperm(x.shape[0])
    train_x, valid_x = torch.index_select(
        torch.tensor(x),
        index=indices,
        dim=0
    ).split([train_cnt, valid_cnt], dim=0)
    train_y, valid_y = torch.index_select(
        torch.tensor(y),
        index=indices,
        dim=0
    ).split([train_cnt, valid_cnt], dim=0)

    train_loader = DataLoader(
        dataset=PredefinedDataset(train_x, train_y, img_size),
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset=PredefinedDataset(valid_x, valid_y, img_size),
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader, valid_loader


def get_test_loader(cfg):
    dataset_name = cfg.dataset_name
    img_size = cfg.img_size
    test_batch_size = cfg.test_batch_size

    if dataset_name == 'mnist':
        x, y = load_mnist(img_size, is_train=False)
    elif dataset_name == 'cifar-10':
        x, y = load_cifar10(img_size, is_train=False)
    elif dataset_name == 'cifar-100':
        x, y = load_cifar100(img_size, is_train=False)
    elif dataset_name == 'imagenet':
        x, y = load_imagenet(img_size, is_train=False)
    else:
        raise ValueError("Use another set or implement for custom dataset")
    test_loader = DataLoader(
        dataset=PredefinedDataset(x, y),
        batch_size=test_batch_size,
        shuffle=False,
    )
    return test_loader
