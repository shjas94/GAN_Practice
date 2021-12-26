import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def load_mnist(is_train=True):
    dataset = datasets.MNIST(
        './mnist',
        train=is_train,
        download=True
    )
    x = dataset.data
    y = dataset.targets
    return x, y


def load_cifarten(is_train=True):
    dataset = datasets.CIFAR10(
        './cifar_10',
        train=is_train,
        download=True
    )
    x = dataset.data
    y = dataset.targets
    return x, y


def load_cifarhundred(is_train=True):
    dataset = datasets.CIFAR100(
        './cifar_100',
        train=is_train,
        download=True
    )
    x = dataset.data
    y = dataset.targets
    return x, y


def load_imagenet(is_train=True):
    dataset = datasets.ImageNet(
        './imagenet',
        train=is_train,
        download=True,
    )
    x = dataset.data
    y = dataset.targets
    return x, y


class PredefinedDataset(Dataset):
    def __init__(self, data, labels, img_size):
        super().__init__()
        self.data = data
        self.labels = labels
        self.transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(0.5, 0.5)
        ])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data[index] / 255.
        x = torch.transpose(x, 2, 0).transpose(1, 2)
        y = self.labels[index]
        x = self.transforms(x)
        return x, y
