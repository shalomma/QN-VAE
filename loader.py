import numpy as np
from torch import manual_seed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import dataset


class Loader:
    seed = 14

    def __init__(self):
        manual_seed(self.seed)
        self.data = dict()

    def get(self, batch_size, pin_memory=True):
        loaders = dict()
        loaders['train'] = DataLoader(self.data['train'], batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        loaders['val'] = DataLoader(self.data['val'], batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        return loaders


class CIFAR10Loader(Loader):
    def __init__(self, transform):
        super(CIFAR10Loader, self).__init__()
        self.data['train'] = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
        self.data['val'] = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)


class MNISTLoader(Loader):
    def __init__(self, transform):
        super(MNISTLoader, self).__init__()
        self.data['train'] = datasets.MNIST('data', train=True, download=True, transform=transform)
        self.data['val'] = datasets.MNIST('data', train=False, transform=transform)


class ImageNetLoader(Loader):
    def __init__(self):
        super(ImageNetLoader, self).__init__()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        compose_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
        compose_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        self.data['val'] = datasets.ImageNet(root="data", split='val', transform=compose_val)
        self.data['train'] = datasets.ImageNet(root="data", split='train', transform=compose_train)


class EncodedLoader(Loader):
    def __init__(self, root_dir, q, transform=None):
        super(EncodedLoader, self).__init__()
        size = 60000
        val_size = 12000
        data_indices = np.arange(0, size)
        val_indices = np.random.choice(data_indices, val_size, replace=False)
        train_indices = np.array(list(set(data_indices) - set(val_indices)))
        self.data['train'] = dataset.Encoded(root_dir, q, train_indices, transform)
        self.data['val'] = dataset.Encoded(root_dir, q, val_indices, transform)
