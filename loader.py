import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class CIFAR10Loader:
    def __init__(self):
        self.data = dict()
        compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))])
        self.data['train'] = datasets.CIFAR10(root="data", train=True, download=True, transform=compose)
        self.data['val'] = datasets.CIFAR10(root="data", train=False, download=True, transform=compose)

    def get(self, batch_size):
        loaders = dict()
        loaders['train'] = DataLoader(self.data['train'], batch_size=batch_size, shuffle=True, pin_memory=True)
        loaders['val'] = DataLoader(self.data['val'], batch_size=batch_size, shuffle=True, pin_memory=True)
        return loaders
