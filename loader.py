from torch import manual_seed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Loader:
    seed = 14

    def __init__(self):
        manual_seed(self.seed)
        self.data = dict()
        self.compose = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))])

    def get(self, batch_size):
        loaders = dict()
        loaders['train'] = DataLoader(self.data['train'], batch_size=batch_size, shuffle=True, pin_memory=True)
        loaders['val'] = DataLoader(self.data['val'], batch_size=batch_size, shuffle=True, pin_memory=True)
        return loaders


class CIFAR10Loader(Loader):
    def __init__(self):
        super(CIFAR10Loader, self).__init__()
        self.data['train'] = datasets.CIFAR10(root="data", train=True, download=True, transform=self.compose)
        self.data['val'] = datasets.CIFAR10(root="data", train=False, download=True, transform=self.compose)


class ImageNetLoader(Loader):
    def __init__(self):
        super(ImageNetLoader, self).__init__()
        self.data['train'] = datasets.ImageNet(root="data", split='train', transform=self.compose)
        self.data['val'] = datasets.ImageNet(root="data", split='val', transform=self.compose)
