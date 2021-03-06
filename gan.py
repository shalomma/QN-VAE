import os
import torch
import argparse
from abc import ABC
import logging.config
import torch.nn as nn
from datetime import datetime
from torchvision import transforms

import loader
import trainer as trn
from qnvae import QNVAE
from utils import save_model, load_model


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module, ABC):
    def __init__(self, channels, in_size, latent_dim):
        super(Generator, self).__init__()

        self.init_size = in_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module, ABC):
    def __init__(self, channels, in_size):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of down sampled image
        ds_size = 1  # in_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('timestamp', type=str, help='models timestamp')
    parser.add_argument('-q', type=float, help='epochs', default=1.0)
    parser.add_argument('--epochs', type=int, help='epochs', default=100)
    args = parser.parse_args()

    timestamp = str(datetime.now())[:-7].replace('-', '_').replace(' ', '_').replace(':', '_')
    load_dir = f'./models/{args.timestamp}'
    save_dir = f'{load_dir}/{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    logging.config.fileConfig('logging.ini', defaults={'logfile': f'{save_dir}/training_prior.log'},
                              disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    params = {
        'epochs': args.epochs,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'latent_dim': 16,
        'n_critic': 1,
        'channels': 1,
        'in_size': 7
    }

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(params['channels'], params['in_size'], params['latent_dim'])
    discriminator = Discriminator(params['channels'], params['in_size'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(device)

    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    model = {
        'generator': generator,
        'discriminator': discriminator
    }

    optimizer = {
        'generator': torch.optim.Adam(generator.parameters(), lr=params['learning_rate']),
        'discriminator': torch.optim.Adam(discriminator.parameters(), lr=params['learning_rate'])
    }

    _, params_qnvae = load_model(QNVAE, 'qnvae', q=args.q, directory=load_dir)
    transform = transforms.Lambda(lambda sample: sample.type(torch.float32) / (params_qnvae['num_embeddings'] - 1))
    loaders = loader.EncodedLoader(load_dir, q=args.q, transform=transform).get(params['batch_size'], pin_memory=False)
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize(mean=(0.5,), std=(1.0,))])
    # loaders = loader.MNISTLoader(transform).get(params['batch_size'])

    trainer = trn.GANTrainer(model, optimizer, loaders, None)
    trainer.loss = adversarial_loss
    trainer.n_critic = params['n_critic']
    trainer.epochs = params['epochs']
    trainer.latent_dim = params['latent_dim']
    trainer.run()
    save_model(model['generator'], params, 'generator', args.q, save_dir)
    save_model(model['discriminator'], params, 'discriminator', args.q, save_dir)
    log.info(f'Done. saved dir: {save_dir}')
