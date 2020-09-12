import os
from datetime import datetime

import numpy as np
import torch
from torch import optim
import torchvision.transforms as transforms
import logging.config
from git import Repo

import loader
from pixelcnn import PixelCNN
from trainer import PriorTrainer
from utils import save_model


if __name__ == '__main__':
    timestamp = str(datetime.now())[:-7]
    timestamp = timestamp.replace('-', '_').replace(' ', '_').replace(':', '_')
    os.makedirs(f'models/{timestamp}')

    logging.config.fileConfig('logging.ini', defaults={'logfile': f'models/{timestamp}/training_prior.log'},
                              disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    root_dir = os.path.join('models', timestamp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(device)

    params = {
        'batch_size': 256,
        'batches': 5000,
        'data_channels': 3,
        'hidden_fmaps': 30,
        'levels': 10,
        'hidden_layers': 6,
        'causal_ksize': 7,
        'hidden_ksize': 7,
        'out_hidden_fmaps': 10,
        'max_norm': 1.,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_embeddings': 512
    }

    def quantize(image):
        return np.digitize(np.array(image) / 255, np.arange(params['levels']) / params['levels']) - 1

    discretize = transforms.Compose([
        transforms.Lambda(quantize),
        transforms.ToTensor(),
        transforms.Lambda(lambda image: image.float())
    ])

    loaders = loader.CIFAR10Loader(discretize).get(params['batch_size'], pin_memory=False)
    prior_model = PixelCNN(params['data_channels'], params['hidden_fmaps'],
                           params['num_embeddings'], params['hidden_layers'],
                           params['causal_ksize'], params['hidden_ksize'], params['out_hidden_fmaps']).to(device)
    optimizer = optim.Adam(prior_model.parameters(), lr=params['learning_rate'],
                           weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, params['learning_rate'],
                                            10 * params['learning_rate'], cycle_momentum=False)
    trainer = PriorTrainer(prior_model, optimizer, loaders, scheduler)
    trainer.max_norm = params['max_norm']
    trainer.levels = params['levels']
    trainer.batches = params['batches']
    trainer.run()
    params['commit'] = Repo('./').head.commit.hexsha[:7]
    params['loss'] = trainer.metrics['loss']
    save_model(prior_model, params, 'pixelcnn', q=0, directory=f'models/{timestamp}')
