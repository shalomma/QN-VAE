import os
import numpy as np
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('timestamp', type=str, help='models timestamp')
    args = parser.parse_args()

    logging.config.fileConfig('logging.ini', defaults={'logfile': f'models/{args.timestamp}/training_prior.log'},
                              disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    root_dir = os.path.join('models', args.timestamp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(device)

    params = {
        'batch_size': 256,
        'batches': 5000,
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

    def quantize(image, levels):
        return np.digitize(image, np.arange(levels) / levels) - 1

    discretize = transforms.Compose([
        transforms.Lambda(lambda image: np.array(image) / params['num_embeddings']),
        transforms.Lambda(lambda image: quantize(image, params['levels'])),
    ])
    discretize = None
    quant_noise_probs = [0.25, 0.5, 0.75, 1]
    for q in quant_noise_probs:
        log.info(f'Train q={q}')
        loaders = loader.EncodedLoader(root_dir, q, discretize).get(params['batch_size'], pin_memory=False)
        prior_model = PixelCNN(params['hidden_fmaps'], params['num_embeddings'], params['hidden_layers'],
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
        save_model(prior_model, params, 'pixelcnn', q, f'models/{args.timestamp}')
