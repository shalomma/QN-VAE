import os
import numpy as np
import argparse
import torch
from torch import optim
import torchvision.transforms as transforms
import logging.config
from git import Repo

import loader
from qnvae import QNVAE
from pixelcnn import PixelCNN
from trainer import PriorTrainer
from utils import save_model, load_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('timestamp', type=str, help='models timestamp')
    parser.add_argument("--reload", dest='reload', help='reload weights', action='store_true')
    args = parser.parse_args()

    logging.config.fileConfig('logging.ini', defaults={'logfile': f'models/{args.timestamp}/training_prior.log'},
                              disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    save_dir = os.path.join('models', args.timestamp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(device)

    params = {
        'batch_size': 32,
        'epochs': 50,
        'data_channels': 1,
        'hidden_fmaps': 120,
        'hidden_layers': 6,
        'causal_ksize': 7,
        'hidden_ksize': 7,
        'out_hidden_fmaps': 120,
        'max_norm': 1.,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
    }
    _, params_qnvae = load_model(QNVAE, 'qnvae', q=0.25, directory=save_dir)
    params['levels'] = params_qnvae['num_embeddings']

    def quantize(image):
        return np.digitize(image, np.arange(params['levels']) / params['levels']) - 1

    discretize = transforms.Compose([
        transforms.Lambda(lambda image: np.array(image) / params['levels']),
        transforms.Lambda(lambda image: quantize(image)),
    ])
    quant_noise_probs = [0.25, 0.5, 0.75, 1]
    for q in quant_noise_probs:
        log.info(f'Train q={q}')
        loaders = loader.EncodedLoader(save_dir, q, discretize).get(params['batch_size'], pin_memory=False)
        prior_model = PixelCNN(params['data_channels'], params['hidden_fmaps'],
                               params['levels'], params['hidden_layers'],
                               params['causal_ksize'], params['hidden_ksize'], params['out_hidden_fmaps']).to(device)
        if args.reload:
            with open(f'{save_dir}/pixelcnn_{q}.pt', 'rb') as f:
                state_dict = torch.load(f, map_location=device)
                prior_model.load_state_dict(state_dict)
        optimizer = optim.Adam(prior_model.parameters(), lr=params['learning_rate'],
                               weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, params['learning_rate'],
                                                10 * params['learning_rate'], cycle_momentum=False)
        qnvae, params_qnvae = load_model(QNVAE, 'qnvae', q=q, directory=save_dir)
        trainer = PriorTrainer(prior_model, optimizer, loaders, scheduler)
        trainer.max_norm = params['max_norm']
        trainer.levels = params['levels']
        trainer.epochs = params['epochs']
        trainer.samples_dir = save_dir
        trainer.decoder = qnvae
        trainer.q = q
        trainer.run()
        params['commit'] = Repo('./').head.commit.hexsha[:7]
        params['metrics'] = trainer.metrics
        save_model(prior_model, params, 'pixelcnn', q, save_dir)
