import os
import glob
import torch
import argparse
import numpy as np
from git import Repo
import logging.config
from torch import optim
from datetime import datetime
import torchvision.transforms as transforms

import loader
from qnvae import QNVAE
from pixelcnn import PixelCNN
from trainer import PriorTrainer
from utils import save_model, load_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('timestamp', type=str, help='models timestamp')
    parser.add_argument('-e', type=int, help='epochs', default=100)
    parser.add_argument("--reload", dest='reload', help='reload weights', action='store_true')
    args = parser.parse_args()

    timestamp = str(datetime.now())[:-7].replace('-', '_').replace(' ', '_').replace(':', '_')
    save_dir = f'./models/{args.timestamp}/{timestamp}'
    load_dir = f'./models/{args.timestamp}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logging.config.fileConfig('logging.ini', defaults={'logfile': f'{save_dir}/training_prior.log'},
                              disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(device)

    params = {
        'batch_size': 32,
        'epochs': args.e,
        'data_channels': 1,
        'levels': 100,
        'hidden_fmaps': 120,
        'hidden_layers': 6,
        'causal_ksize': 7,
        'hidden_ksize': 7,
        'out_hidden_fmaps': 120,
        'max_norm': 1.,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
    }

    quant_noise_probs = [float(q.split('/')[-1][6:-3]) for q in glob.glob(f'{load_dir}/qnvae*.pt')]
    quant_noise_probs = sorted([q for q in quant_noise_probs if q != 0.0])
    _, params_qnvae = load_model(QNVAE, 'qnvae', q=quant_noise_probs[0], directory=load_dir)

    def quantize(image):
        return np.digitize(image, np.arange(params['levels']) / params['levels']) - 1

    discretize = transforms.Compose([
        transforms.Lambda(lambda image: np.array(image) / params_qnvae['num_embeddings']),
        transforms.Lambda(lambda image: quantize(image)),
    ])

    for q in quant_noise_probs:
        log.info(f'Train q={q}')
        loaders = loader.EncodedLoader(load_dir, q, discretize).get(params['batch_size'], pin_memory=False)
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
        qnvae, params_qnvae = load_model(QNVAE, 'qnvae', q=q, directory=load_dir)
        trainer = PriorTrainer(prior_model, optimizer, loaders, scheduler)
        trainer.max_norm = params['max_norm']
        trainer.levels = params['levels']
        trainer.epochs = params['epochs']
        trainer.samples_dir = save_dir
        trainer.decoder = qnvae
        trainer.num_embeddings = params_qnvae['num_embeddings']
        trainer.q = q
        trainer.run()
        params['commit'] = Repo('./').head.commit.hexsha[:7]
        params['metrics'] = trainer.metrics
        save_model(prior_model, params, 'pixelcnn', q, save_dir)
