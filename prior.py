import os
import argparse
import torch
from torch import optim
import logging.config
from git import Repo

import loader
from pixelcnn import PixelCNN
from qnvae import QNVAE
from trainer import PriorTrainer
from utils import save_model
from utils import load_model


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
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_embeddings': 512
    }

    loaders = loader.CIFAR10Loader().get(params['batch_size'], pin_memory=False)
    quant_noise_probs = [0.25, 0.5, 0.75, 1]
    for q in quant_noise_probs:
        log.info(f'Train q={q}')
        model_qnvae, _ = load_model(QNVAE, 'qnvae', q, f'models/{args.timestamp}')
        prior_model = PixelCNN(input_dim=256, dim=64, n_layers=15, n_classes=10).to(device)
        optimizer = optim.Adam(prior_model.parameters(), lr=params['learning_rate'],
                               weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, params['learning_rate'],
                                                10 * params['learning_rate'], cycle_momentum=False)
        trainer = PriorTrainer(prior_model, model_qnvae, optimizer, loaders, scheduler)
        trainer.batches = params['batches']
        trainer.run()
        params['commit'] = Repo('./').head.commit.hexsha[:7]
        params['loss'] = trainer.metrics['loss']
        save_model(prior_model, params, 'pixelcnn', q, f'models/{args.timestamp}')
