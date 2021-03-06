import os
import torch
import argparse
from git import Repo
import logging.config
from torch import optim
from datetime import datetime
from torchvision import transforms

import loader
from qnvae import QNVAE, AE
from utils import save_model
from trainer import VAETrainer

seed = 14
torch.manual_seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('quant', type=str, help='quant list')
    parser.add_argument('--dataset', type=str, help='epochs', default='cifar10')
    parser.add_argument('--epochs', type=int, help='epochs', default=100)
    args = parser.parse_args()

    if not os.path.exists('models'):
        os.makedirs('models')
    timestamp = str(datetime.now())[:-7].replace('-', '_').replace(' ', '_').replace(':', '_')
    os.makedirs(f'models/{timestamp}')

    logging.config.fileConfig('logging.ini', defaults={'logfile': f'models/{timestamp}/training.log'},
                              disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    params = {
        'batch_size': 1024,
        'epochs': args.epochs,
        'num_hidden': 128,
        'num_residual_hidden': 32,
        'num_residual_layers': 2,
        'embedding_dim': 64,
        'num_embeddings': 512,
        'commitment_cost': 0.25,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(device)

    if args.dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0))])
        loaders = loader.CIFAR10Loader(transform).get(params['batch_size'])
        params['in_channels'] = 3
    elif args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5,), std=(1.0,))])
        loaders = loader.MNISTLoader(transform).get(params['batch_size'])
        params['in_channels'] = 1
    else:
        raise Exception('Not a defined dataset')
    params['dataset'] = args.dataset

    qn_model = dict()
    quant_noise_probs = args.quant.split(',')
    quant_noise_probs = [float(q) for q in quant_noise_probs]
    for q in quant_noise_probs:
        qn_model[q] = QNVAE(params['in_channels'], params['num_hidden'], params['num_residual_layers'],
                            params['num_residual_hidden'], params['num_embeddings'],
                            params['embedding_dim'], params['commitment_cost'], quant_noise=q).to(device)
    if 0 in quant_noise_probs:
        qn_model[0] = AE(params['in_channels'], params['num_hidden'], params['num_residual_layers'],
                         params['num_residual_hidden'], params['embedding_dim']).to(device)

    for q, model in qn_model.items():
        log.info(f'Train q={q}')
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        trainer = VAETrainer(model, optimizer, loaders)
        trainer.epochs = params['epochs']
        trainer.run()
        params['commit'] = Repo('./').head.commit.hexsha[:7]
        params['metrics'] = trainer.metrics
        params['quant_noise'] = q
        save_model(model, params, 'qnvae', q, f'models/{timestamp}')

    log.info(f'Done. timestamp: {timestamp}')
