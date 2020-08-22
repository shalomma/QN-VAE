import os
import logging.config
import torch
from torch.optim import Adam
from datetime import datetime
from git import Repo

import loader
from qnvae import QNVAE, AE
import trainer
from utils import save_model


seed = 14
torch.manual_seed(seed)


if __name__ == '__main__':
    if not os.path.exists('models'):
        os.makedirs('models')
    timestamp = str(datetime.now())[:-7]
    timestamp = timestamp.replace('-', '_').replace(' ', '_').replace(':', '_')
    os.makedirs(f'models/{timestamp}')

    logging.config.fileConfig('logging.ini', defaults={'logfile': f'models/{timestamp}/training.log'},
                              disable_existing_loggers=False)
    log = logging.getLogger(__name__)
    
    params = {
        'batch_size': 1024,
        'batches': 2000,
        'num_hidden': 128,
        'num_residual_hidden': 32,
        'num_residual_layers': 2,
        'embedding_dim': 64,
        'num_embeddings': 512,
        'commitment_cost': 0.25,
        'learning_rate': 1e-3
    }

    quant_noise_probs = [0, 0.25, 0.5, 0.75, 1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(device)

    qn_model = dict()
    optimizer = dict()
    for q in quant_noise_probs:
        qn_model[q] = QNVAE(params['num_hidden'], params['num_residual_layers'], params['num_residual_hidden'],
                            params['num_embeddings'], params['embedding_dim'], params['commitment_cost']).to(device)
    qn_model[0] = AE(params['num_hidden'], params['num_residual_layers'],
                     params['num_residual_hidden'], params['embedding_dim']).to(device)
    for q in quant_noise_probs:
        optimizer[q] = Adam(qn_model[q].parameters(), lr=params['learning_rate'], amsgrad=False)

    loaders = loader.CIFAR10Loader().get(params['batch_size'])

    for q in quant_noise_probs:
        log.info(f'Train q={q}')
        trainer = trainer.VAETrainer(qn_model[q], optimizer[q], loaders)
        trainer.batches = params['batches']
        trainer.run()
        params['commit'] = Repo('./').head.commit.hexsha[:7]
        params['loss'] = trainer.train_recon_error
        save_model(qn_model[q], params, q, f'models/{timestamp}')
