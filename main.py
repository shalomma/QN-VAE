import torch
from torch.optim import Adam
import loader
from qn_vae import QNVAE, AE
from trainer import Trainer
from utils import save_model


seed = 14
torch.manual_seed(seed)


if __name__ == '__main__':
    params = {
        'batch_size': 128,
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
    print(device)

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
        print(f'Train q={q}')
        trainer = Trainer(qn_model[q], optimizer[q], loaders)
        trainer.batches = params['batches']
        trainer.run()
        save_model(qn_model[q], params, q)
