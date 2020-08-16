import torch
from torch.optim import Adam
from loader import CIFAR10Loader
from vq_vae import VQ_VAE, AE
from trainer import Trainer


if __name__ == '__main__':
    batch_size = 1024
    epochs = 2000
    num_hidden = 128
    num_residual_hidden = 32
    num_residual_layers = 2
    embedding_dim = 64
    num_embeddings = 512
    commitment_cost = 0.25
    learning_rate = 1e-4

    quant_noise_probs = [0, 0.25, 0.5, 0.75, 1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    qn_model = dict()
    optimizer = dict()
    for q in quant_noise_probs:
        qn_model[q] = VQ_VAE(num_hidden, num_residual_layers, num_residual_hidden,
                             num_embeddings, embedding_dim, commitment_cost).to(device)
    qn_model[0] = AE(num_hidden, num_residual_layers,
                     num_residual_hidden, embedding_dim).to(device)
    for q in quant_noise_probs:
        optimizer[q] = Adam(qn_model[q].parameters(), lr=learning_rate, amsgrad=False)

    loaders = CIFAR10Loader().get(batch_size)

    for q in quant_noise_probs:
        print(f'Train q={q}')
        trainer = Trainer(qn_model[q], optimizer[q], loaders)
        trainer.epochs = epochs
        trainer.run()
        torch.save(qn_model[q].state_dict(), f'model_{q}.pt')
