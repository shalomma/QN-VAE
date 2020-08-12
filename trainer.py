import numpy as np
import torch
import torch.nn.functional as F
from vq_vae import VQ_VAE


class Trainer:
    def __init__(self, model, optimizer, loader):
        self.model = model
        self.optimizer = optimizer
        self.loader = loader
        self.data_variance = np.var(loader['train'].dataset.data / 255.0)
        self.epochs = 50
        self.phases = ['train', 'val']
        self.train_recon_error = []
        self.train_perplexity = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        for i in range(self.epochs):
            to_print = f'Epoch {i:04}: '
            for phase in self.phases:
                self.model.train() if phase == 'train' else self.model.eval()
                epoch_recon_error = 0
                epoch_perplexity = 0
                for k, (samples, _) in enumerate(self.loader[phase]):
                    samples = samples.to(self.device)
                    self.optimizer.zero_grad()
                    vq_loss, data_recon, perplexity = self.model(samples)
                    recon_error = F.mse_loss(data_recon, samples) / self.data_variance
                    epoch_recon_error += recon_error.item() * np.prod(samples.shape)
                    loss = recon_error
                    if isinstance(self.model, VQ_VAE):
                        loss += vq_loss
                        epoch_perplexity += perplexity.item()
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                n_elements = np.prod(self.loader[phase].dataset.data.shape)
                self.train_recon_error.append(epoch_recon_error / n_elements)
                to_print += f'{phase}: '
                to_print += f'recon error: {self.train_recon_error[-1]:.4f}  '
                if isinstance(self.model, VQ_VAE):
                    self.train_perplexity.append(epoch_perplexity / len(self.loader[phase].dataset))
                    to_print += f'perplexity: {np.mean(self.train_perplexity[-1]):.4f} '
            print(to_print)
