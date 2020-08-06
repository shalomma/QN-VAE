import numpy as np
import torch
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, optimizer, loader):
        self.model = model
        self.optimizer = optimizer
        self.loader = loader
        self.epochs = 50
        self.data_variance = None
        self.phases = ['train']
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
                    if phase == 'train':
                        self.optimizer.zero_grad()
                    vq_loss, data_recon, perplexity = self.model(samples)
                    recon_error = F.mse_loss(data_recon, samples, reduction='mean') / self.data_variance
                    loss = recon_error + vq_loss
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    epoch_recon_error += recon_error.item()
                    epoch_perplexity += perplexity.item()
                    break

                self.train_recon_error.append(epoch_recon_error)
                self.train_perplexity.append(epoch_perplexity)
                to_print += f'{phase}: '
                to_print += f'recon error: {np.mean(self.train_recon_error[-100:]):.4f}  '
                to_print += f'perplexity: {np.mean(self.train_perplexity[-100:]):.4f} '

            if i % 100 == 0:
                print(to_print)
