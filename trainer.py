import torch.nn.functional as F


class Trainer:
    def __init__(self, model, optimizer, loader):
        self.model = model
        self.optimizer = optimizer
        self.loader = loader
        self.epochs = 5000
        self.data_variance = None
        self.phases = ['train']
        self.train_recon_error = []
        self.train_perplexity = []

    def train(self):

        for i in range(self.epochs):
            to_print = f'Epoch {i}: '
            for phase in self.phases:
                self.model.train() if phase == 'train' else self.model.eval()
                epoch_recon_error = 0
                epoch_perplexity = 0
                for samples, _ in self.loader[phase]:
                    # samples = samples.to(device)
                    if phase == 'train':
                        self.optimizer.zero_grad()
                    vq_loss, data_recon, perplexity = self.model(samples)
                    recon_error = F.mse_loss(data_recon, samples) / self.data_variance
                    loss = recon_error + vq_loss
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    epoch_recon_error += recon_error.item()
                    epoch_perplexity += perplexity.item()

                self.train_recon_error.append(epoch_recon_error)
                self.train_perplexity.append(epoch_perplexity)
                to_print += f'{phase}: '
                to_print += f'recon error: {epoch_recon_error:.4f}  '
                to_print += f'perplexity: {epoch_perplexity:.4f}: '

            if (i + 1) % 1 == 0:
                print(to_print)
