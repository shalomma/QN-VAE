import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch.backends import cudnn


cudnn.deterministic = True
cudnn.benchmark = False
cudnn.fastest = True


class Trainer(ABC):
    def __init__(self, model, optimizer, loader):
        self.model = model
        self.optimizer = optimizer
        self.loader = loader
        self.batches = 50
        self.phases = ['train', 'val']
        self.train_recon_error = []
        self.train_perplexity = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log = logging.getLogger(__name__).info

    def run(self):
        for i in range(self.batches):
            to_print = f'Batch {i:04}: '
            for phase in self.phases:
                self.model.train() if phase == 'train' else self.model.eval()
                samples, labels = next(iter(self.loader[phase]))
                samples = samples.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                loss = self.compare(samples, labels)

                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

                to_print += f'{phase}: '
                to_print += f'recon error: {np.mean(self.train_recon_error[-100:]):.4f}  '
                # if perplexity is not None:
                #     self.train_perplexity.append(perplexity.item())
                #     to_print += f'perplexity: {np.mean(self.train_perplexity[-100:]):.4f} '
            if i % 100 == 0:
                self.log(to_print)

    @abstractmethod
    def compare(self, samples, labels):
        pass


class VAETrainer(Trainer):
    def __init__(self, model, optimizer, loader):
        super(VAETrainer, self).__init__(model, optimizer, loader)
        self.data_variance = np.var(loader['train'].dataset.data / 255.0)

    def compare(self, samples, labels):
        vq_loss, data_recon, perplexity, encoding = self.model(samples)
        recon_error = F.mse_loss(data_recon, samples) / self.data_variance
        self.train_recon_error.append(recon_error.item())
        loss = recon_error
        if vq_loss is not None:
            loss += vq_loss
        return loss


class PriorTrainer(Trainer):
    def __init__(self, model, optimizer, loader):
        super(PriorTrainer, self).__init__(model, optimizer, loader)

    def compare(self, samples, labels):
        samples = samples.float() / (15 - 1)
        outputs = self.model(samples, labels)
        return F.cross_entropy(outputs, samples)
