import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn


cudnn.deterministic = True
cudnn.benchmark = False
cudnn.fastest = True


class Trainer(ABC):
    def __init__(self, model, optimizer, loader, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loader = loader
        self.scheduler = scheduler
        self.batches = 50
        self.phases = ['train', 'val']
        self.metrics = dict()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log = logging.getLogger(__name__).info

    def run(self):
        for i in range(self.batches):
            to_print = f'Batch {i:04}:'
            for phase in self.phases:
                self.model.train() if phase == 'train' else self.model.eval()
                samples, labels = next(iter(self.loader[phase]))
                samples = samples.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                self.step(samples, labels)

                if phase == 'train' and self.scheduler is not None:
                    self.scheduler.step()

                to_print += f'\t{phase}: '
                for metric, values in self.metrics.items():
                    if values:
                        to_print += f'{metric}: {np.mean(values[-100:]):.4f}  '
            if i % 100 == 0:
                self.log(to_print)

    @abstractmethod
    def step(self, samples, labels):
        pass


class VAETrainer(Trainer):
    def __init__(self, model, optimizer, loader, scheduler):
        super(VAETrainer, self).__init__(model, optimizer, loader, scheduler)
        self.data_variance = np.var(loader['train'].dataset.data / 255.0)
        self.metrics = {
            'loss': [],
            'perplexity': []
        }

    def step(self, samples, labels):
        vq_loss, data_recon, perplexity, encoding = self.model(samples)
        recon_error = F.mse_loss(data_recon, samples) / self.data_variance
        self.metrics['loss'].append(recon_error.item())
        loss = recon_error
        if vq_loss is not None:
            loss += vq_loss
        if perplexity is not None:
            self.metrics['perplexity'].append(perplexity.item())
        if self.model.training:
            loss.backward()
            self.optimizer.step()


class PriorTrainer(Trainer):
    def __init__(self, prior, model, optimizer, loader, scheduler):
        super(PriorTrainer, self).__init__(prior, optimizer, loader, scheduler)
        self.metrics = {
            'loss': []
        }
        self.encoder = model

    def step(self, samples, labels):
        with torch.no_grad():
            _, _, _, latents = self.encoder.encode(samples)
            latents = latents.detach()

        logits = self.model(latents, labels)
        logits = logits.permute(0, 2, 3, 1).contiguous()
        loss = F.cross_entropy(logits.view(-1, self.model.input_dim), latents.view(-1))
        self.metrics['loss'].append(loss.item())
        if self.model.training:
            loss.backward()
            self.optimizer.step()
