import logging
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from utils import save_samples


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
                encoding = self.model.sample((3, 32, 32), 4, label=None, device=self.device)
                save_samples(encoding, './models/', f'encoding_{i}.png')

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
    def __init__(self, model, optimizer, loader, scheduler):
        super(PriorTrainer, self).__init__(model, optimizer, loader, scheduler)
        self.metrics = {
            'loss': []
        }
        self.levels = None
        self.max_norm = None

    def step(self, samples, labels):
        outputs = self.model(samples, labels)
        loss = F.cross_entropy(outputs, samples.long())
        self.metrics['loss'].append(loss.item())
        if self.model.training:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            self.optimizer.step()
