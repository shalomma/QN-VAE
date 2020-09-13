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
    def __init__(self, model, optimizer, loader, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.loader = loader
        self.scheduler = scheduler
        self.root_dir = './'
        self.epochs = 50
        self.phases = ['train', 'val']
        self.metrics = {
            'train': dict(),
            'val': dict(),
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log = logging.getLogger(__name__).info

    def run(self):
        for e in range(self.epochs):
            to_print = f'Epoch {e:04}:'
            for phase in self.phases:
                self.model.train() if phase == 'train' else self.model.eval()
                for data in self.loader[phase]:
                    samples, labels = data
                    samples = samples.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    self.optimizer.zero_grad()
                    self.step(phase, samples, labels)

                to_print += f'\t{phase}: '
                for metric, values in self.metrics[phase].items():
                    if values:
                        to_print += f'{metric}: {np.mean(values[-100:]):.4f}  '

            if self.scheduler is not None:
                self.scheduler.step()
            self.log(to_print)
            self.evaluate(e)

    @abstractmethod
    def step(self, phase, samples, labels):
        pass

    def evaluate(self, epoch):
        pass


class VAETrainer(Trainer):
    def __init__(self, model, optimizer, loader, scheduler=None):
        super(VAETrainer, self).__init__(model, optimizer, loader, scheduler)
        self.data_variance = np.var(loader['train'].dataset.data / 255.0)
        self.metrics = {
            'train': {'loss': [], 'perplexity': []},
            'val': {'loss': [], 'perplexity': []}
        }

    def step(self, phase, samples, labels):
        vq_loss, data_recon, perplexity, encoding = self.model(samples)
        recon_error = F.mse_loss(data_recon, samples) / self.data_variance
        self.metrics[phase]['loss'].append(recon_error.item())
        loss = recon_error
        if vq_loss is not None:
            loss += vq_loss
        if perplexity is not None:
            self.metrics[phase]['perplexity'].append(perplexity.item())
        if self.model.training:
            loss.backward()
            self.optimizer.step()


class PriorTrainer(Trainer):
    def __init__(self, model, optimizer, loader, scheduler):
        super(PriorTrainer, self).__init__(model, optimizer, loader, scheduler)
        self.metrics = {
            'train': {'loss': []},
            'val': {'loss': []}
        }
        self.levels = None
        self.max_norm = None
        self.channels = 1

    def step(self, phase, samples, labels):
        normalized_samples = samples.float() / (self.levels - 1)
        outputs = self.model(normalized_samples, labels)
        loss = F.cross_entropy(outputs, samples.long())
        self.metrics[phase]['loss'].append(loss.item())
        if self.model.training:
            loss.backward()
            if self.max_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            self.optimizer.step()

    def evaluate(self, epoch):
        encoding = self.model.sample((self.channels, 32, 32), 8, label=None, device=self.device)
        save_samples(encoding, self.root_dir, f'encoding_{epoch}.png')
