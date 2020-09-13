import logging
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from utils import save_samples
from tqdm import tqdm


cudnn.deterministic = True
cudnn.benchmark = False
cudnn.fastest = True


class Trainer(ABC):
    def __init__(self, model, optimizer, loader, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loader = loader
        self.scheduler = scheduler
        self.root_dir = './'
        self.epochs = 50
        self.phases = ['train', 'val']
        self.metrics = dict()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log = logging.getLogger(__name__).info

    def run(self):
        for i in range(self.epochs):
            for phase in self.phases:
                self.model.train() if phase == 'train' else self.model.eval()
                for data in tqdm(self.loader[phase], desc=f'Epoch {i}/{self.epochs}'):
                    samples, labels = data
                    samples = samples.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    self.optimizer.zero_grad()
                    self.step(samples, labels)

            self.scheduler.step()
            encoding = self.model.sample((3, 32, 32), 8, label=None, device=self.device)
            save_samples(encoding, self.root_dir, f'encoding_{i}.png')

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
        normalized_samples = samples.float() / (self.levels - 1)
        outputs = self.model(normalized_samples, labels)
        loss = F.cross_entropy(outputs, samples.long())
        self.metrics['loss'].append(loss.item())
        if self.model.training:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            self.optimizer.step()
