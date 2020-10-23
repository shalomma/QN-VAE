import logging
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.backends import cudnn
from utils import save_samples
from torchvision.utils import save_image
import copy


cudnn.deterministic = True
cudnn.benchmark = False
cudnn.fastest = True


class Trainer(ABC):
    def __init__(self, model, optimizer, loader, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.loader = loader
        self.scheduler = scheduler
        self.epochs = 50
        self.phases = ['train', 'val']
        self.metrics = {
            'train': dict(),
            'val': dict(),
        }
        self.metrics_step = {
            'train': dict(),
            'val': dict(),
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log = logging.getLogger(__name__).info
        self.best_weights = None
        self.best_loss = 100

    def run(self):
        for e in range(self.epochs):
            for phase in self.phases:
                self.set_model_phase(phase)
                for data in self.loader[phase]:
                    samples, labels = data
                    samples = samples.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    self.zero_grad()
                    self.step(phase, samples, labels)

            if self.scheduler is not None:
                self.scheduler.step()

            self.evaluate(e)
            self.model_checkpoint()
            self.log_epoch(e)

        self.model.load_state_dict(self.best_weights)

    def set_model_phase(self, phase):
        self.model.train() if phase == 'train' else self.model.eval()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def log_epoch(self, epoch):
        to_print = f'Epoch {epoch:04}:'
        for phase in self.phases:
            to_print += f'\t{phase}: '
            for metric, values in self.metrics[phase].items():
                to_print += f'{metric}: {values[-1]:.4f}  '
        self.log(to_print)

    def model_checkpoint(self):
        loss = self.metrics['val']['loss'][-1]
        if self.best_loss > loss:
            self.best_loss = loss
            self.best_weights = copy.deepcopy(self.model.state_dict())

    @abstractmethod
    def step(self, phase, samples, labels):
        pass

    def evaluate(self, epoch):
        for phase in self.phases:
            for metric, values in self.metrics_step[phase].items():
                self.metrics[phase][metric].append(np.mean(values))


class VAETrainer(Trainer):
    def __init__(self, model, optimizer, loader, scheduler=None):
        super(VAETrainer, self).__init__(model, optimizer, loader, scheduler)
        self.data_variance = np.var(loader['train'].dataset.data.numpy() / 255.0)
        self.metrics = {
            'train': {'loss': [], 'perplexity': []},
            'val': {'loss': [], 'perplexity': []}
        }
        self.metrics_step = {
            'train': {'loss': [], 'perplexity': []},
            'val': {'loss': [], 'perplexity': []}
        }

    def step(self, phase, samples, labels):
        vq_loss, data_recon, perplexity, encoding = self.model(samples)
        recon_error = F.mse_loss(data_recon, samples) / self.data_variance
        self.metrics_step[phase]['loss'].append(recon_error.item())
        loss = recon_error
        if vq_loss is not None:
            loss += vq_loss
        if perplexity is not None:
            self.metrics_step[phase]['perplexity'].append(perplexity.item())
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
        self.metrics_step = {
            'train': {'loss': []},
            'val': {'loss': []}
        }
        self.samples_dir = './'
        self.levels = None
        self.max_norm = None
        self.channels = 1
        self.num_embeddings = 0
        self.decoder = None
        self.q = 0.

    def step(self, phase, samples, labels):
        normalized_samples = samples.float() / (self.levels - 1)
        outputs = self.model(normalized_samples, labels)
        loss = F.cross_entropy(outputs, samples.long())
        self.metrics_step[phase]['loss'].append(loss.item())
        if self.model.training:
            loss.backward()
            if self.max_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            self.optimizer.step()

    def evaluate(self, epoch):
        super(PriorTrainer, self).evaluate(epoch)
        encoding = self.model.sample((1, 7, 7), 4, label=None, device=self.device)
        save_samples(encoding, self.samples_dir, f'latent_{self.q}_{epoch}.png')
        encoding = (encoding * (self.num_embeddings - 1)).long()
        decoded = self.decoder.decode_samples(encoding) + 0.5
        save_samples(decoded, self.samples_dir, f'decoded_{self.q}_{epoch}.png')


class GANTrainer(Trainer):
    def __init__(self, model, optimizer, loader, scheduler):
        super(GANTrainer, self).__init__(model, optimizer, loader, scheduler)
        self.loss = None
        self.latent_dim = None
        self.phases = ['train']
        self.metrics = {
            'generator': {'loss': []},
            'discriminator': {'loss': []}
        }
        self.metrics_step = {
            'generator': {'loss': []},
            'discriminator': {'loss': []}
        }

    def step(self, phase, samples, labels):
        valid = Variable(torch.ones((samples.size(0), 1), device=self.device), requires_grad=False)
        fake = Variable(torch.zeros((samples.size(0), 1), device=self.device), requires_grad=False)
        real_imgs = Variable(samples, requires_grad=False)

        z = Variable(torch.tensor(np.random.normal(0, 1, (samples.shape[0], self.latent_dim)),
                                  dtype=torch.float32, device=self.device))
        gen_imgs = self.model['generator'](z)
        g_loss = self.loss(self.model['discriminator'](gen_imgs), valid)
        g_loss.backward()
        self.optimizer['generator'].step()

        real_loss = self.loss(self.model['discriminator'](real_imgs), valid)
        fake_loss = self.loss(self.model['discriminator'](gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer['discriminator'].step()

        self.metrics_step['discriminator']['loss'].append(d_loss.item())
        self.metrics_step['generator']['loss'].append(g_loss.item())

    def set_model_phase(self, phase):
        self.model['generator'].train() if phase == 'train' else self.model['generator'].eval()
        self.model['discriminator'].train() if phase == 'train' else self.model['discriminator'].eval()

    def zero_grad(self):
        self.optimizer['generator'].zero_grad()
        self.optimizer['discriminator'].zero_grad()

    def log_epoch(self, epoch):
        to_print = f'Epoch {epoch:04}:'
        for phase in ['discriminator', 'generator']:
            to_print += f'\t{phase}: '
            for metric, values in self.metrics[phase].items():
                to_print += f'{metric}: {values[-1]:.4f}  '
        self.log(to_print)

    def evaluate(self, epoch):
        for phase in ['discriminator', 'generator']:
            for metric, values in self.metrics_step[phase].items():
                self.metrics[phase][metric].append(np.mean(values))
        z = Variable(torch.tensor(np.random.normal(0, 1, (25, self.latent_dim)), device=self.device))
        z = z.type(torch.float32)
        gen_imgs = self.model['generator'](z)
        save_image(gen_imgs.data, f"images/{epoch}.png", nrow=5, normalize=True)

    def model_checkpoint(self):
        pass
