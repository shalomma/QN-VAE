import glob
import umap
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy.signal import savgol_filter
from torchvision.utils import make_grid
from qnvae import QNVAE
import loader as ld
from utils import load_model


class Plotter:
    def __init__(self, models, loaders):
        self.models = models
        [m.eval() for _, m in self.models.items()]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaders = loaders
        self.images = dict()
        self.sample_images()

    def sample_images(self):
        for phase, loader in self.loaders.items():
            (samples, _) = next(iter(loader))
            self.images[phase] = samples.to(self.device)

    @staticmethod
    def prepare_images(valid_reconstructions):
        img = make_grid(valid_reconstructions.cpu().data) + 0.5
        img = img.numpy()
        return np.transpose(img, (1, 2, 0))

    @staticmethod
    def smooth(params, metric):
        smoothed = {
            'train': dict(),
            'val': dict()
        }
        for q in quant_noise_probs:
            for p in ['train', 'val']:
                smoothed[p][q] = savgol_filter(params[q]['metrics'][p][metric], 201, 7)
        return smoothed

    def losses(self, params):
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)
        for q in quant_noise_probs:
            ax.plot(params[q]['metrics']['train']['loss'], label=f'train q={q}')
            ax.plot(params[q]['metrics']['val']['loss'], linestyle='dashed', label=f'val q={q}')
        ax.set_yscale('log')
        ax.set_title('Smoothed Reconstruction Error (MSE)')
        ax.set_xlabel('iteration')
        ax.legend()
        plt.savefig(f'loss.png')
        plt.show()

    def perplexity(self, params):
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)
        for q in quant_noise_probs:
            ax.plot(params[q]['metrics']['train']['perplexity'], label=f'train q={q}')
            ax.plot(params[q]['metrics']['val']['perplexity'], linestyle='dashed', label=f'val q={q}')
        ax.set_title('perplexity')
        ax.set_xlabel('iteration')
        ax.legend()
        plt.savefig(f'perplexity.png')
        plt.show()

    def recon_train(self):
        _, train_reconstructions, _, _ = self.models[quant_noise_probs[-1]](self.images['train'])
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        fig.suptitle('Training Reconstruction')
        axs[0].imshow(self.prepare_images(self.images['train']), interpolation='nearest')
        axs[0].set_title('Input')
        axs[1].imshow(self.prepare_images(train_reconstructions), interpolation='nearest')
        axs[1].set_title('Reconstruction')
        plt.tight_layout()
        plt.savefig(f'img_train.png')
        plt.show()

    def recon_val(self):
        images = dict()
        for q, model in self.models.items():
            _, valid_reconstructions, _, _ = model(self.images['val'])
            images[q] = self.prepare_images(valid_reconstructions)
        fig, axs = plt.subplots(2, 3, figsize=(10, 10))
        fig.suptitle('Validation Reconstruction')
        n = len(images)
        q = list(images.keys())
        axs[0, 0].imshow(images[q[0]], interpolation='nearest')
        axs[0, 0].set_title(f'q = {q[0]}')
        if n > 1:
            axs[0, 1].imshow(images[q[1]], interpolation='nearest')
            axs[0, 1].set_title(f'q = {q[1]}')
        elif n > 2:
            axs[0, 2].imshow(images[q[2]], interpolation='nearest')
            axs[0, 2].set_title(f'q = {q[2]}')
        elif n > 3:
            axs[1, 0].imshow(images[q[3]], interpolation='nearest')
            axs[1, 0].set_title(f'q = {q[3]}')
        elif n > 4:
            axs[1, 1].imshow(images[q[4]], interpolation='nearest')
            axs[1, 1].set_title(f'q = {q[4]}')
        axs[1, 2].imshow(self.prepare_images(self.images['val']), interpolation='nearest')
        axs[1, 2].set_title('Input')
        for ax in axs.flat:
            ax.label_outer()
        plt.tight_layout()
        plt.savefig(f'img_val.png')
        plt.show()

    def embedding(self):
        proj = dict()
        for q, model in self.models.items():
            if q != 0:
                proj[q] = umap.UMAP(n_neighbors=3, min_dist=0.1,
                                    metric='cosine').fit_transform(model.vq_vae.embedding.weight.data.cpu())
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        q = list(proj.keys())
        n = len(proj)
        fig.suptitle('UMAP projection of embedding (codebook)')
        axs[0, 0].scatter(proj[q[0]][:, 0], proj[q[0]][:, 1], alpha=0.3)
        axs[0, 0].set_title(f'q = {q[0]}')
        if n > 1:
            axs[0, 1].scatter(proj[q[1]][:, 0], proj[q[1]][:, 1], alpha=0.3)
            axs[0, 1].set_title(f'q = {q[1]}')
        elif n > 2:
            axs[1, 0].scatter(proj[q[2]][:, 0], proj[q[2]][:, 1], alpha=0.3)
            axs[1, 0].set_title(f'q = {q[2]}')
        elif n > 3:
            axs[1, 1].scatter(proj[q[3]][:, 0], proj[q[3]][:, 1], alpha=0.3)
            axs[1, 1].set_title(f'q = {q[3]}')
        for ax in axs.flat:
            ax.label_outer()
        plt.savefig(f'emb.png')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('timestamp', type=str, help='models timestamp')
    args = parser.parse_args()

    load_dir = f'models/{args.timestamp}'
    quant_noise_probs = sorted([float(q.split('/')[-1][6:-3]) for q in glob.glob(f'{load_dir}/qnvae*.pt')])
    qn_model, params_ = dict(), dict()
    for q_ in quant_noise_probs:
        qn_model[q_], params_[q_] = load_model(QNVAE, 'qnvae', q_, load_dir)

    dataset = params_[quant_noise_probs[0]]['dataset']
    if dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0))])
        loaders_ = ld.CIFAR10Loader(transform).get(64)
    elif dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5,), std=(1.0,))])
        loaders_ = ld.MNISTLoader(transform).get(64)
    else:
        raise Exception('Not a defined dataset')

    plotter = Plotter(qn_model, loaders_)
    plotter.losses(params_)
    plotter.perplexity(params_)
    plotter.recon_train()
    plotter.recon_val()
    plotter.embedding()
