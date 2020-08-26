import argparse
import torch
import pickle
import umap
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from torchvision.utils import make_grid
from loader import CIFAR10Loader


class Plotter:
    def __init__(self, models, loaders):
        self.models = models
        [m.eval() for m in self.models]
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
    def losses(params):
        recon_error_smooth = dict()
        for q in quant_noise_probs:
            recon_error_smooth[q] = savgol_filter(params[q]['loss'], 201, 7)
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)
        for q in quant_noise_probs:
            ax.plot(recon_error_smooth[q], label=f'q={q}')
        ax.set_yscale('log')
        ax.set_title('Smoothed Reconstruction Error (MSE)')
        ax.set_xlabel('iteration')
        ax.legend()
        plt.savefig(f'loss.png')
        plt.show()

    def recon_train(self):
        _, train_reconstructions, _, _ = self.models[1](self.images['train'])
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
        axs[0, 0].imshow(images[0], interpolation='nearest')
        axs[0, 0].set_title('q = 0')
        axs[0, 1].imshow(images[0.25], interpolation='nearest')
        axs[0, 1].set_title('q = 0.25')
        axs[0, 2].imshow(images[0.5], interpolation='nearest')
        axs[0, 2].set_title('q = 0.5')
        axs[1, 0].imshow(images[0.75], interpolation='nearest')
        axs[1, 0].set_title('q = 0.75')
        axs[1, 1].imshow(images[1], interpolation='nearest')
        axs[1, 1].set_title('q = 1')
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
        fig.suptitle('UMAP projection of embedding (codebook)')
        axs[0, 0].scatter(proj[0.25][:, 0], proj[0.25][:, 1], alpha=0.3)
        axs[0, 0].set_title('q = 0.25')
        axs[0, 1].scatter(proj[0.5][:, 0], proj[0.5][:, 1], alpha=0.3)
        axs[0, 1].set_title('q = 0.5')
        axs[1, 0].scatter(proj[0.75][:, 0], proj[0.75][:, 1], alpha=0.3)
        axs[1, 0].set_title('q = 0.75')
        axs[1, 1].scatter(proj[1][:, 0], proj[1][:, 1], alpha=0.3)
        axs[1, 1].set_title('q = 1')
        for ax in axs.flat:
            ax.label_outer()
        plt.savefig(f'emb.png')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('timestamp', type=str, help='models timestamp')
    args = parser.parse_args()

    quant_noise_probs = [0, 0.25, 0.5, 0.75, 1]
    qn_model = dict()
    params_ = dict()
    for q_ in quant_noise_probs:
        with open(f'models/{args.timestamp}/vqvae_{q_}_cpu.pkl', 'rb') as f:
            qn_model[q_] = pickle.load(f).cpu()
        with open(f'models/{args.timestamp}/vqvae_params_{q_}.pkl', 'rb') as f:
            params_[q_] = pickle.load(f)

    loaders_ = CIFAR10Loader().get(64)
    plotter = Plotter(qn_model, loaders_)
    plotter.losses(params_)
    plotter.recon_train()
    plotter.recon_val()
    plotter.embedding()
