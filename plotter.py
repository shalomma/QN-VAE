import torch
import umap
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from loader import CIFAR10Loader


class Plotter:
    def __init__(self, models, loaders):
        self.models = models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaders = loaders
        self.images = dict()
        self.sample_images()

    def sample_images(self):
        for loader, phase in self.loaders.items():
            (samples, _) = next(iter(loader))
            self.images[phase] = samples.to(self.device)

    @staticmethod
    def prepare_images(valid_reconstructions):
        img = make_grid(valid_reconstructions.cpu().data) + 0.5
        img = img.numpy()
        return np.transpose(img, (1, 2, 0))

    def recon_train(self):
        _, train_reconstructions, _ = self.models[1](self.images['train'])
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        fig.suptitle('Training Reconstruction')
        axs[0].imshow(self.prepare_images(self.images['train'][:64]), interpolation='nearest')
        axs[0].set_title('Input')
        axs[1].imshow(self.prepare_images(train_reconstructions[:64]), interpolation='nearest')
        axs[1].set_title('Reconstruction')

        plt.tight_layout()
        plt.savefig(f'img_train.png')
        plt.show()

    def recon_val(self):
        images = dict()
        for q, model in self.models.items():
            model[q].eval()
            _, valid_reconstructions, _ = model[q](self.images['val'])
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


if __name__ == '__main__':
    quant_noise_probs = [0, 0.25, 0.5, 0.75, 1]
    qn_model = dict()
    for q_ in quant_noise_probs:
        with open(f'model_{q_}.pt', 'rb') as f:
            qn_model[q_] = pickle.load(f)

    loaders_ = CIFAR10Loader().get(64)
    plotter = Plotter(qn_model, loaders_)
    plotter.recon_train()
    plotter.recon_val()
    plotter.embedding()
