import os
import torch
import argparse
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image

from utils import load_model
from gan import Generator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gan_dir', type=str, help='models timestamp')
    parser.add_argument('--n_samples', type=int, default=64)
    args = parser.parse_args()

    directory = f'models/{args.gan_dir}'
    directory_img = f'{directory}/images'
    os.makedirs(directory_img, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator, params = load_model(Generator, 'generator', 'mnist', directory)
    z = Variable(torch.tensor(np.random.normal(0, 1, (25, params['latent_dim'])), device=device, dtype=torch.float32))
    generated = generator(z)
    for i in range(args.n_samples):
        save_image(generated[i], os.path.join(directory_img, f'{i:04}.png'))
