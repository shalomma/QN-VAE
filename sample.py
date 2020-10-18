import os
import glob
import torch
import argparse
from torchvision.utils import save_image

from utils import load_model, save_samples
from qnvae import QNVAE
from pixelcnn import PixelCNN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('timestamp', type=str, help='models timestamp')
    parser.add_argument('--n_samples', type=int, default=64)
    args = parser.parse_args()

    directory = f'models/{args.timestamp}'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_dir = f'models/{args.timestamp}'
    quant_noise_probs = sorted([float(q.split('/')[-1][6:-3]) for q in glob.glob(f'{load_dir}/qnvae*.pt')])
    for q_ in quant_noise_probs:
        print(f'Sample q={q_}')
        model_pixelcnn, params = load_model(PixelCNN, 'pixelcnn', q_, directory)
        encoding = model_pixelcnn.sample((1, 7, 7), args.n_samples, label=None, device=device)
        save_samples(encoding, directory, f'latent_{q_}.png')
        encoding = (encoding * (params['levels'] - 1)).long()
        model_qnvae, _ = load_model(QNVAE, 'qnvae', q_, directory)
        decoded = model_qnvae.decode_samples(encoding) + 0.5
        os.makedirs(f'{directory}/{q_}')
        for i in range(args.n_samples):
            save_image(decoded[i], os.path.join(f'{directory}/{q_}', f'{i}.png'))
