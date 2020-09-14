import torch
import argparse

from utils import load_model, save_samples
from qnvae import QNVAE
from pixelcnn import PixelCNN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('timestamp', type=str, help='models timestamp')
    args = parser.parse_args()

    directory = f'models/{args.timestamp}'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quant_noise_probs = [0.25, 0.5, 0.75, 1]
    for q_ in quant_noise_probs:
        print(f'Train q={q_}')
        model_pixelcnn, params = load_model(PixelCNN, 'pixelcnn', q_, directory)
        encoding = model_pixelcnn.sample((1, 8, 8), 4, label=None, device=device)
        save_samples(encoding, directory, f'latent_{q_}.png')
        encoding = (encoding * params['levels']).long()
        model_qnvae, _ = load_model(QNVAE, 'qnvae', q_, directory)
        decoded = model_qnvae.decode_samples(encoding) + 0.5
        save_samples(decoded, directory, f'decoded_{q_}.png')
