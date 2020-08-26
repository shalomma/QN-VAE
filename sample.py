import os
import torch
import argparse
from torchvision.utils import save_image

from utils import load_model
from qnvae import QNVAE
from pixelcnn import PixelCNN


def save_samples(samples, dirname, filename):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    count = samples.size()[0]

    count_sqrt = int(count ** 0.5)
    if count_sqrt ** 2 == count:
        nrow = count_sqrt
    else:
        nrow = count

    save_image(samples, os.path.join(dirname, filename), nrow=nrow)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('timestamp', type=str, help='models timestamp')
    args = parser.parse_args()

    directory = f'models/{args.timestamp}'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quant_noise_probs = [0.25, 0.5, 0.75, 1]
    for q_ in quant_noise_probs:
        print(f'Train q={q_}')
        model_pixelcnn = load_model(PixelCNN, 'pixelcnn', q_, directory)
        encoding = model_pixelcnn.sample((1, 8, 8), 64, label=None, device=device)
        save_samples(encoding, directory, f'encoding_{q_}.png')
        model_qnvae = load_model(QNVAE, 'qnvae', q_, directory)
        decoded = model_qnvae.decode_samples(encoding)
        save_samples(decoded, directory, f'decoded_{q_}.png')
