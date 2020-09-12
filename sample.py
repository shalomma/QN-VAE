import torch
import argparse

from utils import load_model, save_samples
from pixelcnn import PixelCNN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('timestamp', type=str, help='models timestamp')
    args = parser.parse_args()

    directory = f'models/{args.timestamp}'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_pixelcnn, _ = load_model(PixelCNN, 'pixelcnn', q=0, directory=directory)
    encoding = model_pixelcnn.sample((3, 32, 32), 4, label=None, device=device)
    save_samples(encoding, directory, f'encoding.png')
