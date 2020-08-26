import os
import torch
import pickle
import argparse
from torchvision.utils import save_image
from qnvae import QNVAE


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quant_noise_probs = [0.25, 0.5, 0.75, 1]
    for q_ in quant_noise_probs:
        print(f'Train q={q_}')
        with open(f'models/{args.timestamp}/pixcelcnn_{q_}_cpu.pkl', 'rb') as f:
            model_pixelcnn = pickle.load(f).cpu()
        encoding = model_pixelcnn.sample((1, 8, 8), 64, label=None, device=device)
        save_samples(encoding, f'models/{args.timestamp}', f'encoding_{q_}.png')
        with open(f'models/{args.timestamp}/params_{q_}.pkl', 'rb') as f:
            params = pickle.load(f)
        model_qnvae = QNVAE(params['num_hidden'], params['num_residual_layers'], params['num_residual_hidden'],
                            params['num_embeddings'], params['embedding_dim'], params['commitment_cost'],
                            quant_noise=q_).to(device)
        model_qnvae.load_state_dict(torch.load(f'models/{args.timestamp}/model_{q_}_state.pt'))
        model_qnvae.eval()
        decoded = model_qnvae.decode_samples(encoding)
        save_samples(decoded, f'models/{args.timestamp}', f'decoded_{q_}.png')
