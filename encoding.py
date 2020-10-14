import glob
import torch
import argparse
from torchvision import transforms

import loader
from qnvae import QNVAE
from utils import load_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('timestamp', type=str, help='models timestamp')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_dir = f'./models/{args.timestamp}'
    batch_size = 1024
    quant_noise_probs = [float(q.split('/')[-1][6:-3]) for q in glob.glob(f'{load_dir}/qnvae*.pt')]
    quant_noise_probs = sorted([q for q in quant_noise_probs if q != 0.0])

    params = None
    qn_model = dict()
    for q_ in quant_noise_probs:
        qn_model[q_], params = load_model(QNVAE, 'qnvae', q_, load_dir)

    if params['dataset'] == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0))])
        loaders = loader.CIFAR10Loader(transform).get(params['batch_size'])
    elif params['dataset'] == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5,), std=(1.0,))])
        loaders = loader.MNISTLoader(transform).get(params['batch_size'])
    else:
        raise Exception('Not a defined dataset')

    for q_, model in qn_model.items():
        print(f'Encoding using {q_} QN-VAE')
        model.eval()
        encode_dataset = torch.tensor([]).long().to(device)
        labels_dataset = torch.tensor([]).long().to(device)
        for phase in ['train', 'val']:
            for samples, labels in loaders[phase]:
                samples, labels = samples.to(device), labels.to(device)
                _, _, _, encoding = model(samples)
                encode_dataset = torch.cat((encode_dataset, encoding))
                labels_dataset = torch.cat((labels_dataset, labels))
        torch.save(encode_dataset, f'{load_dir}/encoded_data_{q_}.pt')
        torch.save(labels_dataset, f'{load_dir}/encoded_labels_{q_}.pt')
