import argparse
import pickle
import torch

import loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('timestamp', type=str, help='models timestamp')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 1024
    quant_noise_probs = [0, 0.25, 0.5, 0.75, 1]
    qn_model = dict()
    for q_ in quant_noise_probs:
        with open(f'models/{args.timestamp}/model_{q_}_cpu.pkl', 'rb') as f:
            qn_model[q_] = pickle.load(f).to(device)

    loaders = loader.CIFAR10Loader().get(batch_size)
    for q_, model in qn_model.items():
        if q_ == 0:
            continue
        print(f'Encoding using {q_} QN-VAE')
        model.eval()
        encode_dataset = torch.tensor([]).to(device)
        labels_dataset = torch.tensor([]).to(device)
        for phase in ['train', 'val']:
            for samples, labels in loaders[phase]:
                samples, labels = samples.to(device), labels.to(device)
                _, _, _, encoding = model(samples)
                encode_dataset = torch.cat((encode_dataset, encoding.float()))
                labels_dataset = torch.cat((labels_dataset, labels.float()))
        torch.save(encode_dataset, f'./models/{args.timestamp}/encoded_{q_}.pt')
        torch.save(labels_dataset, f'./models/{args.timestamp}/encoded_labels_{q_}.pt')
