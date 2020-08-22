import os
import argparse
import torch
from torch.optim import Adam

import loader
from pixelcnn import PixelCNN
import trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('timestamp', type=str, help='models timestamp')
    args = parser.parse_args()

    root_dir = os.path.join('models', args.timestamp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_fmaps = 21
    color_levels = 4
    hidden_layers = 6
    causal_ksize = 7
    hidden_ksize = 7
    out_hidden_fmaps = 10

    learning_rate = 1e-4
    batch_size = 1024
    batches = 1000

    quant_noise_probs = [0.25, 0.5, 0.75, 1]
    for q in quant_noise_probs:
        loaders = loader.EncodedLoader(root_dir, q).get(batch_size)
        prior_model = PixelCNN(hidden_fmaps, color_levels, hidden_layers,
                               causal_ksize, hidden_ksize, out_hidden_fmaps)
        optimizer = Adam(prior_model.parameters(), lr=learning_rate, amsgrad=False)
        trainer = trainer.PriorTrainer(prior_model, optimizer, loaders)
        trainer.batches = batches
        trainer.run()
