import torch
from torch import optim
import torchvision.transforms as transforms
import numpy as np
import logging.config

import loader
from pixelcnn import PixelCNN
from trainer import PriorTrainer
from sample import save_samples


if __name__ == '__main__':
    logging.config.fileConfig('logging.ini', defaults={'logfile': f'training_pixelcnn.log'},
                              disable_existing_loggers=False)
    log = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(device)

    params = {
        'batch_size': 256,
        'batches': 3000,
        'hidden_fmaps': 30,
        'levels': 2,
        'hidden_layers': 6,
        'causal_ksize': 7,
        'hidden_ksize': 7,
        'out_hidden_fmaps': 10,
        'max_norm': 1.,
        'learning_rate': 1e-4,
        'num_embeddings': 512
    }

    def quantize(image, levels): return np.digitize(image, np.arange(levels) / levels) - 1
    to_rgb = transforms.Compose([
        transforms.Lambda(lambda image: quantize(image, params['levels'])),
        transforms.ToTensor(),
        # transforms.Lambda(lambda image_tensor: image_tensor.repeat(3, 1, 1))
    ])

    loaders = loader.MNISTLoader(to_rgb).get(params['batch_size'], pin_memory=False)
    model_pixelcnn = PixelCNN(params['hidden_fmaps'], params['levels'], params['hidden_layers'],
                              params['causal_ksize'], params['hidden_ksize'], params['out_hidden_fmaps']).to(device)
    optimizer = optim.Adam(model_pixelcnn.parameters(), lr=params['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, params['learning_rate'],
                                            10 * params['learning_rate'], cycle_momentum=False)
    trainer = PriorTrainer(model_pixelcnn, optimizer, loaders, scheduler)
    trainer.levels = params['levels']
    trainer.batches = params['batches']
    trainer.max_norm = params['max_norm']
    trainer.run()
    samples = model_pixelcnn.sample((1, 28, 28), 16, label=None, device=device)
    save_samples(samples, './', f'samples.png')
