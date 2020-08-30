import os
import pickle
import torch
import inspect
from torchvision.utils import save_image


def save_model(model, params, prefix, q, directory):
    with open(f'{directory}/{prefix}_params_{q}.pkl', 'wb') as f:
        pickle.dump(params, f)
    torch.save(model.state_dict(), f'{directory}/{prefix}_{q}.pt')


def load_model(model_class, prefix, q, directory):
    with open(f'{directory}/{prefix}_params_{q}.pkl', 'rb') as f:
        params = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = list(inspect.signature(model_class.__init__).parameters.keys())
    params_class = dict()
    for a in args[1:]:
        params_class[a] = params[a]
    model = model_class(**params_class).to(device)
    model.load_state_dict(torch.load(f'{directory}/{prefix}_{q}.pt'))
    model.eval()
    return model, params


def save_samples(samples, dirname, filename):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    count = samples.size()[0]
    count_sqrt = int(count ** 0.5)
    n_row = count_sqrt if count_sqrt ** 2 == count else count
    save_image(samples, os.path.join(dirname, filename), nrow=n_row)
