import pickle
import torch


def save_model(model, params, prefix, q, directory):
    with open(f'{directory}/{prefix}_{q}.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(f'{directory}/{prefix}_{q}_cpu.pkl', 'wb') as f:
        pickle.dump(model.cpu(), f)
    with open(f'{directory}/{prefix}_params_{q}.pkl', 'wb') as f:
        pickle.dump(params, f)
    torch.save(model.state_dict(), f'{directory}/{prefix}_{q}_state.pt')
    torch.save(model, f'{directory}/{prefix}_{q}.pt')
