import pickle
import torch


def save_model(model, params, q):
    with open(f'model_{q}.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(f'model_{q}_cpu.pkl', 'wb') as f:
        pickle.dump(model.cpu(), f)
    with open(f'params_{q}.pkl', 'wb') as f:
        pickle.dump(params, f)
    torch.save(model.state_dict(), f'model_{q}_state.pt')
    torch.save(model, f'model_{q}.pt')
