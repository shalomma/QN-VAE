import pickle
import torch


def save_model(model, params, q, directory):
    with open(f'{directory}/model_{q}.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(f'{directory}/model_{q}_cpu.pkl', 'wb') as f:
        pickle.dump(model.cpu(), f)
    with open(f'{directory}/params_{q}.pkl', 'wb') as f:
        pickle.dump(params, f)
    torch.save(model.state_dict(), f'{directory}/model_{q}_state.pt')
    torch.save(model, f'{directory}/model_{q}.pt')
