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


def load_model(model_class, prefix, q, directory):
    with open(f'{directory}/{prefix}_params_{q}.pkl', 'rb') as f:
        params = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(params['num_hidden'], params['num_residual_layers'], params['num_residual_hidden'],
                        params['num_embeddings'], params['embedding_dim'], params['commitment_cost'],
                        quant_noise=q).to(device)
    model.load_state_dict(torch.load(f'{directory}/{prefix}_{q}_state.pt'))
    model.eval()
    return model
