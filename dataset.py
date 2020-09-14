import os
import torch
import torch.utils.data as data


class Encoded(data.Dataset):
    def __init__(self, root_dir, q, indices, transform=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = torch.load(os.path.join(root_dir, f'encoded_data_{q}.pt'), map_location=device)
        self.data = self.data.view(-1, 1, 8, 8).cpu().numpy()
        self.y = torch.load(os.path.join(root_dir, f'encoded_labels_{q}.pt'), map_location=device)
        self.y = self.y[indices].to(device).long()
        self.data = self.data[indices]
        if transform is not None:
            self.data = transform(self.data)
        self.data = torch.tensor(self.data).to(device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.y[idx]
