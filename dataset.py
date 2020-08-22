import os
import torch
import torch.utils.data as data


class Encoded(data.Dataset):
    def __init__(self, root_dir, q, indices):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = torch.load(os.path.join(root_dir, f'encoded_{q}.pt'), map_location=device)
        self.y = torch.load(os.path.join(root_dir, f'encoded_labels_{q}.pt'), map_location=device)
        self.data = self.data[indices].to(device)
        self.y = self.y[indices].to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.y[idx]
