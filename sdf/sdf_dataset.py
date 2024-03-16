
from torch.utils.data import Dataset
import torch
import numpy as np

class SDFDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        data_tensor = torch.from_numpy(data).float()
        return data_tensor
