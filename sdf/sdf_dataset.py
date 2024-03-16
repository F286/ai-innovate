import numpy as np
from torch.utils.data import Dataset
import torch
from sdf_object import SDFObject  # Adjust the import path as needed

class SDFDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        sdf_object = SDFObject.load(self.file_paths[idx])
        # Ensure the input has a single channel dimension
        edge_voxels = np.expand_dims((sdf_object.sdf_data == 0).astype(np.float32), axis=0)
        target = sdf_object.sdf_data.astype(np.float32)
        return torch.tensor(edge_voxels), torch.tensor(target)