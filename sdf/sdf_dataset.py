import numpy as np
from torch.utils.data import Dataset
import torch
from .sdf_object import SDFObject  # Adjust the import path as needed

class SDFDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        sdf_object = SDFObject.load(self.file_paths[idx])

        edge_voxels = sdf_object.get_edge_voxels_tensor()
        target = sdf_object.get_target_tensor()
        
        assert len(edge_voxels.shape) == 4 and edge_voxels.shape[0] == 1, f"Edge voxels must be rank 4 with shape [1, X, Y, Z], got {edge_voxels.shape}"
        assert len(target.shape) == 4 and target.shape[0] == 1, f"Target must be rank 4 with shape [1, X, Y, Z], got {target.shape}"

        return edge_voxels, target

