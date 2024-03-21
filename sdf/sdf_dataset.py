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
        
        assert edge_voxels.shape == (1, 128, 128), f"Edge voxels shape must be [1, 128, 128], got {edge_voxels.shape}"
        assert target.shape == (1, 128, 128), f"Target shape must be [1, 128, 128], got {target.shape}"

        return edge_voxels, target

