import numpy as np
from torch.utils.data import Dataset
import torch
from sdf_object import SDFObject


class SDFDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        sdf_object = SDFObject.load(self.file_paths[idx])
        edge_voxels = np.expand_dims((sdf_object.sdf_data == 0).astype(np.float32), axis=0)
        target = sdf_object.sdf_data.astype(np.float32)

        # Ensure edge_voxels tensor is correctly shaped as [1, height, width]
        edge_voxels = edge_voxels.reshape(1, edge_voxels.shape[1], edge_voxels.shape[2])

        # Assert to ensure the edge_voxels has the expected single channel
        assert edge_voxels.shape[0] == 1, f"Input should have 1 channel, but got {edge_voxels.shape[0]}"

        return torch.tensor(edge_voxels), torch.tensor(target)