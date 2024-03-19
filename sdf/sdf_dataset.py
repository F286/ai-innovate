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

        # Ensure the input has a single channel dimension
        # edge_voxels = np.expand_dims((sdf_object.sdf_data == 0).astype(np.float32), axis=0)
        # target = sdf_object.sdf_data.astype(np.float32)

        # Assert to ensure the size of edge voxels and target is [1, 1, 100, 100]
        # assert edge_voxels.shape == (1, 100, 100), f"Edge voxels shape must be [1, 100, 100], got {edge_voxels.shape}"
        # assert target.shape == (100, 100), f"Target shape must be [100, 100], got {target.shape}"

        # # Add a batch dimension to match the expected size [1, 1, 100, 100]
        # edge_voxels = np.expand_dims(edge_voxels, axis=0)
        # target = np.expand_dims(target, axis=0)

        # edge_voxels = edge_voxels.unsqueeze(1)
        # target = target.unsqueeze(1)
        # target = target.unsqueeze(1)
        
        assert edge_voxels.shape == (1, 100, 100), f"Edge voxels shape must be [1, 100, 100], got {edge_voxels.shape}"
        assert target.shape == (1, 100, 100), f"Target shape must be [1, 100, 100], got {target.shape}"

        return edge_voxels, target
        # return torch.tensor(edge_voxels), torch.tensor(target)

