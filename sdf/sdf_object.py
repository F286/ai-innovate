import numpy as np
import matplotlib.pyplot as plt
import torch

class SDFObject:
    def __init__(self, sdf_data):
        self.sdf_data = sdf_data

    @staticmethod
    def load(filename):
        sdf_data = np.load(filename)
        return SDFObject(sdf_data)

    def get_edge_voxels(self) -> 'SDFObject':
        """
        This method identifies the edge voxels within the SDF data, defined as voxels with a value of 0.
        It returns a new SDFObject containing only these edge voxels.

        Returns:
            SDFObject: An instance of SDFObject containing the edge voxels.
        """
        # Assuming 'edge' voxels are those with a value of 0
        # np.expand_dims adds an axis, making the array compatible for operations that expect an additional dimension.
        # (self.sdf_data == 0).astype(np.float32) creates a binary mask of the edge voxels, converting True/False to 1.0/0.0.
        edge_voxels = np.expand_dims((self.sdf_data < 0.1).astype(np.float32), axis=0)

        # Instead of returning a tensor, we return a new SDFObject containing the edge voxels.
        return SDFObject(edge_voxels)

    def get_edge_voxels_tensor(self) -> 'torch.Tensor':
        """
        This method is a variant of get_edge_voxels that returns the edge voxels as a PyTorch tensor.

        Returns:
            torch.Tensor: A tensor containing the edge voxels.
        """
        # Call the original get_edge_voxels method to get the SDFObject with edge voxels
        edge_voxels_sdf = self.get_edge_voxels()

        # Convert the numpy array inside the SDFObject to a PyTorch tensor
        edge_voxels_tensor = torch.from_numpy(edge_voxels_sdf.sdf_data)

        # Print the shape of the tensor to understand its structure.
        print(f"Shape of edge_voxels_tensor: {edge_voxels_tensor.shape}")

        return edge_voxels_tensor

    def get_target(self) -> 'SDFObject':
        """
        This method processes the SDF data to be used as a target for training or inference.
        It takes the absolute value of the SDF data, preparing it as a target dataset.
        The method returns a new SDFObject containing this processed data.

        Returns:
            SDFObject: An instance of SDFObject containing the processed target data.
        """
        # np.abs computes the absolute value of each element in the sdf_data.
        # .astype(np.float32) ensures the data is in float32 format, suitable for most deep learning operations.
        target = np.abs(self.sdf_data).astype(np.float32)
        # Instead of returning a tensor, we return a new SDFObject containing the target data.
        return SDFObject(target)

    def get_target_tensor(self) -> 'torch.Tensor':
        """
        This method is a variant of get_target that returns the processed SDF data as a PyTorch tensor.

        Returns:
            torch.Tensor: A tensor containing the processed target data.
        """
        # Call the original get_target method to get the SDFObject with processed target data
        target_sdf = self.get_target()
        # Convert the numpy array inside the SDFObject to a PyTorch tensor
        target_tensor = torch.from_numpy(target_sdf.sdf_data)
        return target_tensor