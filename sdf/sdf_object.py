import numpy as np
import matplotlib.pyplot as plt
import torch

class SDFObject:
    def __init__(self, sdf_data, name: str):
        self.sdf_data = sdf_data
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def load(filename):
        sdf_data = np.load(filename)
        return SDFObject(sdf_data, filename)

    def get_edge_voxels(self) -> 'SDFObject':
        """
        This method identifies the edge voxels within the SDF data, defined as voxels with a value of 0.
        It returns a new SDFObject containing only these edge voxels.

        Returns:
            SDFObject: An instance of SDFObject containing the edge voxels.
        """
        # To identify edge voxels, we're looking for voxels with values less than 0.1 and greater than -1.1.
        # This range captures the desired edge definition for our application.
        # np.logical_and combines two conditions, checking for values both less than 0.1 and greater than -1.1.
        # The result is a boolean mask where True represents voxels within our defined edge criteria.
        # (self.sdf_data < 0.1) & (self.sdf_data > -1.1) creates this boolean mask based on our conditions.
        # .astype(np.float32) converts the boolean mask to a float mask, turning True/False into 1.0/0.0, making it compatible for further processing.
        # np.expand_dims adds an axis, making the array compatible for operations that expect an additional dimension, such as certain neural network inputs.

        # edge_voxels = np.expand_dims(((self.sdf_data < 0.1) & (self.sdf_data > -1.1)).astype(np.float32), axis=0)
        edge_voxels = np.expand_dims((self.sdf_data < 0.1).astype(np.float32), axis=0)

        # Instead of returning a tensor, we return a new SDFObject containing the edge voxels.
        return SDFObject(edge_voxels, "Edge Voxels")

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
        
        # edge_voxels_tensor = edge_voxels_tensor.unsqueeze(0)

        return edge_voxels_tensor

    def get_target(self) -> 'SDFObject':
        """
        This method processes the SDF data to be used as a target for training or inference.
        It takes the absolute value of the SDF data, preparing it as a target dataset.
        The method returns a new SDFObject containing this processed data.

        Returns:
            SDFObject: An instance of SDFObject containing the processed target data.
        """
        # target = np.expand_dims(np.clip((256 - np.abs(self.sdf_data)) / 256, 0, 1).astype(np.float32), axis=0)
        target = np.expand_dims(np.clip((self.sdf_data) / 256, -1, 1).astype(np.float32), axis=0)

        # Instead of returning a tensor, we return a new SDFObject containing the target data.
        return SDFObject(target, "Target")

    def get_target_tensor(self) -> 'torch.Tensor':
        """
        This method is a variant of get_target that returns the processed SDF data as a PyTorch tensor.

        Returns:
            torch.Tensor: A tensor containing the processed target data.
        """
        target_sdf = self.get_target()
        target_tensor = torch.from_numpy(target_sdf.sdf_data)
        return target_tensor