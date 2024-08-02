import torch
import torch.nn as nn
import torch.nn.functional as F

class MinConv2d(nn.Module):
    """
    A PyTorch module for performing 'min convolution' operations.
    This module mimics the interface of nn.Conv2d but replaces the summing operation
    with a minimum operation over the kernel-sized neighborhoods.
    
    Attributes:
        kernel_size (int): The size of the window over which to take the minimum.
        stride (int): The stride of the convolution. Defaults to 1.
        padding (int): Zero-padding added to both sides of the input. Defaults to 1.
    """
    def __init__(self, kernel_size, stride=1, padding=1):
        """
        Initializes the MinConv2d module with the specified kernel size, stride, and padding.
        
        Parameters:
            kernel_size (int): The size of the kernel/window.
            stride (int): The stride with which the window moves across the input tensor.
            padding (int): The amount of zero-padding added to the input tensor.
        """
        super(MinConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the MinConv2d module.
        
        Parameters:
            x (Tensor): The input tensor of shape (batch_size, channels, height, width) or (channels, height, width).
        
        Returns:
            Tensor: The output tensor after applying the min convolution operation.
        """
        # Check if the input tensor has three dimensions and unsqueeze it to add a batch dimension if necessary
        hasExtendedBatch: bool = False
        if x.dim() == 3:
            hasExtendedBatch = True
            x = x.unsqueeze(0)  # Adds a batch dimension with a size of 1 at the beginning
        
        # Unfolding the input tensor to extract sliding windows
        unfolded = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        
        # Reshaping to separate out each window. Each window is flattened out in the last dimension.
        unfolded = unfolded.view(x.shape[0], x.shape[1], self.kernel_size**2, -1)
        
        # Taking the minimum across the window dimensions. This performs the min operation across the kernel.
        min_values, _ = unfolded.min(dim=2)
        
        # Calculating the output dimensions based on the input dimensions, kernel size, stride, and padding
        output_height = (x.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (x.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Reshaping the output tensor to the expected output shape
        output = min_values.view(x.shape[0], x.shape[1], output_height, output_width)
        
        # If the original input tensor had three dimensions, remove the added batch dimension before returning
        if hasExtendedBatch:
            output = output.squeeze(0)  # Removes the batch dimension
        
        return output
