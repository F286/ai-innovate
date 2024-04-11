import torch
import torch.nn as nn
import torch.nn.functional as F

class SDFNet(nn.Module):
    def __init__(self):
        super(SDFNet, self).__init__()

        # Define the number of features per layer
        LAYER_FEATURES = [8, 8, 16, 16, 1]  # Increasing features and reducing to 1 output channel
        KERNEL_SIZES = [3, 3, 3, 3, 3]  # Uniform kernel size to simplify padding calculation

        # Initial convolution layer with padding to preserve dimensions
        self.initial_conv = nn.Conv3d(
            in_channels=1, 
            out_channels=LAYER_FEATURES[0], 
            kernel_size=KERNEL_SIZES[0], 
            stride=1, 
            padding=KERNEL_SIZES[0] // 2  # Padding to maintain dimensions
        )
        
        # Additional convolution layers with padding to preserve dimensions
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(
                    in_channels=LAYER_FEATURES[i], 
                    out_channels=LAYER_FEATURES[i+1], 
                    kernel_size=KERNEL_SIZES[i+1], 
                    stride=1, 
                    padding=KERNEL_SIZES[i+1] // 2  # Padding to maintain dimensions
                ),
                nn.BatchNorm3d(LAYER_FEATURES[i+1]),
                nn.ReLU()
            ) for i in range(len(LAYER_FEATURES) - 2)
        ])

        # Final layer to reduce the feature maps to 1 channel, with padding to preserve dimensions
        self.final_conv = nn.Conv3d(
            in_channels=LAYER_FEATURES[-2], 
            out_channels=LAYER_FEATURES[-1], 
            kernel_size=KERNEL_SIZES[-1], 
            stride=1,
            padding=KERNEL_SIZES[-1] // 2
        )

    def forward(self, x):
        x = self.initial_conv(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.final_conv(x)
        return x