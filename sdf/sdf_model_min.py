import torch
import torch.nn as nn
import torch.nn.functional as F

from sdf.min_conv_2d import MinConv2d

class SDFNet(nn.Module):
    def __init__(self):
        super(SDFNet, self).__init__()

        STORAGE_SIZE = 8
        HIDDEN_SIZE = 8

        # The initial convolution layer remains unchanged.
        self.initial_conv = nn.Conv2d(1, STORAGE_SIZE, kernel_size=1, padding=0, bias=True)

        # Dynamically create 16 separate instances of seq_sampling.
        self.seq_samplings = nn.ModuleList([nn.Sequential(
            MinConv2d(kernel_size=3)
        ) for _ in range(16)])

        # The final convolution layers remain unchanged.
        self.final_conv = nn.Sequential(
            nn.Conv2d(STORAGE_SIZE, 32, kernel_size=3, padding="same", dilation=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, x):
        x = self.initial_conv(x)

        # Iteratively apply each of the 16 seq_sampling instances to the input.
        for seq_sampling in self.seq_samplings:
            x = x + seq_sampling(x)
        
        x = self.final_conv(x)
        return x
