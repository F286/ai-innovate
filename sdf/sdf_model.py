
import torch
import torch.nn as nn
import torch.nn.functional as F

class SDFNet(nn.Module):
    def __init__(self):
        super(SDFNet, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.unsqueeze(1)  # Adding channel dimension
        for _ in range(5):  # Sequential convolutions with the same weights
            x = self.relu(self.conv(x))
        return x.squeeze(1)  # Removing channel dimension for output
