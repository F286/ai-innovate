
import torch
import torch.nn as nn
import torch.nn.functional as F

class SDFNet(nn.Module):
    def __init__(self):
        super(SDFNet, self).__init__()
        self.conv0 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.unsqueeze(1)  # Adding channel dimension

        x = self.relu(self.conv0(x))

        for _ in range(16):  # Sequential convolutions with the same weights
            x = self.relu(self.conv1(x))

        x = self.relu(self.conv2(x))

        return x.squeeze(1)  # Removing channel dimension for output
