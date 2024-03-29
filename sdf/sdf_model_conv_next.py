import torch
import torch.nn as nn
import torch.nn.functional as F

class WideSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(WideSeparableConv, self).__init__()
        self.horizontal_conv = nn.Conv2d(in_channels, out_channels,
                                         kernel_size=(1, kernel_size), padding=(0, kernel_size // 2), bias=False)
        self.vertical_conv = nn.Conv2d(out_channels, out_channels,
                                       kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0), bias=False)
        nn.init.kaiming_normal_(self.horizontal_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.vertical_conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.horizontal_conv(x))
        x = F.relu(self.vertical_conv(x))
        return x

class SDFNet(nn.Module):
    def __init__(self):
        super(SDFNet, self).__init__()
        LAYER_FEATURES = [128, 128, 128]
        KERNEL_SIZE = 64

        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, LAYER_FEATURES[0], kernel_size=1, padding="same", bias=False),
            nn.ReLU()
        )

        # Replace down_convs and up_convs with WideSeparableConv
        self.convs = nn.ModuleList([WideSeparableConv(LAYER_FEATURES[i], LAYER_FEATURES[i+1], KERNEL_SIZE) 
                                    for i in range(len(LAYER_FEATURES)-1)])

        self.middle_convs = nn.Sequential(
            WideSeparableConv(LAYER_FEATURES[-1], LAYER_FEATURES[-1], KERNEL_SIZE),
            WideSeparableConv(LAYER_FEATURES[-1], LAYER_FEATURES[-1], KERNEL_SIZE)
        )

        self.final_conv = nn.Conv2d(LAYER_FEATURES[0], 1, kernel_size=1)

    def forward(self, x):
        x = self.initial_conv(x)

        for conv in self.convs:
            x = conv(x)

        x = self.middle_convs(x)

        out = self.final_conv(x)
        return out