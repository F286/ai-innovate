import torch
import torch.nn as nn
import torch.nn.functional as F


class SDFNet(nn.Module):
    def __init__(self):
        super(SDFNet, self).__init__()

        STORAGE_SIZE = 8
        HIDDEN_SIZE = 16
        EXPAND_SIZE = 8

        self.conv0 = nn.Conv2d(1, STORAGE_SIZE, kernel_size=1, padding=0, bias=True)

        # Sequential operation with kernel sampling from distance 64, preserving input/output size
        self.seq_sampling_64 = nn.Sequential(
            nn.Conv2d(STORAGE_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=64, bias=True),
            nn.Conv2d(HIDDEN_SIZE, EXPAND_SIZE, kernel_size=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(EXPAND_SIZE, STORAGE_SIZE, kernel_size=1, padding=0, bias=True),
        )
        # Sequential operation with kernel sampling from distance 32, preserving input/output size
        self.seq_sampling_32 = nn.Sequential(
            nn.Conv2d(STORAGE_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=32, bias=True),
            nn.Conv2d(HIDDEN_SIZE, EXPAND_SIZE, kernel_size=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(EXPAND_SIZE, STORAGE_SIZE, kernel_size=1, padding=0, bias=True),
        )

        # Sequential operation with kernel sampling from distance 16, preserving input/output size
        self.seq_sampling_16 = nn.Sequential(
            nn.Conv2d(STORAGE_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=16, bias=True),
            nn.Conv2d(HIDDEN_SIZE, EXPAND_SIZE, kernel_size=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(EXPAND_SIZE, STORAGE_SIZE, kernel_size=1, padding=0, bias=True),
        )

        # Sequential operation with kernel sampling from distance 8, preserving input/output size
        self.seq_sampling_8 = nn.Sequential(
            nn.Conv2d(STORAGE_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=8, bias=True),
            nn.Conv2d(HIDDEN_SIZE, EXPAND_SIZE, kernel_size=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(EXPAND_SIZE, STORAGE_SIZE, kernel_size=1, padding=0, bias=True),
        )

        # Sequential operation with kernel sampling from distance 4
        self.seq_sampling_4 = nn.Sequential(
            nn.Conv2d(STORAGE_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=4, bias=True),
            nn.Conv2d(HIDDEN_SIZE, EXPAND_SIZE, kernel_size=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(EXPAND_SIZE, STORAGE_SIZE, kernel_size=1, padding=0, bias=True),
        )

        # Sequential operation with kernel sampling from distance 2
        self.seq_sampling_2 = nn.Sequential(
            nn.Conv2d(STORAGE_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=2, bias=True),
            nn.Conv2d(HIDDEN_SIZE, EXPAND_SIZE, kernel_size=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(EXPAND_SIZE, STORAGE_SIZE, kernel_size=1, padding=0, bias=True),
        )

        # Original sequential operation with stride 1
        self.seq_stride_1 = nn.Sequential(
            nn.Conv2d(STORAGE_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=1, bias=True),
            nn.Conv2d(HIDDEN_SIZE, EXPAND_SIZE, kernel_size=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(EXPAND_SIZE, STORAGE_SIZE, kernel_size=1, padding=0, bias=True),
        )

        self.conv3 = nn.Conv2d(STORAGE_SIZE, 1, kernel_size=1, padding=0, bias=True)
    def forward(self, x):

        x = self.conv0(x)

        # Apply the sequential operations with different strides and take the maximum between the current value and the result for a residual-like behavior
        x = x + self.seq_sampling_64(x)
        x = x + self.seq_sampling_32(x)
        x = x + self.seq_sampling_16(x)
        x = x + self.seq_sampling_8(x)
        x = x + self.seq_sampling_4(x)
        x = x + self.seq_sampling_2(x)
        x = x + self.seq_stride_1(x)
        
        x = self.conv3(x)

        return x
    