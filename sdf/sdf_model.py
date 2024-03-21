import torch
import torch.nn as nn
import torch.nn.functional as F

class SDFNet(nn.Module):
    def __init__(self):
        super(SDFNet, self).__init__()

        HIDDEN_SIZE = 4  # Constant variable for hidden size

        self.conv0 = nn.Conv2d(1, HIDDEN_SIZE, kernel_size=1, padding=0, bias=False)

        # Sequential operation with kernel sampling from distance 32, preserving input/output size
        self.seq_sampling_64 = nn.Sequential(
            nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=64, bias=False),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=1, padding=0, bias=False)
        )
        # Sequential operation with kernel sampling from distance 32, preserving input/output size
        self.seq_sampling_32 = nn.Sequential(
            nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=32, bias=False),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=1, padding=0, bias=False)
        )

        # Sequential operation with kernel sampling from distance 16, preserving input/output size
        self.seq_sampling_16 = nn.Sequential(
            nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=16, bias=False),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=1, padding=0, bias=False)
        )

        # Sequential operation with kernel sampling from distance 8, preserving input/output size
        self.seq_sampling_8 = nn.Sequential(
            nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=8, bias=False),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=1, padding=0, bias=False)
        )

        # Sequential operation with kernel sampling from distance 4
        self.seq_sampling_4 = nn.Sequential(
            nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=4, bias=False),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=1, padding=0, bias=False)
        )

        # Sequential operation with kernel sampling from distance 2
        self.seq_sampling_2 = nn.Sequential(
            nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=1, padding=0, bias=False)
        )

        # Original sequential operation with stride 1
        self.seq_stride_1 = nn.Sequential(
            nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=1, padding=0, bias=False)
        )

        self.conv3 = nn.Conv2d(HIDDEN_SIZE, 1, kernel_size=1, padding=0, bias=False)
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
        
        # x = torch.max(x, self.seq_sampling_64(x))
        # x = torch.max(x, self.seq_sampling_32(x))
        # x = torch.max(x, self.seq_sampling_16(x))
        # x = torch.max(x, self.seq_sampling_8(x))
        # x = torch.max(x, self.seq_sampling_4(x))
        # x = torch.max(x, self.seq_sampling_2(x))
        # x = torch.max(x, self.seq_stride_1(x))

        x = self.conv3(x)

        return x
    
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SDFNet(nn.Module):
#     def __init__(self):
#         super(SDFNet, self).__init__()

#         HIDDEN_SIZE = 4  # Constant variable for hidden size

#         self.conv0 = nn.Conv2d(1, HIDDEN_SIZE, kernel_size=1, padding=0, bias=False)

#         # Sequential operation with kernel sampling from distance 32, preserving input/output size
#         self.seq_sampling_64 = nn.Sequential(
#             nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=64, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=1, padding=0, bias=True)
#         )
#         # Sequential operation with kernel sampling from distance 32, preserving input/output size
#         self.seq_sampling_32 = nn.Sequential(
#             nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=32, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=1, padding=0, bias=True)
#         )

#         # Sequential operation with kernel sampling from distance 16, preserving input/output size
#         self.seq_sampling_16 = nn.Sequential(
#             nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=16, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=1, padding=0, bias=True)
#         )

#         # Sequential operation with kernel sampling from distance 8, preserving input/output size
#         self.seq_sampling_8 = nn.Sequential(
#             nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=8, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=1, padding=0, bias=True)
#         )

#         # Sequential operation with kernel sampling from distance 4
#         self.seq_sampling_4 = nn.Sequential(
#             nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=4, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=1, padding=0, bias=True)
#         )

#         # Sequential operation with kernel sampling from distance 2
#         self.seq_sampling_2 = nn.Sequential(
#             nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=2, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=1, padding=0, bias=True)
#         )

#         # Original sequential operation with stride 1
#         self.seq_stride_1 = nn.Sequential(
#             nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=1, bias=False),
#             nn.ReLU(),
#             # nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding="same", dilation=1, bias=False)
#         )

#         self.conv3 = nn.Conv2d(HIDDEN_SIZE, 1, kernel_size=1, padding=0, bias=True)
#     def forward(self, x):

#         initial_shape = x.shape

#         x = self.conv0(x)

#         # Apply the sequential operations with different strides and add the result to the input for a residual-like behavior
#         x = x + self.seq_sampling_64(x)
#         x = x + self.seq_sampling_32(x)
#         x = x + self.seq_sampling_16(x)
#         x = x + self.seq_sampling_8(x)
#         x = x + self.seq_sampling_4(x)
#         x = x + self.seq_sampling_2(x)
#         x = x + self.seq_stride_1(x)
        
#         x = self.conv3(x)

#         return x