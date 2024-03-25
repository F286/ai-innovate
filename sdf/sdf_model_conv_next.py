import torch
import torch.nn as nn
import torch.nn.functional as F

class SDFNet(nn.Module):
    def __init__(self):
        super(SDFNet, self).__init__()

        # Constants for sizes
        INITIAL_FEATURES = 8
        MIDDLE_SIZE = 16
        FEED_FORWARD_EXPAND = 4
        KERNEL_SIZE = 3 # 7
        INITIAL_EXTENTS = 128
        DOWNSCALE_LAYERS = 5
        MIDDLE_EXTENTS = INITIAL_EXTENTS // 2**DOWNSCALE_LAYERS

        # TODO: The convnext actually does at 2x2 with a stride of 2 to downsample at the very start. can do that
        # TODO: Before any downsample, : one before each downsampling layer, one after the stem.
        # This very first downsample seems to be the 'stem'

        # Initial Convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, INITIAL_FEATURES, kernel_size=1, padding="same", padding_mode="replicate", bias=False),
            nn.LayerNorm([INITIAL_FEATURES, INITIAL_EXTENTS, INITIAL_EXTENTS])
        )

        # Contracting Path (Downscaling)
        self.down_convs = nn.ModuleList()
        for i in range(DOWNSCALE_LAYERS):
            current_extents = INITIAL_EXTENTS // 2**i
            print(current_extents)
            self.down_convs.append(
                nn.Sequential(
                    # Depthwise convolution
                    nn.Conv2d(INITIAL_FEATURES, INITIAL_FEATURES, kernel_size=KERNEL_SIZE, padding="same", groups=INITIAL_FEATURES, bias=False),
                    nn.LayerNorm([INITIAL_FEATURES, current_extents, current_extents]),
                    # Pointwise convolution to expand the channel size
                    nn.Conv2d(INITIAL_FEATURES, INITIAL_FEATURES * FEED_FORWARD_EXPAND, kernel_size=1, bias=False),
                    nn.ReLU(),
                    # Feed-forward layer to contract back to the initial number of features
                    nn.Conv2d(INITIAL_FEATURES * FEED_FORWARD_EXPAND, INITIAL_FEATURES, kernel_size=1, bias=False)
                )
            )
        self.downscale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(INITIAL_FEATURES, INITIAL_FEATURES, kernel_size=2, stride=2, padding=0, bias=False)
            ) for _ in range(DOWNSCALE_LAYERS)
        ])
            
        # Middle Part
        self.middle_convs = nn.Sequential(
                nn.Sequential(
                    # Depthwise convolution
                    nn.Conv2d(INITIAL_FEATURES, MIDDLE_SIZE, kernel_size=KERNEL_SIZE, padding="same", groups=INITIAL_FEATURES, bias=False),
                    nn.LayerNorm([MIDDLE_SIZE, MIDDLE_EXTENTS, MIDDLE_EXTENTS]),
                    # Pointwise convolution to expand the channel size
                    nn.Conv2d(MIDDLE_SIZE, MIDDLE_SIZE * FEED_FORWARD_EXPAND, kernel_size=1, bias=False),
                    nn.ReLU(),
                    # Feed-forward layer to contract back to the initial number of features
                    nn.Conv2d(MIDDLE_SIZE * FEED_FORWARD_EXPAND, INITIAL_FEATURES, kernel_size=1, bias=False)
                )
        )

        # Expanding Path (Upscaling)
        self.up_transposes = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(DOWNSCALE_LAYERS):
            self.up_transposes.append(
                nn.ConvTranspose2d(INITIAL_FEATURES, INITIAL_FEATURES, kernel_size=2, stride=2, bias=False)
            )
            current_extents = INITIAL_EXTENTS // 2**(DOWNSCALE_LAYERS - 1 - i)
            print(current_extents)
            self.up_convs.append(
                nn.Sequential(
                    # Depthwise convolution
                    nn.Conv2d(INITIAL_FEATURES * 2, INITIAL_FEATURES, kernel_size=KERNEL_SIZE, padding="same", groups=INITIAL_FEATURES, bias=False),
                    nn.LayerNorm([INITIAL_FEATURES, current_extents, current_extents]),
                    # Pointwise convolution to expand the channel size
                    nn.Conv2d(INITIAL_FEATURES, INITIAL_FEATURES * FEED_FORWARD_EXPAND, kernel_size=1, bias=False),
                    nn.ReLU(),
                    # Feed-forward layer to contract back to the initial number of features
                    nn.Conv2d(INITIAL_FEATURES * FEED_FORWARD_EXPAND, INITIAL_FEATURES, kernel_size=1, bias=False)
                )
            )

        # Final Convolution
        self.final_conv = nn.Conv2d(INITIAL_FEATURES, 1, kernel_size=1)

    def forward(self, x):
        # Initial Convolution
        x = self.initial_conv(x)

        # Contracting Path
        contracting_path_features = []
        for down_conv, downscale_conv in zip(self.down_convs, self.downscale_convs):
            x = down_conv(x)
            contracting_path_features.append(x)
            x = downscale_conv(x)

        # Middle Part
        x = self.middle_convs(x)

        # Expanding Path
        for up_transpose, up_conv, features in zip(self.up_transposes, self.up_convs, reversed(contracting_path_features)):
            x = up_transpose(x)
            # x = x + features
            x = torch.cat([x, features], dim=-3)
            x = up_conv(x)

        # Final Convolution
        out = self.final_conv(x)
        return out
