import torch
import torch.nn as nn
import torch.nn.functional as F

class SDFNet(nn.Module):
    def __init__(self):
        super(SDFNet, self).__init__()

        # Constants for sizes
        LAYER_FEATURES = [16, 32, 64]  # This array directly specifies the number of features per layer
        # LAYER_FEATURES = [8, 16, 32, 64, 128, 256]  # This array directly specifies the number of features per layer
        FEED_FORWARD_EXPAND = 4
        KERNEL_SIZE = 3 # 7
        INPUT_EXTENTS = 128
        INITIAL_EXTENTS = 32
        DOWNSCALE_LAYERS = len(LAYER_FEATURES) - 1  # The size of LAYER_FEATURES array sets DOWNSCALE_LAYERS
        MIDDLE_EXTENTS = INITIAL_EXTENTS // 2**DOWNSCALE_LAYERS

        # Initial Convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, LAYER_FEATURES[0], kernel_size=1, padding="same", padding_mode="replicate", bias=False),
            nn.LayerNorm([LAYER_FEATURES[0], INPUT_EXTENTS, INPUT_EXTENTS])
        )

        # Contracting Path (Downscaling)
        self.down_convs = nn.ModuleList()
        self.downscale_convs = nn.ModuleList()

        # Stem (initial downscale) 
        # TODO Add stem for convnext, initial downscale. append it manually, and use a 4x4 downscale.

        # Stem (initial downscale) 
        # Implementing the stem for an initial downscale using a 4x4 stride to effectively reduce the spatial dimensions
        self.down_convs.append(nn.Sequential())
        self.downscale_convs.append(
            nn.Sequential(
                nn.Conv2d(LAYER_FEATURES[0], LAYER_FEATURES[0], kernel_size=4, stride=4, padding=0, bias=False)
            ))

        for i in range(DOWNSCALE_LAYERS):
            current_extents = INITIAL_EXTENTS // 2**i
            features_in = LAYER_FEATURES[i]
            features_out = LAYER_FEATURES[i + 1]
            print(f"current_extents {features_in}  features_out {features_out}")
            self.down_convs.append(
                nn.Sequential(
                    # Depthwise convolution
                    nn.Conv2d(features_in, features_in, kernel_size=KERNEL_SIZE, padding="same", groups=features_in, bias=False),
                    nn.LayerNorm([features_in, current_extents, current_extents]),
                    # Pointwise convolution to expand the channel size
                    nn.Conv2d(features_in, features_in * FEED_FORWARD_EXPAND, kernel_size=1, bias=False),
                    nn.GELU(),
                    # Feed-forward layer to contract back to the initial number of features
                    nn.Conv2d(features_in * FEED_FORWARD_EXPAND, features_out, kernel_size=1, bias=False)
                )
            )
            self.downscale_convs.append(
                nn.Sequential(
                    nn.Conv2d(features_out, features_out, kernel_size=2, stride=2, padding=0, bias=False)
                ))

        # Middle Part
        self.middle_convs = nn.Sequential(
            nn.Sequential(
                # Depthwise convolution
                nn.Conv2d(LAYER_FEATURES[-1], LAYER_FEATURES[-1], kernel_size=KERNEL_SIZE, padding="same", groups=LAYER_FEATURES[-1], bias=False),
                nn.LayerNorm([LAYER_FEATURES[-1], MIDDLE_EXTENTS, MIDDLE_EXTENTS]),
                # Pointwise convolution to expand the channel size
                nn.Conv2d(LAYER_FEATURES[-1], LAYER_FEATURES[-1] * FEED_FORWARD_EXPAND, kernel_size=1, bias=False),
                nn.GELU(),
                # Feed-forward layer to contract back to the initial number of features
                nn.Conv2d(LAYER_FEATURES[-1] * FEED_FORWARD_EXPAND, LAYER_FEATURES[-1], kernel_size=1, bias=False)
            )
        )

        # Expanding Path (Upscaling)
        self.up_transposes = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for i in range(DOWNSCALE_LAYERS):
            index = DOWNSCALE_LAYERS - 1 - i
            features_in = LAYER_FEATURES[index + 1]
            features_out = LAYER_FEATURES[index]
            print(f"current_extents {features_in}  features_out {features_out}")
            
            self.up_transposes.append(
                nn.ConvTranspose2d(features_in, features_in, kernel_size=2, stride=2, bias=False)
            )
            current_extents = INITIAL_EXTENTS // 2**index
            print(current_extents)
            self.up_convs.append(
                nn.Sequential(
                    # Depthwise convolution
                    nn.Conv2d(features_in * 2, features_out, kernel_size=KERNEL_SIZE, padding="same", groups=features_out, bias=False),
                    nn.LayerNorm([features_out, current_extents, current_extents]),
                    # Pointwise convolution to expand the channel size
                    nn.Conv2d(features_out, features_out * FEED_FORWARD_EXPAND, kernel_size=1, bias=False),
                    nn.GELU(),
                    # Feed-forward layer to contract back to the initial number of features
                    nn.Conv2d(features_out * FEED_FORWARD_EXPAND, features_out, kernel_size=1, bias=False)
                )
            )

        # Final Convolution
        self.final_conv = nn.Sequential(
                # stem upscale
                nn.ConvTranspose2d(LAYER_FEATURES[0], LAYER_FEATURES[0], kernel_size=4, stride=4, bias=False),
                nn.Conv2d(LAYER_FEATURES[0], 1, kernel_size=1)
            )

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
