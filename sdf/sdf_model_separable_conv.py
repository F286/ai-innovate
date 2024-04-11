import torch
import torch.nn as nn
import torch.nn.functional as F

class SDFNet(nn.Module):
    def __init__(self):
        super(SDFNet, self).__init__()

        # Constants for sizes, adapted for 3D
        LAYER_FEATURES = [4, 8, 16]  # Number of features per layer
        FEED_FORWARD_EXPAND = 2
        KERNEL_SIZE = (3, 3, 3)  # Now a 3-tuple for 3D
        DOWNSCALE_LAYERS = len(LAYER_FEATURES) - 1
        DOWNSCALE_RATE = 4

        # Initial Convolution, adapted for 3D
        INITIAL_LAYER_FEATURES = 4
        self.initial_conv = nn.Sequential(
            nn.Conv3d(1, INITIAL_LAYER_FEATURES, kernel_size=1, padding="same", bias=False),
            # nn.BatchNorm3d(INITIAL_LAYER_FEATURES)
        )

        # Contracting Path (Downscaling), adapted for 3D
        self.down_convs = nn.ModuleList()
        self.downscale_convs = nn.ModuleList()

        # Initial downscale
        self.down_convs.append(nn.Sequential())
        self.downscale_convs.append(
            nn.Sequential(
                nn.Conv3d(INITIAL_LAYER_FEATURES, LAYER_FEATURES[0], kernel_size=DOWNSCALE_RATE, stride=DOWNSCALE_RATE, padding=0, bias=False)
            ))

        for i in range(DOWNSCALE_LAYERS):
            features_in = LAYER_FEATURES[i]
            features_out = LAYER_FEATURES[i + 1]
            self.down_convs.append(
                nn.Sequential(
                    # Depthwise convolution adapted for 3D
                    nn.Conv3d(features_in, features_in, kernel_size=KERNEL_SIZE, padding="same", groups=features_in, bias=False),
                    # nn.BatchNorm3d(features_in),
                    # Pointwise convolution to expand the channel size
                    nn.Conv3d(features_in, features_in * FEED_FORWARD_EXPAND, kernel_size=1, bias=False),
                    nn.GELU(),
                    # Contracting back to the initial number of features
                    nn.Conv3d(features_in * FEED_FORWARD_EXPAND, features_out, kernel_size=1, bias=False),
                )
            )
            self.downscale_convs.append(
                nn.Sequential(
                    nn.Conv3d(features_out, features_out, kernel_size=DOWNSCALE_RATE, stride=DOWNSCALE_RATE, padding=0, bias=False)
                ))

        # Middle Part, adapted for 3D
        self.middle_convs = nn.Sequential(
            nn.Conv3d(LAYER_FEATURES[-1], LAYER_FEATURES[-1], kernel_size=KERNEL_SIZE, padding="same", groups=LAYER_FEATURES[-1], bias=False),
            nn.BatchNorm3d(LAYER_FEATURES[-1]),
            nn.Conv3d(LAYER_FEATURES[-1], LAYER_FEATURES[-1] * FEED_FORWARD_EXPAND, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv3d(LAYER_FEATURES[-1] * FEED_FORWARD_EXPAND, LAYER_FEATURES[-1], kernel_size=1, bias=False),
        )

        # Expanding Path (Upscaling), adapted for 3D
        self.up_transposes = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for i in range(DOWNSCALE_LAYERS):
            index = DOWNSCALE_LAYERS - 1 - i
            features_in = LAYER_FEATURES[index + 1]
            features_out = LAYER_FEATURES[index]
            self.up_transposes.append(
                nn.ConvTranspose3d(features_in, features_in, kernel_size=DOWNSCALE_RATE, stride=DOWNSCALE_RATE, bias=False)
            )
            self.up_convs.append(
                nn.Sequential(
                    # Depthwise convolution adapted for 3D
                    nn.Conv3d(features_in * 2, features_out, kernel_size=KERNEL_SIZE, padding="same", groups=features_out, bias=False),
                    # nn.BatchNorm3d(features_out),
                    # Pointwise convolution to expand the channel size
                    nn.Conv3d(features_out, features_out * FEED_FORWARD_EXPAND, kernel_size=1, bias=False),
                    nn.GELU(),
                    # Contracting back to the initial number of features
                    nn.Conv3d(features_out * FEED_FORWARD_EXPAND, features_out, kernel_size=1, bias=False),
                )
            )

        # Final Convolution, adapted for 3D
        self.final_conv = nn.Sequential(
            nn.ConvTranspose3d(LAYER_FEATURES[0], LAYER_FEATURES[0], kernel_size=DOWNSCALE_RATE, stride=DOWNSCALE_RATE, bias=False),
            # nn.BatchNorm3d(LAYER_FEATURES[0]),
            nn.Conv3d(LAYER_FEATURES[0], LAYER_FEATURES[0] * FEED_FORWARD_EXPAND, kernel_size=1),  # Expands the channel dimension
            nn.ReLU(),  # Activation function for non-linearity
            nn.Conv3d(LAYER_FEATURES[0] * FEED_FORWARD_EXPAND, LAYER_FEATURES[0], kernel_size=1),  # Contracts the channel dimension back
            nn.Conv3d(LAYER_FEATURES[0], 1, kernel_size=1),
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
            x = torch.cat([x, features], dim=1)  # Concatenate along the channel dimension
            x = up_conv(x)

        # Final Convolution
        out = self.final_conv(x)
        return out