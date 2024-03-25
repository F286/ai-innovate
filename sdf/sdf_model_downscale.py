import torch
import torch.nn as nn
import torch.nn.functional as F

class SDFNet(nn.Module):
    def __init__(self):
        super(SDFNet, self).__init__()

        # Constants for sizes
        INITIAL_FEATURES = 16
        MIDDLE_SIZE = 32

        # Initial Convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, INITIAL_FEATURES, kernel_size=3, padding="same", padding_mode="replicate", bias=False),
            nn.BatchNorm2d(INITIAL_FEATURES),
            nn.ReLU()
        )

        # Contracting Path (Downscaling)
        self.down_convs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for _ in range(5):
            self.down_convs.append(
                nn.Sequential(
                    nn.Conv2d(INITIAL_FEATURES, INITIAL_FEATURES, kernel_size=3, padding="same", padding_mode="replicate", bias=False),
                    nn.BatchNorm2d(INITIAL_FEATURES),
                    nn.ReLU(),
                    nn.Conv2d(INITIAL_FEATURES, INITIAL_FEATURES, kernel_size=3, padding="same", padding_mode="replicate", bias=False),
                    nn.BatchNorm2d(INITIAL_FEATURES),
                    nn.ReLU()
                )
            )
            
        # Middle Part
        self.middle_convs = nn.Sequential(
            nn.Conv2d(INITIAL_FEATURES, MIDDLE_SIZE, kernel_size=3, padding="same", padding_mode="replicate", bias=False),
            nn.BatchNorm2d(MIDDLE_SIZE),
            nn.ReLU(),
            nn.Conv2d(MIDDLE_SIZE, MIDDLE_SIZE, kernel_size=3, padding="same", padding_mode="replicate", bias=False),
            nn.BatchNorm2d(MIDDLE_SIZE),
            nn.ReLU(),
            nn.Conv2d(MIDDLE_SIZE, MIDDLE_SIZE, kernel_size=3, padding="same", padding_mode="replicate", bias=False),
            nn.BatchNorm2d(MIDDLE_SIZE),
            nn.ReLU(),
            nn.Conv2d(MIDDLE_SIZE, INITIAL_FEATURES, kernel_size=3, padding="same", padding_mode="replicate", bias=False),
            nn.BatchNorm2d(INITIAL_FEATURES),
            nn.ReLU()
        )

        # Expanding Path (Upscaling)
        self.up_transposes = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for _ in range(5):
            self.up_transposes.append(
                nn.ConvTranspose2d(INITIAL_FEATURES, INITIAL_FEATURES, kernel_size=2, stride=2, bias=False)
            )
            self.up_convs.append(
                nn.Sequential(
                    nn.Conv2d(INITIAL_FEATURES * 2, INITIAL_FEATURES, kernel_size=3, padding="same", padding_mode="replicate", bias=False),
                    nn.BatchNorm2d(INITIAL_FEATURES),
                    nn.ReLU(),
                    nn.Conv2d(INITIAL_FEATURES, INITIAL_FEATURES, kernel_size=3, padding="same", padding_mode="replicate", bias=False),
                    nn.BatchNorm2d(INITIAL_FEATURES),
                    nn.ReLU()
                )
            )

        # Final Convolution
        self.final_conv = nn.Conv2d(INITIAL_FEATURES, 1, kernel_size=1)

    def forward(self, x):
        # Initial Convolution
        x = self.initial_conv(x)

        # Contracting Path
        contracting_path_features = []
        for down_conv in self.down_convs:
            x = down_conv(x)
            contracting_path_features.append(x)
            x = self.pool(x)

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
