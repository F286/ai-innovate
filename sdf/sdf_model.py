# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class PatchEmbedding(nn.Module):
#     def __init__(self, in_channels=1, patch_size=10, emb_size=768, img_size=100):
#         super().__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.n_patches = (img_size // patch_size) ** 2
#         self.conv = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
                
#         x = self.conv(x)  # [B, emb_size, H/P, W/P]
#         x = x.flatten(2)  # [B, emb_size, n_patches]
#         x = x.transpose(1, 2)  # [B, n_patches, emb_size]
#         return x

# class SDFNet(nn.Module):
#     def __init__(self, img_size=100, patch_size=10, emb_size=768, n_heads=8, n_layers=6):
#         super(SDFNet, self).__init__()
#         self.patch_embedding = PatchEmbedding(patch_size=patch_size, emb_size=emb_size, img_size=img_size)
#         self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_embedding.n_patches + 1, emb_size))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
#         encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=n_heads)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
#         self.to_sdf = nn.Linear(emb_size, img_size*img_size)  # Assuming a flattened output

#     def forward(self, x):
#         B = x.shape[0]
#         x = self.patch_embedding(x)
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)  # Add class token
#         x += self.pos_embedding[:, :(self.patch_embedding.n_patches + 1)]
#         x = self.transformer_encoder(x)
#         x = x[:, 0]  # Use the class token
#         x = self.to_sdf(x)
#         x = x.view(B, 1, 100, 100)  # Reshape back to image dimensions
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class SDFNet(nn.Module):
    def __init__(self):
        super(SDFNet, self).__init__()

        self.conv0 = nn.Conv2d(1, 8, kernel_size=1, padding=0, bias=False)

        # Sequential operation with kernel sampling from distance 32, preserving input/output size
        self.seq_sampling_64 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding="same", dilation=64, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding="same", dilation=64, bias=False)
        )
        # Sequential operation with kernel sampling from distance 32, preserving input/output size
        self.seq_sampling_32 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding="same", dilation=32, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding="same", dilation=32, bias=False)
        )

        # Sequential operation with kernel sampling from distance 16, preserving input/output size
        self.seq_sampling_16 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding="same", dilation=16, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding="same", dilation=16, bias=False)
        )

        # Sequential operation with kernel sampling from distance 8, preserving input/output size
        self.seq_sampling_8 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding="same", dilation=8, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding="same", dilation=8, bias=False)
        )


        # Sequential operation with kernel sampling from distance 4
        self.seq_sampling_4 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding="same", dilation=4, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding="same", dilation=4, bias=False)
        )

        # Sequential operation with kernel sampling from distance 2
        self.seq_sampling_2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding="same", dilation=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding="same", dilation=2, bias=False)
        )

        # Original sequential operation with stride 1
        self.seq_stride_1 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding="same", dilation=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding="same", dilation=1, bias=False)
        )

        self.conv3 = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

    def forward(self, x):

        initial_shape = x.shape

        x = self.conv0(x)

        # Apply the sequential operations with different strides
        x = self.seq_sampling_64(x)
        x = self.seq_sampling_32(x)
        x = self.seq_sampling_16(x)
        x = self.seq_sampling_8(x)
        x = self.seq_sampling_4(x)
        x = self.seq_sampling_2(x)
        x = self.seq_stride_1(x)

        x = self.conv3(x)

        return x