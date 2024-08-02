import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, patch_size, embed_dim, img_size, in_chans):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.in_chans = in_chans

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, E, H/P, W/P]
        x = x.flatten(2)  # [B, E, N]
        x = x.transpose(1, 2)  # [B, N, E]
        return x
    
class PatchDeEmbed(nn.Module):
    """Rearrange patch embeddings back to the 2D image (or SDF) layout."""
    def __init__(self, patch_size, embed_dim, img_size, in_chans):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, N, E = x.shape
        H = W = int((self.img_size * self.img_size) ** 0.5)
        x = x.transpose(1, 2).contiguous().view(B, E, H // self.patch_size, W // self.patch_size)
        x = self.proj(x)  # Project back to image space
        return x

class TransformerEncoder(nn.Module):
    """Transformer Encoder to process the sequence of image patches."""
    def __init__(self, embed_dim, depth, num_heads):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, dim_feedforward=embed_dim * 4, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, src):
        return self.encoder(src)

class SDFNet(nn.Module):
    def __init__(self, img_size=128, patch_size=8, embed_dim=128, depth=4, num_heads=8, in_chans=1, out_chans=1):
        super(SDFNet, self).__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.patch_embed = PatchEmbed(patch_size, embed_dim, img_size, in_chans)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))  # Learned positional encoding
        self.transformer_encoder = TransformerEncoder(embed_dim, depth, num_heads)
        # Updated to use PatchDeEmbed instead of a linear head
        self.patch_deembed = PatchDeEmbed(patch_size, embed_dim, img_size, embed_dim)  # Note: Output embed_dim channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=1),  # Expands the channel dimension
            nn.ReLU(),  # Activation function for non-linearity
            nn.Conv2d(embed_dim * 2, out_chans, kernel_size=1)  # Contracts the channel dimension back
        )
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.pos_embed, std=0.02)  # Initialize position embeddings
        
    def forward(self, x):
        # Store the original size of x for later use to ensure output matches input dimensions
        original_size = x.size()

        # Embed patches
        x = self.patch_embed(x)
        # Add position encoding
        x = x + self.pos_embed
        # Transformer Encoder
        x = self.transformer_encoder(x)
        # Reconstruct image/SDF from patches, maintaining embed_dim channels
        x = self.patch_deembed(x)
        # Apply 1x1 convolution to map to the desired number of output channels
        x = self.final_conv(x)

        return x