from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

class SpatialConvProjector(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, pooling_size):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]
        self.num_patches_post = [num // pooling_size for num in self.num_patches_pre]

        # Define the 3D convolutional layer
        self.conv = nn.Conv3d(in_dim, out_dim, kernel_size=pooling_size, stride=pooling_size)

    def forward(self, x):
        B = x.shape[0]  # B*N*D

        # Reshape to 3D tensor
        to_3d = Rearrange("b (p1 p2 p3) d -> b d p1 p2 p3", b=B, d=self.in_dim, p1=self.num_patches_pre[0], p2=self.num_patches_pre[1], p3=self.num_patches_pre[2])
        x = to_3d(x)
        
        # Apply 3D convolution
        x = self.conv(x)
        
        # Reshape back to sequence
        to_seq = Rearrange("b d p1 p2 p3 -> b (p1 p2 p3) d", b=B, d=self.out_dim, p1=self.num_patches_post[0], p2=self.num_patches_post[1], p3=self.num_patches_post[2])
        x = to_seq(x)

        return x

    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_post:
            num *= n
        return num
