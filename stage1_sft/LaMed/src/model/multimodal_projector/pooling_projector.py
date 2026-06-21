from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

class PoolingProjector(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type, layer_num, pooling_type='avg', pooling_size=2):
        super().__init__()
        self.in_dim = in_dim
        self.pooling_size = pooling_size

        self.layer_type = layer_type
        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]
        self.num_patches_post = [num // pooling_size for num in self.num_patches_pre]
            
        depth = int(layer_num)
        modules = [nn.Linear(in_dim, out_dim)]
        for _ in range(1, depth):
            modules.append(nn.Linear(out_dim, out_dim))
        self.projector = nn.Sequential(*modules)
        self.pooling_type = pooling_type

    def forward(self, x):
        B = x.shape[0] # B*N*D
        if self.layer_type == 'avg':
            to_3d = Rearrange("b (p1 p2 p3) d -> b d p1 p2 p3", b=B, d=self.in_dim, p1=self.num_patches_pre[0], p2=self.num_patches_pre[1], p3=self.num_patches_pre[2])
            x = to_3d(x)
            x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)
            to_seq = Rearrange("b d p1 p2 p3 -> b (p1 p2 p3) d", b=B, d=self.in_dim, p1=self.num_patches_post[0], p2=self.num_patches_post[1], p3=self.num_patches_post[2])
            x = to_seq(x)
        elif self.layer_type == 'max':
            to_3d = Rearrange("b (p1 p2 p3) d -> b d p1 p2 p3", b=B, d=self.in_dim, p1=self.num_patches_pre[0], p2=self.num_patches_pre[1], p3=self.num_patches_pre[2])
            x = to_3d(x)
            x = F.max_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)
            to_seq = Rearrange("b d p1 p2 p3 -> b (p1 p2 p3) d", b=B, d=self.in_dim, p1=self.num_patches_post[0], p2=self.num_patches_post[1], p3=self.num_patches_post[2])
            x = to_seq(x)

        x = rearrange(x, "b n d -> (b n) d")
        x = self.projector(x)
        x = rearrange(x, "(b n) d -> b n d", b=B)

        return x

    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_post:
            num *= n
        return num