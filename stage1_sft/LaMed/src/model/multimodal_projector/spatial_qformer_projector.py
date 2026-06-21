from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
import torch

class SpatialConvLinearQformerProjector(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type, layer_num, pooling_size=2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]
        self.num_patches_post = [num // pooling_size for num in self.num_patches_pre]

        token_num = 64
        self.queries = nn.Parameter(torch.randn((token_num, in_dim)))

        self.cross_attn = nn.MultiheadAttention(in_dim, num_heads=4, batch_first=True)
        self.attention_projector = nn.Linear(in_dim, out_dim)

        if layer_type == 'linear':
            depth = int(layer_num)
            modules = [nn.Conv3d(in_dim, out_dim, kernel_size=pooling_size, stride=pooling_size),
                       Rearrange('b d p1 p2 p3 -> b (p1 p2 p3) d')]
            for _ in range(1, depth):
                modules.append(nn.Linear(out_dim, out_dim))
            self.projector = nn.Sequential(*modules)
        elif layer_type == 'mlp':
            depth = int(layer_num)
            modules = [nn.Conv3d(in_dim, out_dim, kernel_size=pooling_size, stride=pooling_size),
                        Rearrange('b d p1 p2 p3 -> b (p1 p2 p3) d')]
            for _ in range(1, depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(out_dim, out_dim))
            self.projector = nn.Sequential(*modules)
        else:
            raise ValueError("layer_type must be 'linear' or 'mlp'")
    
    def global_forward(self, image_features):
        text_features = self.queries
        batch_size = image_features.shape[0]
        text_features = self.queries.unsqueeze(0).repeat(batch_size, 1, 1)
        global_features, atten_map = self.cross_attn(text_features, image_features, image_features)
        out = self.attention_projector(global_features)
        return out



    def forward(self, x):
        B = x.shape[0]  # B*N*D

        global_features = self.global_forward(x)
        
        # Reshape to 3D tensor
        to_3d = Rearrange("b (p1 p2 p3) d -> b d p1 p2 p3", b=B, d=self.in_dim, p1=self.num_patches_pre[0], p2=self.num_patches_pre[1], p3=self.num_patches_pre[2])
        x = to_3d(x)
        
        # Apply convolutional layer
        x = self.projector(x)
        
        # # Reshape back to sequence
        # to_seq = Rearrange("b d p1 p2 p3 -> b (p1 p2 p3) d", b=B, d=self.out_dim, p1=self.num_patches_post[0], p2=self.num_patches_post[1], p3=self.num_patches_post[2])
        # x = to_seq(x)
        
        # normalize for convergence
        # x = F.normalize(x, dim=-1)
        x = torch.cat((x, global_features), dim=1)
        return x

    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_post:
            num *= n
        num += 64
        return num
