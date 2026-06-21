from torch import nn
import torch

class MLPMixerProjector(nn.Module):
    def __init__(self, 
                  image_size, patch_size, in_dim, out_dim, layer_num=1, pooling_size=2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pooling_size = pooling_size

        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]
        self.num_patches_post = [num // pooling_size for num in self.num_patches_pre]

        depth = int(layer_num)
        self.path_prob = nn.Linear(self.num_patches_pre[0] * self.num_patches_pre[1] * self.num_patches_pre[2], self.num_patches_post[0] * self.num_patches_post[1] * self.num_patches_post[2])
        self.gule = nn.GELU()
        self.norm = nn.LayerNorm(in_dim)
        self.linear_prob = nn.Linear(in_dim, out_dim)


    def forward(self, x):
        B = x.shape[0]  # B*N*D
        x = x.permute(0, 2, 1)  # (b, d, n)
        x = self.path_prob(x)
        x = self.gule(x)
        x = x.permute(0, 2, 1)  # (b, n, d)
        x = self.norm(x)
        x = self.linear_prob(x)
        return x
    
    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_post:
            num *= n
        return num

