from torch import nn
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class SpatialConvLinearDictProjector(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type, layer_num, pooling_size=2, dict_size=2048):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dict_size = dict_size

        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]
        self.num_patches_post = [num // pooling_size for num in self.num_patches_pre]

        # 可学习的字典
        self.dictionary = nn.Embedding(dict_size, out_dim)
        nn.init.orthogonal_(self.dictionary.weight)

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

    def forward(self, x, text_feature):
        B = x.shape[0]  # B*N*D

        # Reshape to 3D tensor
        to_3d = Rearrange("b (p1 p2 p3) d -> b d p1 p2 p3", 
                          b=B, d=self.in_dim, 
                          p1=self.num_patches_pre[0], 
                          p2=self.num_patches_pre[1], 
                          p3=self.num_patches_pre[2])
        x = to_3d(x)
        
        # Apply convolutional layer
        x = self.projector(x)  # (B, N, D)

        # 对图像特征和文本特征进行字典映射
        x_reconstructed = self.reconstruct(x)
        text_reconstructed = self.reconstruct(text_feature)

        return x, x_reconstructed, text_reconstructed

    def reconstruct(self, feature):
        """
        通过字典进行投影和重建
        """
        # 计算特征与字典的相似度
        similarity = torch.matmul(feature, self.dictionary.weight.T)  # (B, N, dict_size)

        # 计算 softmax 权重
        weights = F.softmax(similarity, dim=-1)  # (B, N, dict_size)

        # 加权求和，进行特征重建
        reconstructed = torch.matmul(weights, self.dictionary.weight)  # (B, N, D)

        return reconstructed

    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_post:
            num *= n
        return num
