from torch import nn
import os
import torch.nn.functional as F
import torch
import numpy as np

def get_3d_sincos_pos_embed(embed_dim, grid_d, grid_h, grid_w):
    
    grid_z = np.linspace(0, 1, num=grid_d, dtype=np.float32)
    grid_y = np.linspace(0, 1, num=grid_h, dtype=np.float32)
    grid_x = np.linspace(0, 1, num=grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    grid = np.stack(grid, axis=-1).reshape(-1, 3)

    dim_each = embed_dim // 3
    pos_embed = []
    for i in range(3):
        pos = grid[:, i]
        omega = np.power(10000, -np.arange(0, dim_each // 2, dtype=np.float32) / (dim_each // 2))
        out = np.outer(pos, omega)
        pos_embed.append(np.concatenate([np.sin(out), np.cos(out)], axis=1))

    pos_embed = np.concatenate(pos_embed, axis=1)

    # pad or truncate
    if pos_embed.shape[1] < embed_dim:
        pad = embed_dim - pos_embed.shape[1]
        pos_embed = np.pad(pos_embed, ((0, 0), (0, pad)))
    elif pos_embed.shape[1] > embed_dim:
        pos_embed = pos_embed[:, :embed_dim]

    return torch.from_numpy(pos_embed).float().unsqueeze(0)  # [1, N, D]




class SpatialConvLinearReshapeProjector(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        in_dim,
        out_dim,
        layer_type,
        layer_num,
        pooling_size=2,
        mean_prompt_template_path=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]
        self.num_patches_post = [num // pooling_size for num in self.num_patches_pre]
        # self.num_patches_post = self.num_patches_pre[:-1] + [self.num_patches_pre[-1] // pooling_size]
        # self.num_patches_post = [num // pooling_size for num in self.num_patches_pre[:2]] + [self.num_patches_pre[2]]


        self.reduce = nn.Conv3d(in_dim, out_dim, kernel_size=pooling_size, stride=pooling_size)
        # self.reduce = nn.Conv3d(in_dim, out_dim, kernel_size=(1, 1, pooling_size), stride=(1, 1, pooling_size))
        # self.reduce = nn.Conv3d(in_dim, out_dim, kernel_size=(pooling_size, pooling_size, 1), stride=(pooling_size, pooling_size, 1))


        # self.reduce = nn.AvgPool3d(kernel_size=(pooling_size, pooling_size, 1), stride=(pooling_size, pooling_size, 1))
        # self.map = nn.Linear(in_dim, out_dim)
        if layer_type == 'linear':
            depth = int(layer_num)
            modules = []
            for _ in range(1, depth):
                modules.append(nn.Linear(out_dim, out_dim))
                modules.append(nn.LayerNorm(out_dim))
            self.projector = nn.Sequential(*modules)
        elif layer_type == 'mlp':
            depth = int(layer_num)
            modules = []
            for _ in range(1, depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(out_dim, out_dim))
                modules.append(nn.LayerNorm(out_dim))
            self.projector = nn.Sequential(*modules)
        else:
            raise ValueError("layer_type must be 'linear' or 'mlp'")

        # This tensor is downloaded separately and intentionally not committed to GitHub.
        self.mean_prompt_template = nn.Parameter(torch.empty(1, out_dim), requires_grad=False)

        if mean_prompt_template_path is None:
            mean_prompt_template_path = "./mean_prompt_template_qwen2.5.pt"
        if not os.path.exists(mean_prompt_template_path):
            raise FileNotFoundError(
                f"mean prompt template not found: {mean_prompt_template_path}. "
                "Download it separately and pass --mean_prompt_template_path if it is stored elsewhere."
            )
        template = torch.load(mean_prompt_template_path, map_location="cpu")
        if not torch.is_tensor(template):
            raise TypeError(f"mean prompt template must be a tensor, got {type(template)!r}.")
        if template.ndim == 1:
            template = template.unsqueeze(0)
        if tuple(template.shape) != tuple(self.mean_prompt_template.shape):
            raise ValueError(
                f"mean prompt template shape {tuple(template.shape)} does not match "
                f"expected {tuple(self.mean_prompt_template.shape)}."
            )
        with torch.no_grad():
            self.mean_prompt_template.copy_(template.float())

        # 门控 MLP（向量级gate）
        self.gate_mlp = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(out_dim)  #add layer norm

        # 加入位置编码参数（sin-cos 初始化 + 可学习）
        pos_embed = get_3d_sincos_pos_embed(out_dim, *self.num_patches_post)
        self.pos_embed = nn.Parameter(pos_embed, requires_grad=True)

    def forward(self, x):
        B = x.shape[0]  # B*N*D

        # Reshape to 3D tensor
        x = x.reshape(B, *self.num_patches_pre, self.in_dim).contiguous()

        x = x.permute(0, 4, 1, 2, 3)  # [B, D, D1, D2, D3]

        # x = x.reshape(B, self.in_dim, *self.num_patches_pre)

        x = self.reduce(x)
        
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        # x = self.map(x)  # [B, C, D, H, W]
        
        x = self.projector(x)
    
        # Reshape back to sequence
        x = x.reshape(B, -1, self.out_dim)

        # anchor: [1, D] -> [B, N, D]
        anchor = self.mean_prompt_template.unsqueeze(0).expand(B, x.shape[1], -1)  # broadcast to sequence length

        # 计算门控值（每个 token 一个 gate）
        gate_input = torch.cat([x, anchor], dim=-1)  # [B, N, 2D]
        gate = self.gate_mlp(gate_input)             # [B, N, 1]

        # # 门控融合
        x = gate * anchor + (1 - gate) * x     # [B, N, D]

        # # 加入视觉位置编码
        x = x + self.pos_embed

        # 归一化
        x = self.norm(x)                 # [B, N, D]

        return x

    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_post:
            num *= n
        return num


# from torch import nn
# import torch.nn.functional as F
# import torch

# class SpatialConvLinearReshapeProjector(nn.Module):
#     def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type, layer_num, pooling_size=2):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim

#         self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]
#         self.num_patches_post = [num // pooling_size for num in self.num_patches_pre]

#         self.reduce = nn.Conv3d(in_dim, out_dim, kernel_size=pooling_size, stride=pooling_size)
#         if layer_type == 'linear':
#             depth = int(layer_num)
#             modules = []
#             for _ in range(1, depth):
#                 modules.append(nn.Linear(out_dim, out_dim))
#             self.projector = nn.Sequential(*modules)
#         elif layer_type == 'mlp':
#             depth = int(layer_num)
#             modules = []
#             for _ in range(1, depth):
#                 modules.append(nn.GELU())
#                 modules.append(nn.Linear(out_dim, out_dim))
#             self.projector = nn.Sequential(*modules)
#         else:
#             raise ValueError("layer_type must be 'linear' or 'mlp'")

#         # 构造一个可以自动迁移的模型参数
#         self.mean_prompt_template = nn.Parameter(torch.empty(1, out_dim), requires_grad=False)  # 不训练

#         template = torch.load(mean_prompt_template_path, map_location="cpu")  # float32, [N, D] or [1, D]
            
#             # 初始化这个参数
#         with torch.no_grad():
#             self.mean_prompt_template.copy_(template)
            

#         # 门控 MLP（向量级gate）
#         self.gate_mlp = nn.Sequential(
#             nn.Linear(out_dim * 2, out_dim),
#             nn.ReLU(),
#             nn.Linear(out_dim, out_dim),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         B = x.shape[0]  # B*N*D

#         # Reshape to 3D tensor
#         x = x.reshape(B, self.in_dim, *self.num_patches_pre)

#         x = self.reduce(x)
        
#         x = x.permute(0, 2, 3, 4, 1).contiguous()
        
#         x = self.projector(x)
        
#         # Reshape back to sequence
#         x = x.reshape(B, -1, self.out_dim)

#         # anchor: [1, D] -> [B, N, D]
#         anchor = self.mean_prompt_template.unsqueeze(0).expand(B, x.shape[1], -1)  # broadcast to sequence length

#         # 计算门控值（每个 token 一个 gate）
#         gate_input = torch.cat([x, anchor], dim=-1)  # [B, N, 2D]
#         gate = self.gate_mlp(gate_input)             # [B, N, 1]

#         # 门控融合
#         x_fused = gate * anchor + (1 - gate) * x     # [B, N, D]

#         return x_fused

#     @property
#     def proj_out_num(self):
#         num = 1
#         for n in self.num_patches_post:
#             num *= n
#         return num
