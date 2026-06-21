import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from functools import partial

class SelfAttentionLayer(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, num_heads=12, mlp_ratio=4.0, qkv_bias=True, weight_path=None, load_block_id=11):
        super().__init__()
        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]

        self.attn_block = Block(
            dim=in_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        self.projector = nn.Linear(in_dim, out_dim)


        if weight_path is not None:
            print(f"Loading block.{load_block_id} weights from: {weight_path}")
            vision_model_weights = torch.load(weight_path, map_location='cpu')

            # 提取 block.{load_block_id}. 的子权重
            prefix = f'blocks.{load_block_id}.'
            block_state_dict = {
                key[len(prefix):]: val
                for key, val in vision_model_weights.items()
                if key.startswith(prefix)
            }

            # 加载权重
            self.attn_block.load_state_dict(block_state_dict, strict=True)
            print(f"Block.{load_block_id} weights loaded into self.attn_block.")



    def forward(self, x):
        # x: [B, N, D] where N is the number of tokens and D is hidden size
        x =  self.attn_block(x)

        x = self.projector(x)  # [B, N, D_out]

        return x
    
    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_pre:
            num *= n
        return num
