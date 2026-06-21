from torch import nn
import torch

# class AttentiveQueryGenerator(nn.Module):
#     def __init__(self, in_dim, out_token_num):
#         super().__init__()
#         self.out_token_num = out_token_num
#         self.attn_proj = nn.Linear(in_dim, out_token_num)  # 每个 token → Q 个权重

#     def forward(self, x):  # x: [B, N, D]
#         B, N, D = x.shape
#         attn_weights = self.attn_proj(x)  # [B, N, Q]
#         attn_weights = torch.softmax(attn_weights, dim=1)  # [B, N, Q]
#         x_t = x.transpose(1, 2)  # [B, D, N]
#         query_tokens = torch.matmul(x_t, attn_weights)  # [B, D, Q]
#         return query_tokens.transpose(1, 2)  # [B, Q, D]


class AttentiveQueryGenerator(nn.Module):
    def __init__(self, in_dim, out_token_num):
        super().__init__()
        self.out_token_num = out_token_num
        self.query_directions = nn.Parameter(torch.randn(out_token_num, in_dim))  # Q个query方向向量
        self.token_proj = nn.Linear(in_dim, in_dim)

    def forward(self, x):  # x: [B, N, D]
        B, N, D = x.shape
        x_proj = self.token_proj(x)  # [B, N, D]
        query = self.query_directions  # [Q, D]

        # 计算 token 与 query 方向的相似度
        sim = torch.einsum('bnd,qd->bnq', x_proj, query)  # [B, N, Q]
        attn_weights = torch.softmax(sim, dim=1)  # token维归一化

        # 聚合 Q 个 query，每个是 token 的加权组合
        x_t = x.transpose(1, 2)  # [B, D, N]
        query_tokens = torch.matmul(x_t, attn_weights)  # [B, D, Q]
        return query_tokens.transpose(1, 2)  # [B, Q, D]

# class AttentiveQueryGenerator(nn.Module):
#     def __init__(self, in_dim, out_token_num, num_heads=2):
#         super().__init__()
#         assert out_token_num % num_heads == 0, "out_token_num 必须能被 num_heads 整除"
#         self.out_token_num = out_token_num
#         self.num_heads = num_heads
#         self.tokens_per_head = out_token_num // num_heads

#         self.query_directions = nn.Parameter(torch.randn(num_heads, self.tokens_per_head, in_dim))
#         self.token_proj = nn.Linear(in_dim, in_dim)

#     def forward(self, x):  # x: [B, N, D]
#         B, N, D = x.shape
#         x_proj = self.token_proj(x)  # [B, N, D]

#         # 多头相似度计算：[B, N, D] · [H, Q, D] → [B, N, H, Q]
#         sim = torch.einsum('bnd,hqd->bnhq', x_proj, self.query_directions)  # [B, N, H, Q]
#         attn_weights = torch.softmax(sim, dim=1)  # softmax over tokens

#         # x 转为 [B, H, D, N]
#         x_t = x.transpose(1, 2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).contiguous()  # [B, H, D, N]

#         # attn_weights 转为 [B, H, N, Q]
#         attn_weights = attn_weights.permute(0, 2, 1, 3).contiguous()  # [B, H, N, Q]

#         # 聚合： [B, H, D, N] @ [B, H, N, Q] = [B, H, D, Q]
#         query_tokens = torch.matmul(x_t, attn_weights)

#         # reshape → [B, Q_all, D]
#         query_tokens = query_tokens.permute(0, 1, 3, 2).contiguous().view(B, self.out_token_num, D)
#         return query_tokens




class AdapterCrossAttentionProjector(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type='global', layer_num=1, pooling_type='spatial', pooling_size=2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pooling_size = pooling_size
        self.layer_type = layer_type

        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]
        self.num_patches_post = [num // pooling_size for num in self.num_patches_pre[:2]] + [self.num_patches_pre[2]]
        # self.num_patches_post = [num // pooling_size for num in self.num_patches_pre[:1]] + self.num_patches_pre[1:3]
        self.token_num = self.num_patches_post[0] * self.num_patches_post[1] * self.num_patches_post[2]

        # 改为使用结构感知的 attentive query 生成器
        self.query_generator = AttentiveQueryGenerator(in_dim=in_dim, out_token_num=self.token_num)

        self.norm = nn.LayerNorm(out_dim)
        self.cross_attn = nn.MultiheadAttention(in_dim, num_heads=4, batch_first=True)

        if layer_type in ['global', 'local']:
            modules = [nn.Linear(in_dim, out_dim)]
            self.projector = nn.Sequential(*modules)
        else:
            raise ValueError("Invalid layer type. Use 'global' or 'local'.")

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, image_features):
        if self.layer_type == 'global':
            return self.global_forward(image_features)
        elif self.layer_type == 'local':
            return self.local_forward(image_features)

    def global_forward(self, image_features):
        B, N, D = image_features.shape
        queries = self.query_generator(image_features)  # [B, token_num, D]
        global_features, _ = self.cross_attn(queries, image_features, image_features)  # [B, token_num, D]
        out = self.projector(global_features)
        out = self.norm(out)
        return out

    def local_forward(self, image_features):
        B, N, D = image_features.shape
        expected_tokens = self.num_patches_pre[0] * self.num_patches_pre[1] * self.num_patches_pre[2]
        group_vision_tokens = self.num_patches_post[0] * self.num_patches_post[1] * self.num_patches_post[2]
        assert N == expected_tokens, "Number of tokens does not match expected number from patches."

        image_features = image_features.view(B, *self.num_patches_post, self.pooling_size, self.pooling_size, self.pooling_size, D)
        image_features = image_features.permute(0, 1, 2, 3, 7, 4, 5, 6).contiguous()
        image_features = image_features.view(B, self.token_num, -1, D)
        image_features = image_features.view(B * group_vision_tokens, -1, D)

        # 原 local 模式仍使用共享 query（如需替换可再加）
        queries = self.query_generator.attn_proj.weight.new_zeros((B * group_vision_tokens, 1, D))
        merge_feature, _ = self.cross_attn(queries, image_features, image_features)
        merge_feature = merge_feature.view(B, group_vision_tokens, D)
        merge_feature = self.projector(merge_feature)
        return merge_feature

    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_post:
            num *= n
        return num
