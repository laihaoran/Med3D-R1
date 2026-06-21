from torch import nn
import torch

class TwiceCrossAttentionProjector(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type='global', layer_num=1, pooling_type='spatial', pooling_size=2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pooling_size = pooling_size
        self.layer_type = layer_type

        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]
        # self.num_patches_post = [num // pooling_size for num in self.num_patches_pre[:1]] + self.num_patches_pre[1:3]
        self.num_patches_post = [num // pooling_size for num in self.num_patches_pre[0:2]] + self.num_patches_pre[2:3]
        self.token_num = self.num_patches_post[0] * self.num_patches_post[1] * self.num_patches_post[2]

        # 替代 AttentiveQueryGenerator：learnable query + 两次 cross-attn
        self.query_embed = nn.Parameter(torch.randn(self.token_num, in_dim))  # [Q, D]
        self.cross_attn1 = nn.MultiheadAttention(in_dim, num_heads=4, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(in_dim, num_heads=4, batch_first=True)

        self.norm = nn.LayerNorm(out_dim)
        self.projector = nn.Linear(in_dim, out_dim)

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
        query = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]
        query1, _ = self.cross_attn1(query, image_features, image_features)  # [B, Q, D]
        query2, _ = self.cross_attn2(query1, image_features, image_features)  # [B, Q, D]
        out = self.projector(query2)
        out = self.norm(out)
        return out

    def local_forward(self, image_features):
        B, N, D = image_features.shape
        expected_tokens = self.num_patches_pre[0] * self.num_patches_pre[1] * self.num_patches_pre[2]
        group_vision_tokens = self.num_patches_post[0] * self.num_patches_post[1] * self.num_patches_post[2]
        assert N == expected_tokens, "Number of tokens does not match expected number from patches."

        image_features = image_features.view(B, *self.num_patches_post, self.pooling_size, self.pooling_size, self.pooling_size, D)
        image_features = image_features.permute(0, 1, 2, 3, 7, 4, 5, 6).contiguous()
        image_features = image_features.view(B, self.token_num, -1, D).view(B * group_vision_tokens, -1, D)

        query = self.query_embed.new_zeros((B * group_vision_tokens, 1, D))  # zero query
        out, _ = self.cross_attn1(query, image_features, image_features)
        out = self.projector(out.squeeze(1)).view(B, group_vision_tokens, D)
        return out

    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_post:
            num *= n
        return num
