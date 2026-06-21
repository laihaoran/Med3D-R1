from torch import nn
import torch

class CrossAttentionProjector(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type='global', layer_num=1, pooling_type='spatial', pooling_size=2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pooling_size = pooling_size
        self.layer_type = layer_type

        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]
        # self.num_patches_post = [num // pooling_size for num in self.num_patches_pre]
        self.num_patches_post = [num // pooling_size for num in self.num_patches_pre[:2]] + [self.num_patches_pre[2]]

        token_num = self.num_patches_post[0] * self.num_patches_post[1] * self.num_patches_post[2]
        # self.queries = nn.Parameter(torch.randn((token_num, in_dim)))
        self.queries = nn.Parameter(torch.empty(token_num, in_dim))
        nn.init.xavier_uniform_(self.queries)  # 推荐默认方式

        self.norm = nn.LayerNorm(out_dim)

        self.cross_attn = nn.MultiheadAttention(in_dim, num_heads=4, batch_first=True)

        if layer_type in ['global', 'local']:
            # depth = int(layer_num)
            modules = [nn.Linear(in_dim, out_dim)]
            # for _ in range(1, depth):
            #     modules.append(nn.Linear(out_dim, out_dim))
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
        text_features = self.queries
        batch_size = image_features.shape[0]
        text_features = self.queries.unsqueeze(0).repeat(batch_size, 1, 1)
        global_features, atten_map = self.cross_attn(text_features, image_features, image_features)
        out = self.projector(global_features)
        out = self.norm(out)  # 加 LayerNorm 稳定训练
        return out

    def local_forward(self, image_features):
        B, N, D = image_features.shape
        expected_tokens = self.num_patches_pre[0] * self.num_patches_pre[1] * self.num_patches_pre[2]
        group_vision_tokens = self.num_patches_post[0] * self.num_patches_post[1] * self.num_patches_post[2]
        assert N == expected_tokens, "Number of tokens does not match expected number from patches."

        image_features = image_features.view(B, *self.num_patches_post, self.pooling_size, self.pooling_size, self.pooling_size, D)
        image_features = image_features.permute(0, 1, 2, 3, 7, 4, 5, 6).contiguous()
        image_features = image_features.view(B, self.num_patches_post[0] * self.num_patches_post[1] * self.num_patches_post[2], -1, D)

        image_features = image_features.view(B * group_vision_tokens, -1, D)

        queries = self.queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        queries = queries.view(B * group_vision_tokens, -1, D)

        merge_feature, weighted_patches = self.cross_attn(queries, image_features, image_features)
        merge_feature = merge_feature.view(B, group_vision_tokens, D)
        merge_feature = self.projector(merge_feature)
        return merge_feature

    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_post:
            num *= n
        return num
