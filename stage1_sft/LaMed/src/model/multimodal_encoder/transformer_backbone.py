from functools import partial
import torch
import torch.nn as nn
from torchvision.transforms.functional import InterpolationMode
from timm.models.vision_transformer import PatchEmbed, Block
import numpy as np

def get_3d_sincos_pos_embed(embed_dim, grid_d, grid_h, grid_w, cls_token=False):
    """
    Create 3D sine-cosine positional embeddings ensuring exact embedding dimensions.
    grid_d, grid_h, grid_w: dimensions of the grid depth, height, and width.
    cls_token: whether to include a class token.
    """
    # Create a 3D grid for position encoding
    grid_z = np.linspace(0, 1, num=grid_d, dtype=np.float32)
    grid_y = np.linspace(0, 1, num=grid_h, dtype=np.float32)
    grid_x = np.linspace(0, 1, num=grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')  # Change to ij for consistency
    grid = np.stack(grid, axis=-1).reshape(-1, 3)  # Flatten grid

    # Calculate position encoding for each dimension
    pos_embed = []
    dim_per_axis = embed_dim // 3  # Divide the dimensions equally among the axes
    for i in range(3):
        pos = grid[:, i]
        omega = np.power(10000, -np.arange(0, dim_per_axis // 2) / (dim_per_axis // 2))
        sin_emb = np.sin(pos[:, None] * omega[None, :])
        cos_emb = np.cos(pos[:, None] * omega[None, :])
        pos_embed.append(np.concatenate([sin_emb, cos_emb], axis=1))

    pos_embed = np.concatenate(pos_embed, axis=1)

    # Ensure the positional embedding exactly matches the embedding dimension
    if pos_embed.shape[1] > embed_dim:
        pos_embed = pos_embed[:, :embed_dim]
    elif pos_embed.shape[1] < embed_dim:
        extra_dims = embed_dim - pos_embed.shape[1]
        extra_emb = np.zeros((pos_embed.shape[0], extra_dims))
        pos_embed = np.concatenate([pos_embed, extra_emb], axis=1)

    if cls_token:
        # Prepend a class token if required
        cls_embed = np.zeros((1, embed_dim))
        pos_embed = np.concatenate([cls_embed, pos_embed], axis=0)

    return pos_embed

class PatchEmbed3D(nn.Module):
    """Compute 3D patch embeddings for non-cubic volumes."""
    def __init__(self, img_size=(224, 224, 112), patch_size=(16, 16, 8), in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Calculate the number of patches along each dimension
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)


    def forward(self, x):
        x = self.proj(x)  # (N, C, D', H', W')
        x = x.flatten(2)  # (N, C, D'*H'*W')
        x = x.transpose(1, 2)  # (N, D'*H'*W', C)
        return x


class ViT3D(nn.Module):
    """Vision Transformer (ViT) with 3D support."""
    def __init__(
        self,
        in_channels: int,
        img_size: tuple,
        patch_size: tuple,
        hidden_size: int = 768, #1536
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = True,
        save_attn: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.classification = classification
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_channels, hidden_size)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_dim // hidden_size, qkv_bias=qkv_bias, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)
        self.initialize_weights()

        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            if post_activation == "Tanh":
                self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            else:
                self.classification_head = nn.Linear(hidden_size, num_classes)

    def initialize_weights(self):
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.img_size[0] // self.patch_embed.patch_size[0],
                                            self.patch_embed.img_size[1] // self.patch_embed.patch_size[1],
                                            self.patch_embed.img_size[2] // self.patch_embed.patch_size[2], cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)

        x = self.norm(x)
        if self.classification:
            x = self.classification_head(x[:, 0])
        return x, hidden_states_out


class ViT3DMAETower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        self.vision_tower = ViT3D(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="conv",
            spatial_dims=len(self.config.patch_size),
            classification=False
        )

    def forward(self, images):
        last_feature, hidden_states = self.vision_tower(images)
        if self.select_layer == -1:
            image_features = last_feature
        elif self.select_layer < -1:
            image_features = hidden_states[self.select_feature]
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size


def update_weights(model, pretrained_dict):
    model_dict = model.state_dict()
    # 过滤掉不匹配的权重
    pretrained_dict = {k[len('vision_encoder.'):]: v for k, v in pretrained_dict.items() if k[len('vision_encoder.'):]  in model_dict}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print('matched keys:', len(pretrained_dict))
    # 更新模型参数字典x
    model_dict.update(pretrained_dict)
    return model_dict


# 加载权重函数
def load_weight(model, path, save_path):
    update_dict = torch.load(path, map_location=torch.device('cpu'))
    # update_dict = {k: v for k, v in pretrain_dict['model'].items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape} for mae pretrain model
    # update_dict = update_weights(model, pretrain_dict['state_dict']) # for ma3e pretrain model
    
    # 打印预训练权重和当前模型中不匹配参数的名称和形状
    # model_dict = model.state_dict()
    # missing_keys = []
    # for name, param in pretrain_dict['model'].items():
    #     if name in model_dict:
    #         if model_dict[name].shape != param.shape:
    #             print(f"Shape mismatch for layer: {name}")
    #             print(f"Pretrained shape: {param.shape}, Model shape: {model_dict[name].shape}")
    #             missing_keys.append(name)

    # # 更新字典以仅包含匹配的权重
    # update_dict = {k: v for k, v in pretrain_dict['model'].items() if k in model_dict and model_dict[k].shape == v.shape}

    # # 打印哪些层的参数缺失
    # if missing_keys:
    #     print("\nThe following parameters are missing or have mismatched shapes:")
    #     for key in missing_keys:
    #         print(key)

    model.load_state_dict(update_dict, strict=True)
    print(f"Loaded {len(update_dict)} weights from {path}")
    torch.save(update_dict, save_path)
    return model










# import torch
# import torchvision
# import torch.nn as nn
# from torchvision.transforms.functional import InterpolationMode
# from timm.models.vision_transformer import PatchEmbed, Block
# import numpy as np

# class DepthwiseSeparableConv3D(nn.Module):
#     """3D Depthwise Separable Convolution"""
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
#         super(DepthwiseSeparableConv3D, self).__init__()
#         self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
#         self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x


# class PatchEmbed3D(nn.Module):
#     """Compute 3D patch embeddings for non-cubic volumes."""
#     def __init__(self, img_size=(224, 224, 112), patch_size=(16, 16, 8), in_chans=1, embed_dim=768):
#         super().__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.in_chans = in_chans
#         self.embed_dim = embed_dim

#         # Calculate the number of patches along each dimension
#         self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
#         self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         # self.proj = DepthwiseSeparableConv3D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        

#     def forward(self, x):
#         x = self.proj(x)  # (N, C, D', H', W')
#         x = x.flatten(2)  # (N, C, D'*H'*W')
#         x = x.transpose(1, 2)  # (N, D'*H'*W', C)
#         return x




# class MRM(nn.Module):
#     """Masked Autoencoder with 3D VisionTransformer backbone."""
#     def __init__(self, img_size=(224, 224, 112), patch_size=(16, 16, 8), in_chans=1,
#                  embed_dim=768, depth=12, num_heads=12,
#                  decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#                  mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
#         super().__init__()
        
#         # define value
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.embed_dim = embed_dim
#         self.decoder_embed_dim = decoder_embed_dim

#         #defien function
#         self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, embed_dim)
#         num_patches = self.patch_embed.num_patches
        
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
#         self.blocks = nn.ModuleList([
#             Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
#             for i in range(depth)])
#         self.norm = norm_layer(embed_dim)
        
#         self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
#         self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
#         self.decoder_blocks = nn.ModuleList([
#             Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
#             for i in range(decoder_depth)])
        
#         self.decoder_norm = norm_layer(decoder_embed_dim)
#         self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0]*patch_size[1]*patch_size[2]*in_chans, bias=True)
        
#         self.norm_pix_loss = norm_pix_loss
#         self.initialize_weights()

#     def initialize_weights(self):
#         # Initialize weights as needed, especially the 3D positional embeddings
#         grid_d, grid_h, grid_w = (self.img_size[0] // self.patch_size[0], 
#                                   self.img_size[1] // self.patch_size[1], 
#                                   self.img_size[2] // self.patch_size[2])
#         pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], grid_d, grid_h, grid_w, cls_token=True)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
#         decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_embed_dim, grid_d, grid_h, grid_w, cls_token=True)
#         self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

#         # Other weight initialization as before
#         # Initialize nn.Linear and nn.LayerNorm here as well, similar to previous initialization logic
#           # initialize patch_embed like nn.Linear (instead of nn.Conv2d)

#         w = self.patch_embed.proj.weight.data
#         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

#         # # for DepthwiseSeparableConv3D
#         # for module in self.patch_embed.proj.modules():
#         #     if isinstance(module, nn.Conv3d):
#         #         w = module.weight.data
#         #         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
#         #         if module.bias is not None:
#         #             torch.nn.init.constant_(module.bias, 0)


#         # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#         torch.nn.init.normal_(self.cls_token, std=.02)
#         torch.nn.init.normal_(self.mask_token, std=.02)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def patchify(self, imgs):
#         """
#         imgs: (N, C, D, H, W)
#         x: (N, L, patch_size[0]*patch_size[1]*patch_size[2]*C)
#         """
#         # 获取每个维度的patch大小
#         pd, ph, pw = self.patch_embed.patch_size
#         assert imgs.shape[2] % pd == 0 and imgs.shape[3] % ph == 0 and imgs.shape[4] % pw == 0
        
#         d = imgs.shape[2] // pd
#         h = imgs.shape[3] // ph
#         w = imgs.shape[4] // pw
#         x = imgs.reshape(imgs.shape[0], imgs.shape[1], d, pd, h, ph, w, pw)
#         # x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()  # Reorder dimensions
#         x = torch.einsum('ncdohpwq->ndhwopqc', x)
#         x = x.reshape(imgs.shape[0], d * h * w, pd * ph * pw * imgs.shape[1])  # Flatten
#         return x

#     def unpatchify(self, x):
#         """
#         x: (N, L, patch_size[0]*patch_size[1]*patch_size[2]*C)
#         imgs: (N, C, D, H, W)
#         """
#         pd, ph, pw = self.patch_embed.patch_size
#         N, L, D = x.shape
#         d = h = w = int(L ** (1/3))  # Assuming cubic patches for simplicity
#         assert d * h * w == L, "Number of patches does not match expected volume"
        
#         x = x.reshape(shape=(N, d, h, w, pd, ph, pw, self.in_chans))
#         # x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()  # Reorder dimensions back
#         x = torch.einsum('ndhwopqc->ncdohpwq', x)
#         imgs = x.reshape(N, self.in_chans, d * pd, h * ph, w * pw)  # Reshape back to original dimensions
#         return imgs

#     def random_masking(self, x, mask_ratio):
#         """
#         Perform per-sample random masking by per-sample shuffling.
#         Per-sample shuffling is done by argsort random noise.
#         x: [N, L, D], sequence
#         """
#         N, L, D = x.shape  # batch, length, dim
#         len_keep = int(L * (1 - mask_ratio))
        
#         noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
#         # sort noise for each sample
#         ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
#         ids_restore = torch.argsort(ids_shuffle, dim=1)

#         # keep the first subset
#         ids_keep = ids_shuffle[:, :len_keep]
#         x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

#         # generate the binary mask: 0 is keep, 1 is remove
#         mask = torch.ones([N, L], device=x.device)
#         mask[:, :len_keep] = 0
#         # unshuffle to get the binary mask
#         mask = torch.gather(mask, dim=1, index=ids_restore)

#         return x_masked, mask, ids_restore

#     def image_encoder(self, x, mask_ratio):
#         # embed patches
#         x = self.patch_embed(x)

#         # add pos embed w/o cls token
#         x = x + self.pos_embed[:, 1:, :]

#         # masking: length -> length * mask_ratio
#         # x, mask, ids_restore = self.random_masking(x, mask_ratio)

#         # append cls token
#         cls_token = self.cls_token + self.pos_embed[:, :1, :]
#         cls_tokens = cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)

#         # apply Transformer blocks
#         for blk in self.blocks:
#             x = blk(x)
#         x = self.norm(x)

#         return x
    
#     def forward_decoder(self, x, ids_restore):
#         # embed tokens
#         x = self.decoder_embed(x)

#         # append mask tokens to sequence
#         mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
#         x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
#         x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
#         x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

#         # add pos embed
#         x = x + self.decoder_pos_embed

#         # apply Transformer blocks
#         for blk in self.decoder_blocks:
#             x = blk(x)
#         x = self.decoder_norm(x)

#         # predictor projection
#         x = self.decoder_pred(x)

#         # remove cls token
#         x = x[:, 1:, :]

#         return x
    
#     def forward_loss(self, imgs, pred, mask):
#         """
#         imgs: [N, 3, H, W]
#         pred: [N, L, p*p*3]
#         mask: [N, L], 0 is keep, 1 is remove, 
#         """
#         target = self.patchify(imgs)
#         if self.norm_pix_loss:
#             mean = target.mean(dim=-1, keepdim=True)
#             var = target.var(dim=-1, keepdim=True)
#             target = (target - mean) / (var + 1.e-6)**.5

#         loss = (pred - target) ** 2
#         loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

#         loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
#         return loss

#     def forward(self, imgs, mask_ratio=0.75):
#         # imgs = imgs.cuda()

#         latent = self.image_encoder(imgs, mask_ratio)

#         # pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
#         # loss = self.forward_loss(imgs, pred, mask)

#         return latent


# def mrm_vit_b16(**kwargs):
#     model = MRM(
#         img_size=(224, 224, 112), patch_size=(16, 16, 8), in_chans=1, embed_dim=1536, depth=12, num_heads=12,
#         decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model



# # 定义模型配置
# config = {
#     'in_channels': 1,
#     'img_size': (224, 224, 112),
#     'patch_size': (16, 16, 8),
#     'hidden_size': 768,   # 1536
#     'mlp_dim': 3072,   # 6144
#     'num_layers': 12,
#     'num_heads': 12,
#     'pos_embed': 'conv',
#     'classification': False,
#     'num_classes': 2,
#     'dropout_rate': 0.0,
#     'spatial_dims': 3,
#     'post_activation': 'Tanh',
#     'qkv_bias': True,
#     'save_attn': False,
# }

# model = ViT3D(**config)


# pretrain_dict = torch.load('/path/to/vision_model_weights.pth', map_location=torch.device('cpu'))
# model.load_state_dict(pretrain_dict, strict=True)
# write new weight
# model = load_weight(model, "/path/to/input/checkpoint.pth", '/path/to/output/vit_b16_3D_epoch_116.pth')

# M3AE
# model = load_weight(model, "/path/to/m3ae/checkpoint.ckpt", '/path/to/output/vit_b16_3D_m3ae_epoch_3.pth')

# model1 = mrm_vit_b16().cuda()
# model1 = load_weight(model1, "/path/to/input/checkpoint.pth", '/path/to/output/vit_b16_3D_embedding1536_epoch_442.pth')


# x = torch.ones([2, 1, 224, 224, 112]).cuda()




# x1 = model(x)
# x2 = model1(x)




