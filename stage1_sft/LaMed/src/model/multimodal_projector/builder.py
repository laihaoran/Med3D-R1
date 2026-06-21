from torch import nn
from .spatial_pooling_projector import SpatialPoolingProjector
from .spatial_conv_projector import SpatialConvProjector
from .spatial_multi_conv_projector import SpatialConvLinearProjector
from .spatial_qformer_projector import SpatialConvLinearQformerProjector
from .qformer_projector import CrossAttentionProjector
from .mixmlp_projector import MLPMixerProjector
from .pooling_projector import PoolingProjector
from .smart_projector import SmartProjector
from .spatial_conv_view_projector import SpatialConvLinearReshapeProjector
from .adapt_qformer_projector import AdapterCrossAttentionProjector
from .self_attention_block import SelfAttentionLayer
from .twice_attention import TwiceCrossAttentionProjector

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, *args, **kwargs):
        return x
    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class Minigpt(nn.Module):
    def __init__(self, config=None):
        super(Minigpt, self).__init__()
        # c*4 is the input size, and c is the output size for the linear layer
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.linear = nn.Linear(inc * 4, ouc)

    def forward(self, x):
        # x is the input tensor with shape [b, num_tokens, c]
        b, num_tokens, c = x.shape

        # Check if num_tokens is divisible by 4
        if num_tokens % 4 != 0:
            raise ValueError("num_tokens must be divisible by 4")

        # Reshape x to [b, num_tokens/4, c*4]
        x = x.view(b, num_tokens // 4, c * 4)

        # Apply the linear transformation
        x = self.linear(x)
        return x


class Vanilla(nn.Module):
    def __init__(self, config=None):
        super(Vanilla, self).__init__()
        # c*4 is the input size, and c is the output size for the linear layer
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.linear = nn.Linear(inc * 4, ouc)

    def forward(self, x):
        b, num_tokens, c = x.shape

        # Check if num_tokens is divisible by 4
        if num_tokens % 4 != 0:
            raise ValueError("num_tokens must be divisible by 4")

        # First, reshape to [b, num_tokens//4, 4, c]
        x = x.view(b, num_tokens // 4, 4, c)

        # Then, permute to interleave the tokens
        x = x.permute(0, 1, 3, 2).contiguous()

        # Finally, reshape to [b, num_tokens//4, c*4] to interleave features of 4 tokens
        x = x.view(b, num_tokens // 4, c * 4)

        # Apply the linear transformation
        x = self.linear(x)
        return x





class FullLinear(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type, layer_num):
        super(FullLinear, self).__init__()
        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]

        if layer_type == 'linear':
            self.projector = nn.Linear(in_dim, out_dim)
        elif layer_type == 'mlp':
            depth = int(layer_num)
            modules = [nn.Linear(in_dim, out_dim // 2)]
            modules.append(nn.LayerNorm(out_dim // 2))
            for _ in range(1, depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(out_dim // 2, out_dim // 2))
                modules.append(nn.LayerNorm(out_dim // 2))
            modules.append(nn.GELU())
            modules.append(nn.Linear(out_dim // 2, out_dim))
            modules.append(nn.LayerNorm(out_dim))
            self.projector = nn.Sequential(*modules)

    def forward(self, x):
        x = self.projector(x)
        return x
    
    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_pre:
            num *= n
        return num


def build_mm_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type')

    if projector_type == 'linear':
        return FullLinear(image_size=config.image_size,
                         patch_size=config.patch_size,
                         in_dim=config.mm_hidden_size,
                        out_dim=config.hidden_size,
                        layer_type=config.proj_layer_type,
                        layer_num=config.proj_layer_num)

    elif projector_type == 'spp':
        return SpatialPoolingProjector(image_size=config.image_size,
                                        patch_size=config.patch_size,
                                        in_dim=config.mm_hidden_size,
                                        out_dim=config.hidden_size,
                                        layer_type=config.proj_layer_type,
                                        layer_num=config.proj_layer_num,
                                        pooling_type=config.proj_pooling_type,
                                        pooling_size=config.proj_pooling_size)
    elif projector_type == 'conv':
        return SpatialConvProjector(image_size=config.image_size,
                                        patch_size=config.patch_size,
                                        in_dim=config.mm_hidden_size,
                                        out_dim=config.hidden_size,
                                        pooling_size=config.proj_pooling_size)
    elif projector_type == 'convlinear':
        return SpatialConvLinearProjector(image_size=config.image_size,
                                        patch_size=config.patch_size,
                                        in_dim=config.mm_hidden_size,
                                        out_dim=config.hidden_size,
                                        layer_type=config.proj_layer_type,
                                        layer_num=config.proj_layer_num,
                                        pooling_size=config.proj_pooling_size)
    elif projector_type == 'convreshape':
        return SpatialConvLinearReshapeProjector(image_size=config.image_size,
                                        patch_size=config.patch_size,
                                        in_dim=config.mm_hidden_size,
                                        out_dim=config.hidden_size,
                                        layer_type=config.proj_layer_type,
                                        layer_num=config.proj_layer_num,
                                        pooling_size=config.proj_pooling_size,
                                        mean_prompt_template_path=getattr(config, "mean_prompt_template_path", None))
    elif projector_type == 'convqformer':
        return SpatialConvLinearQformerProjector(image_size=config.image_size,
                                        patch_size=config.patch_size,
                                        in_dim=config.mm_hidden_size,
                                        out_dim=config.hidden_size,
                                        layer_type=config.proj_layer_type,
                                        layer_num=config.proj_layer_num,
                                        pooling_size=config.proj_pooling_size)
    elif projector_type == 'qformer':
        return CrossAttentionProjector(image_size=config.image_size,
                                        patch_size=config.patch_size,
                                        in_dim=config.mm_hidden_size,
                                        out_dim=config.hidden_size,
                                        layer_type=config.proj_layer_type,
                                        layer_num=config.proj_layer_num,
                                        pooling_size=config.proj_pooling_size)
    elif projector_type == 'adapter_qformer':
        return AdapterCrossAttentionProjector(image_size=config.image_size,
                                        patch_size=config.patch_size,
                                        in_dim=config.mm_hidden_size,
                                        out_dim=config.hidden_size,
                                        layer_type=config.proj_layer_type,
                                        layer_num=config.proj_layer_num,
                                        pooling_size=config.proj_pooling_size)
    elif projector_type == 'twice_qformer':
        return TwiceCrossAttentionProjector(image_size=config.image_size,
                                        patch_size=config.patch_size,
                                        in_dim=config.mm_hidden_size,
                                        out_dim=config.hidden_size,
                                        layer_type=config.proj_layer_type,
                                        layer_num=config.proj_layer_num,
                                        pooling_size=config.proj_pooling_size)

    elif projector_type == 'self_attention':
        return SelfAttentionLayer(image_size=config.image_size,
                                 patch_size=config.patch_size,
                                 in_dim=config.mm_hidden_size,
                                 out_dim=config.hidden_size)
    elif projector_type == 'mixmlp':
        return MLPMixerProjector(image_size=config.image_size,
                                        patch_size=config.patch_size,
                                        in_dim=config.mm_hidden_size,
                                        out_dim=config.hidden_size,
                                        layer_num=config.proj_layer_num,
                                        pooling_size=config.proj_pooling_size)
    elif projector_type == 'pooling':
        print("building pooling projector")
        return PoolingProjector(image_size=config.image_size,
                                        patch_size=config.patch_size,
                                        in_dim=config.mm_hidden_size,
                                        out_dim=config.hidden_size,
                                        layer_type=config.proj_layer_type,
                                        layer_num=config.proj_layer_num,
                                        pooling_type=config.proj_pooling_type,
                                        pooling_size=config.proj_pooling_size)
    elif projector_type == 'Smart':
        print("building Smart projector")
        return SmartProjector(image_size=config.image_size,  
                               patch_size=config.patch_size, 
                               image_dim=config.mm_hidden_size, 
                               hidden_dim=config.hidden_size, 
                               pooling_size=config.proj_pooling_size,
                                topk=config.set_proj_num,
                                use_random=config.use_random,
                                random_num=config.random_num,
                                decoder_block_path=getattr(config, "decoder_block_path", None),)

    elif projector_type == 'identity':
        return IdentityMap()
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')
