from .vit import ViT3DTower
from .transformer_backbone import ViT3DMAETower

def build_vision_tower(config, **kwargs):
    vision_tower = getattr(config, 'vision_tower', None)
    if 'vit3d' in vision_tower.lower():
        return ViT3DTower(config, **kwargs)
    elif 'vitb3d' in vision_tower.lower():
        return ViT3DMAETower(config, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')