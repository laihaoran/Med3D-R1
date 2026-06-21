from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_mm_projector
from .segmentation_module.builder import build_segmentation_module
from LaMed.src.model.loss import BCELoss, BinaryDiceLoss
# import deepspeed

import torch.nn.functional as F

def resize_pos_embed_3d(pos_embed, old_grid, new_grid, cls_token=True):
    """
    插值3D位置编码，从旧的grid上采样到新的grid。
    - pos_embed: torch.Tensor, shape [1, N+1, C]（含cls_token）
    - old_grid: tuple[int], e.g., (14, 14, 14)
    - new_grid: tuple[int], e.g., (28, 28, 28)
    """
    if cls_token:
        cls_tok, pos = pos_embed[:, :1, :], pos_embed[:, 1:, :]
    else:
        cls_tok, pos = None, pos_embed

    D_old, H_old, W_old = old_grid
    D_new, H_new, W_new = new_grid
    C = pos.shape[-1]

    pos = pos.reshape(1, D_old, H_old, W_old, C).permute(0, 4, 1, 2, 3)  # (1, C, D, H, W)
    pos_resized = F.interpolate(pos, size=(D_new, H_new, W_new), mode='trilinear', align_corners=False)
    pos_resized = pos_resized.permute(0, 2, 3, 4, 1).reshape(1, -1, C)

    if cls_token:
        return torch.cat((cls_tok, pos_resized), dim=1)
    else:
        return pos_resized



class LamedMetaModel:
    def __init__(self, config):
        super(LamedMetaModel, self).__init__(config)

        self.config = config
        self.seg_enable = False

        if hasattr(config, "vision_tower"):
            self.vision_tower = build_vision_tower(config)
            self.mm_projector = build_mm_projector(config)

        if hasattr(config, "segmentation_module") and config.segmentation_module is not None:
            self.seg_enable = True
            self.seg_module = build_segmentation_module(config)

            self.seg_projector = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_size, config.mm_hidden_size),
                nn.Dropout(0.1),
            )

            self.dice_loss = BinaryDiceLoss()
            self.bce_loss = BCELoss()

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        return vision_tower

    def initialize_vision_modules(self, model_args):
        self.config.image_channel = model_args.image_channel
        self.config.image_size = model_args.image_size
        self.config.patch_size = model_args.patch_size

        self.config.vision_tower = model_args.vision_tower
        self.config.vision_select_layer = model_args.vision_select_layer
        self.config.vision_select_feature = model_args.vision_select_feature
        self.config.img_token_id = model_args.img_token_id

        self.config.mm_projector_type = model_args.mm_projector_type
        self.config.proj_layer_type = model_args.proj_layer_type
        self.config.proj_layer_num = model_args.proj_layer_num
        self.config.proj_pooling_type = model_args.proj_pooling_type
        self.config.proj_pooling_size = model_args.proj_pooling_size
        self.config.set_proj_num = model_args.set_proj_num
        self.config.use_random = model_args.use_random
        self.config.random_num = model_args.random_num
        self.config.mean_prompt_template_path = model_args.mean_prompt_template_path

        # vision tower
        if self.get_vision_tower() is None:
            self.vision_tower = build_vision_tower(self.config)
            # If you have a more robust vision encoder, try freezing the vision tower by requires_grad_(False)
            self.vision_tower.requires_grad_(not model_args.freeze_vision_tower)


        if model_args.pretrain_vision_model is not None:
            vision_model_weights = torch.load(model_args.pretrain_vision_model, map_location='cpu')
            self.vision_tower.vision_tower.load_state_dict(vision_model_weights, strict=True)
            # print(f"[Info] Loading vision model from: {model_args.pretrain_vision_model}")
            # vision_model_weights = torch.load(model_args.pretrain_vision_model, map_location='cpu')

            # model_state_dict = self.vision_tower.vision_tower.state_dict()
            # new_state_dict = {}

            # for key in model_state_dict:
            #     if key in vision_model_weights:
            #         if vision_model_weights[key].shape == model_state_dict[key].shape:
            #             new_state_dict[key] = vision_model_weights[key]
            #         elif "pos_embed" in key:
            #             print(f"[Warning] Resizing pos_embed from {vision_model_weights[key].shape} to {model_state_dict[key].shape}...")
            #             # 自动上采样位置编码
            #             old_pos = vision_model_weights[key]
            #             new_pos = model_state_dict[key]
            #             has_cls_token = old_pos.shape[1] == model_state_dict[key].shape[1]  # usually True
            #             num_old = old_pos.shape[1] - int(has_cls_token)
            #             num_new = new_pos.shape[1] - int(has_cls_token)

            #             # 估算旧grid与新grid
            #             D_old = H_old = W_old = int(round(num_old ** (1 / 3)))
            #             D_new = H_new = W_new = int(round(num_new ** (1 / 3)))

            #             resized = resize_pos_embed_3d(old_pos, (D_old, H_old, W_old), (D_new, H_new, W_new), cls_token=has_cls_token)
            #             new_state_dict[key] = resized
            #         else:
            #             print(f"[Warning] Shape mismatch for {key}, skipping.")
            #     else:
            #         print(f"[Warning] {key} not found in checkpoint.")

            # msg = self.vision_tower.vision_tower.load_state_dict(new_state_dict, strict=False)
            # print(f"[Info] Vision model loaded with message: {msg}")



        self.config.mm_hidden_size = self.vision_tower.hidden_size

        # mm_projector
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_mm_projector(self.config)

        if model_args.pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=True)

    def initialize_seg_modules(self, model_args):
        self.config.segmentation_module = model_args.segmentation_module

        # segmentation_module
        if getattr(self, 'seg_module', None) is None:
            self.seg_module = build_segmentation_module(self.config)
            self.seg_projector = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.hidden_size, self.config.mm_hidden_size),
                nn.Dropout(0.1),
            )
            self.seg_enable = True

        if model_args.pretrain_seg_module is not None:
            seg_module_weights = torch.load(model_args.pretrain_seg_module, map_location='cpu')
            new_state_dict = {}
            for key, value in seg_module_weights.items():
                if key.startswith('model.text_encoder.') or key.startswith('text_encoder.'):
                    continue
                if key.startswith('model.'):
                    new_key = key[len('model.'):]
                    new_state_dict[new_key] = value
            self.seg_module.load_state_dict(new_state_dict, strict=True)

        self.dice_loss = BinaryDiceLoss()
        self.bce_loss = BCELoss()


class LamedMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    # modify the for change the tokens used in llm
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        else:
            image_features = self.encode_images(images)
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            # image_features = self.get_model().mm_projector(image_features, inputs_embeds, labels, attention_mask)

            # cat image features to input embeddings in simple format
            # inputs_embeds = torch.cat(
            #     (inputs_embeds[:, :1, :], image_features, inputs_embeds[:, (image_features.shape[1] + 1):, :]), dim=1)
            
            image_token_id = getattr(self.config, "img_token_id", None)
            if image_token_id is None:
                raise ValueError("config.img_token_id is not set. Initialize the tokenizer before training.")
            image_idx = torch.where(input_ids == image_token_id)
            if image_idx[0].numel() == 0:
                raise ValueError(f"Image token id {image_token_id} was not found in input_ids.")
            idx = image_idx[1][0].item()
            # cat image features to input embeddings in usr/n<im_start><im_patch><im_end>
            inputs_embeds = torch.cat(
                (inputs_embeds[:, : idx , :], image_features, inputs_embeds[:, (image_features.shape[1] + idx):, :]), dim=1)
            # T = image_features.shape[1]  # 图像 token 长度
            # B, L = input_ids.shape

            # vision_expert_mask = torch.zeros((B, L), dtype=torch.bool, device=input_ids.device)

            # img_idx = 0
            # for b in range(B):
            #     image_idx = torch.where(input_ids == 151665)
            #     idx = image_idx[1][0].item()
            #     inputs_embeds = torch.cat(
            #     (inputs_embeds[:, : idx , :], image_features, inputs_embeds[:, (image_features.shape[1] + idx):, :]), dim=1)
            #     vision_expert_mask[b, idx : idx + T] = True
            #     img_idx += 1

        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = model_args.num_new_tokens

        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
                # self.get_input_embeddings().weight.requires_grad = True
                # self.get_output_embeddings().weight.requires_grad = False
            else:
                # we add 4 new tokens
                # if new tokens need input, please train input_embeddings
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                # if new tokens need predict, please train output_embeddings
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings = embed_tokens_weight
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
            
    # def initialize_vision_tokenizer(self, model_args, tokenizer):
    #     num_new_tokens = model_args.num_new_tokens

    #     # 调整token嵌入层
    #     self.resize_token_embeddings(len(tokenizer))
    #     if num_new_tokens > 0:
    #         input_embeddings = self.get_input_embeddings().weight.data
    #         output_embeddings = self.get_output_embeddings().weight.data

    #         # 初始化新 token 的 embedding 为平均值
    #         input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    #         output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

    #         input_embeddings[-num_new_tokens:] = input_embeddings_avg
    #         output_embeddings[-num_new_tokens:] = output_embeddings_avg

    #         # 根据参数选择是否训练 embedding 和 lm_head
    #         if model_args.tune_mm_mlp_adapter:
    #             for p in self.get_input_embeddings().parameters():
    #                 p.requires_grad = True
    #             for p in self.get_output_embeddings().parameters():
    #                 p.requires_grad = False                        # never train output_embeddings
    #         else:
    #             # 允许训练所有相关参数
    #             for p in self.get_input_embeddings().parameters():
    #                 p.requires_grad = True
    #             for p in self.get_output_embeddings().parameters():
    #                 p.requires_grad = False

    #     # 加载多模态 MLP 适配器的权重
    #     if model_args.pretrain_mm_mlp_adapter:
    #         mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
    #         embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
    #         output_embed_tokens_weight = mm_projector_weights['lm_head.weight']

    #         if input_embeddings.shape == embed_tokens_weight.shape:
    #             input_embeddings = embed_tokens_weight
    #         elif embed_tokens_weight.shape[0] == num_new_tokens:
    #             input_embeddings[-num_new_tokens:] = embed_tokens_weight
    #         else:
    #             raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Number of new tokens: {num_new_tokens}.")
            
    #         if output_embeddings.shape == output_embed_tokens_weight.shape:
    #             output_embeddings = output_embed_tokens_weight
    #         elif output_embed_tokens_weight.shape[0] == num_new_tokens:
    #             output_embeddings[-num_new_tokens:] = output_embed_tokens_weight
    #         else:
    #             raise ValueError(f"Unexpected output_embed_tokens_weight shape. Pretrained: {output_embed_tokens_weight.shape}. Current: {output_embeddings.shape}. Number of new tokens: {num_new_tokens}.")


class LamedMetaWoVisionModel:
    def __init__(self, config):
        super(LamedMetaWoVisionModel, self).__init__(config)
        self.config = config
        if hasattr(config, "vision_tower"):
            self.mm_projector = build_mm_projector(config)


    def get_mm_projector(self):
        return self.mm_projector

    def initialize_mm_projector(self, model_args):
        self.config.image_channel = model_args.image_channel
        self.config.image_size = model_args.image_size
        self.config.patch_size = model_args.patch_size

        self.config.vision_tower = model_args.vision_tower
        self.config.vision_select_layer = model_args.vision_select_layer
        self.config.vision_select_feature = model_args.vision_select_feature

        self.config.mm_projector_type = model_args.mm_projector_type
        self.config.proj_layer_type = model_args.proj_layer_type
        self.config.proj_layer_num = model_args.proj_layer_num
        self.config.proj_pooling_type = model_args.proj_pooling_type
        self.config.proj_pooling_size = model_args.proj_pooling_size
    

        self.config.mm_projector_type = model_args.mm_projector_type
        self.config.proj_layer_type = model_args.proj_layer_type
        self.config.proj_layer_num = model_args.proj_layer_num
        self.config.proj_pooling_type = model_args.proj_pooling_type
        self.config.proj_pooling_size = model_args.proj_pooling_size

        # self.config.mm_hidden_size = 1536

        # mm_projector
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_mm_projector(self.config)

        if model_args.pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=True)


class LamedMetaWoVisionForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_mm_projector(self):
        return self.get_model().get_mm_projector()

    def prepare_inputs_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels
    ):
        # Placeholder for multimodal input processing without vision component
        inputs_embeds = self.get_model().embed_tokens(input_ids)
        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = model_args.num_new_tokens

        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():   # 本质上是Lm_head层， 做adapter的时候，我们使用原始的lm_head，所以不具备生成新token的能力。
                    p.requires_grad = False
            else:
                # we add 4 new tokens
                # if new tokens need input, please train input_embeddings
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                # if new tokens need predict, please train output_embeddings
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']

            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings = embed_tokens_weight
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
            


        num_new_tokens = model_args.num_new_tokens

        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():   # do not modify output_embeddings in tune_mm_mlp_adapter, maybe we can train it 
                    p.requires_grad = False
            else:
                # Enable training for new tokens
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']

            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings = embed_tokens_weight
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Number of new tokens: {num_new_tokens}.")
