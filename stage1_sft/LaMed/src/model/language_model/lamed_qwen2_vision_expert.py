from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..lamed_arch import LamedMetaModel, LamedMetaForCausalLM
from .mixin import LlamaVisionExpertFCMixin, LlamaVisionExpertAttnMixin


class LamedConfig(Qwen2Config):
    model_type = "lamed_qwen2"

class LamedQwen2Model(LamedMetaModel, Qwen2Model):
    config_class = LamedConfig
    def __init__(self, config: LamedConfig):
        super(LamedQwen2Model, self).__init__(config)
        


class LamedQwen2VisionExpertForCausalLM(LamedMetaForCausalLM, Qwen2ForCausalLM):
    config_class = LamedConfig

    def __init__(self, config: LamedConfig):
        # Qwen的初始化流程修正
        super(Qwen2ForCausalLM, self).__init__(config)

        config._attn_implementation = "flash_attention_2"
        self.rec_enable = False
        self.model = LamedQwen2Model(config)

        # 获取 transformer 层结构（假设就是 model.layers）
        transformer_layers = self.model  # 这是一个 ModuleList


        vision_layer_range = getattr(config, "vision_layer_range", None)
        # 注册专家模块
        self.vision_fc_expert = LlamaVisionExpertFCMixin(
            in_features=config.hidden_size,
            hidden_features=config.intermediate_size,
            num_layers=config.num_hidden_layers,
            num_vision_layers=config.num_hidden_layers,
            vision_layer_range=vision_layer_range,
            params_dtype=torch.float,
            transformer_layers=transformer_layers,  # 传入 transformer 层结构
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        self.vision_attn_expert = LlamaVisionExpertAttnMixin(
            hidden_size=config.hidden_size,
            # num_heads=config.num_attention_heads,
            num_layers=config.num_hidden_layers,
            num_vision_layers=config.num_hidden_layers,
            use_vision_expert=True,
            vision_layer_range=vision_layer_range,
            params_dtype=torch.float,
            transformer_layers=transformer_layers,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        # # 加入视觉专家模块
        # vision_layer_range = getattr(config, "vision_layer_range", None)
        # self.model.add_mixin(
        #     "mlp",
        #     LlamaVisionExpertFCMixin(
        #         in_features=config.hidden_size,
        #         hidden_features=config.intermediate_size,
        #         num_layers=config.num_hidden_layers,
        #         num_vision_layers=config.num_hidden_layers,  # 或者部分层
        #         vision_layer_range=vision_layer_range,  # 可选，控制哪些层启用
        #         params_dtype=torch.float,
        #         device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     )
        # )
        # # 加入旋转位置编码的视觉专家模块
        # self.model.add_mixin(
        #     "rotary",
        #     LlamaVisionExpertAttnMixin(
        #         hidden_size=config.hidden_size,
        #         num_heads=config.num_attention_heads,
        #         num_layers=config.num_hidden_layers,
        #         num_vision_layers=config.num_hidden_layers,
        #         use_vision_expert=True,
        #         vision_layer_range=vision_layer_range,  # 如 [0, 1, 2, ..., 11]
        #         params_dtype=torch.float,
        #         device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     )
        # )

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        images: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        segs: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_weights: Optional[torch.LongTensor] = None,
        **kwargs  # 兼容Qwen的额外参数
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        input_ids_pre = input_ids
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                vision_expert_mask
            ) = self.prepare_inputs_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
            )
        try:
            seg_ids = torch.nonzero(torch.sum(segs, dim=(1, 2, 3, 4))).flatten().tolist()
        except:
            seg_ids = []

        if self.get_model().seg_enable and seg_ids:
            outputs = super().forward(
                                    input_ids=input_ids,
                                    inputs_embeds=inputs_embeds,
                                    attention_mask=attention_mask,
                                    labels=labels,
                                    output_hidden_states=True,
                                    position_ids=position_ids,
                                    past_key_values=past_key_values,
                                    use_cache=use_cache,
                                    output_attentions=output_attentions,
                                    return_dict=return_dict
                                )

            output_hidden_states = outputs.hidden_states

            last_hidden_state = output_hidden_states[-1]

            seg_token_mask = input_ids_pre[:, 1:] == self.config.seg_token_id
            seg_token_mask = torch.cat(
                [
                    seg_token_mask,
                    torch.zeros((seg_token_mask.shape[0], 1), dtype=seg_token_mask.dtype).cuda(),
                ],
                dim=1,
            )

            seg_prompts = []
            for i in seg_ids:
                if torch.sum(seg_token_mask[i]) == 1:
                    seg_token = last_hidden_state[i][seg_token_mask[i]]
                    seg_prompt = self.get_model().seg_projector(seg_token)
                elif torch.sum(seg_token_mask[i]) > 1:
                    seg_tokens = last_hidden_state[i][seg_token_mask[i]]
                    seg_token = torch.mean(seg_tokens, dim=0, keepdim=True)
                    seg_prompt = self.get_model().seg_projector(seg_token)
                else:
                    seg_prompt = torch.zeros([1, self.config.mm_hidden_size], dtype=last_hidden_state.dtype,
                                             device=last_hidden_state.device)
                seg_prompts.append(seg_prompt)

            seg_prompts = torch.cat(seg_prompts, dim=0)
            logits = self.get_model().seg_module(images[seg_ids], text_emb=seg_prompts)
            loss_dice = self.get_model().dice_loss(logits, segs[seg_ids])
            loss_bce = self.get_model().bce_loss(logits, segs[seg_ids])
            seg_loss = loss_dice + loss_bce
            outputs.loss = outputs.loss + seg_loss
            return outputs
        elif self.rec_enable:
            outputs = super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                                )

            image_features = image_features.view(-1, image_features.size(-1))  # (B*N, D)
              # 计算特征与字典的相似度
            similarity = torch.matmul(image_features, self.get_model().embed_tokens.weight.T)  # (B*N, dict_size)
            temperature = 0.07  # 可以调整
            similarity = similarity / temperature
            # 计算 softmax 权重
            weights = F.softmax(similarity, dim=-1)  # (B, N, dict_size)

            # 加权求和，进行特征重建
            reconstructed = torch.matmul(weights, self.get_model().embed_tokens.weight)  # (B*N, D)

            # rec_loss = F.mse_loss(reconstructed, image_features)
            mae_rec_loss = F.l1_loss(reconstructed, image_features)
            outputs.loss = outputs.loss + mae_rec_loss
            return outputs
        else:
            hidden_states = inputs_embeds

            # # 如果使用 expert，就注册一次新的 forward 函数
            # if vision_expert_mask is not None and self.vision_fc_expert is not None:
            #     for i in range(self.config.num_hidden_layers):
            #         self.model.layers[i].mlp.forward = lambda hidden_states, **inner_kwargs: \
            #             self.vision_fc_expert.mlp_forward(hidden_states, layer_id=i, vision_expert_mask=vision_expert_mask)
            #         print(i, " vision_fc_expert forward registered")

            #     # 注册专家前向（只在有 mask 时才执行）
            # if vision_expert_mask is not None and self.vision_fc_expert is not None:
            #     def make_mlp_forward(expert_module, layer_id, vision_expert_mask):
            #         def new_forward(hidden_states, **inner_kwargs):
            #             return expert_module.mlp_forward(
            #                 hidden_states,
            #                 layer_id=layer_id,
            #                 vision_expert_mask=vision_expert_mask,
            #                 **inner_kwargs
            #             )
            #         return new_forward

            def make_mlp_forward(expert_module, layer_id, vision_expert_mask):
                def new_forward(hidden_states, **kwargs):
                    return expert_module.mlp_forward(
                        hidden_states,
                        layer_id=layer_id,
                        vision_expert_mask=vision_expert_mask,
                        **kwargs
                    )
                return new_forward
            
            if vision_expert_mask is not None:
                if self.vision_fc_expert is not None:
                    for i in range(self.config.num_hidden_layers):
                        self.model.layers[i].mlp.forward = make_mlp_forward(
                            self.vision_fc_expert, i, vision_expert_mask
                        )

                if self.vision_attn_expert is not None:
                    for i in range(self.config.num_hidden_layers):
                        self.model.layers[i].self_attn.forward = make_attn_forward(
                            self.vision_attn_expert, i, vision_expert_mask
                        )




            #     for i in range(self.config.num_hidden_layers):
            #         block = self.model.layers[i]  # 有些模型是 self.model.model.layers[i]
            #         block.mlp.forward = make_mlp_forward(self.vision_fc_expert, i, vision_expert_mask)


            if vision_expert_mask is not None and self.vision_attn_expert is not None:
                for i in range(self.config.num_hidden_layers):
                    self.model.layers[i].self_attn.forward = lambda hidden_states, attention_mask, **inner_kwargs: \
                        self.vision_attn_expert.attention_forward(
                            hidden_states, attention_mask,
                            layer_id=i,
                            vision_expert_mask=vision_expert_mask,
                            **inner_kwargs
                        )



            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            if labels is not None and token_weights is not None:
                # shift logits and labels for causal LM loss
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_weights = token_weights[..., 1:].contiguous()  # 对齐 labels

                # Flatten
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1))  # (B*L,)
                # Apply weights
                weighted_loss = (loss * shift_weights.view(-1)).sum() / shift_weights.view(-1).sum()
                outputs.loss = weighted_loss
            return outputs


    @torch.no_grad()
    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        seg_enable: bool = False,
        vision_expert_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor, Any]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                vision_expert_mask
            ) = self.prepare_inputs_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if seg_enable:
            outputs = super().generate(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs
            )

            output_hidden_states = outputs.hidden_states
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.config.seg_token_id

            last_tensors = [tuple[-1] for tuple in output_hidden_states]
            last_hidden_state = torch.cat(last_tensors[1:], dim=1)

            seg_prompts = []
            noseg_ids = []
            for i in range(len(seg_token_mask)):
                if torch.sum(seg_token_mask[i]) == 1:
                    seg_token = last_hidden_state[i][seg_token_mask[i]]
                    seg_prompt = self.get_model().seg_projector(seg_token)
                elif torch.sum(seg_token_mask[i]) > 1:
                    seg_tokens = last_hidden_state[i][seg_token_mask[i]]
                    seg_token = torch.mean(seg_tokens, dim=0, keepdim=True)
                    seg_prompt = self.get_model().seg_projector(seg_token)
                else:
                    noseg_ids.append(i)
                    seg_prompt = torch.zeros([1, self.config.mm_hidden_size], dtype=last_hidden_state.dtype,
                                             device=last_hidden_state.device)
                seg_prompts.append(seg_prompt)

            seg_prompts = torch.cat(seg_prompts, dim=0)
            logits = self.get_model().seg_module(images, seg_prompts)
            logits[noseg_ids] = -torch.inf

            return output_ids, logits
        else:
            output_ids = super().generate(
                inputs_embeds=inputs_embeds,
                vision_expert_mask=vision_expert_mask,
                **kwargs
            )
            return output_ids


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        return inputs



AutoConfig.register("lamed_qwen2", LamedConfig)
AutoModelForCausalLM.register(LamedConfig, LamedQwen2VisionExpertForCausalLM)