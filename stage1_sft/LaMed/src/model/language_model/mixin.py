import os
import sys
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sat.transformer_defaults import attention_fn_default
from sat.model.base_model import BaseModel, BaseMixin, non_conflict
from sat.model import ChatGLMModel
from sat.mpu.layers import ColumnParallelLinear, RowParallelLinear
from sat.mpu.utils import split_tensor_along_last_dim, gelu
from sat import mpu
import torch.distributed as dist

# 确保torch.distributed已经初始化（比如你用 torchrun 启动的话）
if not dist.is_initialized():
    dist.init_process_group(backend='nccl')  # 或 'gloo'，根据实际环境

# 然后初始化模型并行组
mpu.initialize_model_parallel(
    model_parallel_size_=1,  # 如果不并行就设为1
)

class LlamaVisionExpertFCMixin(BaseMixin):
    def __init__(self, in_features, hidden_features, num_layers=32, num_vision_layers=0, vision_layer_range=None,
                 params_dtype=torch.float, transformer_layers=None, device=torch.device('cpu')):
        super().__init__()

        self.num_layers = num_layers
        self.num_vision_layers = num_vision_layers
        if vision_layer_range is None:
            vision_layer_range = [i for i in range(min(num_vision_layers, num_layers))]
        self.transformer=transformer_layers
        self.vision_layer_range = vision_layer_range
        self.gate_proj = nn.ModuleList([ColumnParallelLinear(
            in_features,
            hidden_features,
            gather_output=False,
            init_method=None,
            bias=False,
            params_dtype=params_dtype,
            module=self,
            name="dense_h_to_4h_gate",
            skip_init=True,
            device=device
        ) for i in range(num_layers)])
        # Trainable vision expert parameters
        vision_dense_h_to_4h_list = []
        vision_dense_4h_to_h_list = []
        gate_proj_list = []


        for i in vision_layer_range:
            vision_dense_h_to_4h = ColumnParallelLinear(
                in_features,
                hidden_features,
                gather_output=False,
                init_method=None,
                bias=False,
                params_dtype=params_dtype,
                module=self,
                name="vision_dense_h_to_4h",
                skip_init=True,
                device=device
            )

            # Project back to h.
            vision_dense_4h_to_h = RowParallelLinear(
                hidden_features,
                in_features,
                input_is_parallel=True,
                init_method=None,
                bias=False,
                params_dtype=params_dtype,
                module=self,
                name="vision_dense_4h_to_h",
                skip_init=True,
                device=device
            )

            gate_proj = ColumnParallelLinear(
                in_features,
                hidden_features,
                gather_output=False,
                init_method=None,
                bias=False,
                params_dtype=params_dtype,
                module=self,
                name="vision_gate_proj",
                skip_init=True,
                device=device
            )

            vision_dense_h_to_4h_list.append(vision_dense_h_to_4h)
            vision_dense_4h_to_h_list.append(vision_dense_4h_to_h)
            gate_proj_list.append(gate_proj)

        self.vision_dense_h_to_4h_list = nn.ModuleDict([
            (str(layer_id), vision_dense_h_to_4h)
            for layer_id, vision_dense_h_to_4h in zip(vision_layer_range, vision_dense_h_to_4h_list)
        ])
        self.vision_dense_4h_to_h_list = nn.ModuleDict([
            (str(layer_id), vision_dense_4h_to_h)
            for layer_id, vision_dense_4h_to_h in zip(vision_layer_range, vision_dense_4h_to_h_list)
        ])
        self.vision_gate_proj = nn.ModuleDict([
            (str(layer_id), gate_proj)
            for layer_id, gate_proj in zip(vision_layer_range, gate_proj_list)
        ])

    def mlp_forward(self, hidden_states, **kw_args):
        mixin_self = self
        self = self.transformer.layers[kw_args['layer_id']].mlp
        if "vision_expert_mask" in kw_args:
            vision_expert_mask = kw_args['vision_expert_mask']
        else:
            vision_expert_mask = None

        layer_id_key = str(int(kw_args['layer_id']))

        if kw_args['layer_id'] in mixin_self.vision_layer_range and (vision_expert_mask is not None) and vision_expert_mask.any():
            vision_dense_h_to_4h = mixin_self.vision_dense_h_to_4h_list[layer_id_key]
            vision_dense_4h_to_h = mixin_self.vision_dense_4h_to_h_list[layer_id_key]
            vision_gate_proj = mixin_self.vision_gate_proj[layer_id_key]
            output = torch.empty(hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device)

            language_hidden_state = hidden_states[~vision_expert_mask.bool()]
            language_intermediate_parallel = self.activation_func(mixin_self.gate_proj[kw_args['layer_id']](language_hidden_state)) * self.dense_h_to_4h(language_hidden_state)
            output[~vision_expert_mask.bool()] = self.dense_4h_to_h(language_intermediate_parallel)  # language_output

            vision_hidden_state = hidden_states[vision_expert_mask.bool()]
            vision_intermediate_parallel = vision_dense_h_to_4h(vision_hidden_state)
            gate_output = vision_gate_proj(vision_hidden_state)

            vision_intermediate_parallel *= self.activation_func(gate_output)
            output[vision_expert_mask.bool()] = vision_dense_4h_to_h(vision_intermediate_parallel)  # vision_output
        else:
            intermediate_parallel = self.activation_func(mixin_self.gate_proj[kw_args['layer_id']](hidden_states)) * self.dense_h_to_4h(hidden_states)
            output = self.dense_4h_to_h(intermediate_parallel)

        return output.contiguous()

    def copy_param(self):
        with torch.no_grad():
            for i in self.vision_layer_range:
                self.vision_gate_proj[str(i)].weight.data.copy_(self.gate_proj[i].weight.data)
                self.vision_dense_4h_to_h_list[str(i)].weight.data.copy_(self.transformer.layers[i].mlp.dense_4h_to_h.weight.data)
                self.vision_dense_h_to_4h_list[str(i)].weight.data.copy_(self.transformer.layers[i].mlp.dense_h_to_4h.weight.data)

from sat.mpu import get_model_parallel_world_size
from sat.mpu.utils import divide
from sat.model.position_embedding.triton_rotary_embeddings import FastRotaryEmbedding

class LlamaVisionExpertAttnMixin(BaseMixin):
    def __init__(
        self,
        hidden_size,
        num_layers=28,
        num_vision_layers=0,
        use_vision_expert=True,
        vision_layer_range=None,
        params_dtype=torch.float,
        transformer_layers=None,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.transformer = transformer_layers
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_vision_expert = use_vision_expert

        if vision_layer_range is None:
            vision_layer_range = list(range(min(num_vision_layers, num_layers)))
        self.vision_layer_range = vision_layer_range

        if use_vision_expert:
            self.vision_q_proj = nn.ModuleDict({
                str(i): nn.Linear(hidden_size, hidden_size, bias=True).to(device=device, dtype=params_dtype)
                for i in vision_layer_range
            })
            self.vision_k_proj = nn.ModuleDict({
                str(i): nn.Linear(hidden_size, hidden_size // 8, bias=True).to(device=device, dtype=params_dtype)
                for i in vision_layer_range
            })
            self.vision_v_proj = nn.ModuleDict({
                str(i): nn.Linear(hidden_size, hidden_size // 8, bias=True).to(device=device, dtype=params_dtype)
                for i in vision_layer_range
            })
            self.vision_o_proj = nn.ModuleDict({
                str(i): nn.Linear(hidden_size, hidden_size, bias=False).to(device=device, dtype=params_dtype)
                for i in vision_layer_range
            })

    def replace_attention_forward(self, vision_expert_mask):
        def make_attn_forward(layer_id):
            def new_forward(self_attn, hidden_states, attention_mask=None, **kwargs):
                vision_mask = vision_expert_mask.bool()
                lang_mask = ~vision_mask
                vision_h = hidden_states[vision_mask]
                lang_h = hidden_states[lang_mask]

                # Projections
                q = torch.empty_like(hidden_states)
                k = torch.empty(hidden_states.size(0), hidden_states.size(1), self_attn.k_proj.out_features, device=hidden_states.device, dtype=hidden_states.dtype)
                v = torch.empty_like(k)

                # Language tokens use original projection
                q[lang_mask] = self_attn.q_proj(lang_h)
                k[lang_mask] = self_attn.k_proj(lang_h)
                v[lang_mask] = self_attn.v_proj(lang_h)

                # Vision tokens use expert projection
                q[layer_id][vision_mask] = self.vision_q_proj[str(layer_id)](vision_h)
                k[layer_id][vision_mask] = self.vision_k_proj[str(layer_id)](vision_h)
                v[layer_id][vision_mask] = self.vision_v_proj[str(layer_id)](vision_h)

                # Compute attention scores and apply attention
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
                if attention_mask is not None:
                    attn_scores = attn_scores + attention_mask
                attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
                attn_output = torch.matmul(attn_probs, v)

                # Output projection
                o_proj = self_attn.o_proj if str(layer_id) not in self.vision_o_proj else self.vision_o_proj[str(layer_id)]
                return o_proj(attn_output)

            return new_forward

        # Inject new forward into each attention layer
        for i in self.vision_layer_range:
            block = self.transformer[i]
            block.self_attn.forward = make_attn_forward(i).__get__(block.self_attn, block.self_attn.__class__)

    def copy_param(self):
        with torch.no_grad():
            for i in self.vision_layer_range:
                self.vision_q_proj[str(i)].weight.data.copy_(self.transformer[i].self_attn.q_proj.weight.data)
                self.vision_k_proj[str(i)].weight.data.copy_(self.transformer[i].self_attn.k_proj.weight.data)
                self.vision_v_proj[str(i)].weight.data.copy_(self.transformer[i].self_attn.v_proj.weight.data)
                self.vision_o_proj[str(i)].weight.data.copy_(self.transformer[i].self_attn.o_proj.weight.data)
